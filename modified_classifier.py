import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
import cv2  # OpenCV for image ops
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image


# =============== 1) Load Your Model ===================
model = MobileNetV2(weights="imagenet")


# =============== 2) GRAD-CAM Utilities ===================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for the predicted (or specified) class.

    Args:
        img_array (np.array): Preprocessed input image array of shape (1, 224, 224, 3).
        model (tf.keras.Model): Trained model (e.g., MobileNetV2).
        last_conv_layer_name (str): Name of the last convolutional layer in the model.
        pred_index (int): Optional specific class index for which to generate Grad-CAM.
                          If None, uses the top predicted class.

    Returns:
        heatmap (np.array): A 2D numpy array (values normalized between 0 and 1)
                            that can be used to locate important regions.
    """
    # 1. Create sub-model for the final conv outputs + final predictions
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    # 2. Record ops for gradient calculation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # 3. Compute gradients of the target class w.r.t. last conv layer
    grads = tape.gradient(class_channel, conv_outputs)

    # 4. Global average pool the gradients across spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Weight the feature maps
    conv_outputs = conv_outputs[0]  # shape: (H, W, channels)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. ReLU & normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def display_gradcam(img_path, heatmap, alpha=0.4):
    """
    Displays Grad-CAM heatmap over original image.

    Args:
        img_path (str): Path to the original image file.
        heatmap (np.array): 2D array (shape: (224, 224)) of normalized heatmap values.
        alpha (float): Transparency for heatmap overlay.
    """
    # 1. Load original image as array
    img = image.load_img(img_path)
    img = np.array(img)

    # 2. Resize heatmap to match original dimensions
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # 3. Convert heatmap to RGB
    heatmap_255 = np.uint8(255 * heatmap_resized)
    heatmap_color = cm.jet(heatmap_255)[:, :, :3]  # shape: (H, W, 3) in [0, 1]

    # 4. Create overlay
    overlay = heatmap_color * alpha + img / 255.0 * (1 - alpha)

    # 5. Display original + heatmap side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("Original Image")
    ax1.imshow(img)
    ax1.axis("off")

    ax2.set_title("Grad-CAM")
    ax2.imshow(overlay)
    ax2.axis("off")
    plt.tight_layout()
    plt.show()


# =============== 3) Occlusion Functions ===================
def occlude_black_box(img, mask):
    """
    Apply a solid black box covering the bounding box of the mask region.
    img: Original image array (H, W, 3)
    mask: Binary mask (H, W), 1 where we want to occlude.
    """
    # Find bounding box
    y_indices, x_indices = np.where(mask == 1)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return img  # No occlusion needed

    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    occluded_img = img.copy()
    occluded_img[y_min:y_max+1, x_min:x_max+1, :] = 0  # Fill with black
    return occluded_img


def occlude_blur(img, mask):
    """
    Apply Gaussian blur to the bounding box of the mask region.
    """
    y_indices, x_indices = np.where(mask == 1)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return img

    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    occluded_img = img.copy()
    region = occluded_img[y_min:y_max+1, x_min:x_max+1, :]

    # Blur the region
    region_blurred = cv2.GaussianBlur(region, ksize=(21, 21), sigmaX=0)
    occluded_img[y_min:y_max+1, x_min:x_max+1, :] = region_blurred
    return occluded_img


def occlude_pixelate(img, mask):
    """
    Pixelate (mosaic) the bounding box of the mask region.
    """
    y_indices, x_indices = np.where(mask == 1)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return img

    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    occluded_img = img.copy()
    region = occluded_img[y_min:y_max+1, x_min:x_max+1, :]

    # Choose scale factor for pixelation
    scale_factor = 0.1
    small_h = max(1, int(region.shape[0] * scale_factor))
    small_w = max(1, int(region.shape[1] * scale_factor))

    # Downsample & then upsample
    region_small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    region_pixelated = cv2.resize(region_small, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    occluded_img[y_min:y_max+1, x_min:x_max+1, :] = region_pixelated
    return occluded_img


def apply_occlusions(img_path, heatmap, threshold=0.5):
    """
    1. Threshold the Grad-CAM heatmap to create a binary mask
    2. Apply three types of occlusions (black box, blur, pixelation)
       to the bounding box region of the mask.
    3. Display all results side-by-side.
    """
    # 1. Load original
    original_img = image.load_img(img_path)
    original_img = np.array(original_img)

    # 2. Resize heatmap to match original
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # 3. Create binary mask: 1 where heatmap > threshold
    mask = np.where(heatmap_resized > threshold, 1, 0).astype(np.uint8)

    # 4. Apply each occlusion
    occluded_black = occlude_black_box(original_img, mask)
    occluded_blur = occlude_blur(original_img, mask)
    occluded_pixel = occlude_pixelate(original_img, mask)

    # 5. Display: Original + black box + blur + pixelation
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    ax_list = axes.ravel()

    titles = ["Original", "Black Box", "Blurred", "Pixelated"]
    images = [original_img, occluded_black, occluded_blur, occluded_pixel]

    for ax, img_disp, title in zip(ax_list, images, titles):
        ax.set_title(title)
        ax.imshow(img_disp)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# =============== 4) Main Classification + Grad-CAM ===============
def classify_image(image_path):
    """Classify an image, display predictions and Grad-CAM, then apply occlusions."""
    try:
        # 1. Load & preprocess for model
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # 2. Predict
        predictions = model(img_array, training=False)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # 3. Print top 3
        print("Top-3 Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            print(f"{i + 1}: {label} ({score:.2f})")

        # 4. Get index of top predicted class
        pred_index = np.argmax(predictions[0])

        # 5. Generate Grad-CAM
        heatmap = make_gradcam_heatmap(
            img_array,
            model,
            last_conv_layer_name="Conv_1", 
            pred_index=pred_index
        )

        # 6. Display Grad-CAM
        display_gradcam(image_path, heatmap, alpha=0.4)

        # 7. Apply occlusions
        apply_occlusions(image_path, heatmap, threshold=0.5)

    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    image_path = "test1.jpg"
    classify_image(image_path)

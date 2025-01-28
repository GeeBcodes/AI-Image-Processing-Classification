from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

def apply_filters(image_path):
    """
    Load an image, then apply four filters:
      1. Edge Detection
      2. Sharpen
      3. Emboss
      4. Black & White (Grayscale)
    and save each result as a new file.
    """
    # 1. Load the original image
    img = Image.open(image_path)

    # 2. Apply Edge Detection
    edges_img = img.filter(ImageFilter.FIND_EDGES)
    edges_output_path = "edges_result.jpg"
    edges_img.save(edges_output_path)
    print(f"Saved edge detection result to {edges_output_path}")

    # 3. Apply Sharpen
    sharpened_img = img.filter(ImageFilter.SHARPEN)
    sharpen_output_path = "sharpen_result.jpg"
    sharpened_img.save(sharpen_output_path)
    print(f"Saved sharpen result to {sharpen_output_path}")

    # 4. Apply Emboss
    embossed_img = img.filter(ImageFilter.EMBOSS)
    emboss_output_path = "emboss_result.jpg"
    embossed_img.save(emboss_output_path)
    print(f"Saved emboss result to {emboss_output_path}")
    
    # 5. Convert to Black & White (Grayscale)
    bw_img = img.convert("L")   # "L" mode indicates grayscale
    bw_output_path = "bw_result.jpg"
    bw_img.save(bw_output_path)
    print(f"Saved black & white result to {bw_output_path}")


if __name__ == "__main__":
    # Example image file path
    input_image_path = "test1.jpg"
    apply_filters(input_image_path)

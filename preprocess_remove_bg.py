from PIL import Image
import numpy as np
import os

# Function to remove white background from an image
def remove_white_background(input_path, output_path, threshold=240):
    # Open the image and ensure it has an alpha channel (RGBA)
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)

    # Split the RGBA channels
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    # Create a mask for pixels that are close to white
    white_mask = (r > threshold) & (g > threshold) & (b > threshold)

    # Set the alpha channel of white pixels to 0 (transparent)
    data[white_mask, 3] = 0

    # Save the updated image
    Image.fromarray(data).save(output_path)
    print(f"Saved: {output_path}")

# Function to process all images in a folder (including subfolders)
def batch_process_recursive(input_root, output_root, threshold=240):
    for root, _, files in os.walk(input_root):
        for filename in files:
            print(f"Found file: {filename}")
            ext = filename.lower().split('.')[-1]
            # Only process image files
            if ext in ["png", "jpg", "jpeg"]:
                # Keep the same folder structure in the output
                relative_path = os.path.relpath(root, input_root)
                in_path = os.path.join(root, filename)
                out_dir = os.path.join(output_root, relative_path)
                os.makedirs(out_dir, exist_ok=True)

                # Save all processed images as PNG
                name_wo_ext = os.path.splitext(filename)[0]
                out_path = os.path.join(out_dir, name_wo_ext + ".png")

                print(f"Processing: {in_path} -> {out_path}")
                remove_white_background(in_path, out_path, threshold)

# Main entry point
if __name__ == "__main__":
    print("Starting background removal...")
    input_folder = "images/raw-images/carrot"   # Input folder with original images
    output_folder = "images/train-images/carrot" # Output folder for processed images
    batch_process_recursive(input_folder, output_folder)

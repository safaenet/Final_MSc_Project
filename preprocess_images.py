from PIL import Image
import numpy as np
import os

def remove_white_background(input_path, output_path, threshold=240):
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)

    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    white_mask = (r > threshold) & (g > threshold) & (b > threshold)
    data[white_mask, 3] = 0  # make white-ish pixels transparent

    Image.fromarray(data).save(output_path)
    print(f"Saved: {output_path}")

def batch_process_recursive(input_root, output_root, threshold=240):
    for root, _, files in os.walk(input_root):
        for filename in files:
            print(f"Found file: {filename}")
            ext = filename.lower().split('.')[-1]
            if ext in ["png", "jpg", "jpeg"]:
                relative_path = os.path.relpath(root, input_root)
                in_path = os.path.join(root, filename)
                out_dir = os.path.join(output_root, relative_path)
                os.makedirs(out_dir, exist_ok=True)

                # Always save as PNG
                name_wo_ext = os.path.splitext(filename)[0]
                out_path = os.path.join(out_dir, name_wo_ext + ".png")

                print(f"Processing: {in_path} -> {out_path}")
                remove_white_background(in_path, out_path, threshold)

if __name__ == "__main__":
    print("Starting background removal...")
    input_folder = "images/raw-images"
    output_folder = "images/train-images"
    batch_process_recursive(input_folder, output_folder)

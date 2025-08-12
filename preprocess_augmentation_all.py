
# ------------------------------------------------------------
# Augmentation script for all classes
# This script creates augmented images by placing objects
# (e.g., fruits) from all class folders onto random backgrounds.
# ------------------------------------------------------------

import os
import random
from PIL import Image

# Parameters
fruit_root = "images/train-images"       # Folder containing one subfolder per class
background_dir = "images/backgrounds"    # Folder with background images
output_root = "images/train-images-augmented"  # Folder to save augmented images
num_per_image = 3  # Number of augmented images to create for each original

# Load all background images once, keeping transparency info (RGBA)
backgrounds = [
    Image.open(os.path.join(background_dir, f)).convert("RGBA") 
    for f in os.listdir(background_dir)
]

# Create main output folder if it does not exist
os.makedirs(output_root, exist_ok=True)

# Go through each class folder in the dataset
for fruit_class in os.listdir(fruit_root):
    class_input_dir = os.path.join(fruit_root, fruit_class)
    class_output_dir = os.path.join(output_root, fruit_class)

    # Create output subfolder for this class if it doesn’t exist
    os.makedirs(class_output_dir, exist_ok=True)

    # Go through each image in the class folder
    for fruit_file in os.listdir(class_input_dir):
        fruit_path = os.path.join(class_input_dir, fruit_file)
        fruit_img = Image.open(fruit_path).convert("RGBA")  # Keep transparency for blending

        # Create several augmented versions
        for i in range(num_per_image):
            # Pick a random background and make a copy
            bg = random.choice(backgrounds).copy()

            # Random position (ensuring the object fits fully on background)
            x_offset = random.randint(0, max(0, bg.width - fruit_img.width))
            y_offset = random.randint(0, max(0, bg.height - fruit_img.height))

            # Paste object image on background using alpha channel mask
            bg.paste(fruit_img, (x_offset, y_offset), fruit_img)

            # Save output as JPEG (no alpha channel)
            base_name = os.path.splitext(fruit_file)[0]
            out_path = os.path.join(class_output_dir, f"{base_name}_{i}.jpg")
            bg.convert("RGB").save(out_path, quality=90)
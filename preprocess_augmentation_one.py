
# ------------------------------------------------------------
# Augmentation script to create extra images for one class
# This adds variety to the dataset by placing the object image
# onto different random backgrounds.
# ------------------------------------------------------------

import os
import random
from PIL import Image

# Parameters
class_name = "carrot"  # Name of the class to process
class_root = f"images/support-images/{class_name}"  # Folder containing original object images
background_dir = "images/backgrounds"               # Folder with background images
output_root = f"images/support-images-augmented/{class_name}"  # Folder to save augmented images
num_per_image = 3  # Number of augmented images to generate per original image

# Load all background images once, as RGBA (keep transparency info)
backgrounds = [
    Image.open(os.path.join(background_dir, f)).convert("RGBA") 
    for f in os.listdir(background_dir)
]

# Create output folder if it doesn't exist
os.makedirs(output_root, exist_ok=True)

# Loop through each object image in the chosen class
for fruit_file in os.listdir(class_root):
    fruit_path = os.path.join(class_root, fruit_file)
    fruit_img = Image.open(fruit_path).convert("RGBA")  # Keep transparency for blending

    # Create several augmented versions of each object image
    for i in range(num_per_image):
        # Pick a random background and make a copy
        bg = random.choice(backgrounds).copy()

        # Random position (ensuring object fits within background)
        x_offset = random.randint(0, max(0, bg.width - fruit_img.width))
        y_offset = random.randint(0, max(0, bg.height - fruit_img.height))

        # Paste the object onto the background using its alpha channel as a mask
        bg.paste(fruit_img, (x_offset, y_offset), fruit_img)

        # Build output file path
        base_name = os.path.splitext(fruit_file)[0]
        out_path = os.path.join(output_root, f"{base_name}_{i}.jpg")

        # Save as JPEG (remove alpha channel)
        bg.convert("RGB").save(out_path, quality=90)

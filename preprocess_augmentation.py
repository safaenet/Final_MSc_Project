# # Augmentation code to generate augmented images of all classes.
# import os
# import random
# from PIL import Image

# # Parameters
# fruit_root = "images/train-images"
# background_dir = "images/backgrounds"
# output_root = "images/train-images-augmented"
# num_per_image = 3  # Generate 3 augmented images per fruit image

# # Load backgrounds once
# backgrounds = [Image.open(os.path.join(background_dir, f)).convert("RGBA") 
#             for f in os.listdir(background_dir)]

# # Create output root folder
# os.makedirs(output_root, exist_ok=True)

# # Process each fruit class
# for fruit_class in os.listdir(fruit_root):
#     class_input_dir = os.path.join(fruit_root, fruit_class)
#     class_output_dir = os.path.join(output_root, fruit_class)
#     # os.makedirs(class_output_dir, exist_ok=True)

#     for fruit_file in os.listdir(class_input_dir):
#         fruit_path = os.path.join(class_input_dir, fruit_file)
#         fruit_img = Image.open(fruit_path).convert("RGBA")

#         for i in range(num_per_image):
#             bg = random.choice(backgrounds).copy()

#             # Optional: random offset (ensure fruit fits)
#             max_offset = 20
#             x_offset = random.randint(0, max(0, bg.width - fruit_img.width))
#             y_offset = random.randint(0, max(0, bg.height - fruit_img.height))

#             # Paste fruit on background
#             bg.paste(fruit_img, (x_offset, y_offset), fruit_img)

#             # Save
#             base_name = os.path.splitext(fruit_file)[0]
#             out_path = os.path.join(class_output_dir, f"{base_name}_{i}.jpg")
#             bg.convert("RGB").save(out_path, quality=90)

# Augmentation code to generate augmented images of of one specific class.
import os
import random
from PIL import Image

# Parameters
class_name = "carrot"  # <-- only process this class
class_root = f"images/support-images/{class_name}"
background_dir = "images/backgrounds"
output_root = f"images/support-images-augmented/{class_name}"
num_per_image = 3  # Generate 3 augmented images per fruit image

# Load backgrounds once
backgrounds = [
    Image.open(os.path.join(background_dir, f)).convert("RGBA") 
    for f in os.listdir(background_dir)
]

# Create output folder
os.makedirs(output_root, exist_ok=True)

# Process each fruit image in the specified class
for fruit_file in os.listdir(class_root):
    fruit_path = os.path.join(class_root, fruit_file)
    fruit_img = Image.open(fruit_path).convert("RGBA")

    for i in range(num_per_image):
        bg = random.choice(backgrounds).copy()

        # Optional: random offset (ensure fruit fits)
        x_offset = random.randint(0, max(0, bg.width - fruit_img.width))
        y_offset = random.randint(0, max(0, bg.height - fruit_img.height))

        # Paste fruit on background
        bg.paste(fruit_img, (x_offset, y_offset), fruit_img)

        # Save output
        base_name = os.path.splitext(fruit_file)[0]
        out_path = os.path.join(output_root, f"{base_name}_{i}.jpg")
        bg.convert("RGB").save(out_path, quality=90)

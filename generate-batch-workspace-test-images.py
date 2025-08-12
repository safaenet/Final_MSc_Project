import os
import cv2
import random
import pandas as pd
import numpy as np
from datetime import datetime

# CONFIGURATION
CLASS_FOLDER = 'images/train-images/'       # Input folder containing class subfolders
OUTPUT_FOLDER = 'images/workspace-images/'  # Output folder for generated workspace images
CSV_FOLDER = 'images/ground-truth-csv/'     # Output folder for CSV files with ground-truth coordinates
WORKSPACE_SIZE = (800, 600)                 # Workspace image size (width, height)
NUM_CLASSES = 10                            # Number of objects to place per workspace image

# Create output folders if they do not already exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CSV_FOLDER, exist_ok=True)

# Collect a list of tuples (image_path, class_name) from the class folders
class_images = []
for class_name in os.listdir(CLASS_FOLDER):
    class_dir = os.path.join(CLASS_FOLDER, class_name)
    if os.path.isdir(class_dir):
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                class_images.append((os.path.join(class_dir, file), class_name))

# Function to place an object image into the workspace at (x, y)
def place_object(workspace, class_img, x, y):
    h, w = class_img.shape[:2]
    # Skip if the object would go outside the workspace
    if y + h > workspace.shape[0] or x + w > workspace.shape[1]:
        return workspace
    # If image has an alpha channel, use it for blending
    alpha = class_img[:, :, 3] / 255.0 if class_img.shape[2] == 4 else np.ones((h, w))
    for c in range(3):
        workspace[y:y+h, x:x+w, c] = (1 - alpha) * workspace[y:y+h, x:x+w, c] + alpha * class_img[:, :, c]
    return workspace

# Function to generate one workspace image and save its CSV
def generate_workspace_image(index=0):
    # Start with a white background
    workspace = np.ones((WORKSPACE_SIZE[1], WORKSPACE_SIZE[0], 3), dtype=np.uint8) * 255
    records = []         # Stores class and center coordinates for CSV
    used_positions = []  # Stores placed object bounding boxes to prevent overlaps

    for _ in range(NUM_CLASSES):
        # Pick a random object from the dataset
        class_path, class_name = random.choice(class_images)
        class_img = cv2.imread(class_path, cv2.IMREAD_UNCHANGED)
        if class_img is None:
            continue

        h, w = class_img.shape[:2]
        max_x = WORKSPACE_SIZE[0] - w
        max_y = WORKSPACE_SIZE[1] - h

        # Try up to 50 times to place object without overlap
        for attempt in range(50):
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            box = (x, y, x + w, y + h)
            overlap = any(
                (x1 < box[2] and x2 > box[0] and y1 < box[3] and y2 > box[1])
                for x1, y1, x2, y2 in used_positions
            )
            if not overlap:
                used_positions.append(box)
                break
        else:
            # If overlap-free placement fails after 50 tries, skip this object
            continue

        # Place object in the workspace
        workspace = place_object(workspace, class_img, x, y)

        # Save center coordinates for CSV
        center_x = x + w // 2
        center_y = y + h // 2
        records.append({'class': class_name, 'x': center_x, 'y': center_y})

    # Create a timestamp for unique file names
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    image_path = os.path.join(OUTPUT_FOLDER, f'workspace_{timestamp}.png')
    csv_path = os.path.join(CSV_FOLDER, f'workspace_{timestamp}.csv')

    # Save workspace image and ground truth CSV
    cv2.imwrite(image_path, workspace)
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f'✅ Saved: {image_path}, {csv_path}')

# Generate multiple workspace images
if __name__ == '__main__':
    NUMBER_OF_IMAGES = 50  # Total images to generate
    for i in range(NUMBER_OF_IMAGES):
        generate_workspace_image(i)

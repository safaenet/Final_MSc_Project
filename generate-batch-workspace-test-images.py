import os
import cv2
import random
import pandas as pd
import numpy as np
from datetime import datetime

# CONFIGURATION
CLASS_FOLDER = 'images/train-images/'       # <== Your input folder
OUTPUT_FOLDER = 'images/workspace-images/'  # <== Output images
CSV_FOLDER = 'images/ground-truth-csv/'     # <== Output CSVs
WORKSPACE_SIZE = (800, 600)                 # Width, Height
NUM_CLASSES = 10                              # Number of classes per image

# Create output folders if they don't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CSV_FOLDER, exist_ok=True)

# Get list of (image_path, class_name) tuples
class_images = []
for class_name in os.listdir(CLASS_FOLDER):
    class_dir = os.path.join(CLASS_FOLDER, class_name)
    if os.path.isdir(class_dir):
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                class_images.append((os.path.join(class_dir, file), class_name))

# Function to overlay class image onto workspace at a position
def place_object(workspace, class_img, x, y):
    h, w = class_img.shape[:2]
    if y + h > workspace.shape[0] or x + w > workspace.shape[1]:
        return workspace  # Skip if out of bounds
    alpha = class_img[:, :, 3] / 255.0 if class_img.shape[2] == 4 else np.ones((h, w))
    for c in range(3):
        workspace[y:y+h, x:x+w, c] = (1 - alpha) * workspace[y:y+h, x:x+w, c] + alpha * class_img[:, :, c]
    return workspace

# Generate one workspace image
def generate_workspace_image(index=0):
    workspace = np.ones((WORKSPACE_SIZE[1], WORKSPACE_SIZE[0], 3), dtype=np.uint8) * 255
    records = []
    used_positions = []

    for _ in range(NUM_CLASSES):
        class_path, class_name = random.choice(class_images)
        class_img = cv2.imread(class_path, cv2.IMREAD_UNCHANGED)
        if class_img is None:
            continue

        h, w = class_img.shape[:2]
        max_x = WORKSPACE_SIZE[0] - w
        max_y = WORKSPACE_SIZE[1] - h

        for attempt in range(50):  # Avoid overlapping
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            box = (x, y, x + w, y + h)
            overlap = any((x1 < box[2] and x2 > box[0] and y1 < box[3] and y2 > box[1]) for x1, y1, x2, y2 in used_positions)
            if not overlap:
                used_positions.append(box)
                break
        else:
            continue

        workspace = place_object(workspace, class_img, x, y)
        center_x = x + w // 2
        center_y = y + h // 2
        records.append({'class': class_name, 'x': center_x, 'y': center_y})

    # Save image and CSV
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    image_path = os.path.join(OUTPUT_FOLDER, f'workspace_{timestamp}.png')
    csv_path = os.path.join(CSV_FOLDER, f'workspace_{timestamp}.csv')

    cv2.imwrite(image_path, workspace)
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f'✅ Saved: {image_path}, {csv_path}')

# Generate multiple images
if __name__ == '__main__':
    NUMBER_OF_IMAGES = 50  # Change this number as needed
    for i in range(NUMBER_OF_IMAGES):  # Change this number as needed
        generate_workspace_image(i)

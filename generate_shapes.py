import cv2
import os
import numpy as np

output_path = "images/synthetic-shapes"
classes = ["circle", "rectangle"]
img_size = 100
num_images_per_class = 440

os.makedirs(output_path, exist_ok=True)

def rotate_image(img, angle):
    center = (img.shape[1] // 2, img.shape[0] // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

for cls in classes:
    cls_path = os.path.join(output_path, cls)
    os.makedirs(cls_path, exist_ok=True)
    
    for i in range(num_images_per_class):
        img = np.zeros((img_size, img_size, 4), dtype=np.uint8)

        if cls == "circle":
            center = (np.random.randint(30, 70), np.random.randint(30, 70))
            radius = np.random.randint(10, 20)
            color = (0, 0, 255, 255)  # Red
            cv2.circle(img, center, radius, color, -1)

        elif cls == "rectangle":
            pt1 = (np.random.randint(10, 40), np.random.randint(10, 40))
            pt2 = (pt1[0] + np.random.randint(20, 40), pt1[1] + np.random.randint(20, 40))
            color = (255, 0, 0, 255)  # Blue
            cv2.rectangle(img, pt1, pt2, color, -1)

        angle = np.random.randint(0, 360)
        img_rotated = rotate_image(img, angle)
        
        filename = os.path.join(cls_path, f"img{i}.png")
        cv2.imwrite(filename, img_rotated)

print("Generated 440 rotated transparent images per class.")

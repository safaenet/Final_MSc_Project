import cv2
import os
import numpy as np

output_path = "synthetic-shapes"
classes = ["circle", "rectangle"]
img_size = 84
num_images_per_class = 50

os.makedirs(output_path, exist_ok=True)

for cls in classes:
    cls_path = os.path.join(output_path, cls)
    os.makedirs(cls_path, exist_ok=True)
    
    for i in range(num_images_per_class):
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        if cls == "circle":
            center = (np.random.randint(20, 64), np.random.randint(20, 64))
            radius = np.random.randint(10, 20)
            color = (0, 0, 255)
            cv2.circle(img, center, radius, color, -1)
        
        elif cls == "rectangle":
            pt1 = (np.random.randint(10, 40), np.random.randint(10, 40))
            pt2 = (pt1[0] + np.random.randint(20, 40), pt1[1] + np.random.randint(20, 40))
            color = (255, 0, 0)
            cv2.rectangle(img, pt1, pt2, color, -1)
        
        filename = os.path.join(cls_path, f"img{i}.png")
        cv2.imwrite(filename, img)

print("Synthetic shape dataset generated.")

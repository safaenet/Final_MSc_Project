import cv2
import numpy as np
from PIL import Image

# Load image
img = cv2.imread("images/banana_7.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to HSV for better color segmentation
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define a color range for banana (yellow-ish)
lower_yellow = np.array([15, 40, 40])
upper_yellow = np.array([35, 255, 255])

# Create mask where yellow parts (banana) are white
banana_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Clean up the mask
kernel = np.ones((3, 3), np.uint8)
banana_mask = cv2.morphologyEx(banana_mask, cv2.MORPH_OPEN, kernel, iterations=2)
banana_mask = cv2.morphologyEx(banana_mask, cv2.MORPH_DILATE, kernel, iterations=1)

# Create alpha channel: 255 for banana, 0 for background
alpha = np.where(banana_mask == 255, 255, 0).astype(np.uint8)
rgba = cv2.merge((img_rgb, alpha))

# Find contours to crop tightly around banana
contours, _ = cv2.findContours(banana_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped = rgba[y:y+h, x:x+w]
else:
    cropped = rgba  # fallback

# Resize to 100x100
resized = cv2.resize(cropped, (100, 100), interpolation=cv2.INTER_AREA)

# Save as PNG with transparency
cv2.imwrite("images/banana_cleaned_100x100.png", cv2.cvtColor(resized, cv2.COLOR_RGBA2BGRA))

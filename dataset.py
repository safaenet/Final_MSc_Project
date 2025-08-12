import os
from PIL import Image
import glob
import torch
from torchvision import transforms

# A custom dataset class for loading shape images with different scales
class ShapesDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        # Store the transform function (if any)
        self.transform = transform
        # List to store image file paths, labels, and scale sizes
        self.data = []
        # List to store labels (not directly used here but kept for structure)
        self.labels = []
        # Dictionary to map class names to numeric labels
        self.label_to_index = {}
        # List of scales to resize images before cropping
        self.scale_list = list(range(120, 251, 10))  # 120, 130, ..., 250 pixels

        # Go through each class folder inside the root directory
        class_folders = sorted(os.listdir(root_dir))
        for idx, class_name in enumerate(class_folders):
            # Assign a numeric label to each class
            self.label_to_index[class_name] = idx
            class_path = os.path.join(root_dir, class_name)
            # Search for image files with these extensions
            for ext in ('*.png', '*.jpg', '*.jpeg'):
                for img_path in glob.glob(os.path.join(class_path, ext)):
                    # For each image, store multiple entries with different scales
                    for scale in self.scale_list:
                        self.data.append((img_path, idx, scale))

    def __len__(self):
        # Return the total number of items in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get the file path, label, and scale for this index
        img_path, label, scale = self.data[idx]
        # Open the image and make sure it's in RGB format
        image = Image.open(img_path).convert("RGB")

        # Resize the image to the given scale
        scaled = transforms.Resize((scale, scale))(image)
        # Crop the center part of the image to 84x84 pixels
        cropped = transforms.CenterCrop(84)(scaled)

        # If a transform function was provided, apply it
        if self.transform:
            cropped = self.transform(cropped)

        # Return the processed image and its label
        return cropped, label

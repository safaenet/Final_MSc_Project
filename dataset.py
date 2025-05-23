import os
from PIL import Image
from torch.utils.data import Dataset
import glob
import torch
from torchvision import transforms

class ShapesDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        self.label_to_index = {}
        self.scale_list = list(range(120, 251, 10))  # e.g. 120, 130, ..., 250

        class_folders = sorted(os.listdir(root_dir))
        for idx, class_name in enumerate(class_folders):
            self.label_to_index[class_name] = idx
            class_path = os.path.join(root_dir, class_name)
            # for img_path in glob.glob(os.path.join(class_path, '*.png')):
            #     for scale in self.scale_list:
            #         self.data.append((img_path, idx, scale))
            for ext in ('*.png', '*.jpg', '*.jpeg'):
                for img_path in glob.glob(os.path.join(class_path, ext)):
                    for scale in self.scale_list:
                        self.data.append((img_path, idx, scale))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, scale = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        scaled = transforms.Resize((scale, scale))(image)
        cropped = transforms.CenterCrop(84)(scaled)

        if self.transform:
            cropped = self.transform(cropped)

        return cropped, label

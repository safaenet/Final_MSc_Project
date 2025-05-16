import os
from PIL import Image
from torch.utils.data import Dataset
import glob

class ShapesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        self.label_to_index = {}
        class_folders = sorted(os.listdir(root_dir))
        
        for idx, class_name in enumerate(class_folders):
            self.label_to_index[class_name] = idx
            class_path = os.path.join(root_dir, class_name)
            for img_path in glob.glob(os.path.join(class_path, '*.png')):
                self.data.append(img_path)
                self.labels.append(idx)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

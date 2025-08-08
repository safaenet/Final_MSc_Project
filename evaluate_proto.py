import os
import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from detect_object_multiscale import detect_object_multiscale
import torch
from torchvision import transforms
from prototypical_net import ConvNet
from PIL import Image
import random

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patch_sizes = [84, 96, 112, 128]
stride = 20
class_list = ["apple", "kiwi", "carrot"]
workspace_dir = "images/workspace-images"
ground_truth_dir = "images/ground-truth-csv"
distance_threshold = 30

# --- Load model ---
model = ConvNet().to(device)
model.load_state_dict(torch.load("saved_models/prototypical_net_model.pth", map_location=device))
model.eval()

# --- Define transform ---
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

all_true = []
all_pred = []

# --- Evaluate across images and classes ---
for img_file in os.listdir(workspace_dir):
    if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(workspace_dir, img_file)
    gt_csv_path = os.path.join(ground_truth_dir, os.path.splitext(img_file)[0] + ".csv")

    if not os.path.exists(gt_csv_path):
        print(f"[!] Ground truth missing for {img_file}, skipping.")
        continue

    workspace = cv2.imread(image_path)
    workspace = cv2.cvtColor(workspace, cv2.COLOR_BGR2RGB)
    gt_df = pd.read_csv(gt_csv_path)

    for class_name in class_list:
        support_dir = os.path.join("images/support-images-augmented", class_name)
        support_paths = [
            os.path.join(support_dir, f)
            for f in os.listdir(support_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if len(support_paths) == 0:
            continue

        selected_paths = random.sample(support_paths, k=min(10, len(support_paths)))

        embeddings = []
        for path in selected_paths:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(tensor)
                embeddings.append(emb)
        support_embedding = torch.mean(torch.stack(embeddings), dim=0)

        detected_list = detect_object_multiscale(
            model=model,
            support_path_list=selected_paths,
            workspace_path=image_path,
            patch_sizes=patch_sizes,
            stride=stride,
            device=device,
            distance=4
        )

        pred_points = [d['location'] for d in detected_list]
        gt_points = gt_df[gt_df['class'] == class_name][['x', 'y']].values.tolist()
        matched = set()

        for gx, gy in gt_points:
            found = False
            for i, (px, py) in enumerate(pred_points):
                if i in matched:
                    continue
                dist = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                if dist < distance_threshold:
                    all_true.append(class_name)
                    all_pred.append(class_name)
                    matched.add(i)
                    found = True
                    break
            if not found:
                all_true.append(class_name)
                all_pred.append("none")

        for i in range(len(pred_points)):
            if i not in matched:
                all_true.append("none")
                all_pred.append(class_name)

# --- Final Report ---
print("\n=== Final Evaluation Across All Classes ===")
print(classification_report(all_true, all_pred, zero_division=0))

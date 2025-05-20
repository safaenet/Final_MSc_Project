import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

def detect_object_multiscale(model, support_path, workspace_path, patch_sizes=[84, 96, 112, 128], stride=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load and embed support image
    support_img = Image.open(support_path).convert("RGB")
    support_tensor = transform(support_img).unsqueeze(0).to(device)
    with torch.no_grad():
        support_embedding = model(support_tensor)

    # Load workspace image
    workspace = cv2.imread(workspace_path)
    workspace = cv2.cvtColor(workspace, cv2.COLOR_BGR2RGB)
    h, w, _ = workspace.shape

    min_dist = float('inf')
    best_pos = None
    best_patch_size = None

    # Loop over multiple patch sizes
    for patch_size in patch_sizes:
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = workspace[y:y+patch_size, x:x+patch_size]
                patch_resized = cv2.resize(patch, (84, 84))
                patch_pil = Image.fromarray(patch_resized)
                patch_tensor = transform(patch_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    patch_embedding = model(patch_tensor)

                dist = torch.norm(support_embedding - patch_embedding).item()
                if dist < min_dist:
                    min_dist = dist
                    best_pos = (x + patch_size // 2, y + patch_size // 2)
                    best_patch_size = patch_size

    return {
        'location': best_pos,
        'distance': min_dist,
        'patch_size': best_patch_size,
        'estimated_object_size': best_patch_size
    }

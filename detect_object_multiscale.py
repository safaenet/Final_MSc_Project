import torch
import cv2
from PIL import Image
from torchvision import transforms
from detect_overlap import is_overlap

# This function searches for objects in a workspace image using multiple patch sizes.
# It compares each patch's embedding with the average embedding of the support images.
def detect_object_multiscale(
    model, 
    support_path_list, 
    workspace_path, 
    patch_sizes=[84, 96, 112, 128], 
    stride=10, 
    device='cuda' if torch.cuda.is_available() else 'cpu', 
    distance=2, 
    threshold=0.5, 
    confidence=0.5
):
    # Set the model to evaluation mode (no training updates)
    model.eval()

    # Define how images are preprocessed before going into the model
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Store embeddings for each support image
    embeddings = []
    for path in support_path_list:
        img = Image.open(path).convert("RGB")  # Open image and make sure it is RGB
        tensor = transform(img).unsqueeze(0).to(device)  # Transform and add batch dimension
        with torch.no_grad():  # No gradient calculation
            emb = model(tensor)  # Get embedding from model
            embeddings.append(emb)

    # Compute the average embedding (prototype) for the support images
    support_embedding = torch.mean(torch.stack(embeddings), dim=0)

    # Load the workspace image with OpenCV
    workspace = cv2.imread(workspace_path)
    workspace = cv2.cvtColor(workspace, cv2.COLOR_BGR2RGB)  # Convert to RGB
    h, w, _ = workspace.shape

    matches = []  # Store all detected matches

    # Loop through each patch size
    for patch_size in patch_sizes:
        # Slide over the workspace image with the given stride
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                # Crop patch from workspace
                patch = workspace[y:y+patch_size, x:x+patch_size]
                # Resize patch to model input size
                patch_resized = cv2.resize(patch, (84, 84))
                patch_pil = Image.fromarray(patch_resized)
                patch_tensor = transform(patch_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    patch_embedding = model(patch_tensor)

                # Calculate distance between support and patch embeddings
                dist = torch.norm(support_embedding - patch_embedding).item()

                # If distance is below threshold, consider it a match
                if dist < distance:
                    match_center = (x + patch_size // 2, y + patch_size // 2)
                    matches.append({
                        'location': match_center,  # Center coordinates of patch
                        'distance': dist,          # Embedding distance
                        'patch_size': patch_size,  # Size of the patch
                        'confidence': 1 / (1 + dist)  # Confidence score
                    })

    final_matches = []  # Store matches after filtering overlaps
    # Sort matches by smallest distance (most similar first)
    for match in sorted(matches, key=lambda x: x['distance']):
        keep = True
        for kept in final_matches:
            # Remove match if confidence is too low or overlaps with an existing match
            if match['confidence'] < confidence or is_overlap(
                (match['location'][0], match['location'][1], match['patch_size']),
                (kept['location'][0], kept['location'][1], kept['patch_size']),
                threshold=threshold
            ):
                keep = False
                break
        if keep:
            final_matches.append(match)

    return final_matches  # Return the list of detected objects

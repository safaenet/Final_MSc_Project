import os
import random
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from prototypical_net import ConvNet

def get_class_images(data_root, class_name, extensions=('.png', '.jpg', '.jpeg')):
    folder = os.path.join(data_root, class_name)
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(extensions)
    ]

def sample_two_way_episode(data_root, primary_class, neutral_class="rectangle", k_shot=5, n_query=5, image_size=100, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    def load_samples(cls):
        paths = get_class_images(data_root, cls)
        samples = random.sample(paths, k_shot + n_query)
        support = [transform(Image.open(p).convert("RGB")) for p in samples[:k_shot]]
        query = [transform(Image.open(p).convert("RGB")) for p in samples[k_shot:]]
        return support, query

    support_A, query_A = load_samples(primary_class)
    support_B, query_B = load_samples(neutral_class)

    support = torch.stack(support_A + support_B).to(device)
    query = torch.stack(query_A + query_B).to(device)
    labels = torch.tensor([0]*n_query + [1]*n_query).to(device)

    return support, query, labels

def sample_one_way_episode(data_root, classes, k_shot=5, n_query=5, image_size=100, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    selected_class = random.choice(classes)
    image_paths = get_class_images(data_root, selected_class)
    total_needed = k_shot + n_query

    if len(image_paths) < total_needed:
        raise ValueError(f"Not enough images in class {selected_class}. Found {len(image_paths)}, need {total_needed}")

    sampled = random.sample(image_paths, total_needed)
    support_paths = sampled[:k_shot]
    query_paths = sampled[k_shot:]

    support_images = []
    for path in support_paths:
        img = Image.open(path).convert("RGB")
        support_images.append(transform(img))

    query_images = []
    for path in query_paths:
        img = Image.open(path).convert("RGB")
        query_images.append(transform(img))

    support_tensor = torch.stack(support_images).to(device)
    query_tensor = torch.stack(query_images).to(device)
    query_labels = torch.zeros(n_query, dtype=torch.long).to(device)

    return {
        'class': selected_class,
        'support': support_tensor,
        'query': query_tensor,
        'query_labels': query_labels
    }

def train_prototypical(model, optimizer, data_root, classes, device, k_shot=5, n_query=5, num_episodes=1000):
    model.train()

    for episode in tqdm(range(num_episodes)):
        target_class = random.choice(classes)
        support, query, labels = sample_two_way_episode(
            data_root, primary_class=target_class, neutral_class="rectangle",
            k_shot=k_shot, n_query=n_query, device=device
        )

        embeddings = model(support)
        prototype_A = embeddings[:k_shot].mean(dim=0, keepdim=True)
        prototype_B = embeddings[k_shot:].mean(dim=0, keepdim=True)
        prototypes = torch.cat([prototype_A, prototype_B], dim=0)

        query_embeddings = model(query)
        dists = torch.cdist(query_embeddings, prototypes)  # [num_query, 2]
        logits = -dists

        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"[Episode {episode}] Loss: {loss.item():.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data_root = "images/train-images"
    classes = ["apple", "circle"]
    train_prototypical(model, optimizer, data_root, classes, device)
    # Save model at end of training
    torch.save(model.state_dict(), "saved_models/protonet_class_independent.pth")
    print("✅ Model saved to saved_models/protonet_class_independent.pth")

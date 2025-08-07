import torch
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from learn2learn.data import MetaDataset, TaskDataset
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
from dataset import ShapesDataset  # your custom dataset
from prototypical_net import ConvNet  # your model class

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_root = "images/augmented-images"
model_path = "saved_models/normal_model.pth"
n_ways = 2
k_shot = 1
k_query = 5
episodes = 50

# --- Dataset ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
dataset = ShapesDataset(test_root, transform=transform)
meta_dataset = MetaDataset(dataset)

taskset = TaskDataset(
    meta_dataset,
    task_transforms=[
        NWays(meta_dataset, n=n_ways),
        KShots(meta_dataset, k=k_shot + k_query),
        LoadData(meta_dataset),
        RemapLabels(meta_dataset),
    ],
    num_tasks=episodes,
)

# --- Load Model ---
model = ConvNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Evaluation Loop ---
accuracies, precisions, recalls, f1s = [], [], [], []

for task in taskset:
    data, labels = task
    data, labels = data.to(device), labels.to(device)

    support_data = []
    support_labels = []
    query_data = []
    query_labels = []

    for class_idx in range(n_ways):
        class_mask = labels == class_idx
        class_indices = torch.nonzero(class_mask).squeeze()
        support_idx = class_indices[:k_shot]
        query_idx = class_indices[k_shot:k_shot + k_query]

        support_data.append(data[support_idx])
        support_labels.append(labels[support_idx])
        query_data.append(data[query_idx])
        query_labels.append(labels[query_idx])

    support_data = torch.cat(support_data, dim=0)
    support_labels = torch.cat(support_labels, dim=0)
    query_data = torch.cat(query_data, dim=0)
    query_labels = torch.cat(query_labels, dim=0)

    # Get embeddings
    embeddings = model(torch.cat([support_data, query_data], dim=0))
    support_embeddings = embeddings[:len(support_data)]
    query_embeddings = embeddings[len(support_data):]

    # Compute prototypes
    prototypes = []
    for i in range(n_ways):
        cls_mask = support_labels == i
        cls_embeds = support_embeddings[cls_mask]
        proto = cls_embeds.mean(dim=0)
        prototypes.append(proto)
    prototypes = torch.stack(prototypes)

    # Classify query samples
    dists = torch.cdist(query_embeddings, prototypes)
    preds = torch.argmin(dists, dim=1)

    # Metrics
    y_true = query_labels.cpu().numpy()
    y_pred = preds.cpu().numpy()
    accuracies.append(accuracy_score(y_true, y_pred))
    precisions.append(precision_score(y_true, y_pred, average='macro', zero_division=0))
    recalls.append(recall_score(y_true, y_pred, average='macro', zero_division=0))
    f1s.append(f1_score(y_true, y_pred, average='macro', zero_division=0))

# --- Results ---
print("\n=== 2-Way {}-Shot Evaluation over {} Episodes ===".format(k_shot, episodes))
print("Accuracy:  {:.2f}%".format(100 * sum(accuracies) / episodes))
print("Precision: {:.2f}%".format(100 * sum(precisions) / episodes))
print("Recall:    {:.2f}%".format(100 * sum(recalls) / episodes))
print("F1-Score:  {:.2f}%".format(100 * sum(f1s) / episodes))

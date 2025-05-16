import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, output_size=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, output_size, 3, padding=1), nn.BatchNorm2d(output_size), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

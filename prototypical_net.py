import torch.nn as nn

# A small convolutional neural network (ConvNet) used as the feature extractor
# for a Prototypical Network in few-shot learning.
class ConvNet(nn.Module):
    def __init__(self, output_size=64):
        super().__init__()
        # Sequential layers for feature extraction
        self.features = nn.Sequential(
            # First block: Conv -> BatchNorm -> ReLU -> MaxPool
            nn.Conv2d(3, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(2),

            # Second block
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(2),

            # Third block
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(2),

            # Fourth block: output feature maps reduced to output_size channels
            nn.Conv2d(64, output_size, 3, padding=1), 
            nn.BatchNorm2d(output_size), 
            nn.ReLU(), 
            nn.AdaptiveAvgPool2d(1)  # Output is (output_size, 1, 1)
        )

    def forward(self, x):
        # Pass input through the feature extractor
        x = self.features(x)
        # Flatten output to shape (batch_size, output_size)
        return x.view(x.size(0), -1)

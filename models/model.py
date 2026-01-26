import torch.nn as nn
import torchvision.models as models


class ResNet18_MCDropout(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0.5):
        super(ResNet18_MCDropout, self).__init__()
        self.backbone = models.resnet18(weights=None)
        # Adapt for 1-channel grayscale input
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()  # Remove maxpool for 28x28 images

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

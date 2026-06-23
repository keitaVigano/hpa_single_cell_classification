import torchvision.models as models
import torch.nn as nn
import torch

class HPAResNet(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()

        # carica ResNet50 pretrained
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # sostituisci il primo layer: 3 canali → 4 canali
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        with torch.no_grad():
            backbone.conv1.weight[:, :3] = old_conv.weight
            backbone.conv1.weight[:, 3] = old_conv.weight.mean(dim=1)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)

        self.model = backbone

    def forward(self, x):
        return self.model(x)
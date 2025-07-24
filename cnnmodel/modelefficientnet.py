import torch
import torch.nn as nn
from torchvision import models

class ImageOnlyEfficientNetV2S(nn.Module):
    def __init__(self, num_classes=5):
        super(ImageOnlyEfficientNetV2S, self).__init__()
        self.model = models.efficientnet_v2_s(pretrained=True)  # ใช้ EfficientNetV2-S
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)  # ปรับ FC layer

    def forward(self, x):
        return self.model(x)

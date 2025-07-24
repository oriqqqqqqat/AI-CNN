import torch
import torch.nn as nn
from torchvision import models



class ImageOnlyResNet50(nn.Module):
    def __init__(self, num_classes=5):
        super(ImageOnlyResNet50, self).__init__()
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)

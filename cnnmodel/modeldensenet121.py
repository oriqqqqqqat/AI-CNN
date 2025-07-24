import torch
import torch.nn as nn
from torchvision import models

class ImageOnlyDenseNet121(nn.Module):
    def __init__(self, num_classes=5):
        super(ImageOnlyDenseNet121, self).__init__()
        self.model = models.densenet121(pretrained=True)  # ใช้ DenseNet121
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)  # ปรับ FC layer

    def forward(self, x):
        return self.model(x)

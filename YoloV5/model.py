import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class YOLOv5WithResNet50(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv5WithResNet50, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.conv_head = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.backbone(x)
        output = self.conv_head(features)
        output = self.sigmoid(output)
        return output

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        intersection = (predictions * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        return dice_loss

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.cross_entropy_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        dice_loss = self.dice_loss(predictions, targets)
        ce_loss = self.cross_entropy_loss(predictions, targets)
        total_loss = dice_loss + ce_loss
        return total_loss

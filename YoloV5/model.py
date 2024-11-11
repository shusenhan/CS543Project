import torch
import torch.nn as nn
from torchvision.models import resnet50
# 确保YOLOv5的正确安装并导入，若没有安装YOLOv5库，可使用其他替代方案
# from yolov5 import YOLOv5  # 您可以根据需要导入合适的YOLOv5模块

class YOLOv5WithResNet50(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv5WithResNet50, self).__init__()
        # 使用预训练的ResNet-50作为backbone
        self.backbone = resnet50(pretrained=True)
        # 去掉ResNet-50的最后一个全连接层，保留卷积特征层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # YOLOv5的head部分，用于分割任务
        # self.yolo_head = YOLOv5(num_classes=num_classes)  # 请确认YOLOv5的安装
        # 暂时注释掉YOLOv5部分，可以根据安装情况替换此处

        # 使用Sigmoid激活函数，将输出映射到概率空间
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 使用ResNet-50提取特征
        features = self.backbone(x)
        # 将特征传入YOLOv5的head
        # output = self.yolo_head(features)  # 根据需要调用YOLOv5
        output = features  # 替换为features，仅供测试，实际应用中使用YOLOv5 head
        output = self.sigmoid(output)  # 应用Sigmoid激活函数
        return output

# 定义Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (predictions * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        return dice_loss

# 定义组合损失函数，包括Dice Loss和Cross-Entropy Loss
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()  # 使用Dice Loss
        self.cross_entropy_loss = nn.CrossEntropyLoss()  # 使用Cross-Entropy Loss

    def forward(self, predictions, targets):
        dice_loss = self.dice_loss(predictions, targets)
        ce_loss = self.cross_entropy_loss(predictions, targets)
        total_loss = dice_loss + ce_loss  # 组合损失
        return total_loss

# 测试模型结构
if __name__ == "__main__":
    # 假设每个图像包含2个分类
    model = YOLOv5WithResNet50(num_classes=2)

    # 创建一个随机输入，模拟单张640x640的RGB图像
    test_input = torch.randn(1, 3, 640, 640)

    # 前向传播，获取输出
    output = model(test_input)
    print("模型输出形状:", output.shape)

import os
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import LaneDetectionDataset  # 从dataset.py中导入自定义数据集
from torchvision import transforms

# 获取当前工作目录的根路径
CURRENT_DIR = Path(__file__).resolve().parent
DATA_ROOT = CURRENT_DIR / 'CULane'  # 假设数据集放在当前目录的CULane文件夹下

# 训练和验证的子目录路径
TRAIN_IMAGES_DIR = DATA_ROOT / 'driver_37_30frame/train'
TRAIN_LABELS_DIR = DATA_ROOT / 'laneseg_label_w16/train'
VAL_IMAGES_DIR = DATA_ROOT / 'driver_24_32frame/val'
VAL_LABELS_DIR = DATA_ROOT / 'laneseg_label_w16/val'

# 数据集实例化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640))
])

train_dataset = LaneDetectionDataset(str(TRAIN_IMAGES_DIR), str(TRAIN_LABELS_DIR), transform=transform)
val_dataset = LaneDetectionDataset(str(VAL_IMAGES_DIR), str(VAL_LABELS_DIR), transform=transform)

# 数据加载器配置
BATCH_SIZE = 8  # 批大小
NUM_WORKERS = 4  # 子进程数

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

if __name__ == "__main__":
    # 测试数据加载器是否工作正常
    for images, labels in train_loader:
        print("Image batch shape:", images.size())
        print("Label batch shape:", len(labels))
        break
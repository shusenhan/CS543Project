import os
from torch.utils.data import DataLoader
from dataset import LaneDetectionDataset  # 从dataset.py中导入自定义数据集

# 数据根目录
DATA_ROOT = '/path/to/dataset'  # 在此处设置数据集的根目录

# 训练和验证的子目录
TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, 'train/images')
TRAIN_LABELS_DIR = os.path.join(DATA_ROOT, 'train/labels')
VAL_IMAGES_DIR = os.path.join(DATA_ROOT, 'val/images')
VAL_LABELS_DIR = os.path.join(DATA_ROOT, 'val/labels')

# 数据集实例化
train_dataset = LaneDetectionDataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, transform=None)
val_dataset = LaneDetectionDataset(VAL_IMAGES_DIR, VAL_LABELS_DIR, transform=None)

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

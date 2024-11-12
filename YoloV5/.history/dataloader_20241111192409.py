from pathlib import Path
from torch.utils.data import DataLoader
from dataset import LaneDetectionDataset
from torchvision import transforms

# 获取当前工作目录
current_dir = Path(__file__).resolve().parent

# 数据路径配置
TRAIN_IMAGES_DIR = current_dir / 'driver_37_30frame_frames'
TRAIN_LABELS_DIR = current_dir / 'masks'
VAL_IMAGES_DIR = current_dir / 'driver_24_32frame_frames'  # 假设有类似的验证集
VAL_LABELS_DIR = current_dir / 'masks'

# 定义图像转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640))
])

# 实例化数据集
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
        print("Label batch shape:", labels.size())
        break
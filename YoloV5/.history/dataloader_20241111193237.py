from pathlib import Path
from torch.utils.data import DataLoader
from dataset import LaneDetectionDataset
from torchvision import transforms

# 获取当前工作目录
current_dir = Path(__file__).resolve().parent

# 设置图像和标注的目录路径
TRAIN_IMAGES_DIR = current_dir / 'CULane' / 'driver_24_32frame'
TRAIN_LABELS_DIR = current_dir / 'CULane' / 'driver_24_32frame'
VAL_IMAGES_DIR = current_dir / 'CULane' / 'driver_24_32frame'  # 假设验证集结构相同
VAL_LABELS_DIR = current_dir / 'CULane' / 'driver_24_32frame'

# 定义图像转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640))
])

# 实例化数据集和数据加载器
train_dataset = LaneDetectionDataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, transform=transform)
val_dataset = LaneDetectionDataset(VAL_IMAGES_DIR, VAL_LABELS_DIR, transform=transform)

BATCH_SIZE = 8
NUM_WORKERS = 4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
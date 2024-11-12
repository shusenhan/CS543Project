from pathlib import Path
from torch.utils.data import DataLoader
from dataset import LaneDetectionDataset
from torchvision import transforms

# 获取当前工作目录
current_dir = Path(__file__).resolve().parent

# 设置训练和验证数据的根目录路径
TRAIN_ROOT_DIR = current_dir / 'CULane' / 'driver_24_32frame'
VAL_ROOT_DIR = current_dir / 'CULane' / 'driver_37_30frame'

# 定义图像转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640))
])

# 实例化训练和验证数据集（仅传入两个参数）
train_dataset = LaneDetectionDataset(TRAIN_ROOT_DIR, transform=transform)
val_dataset = LaneDetectionDataset(VAL_ROOT_DIR, transform=transform)

# 数据加载器配置
BATCH_SIZE = 8
NUM_WORKERS = 4

# 创建训练和验证数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
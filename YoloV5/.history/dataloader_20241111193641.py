from pathlib import Path
from torch.utils.data import DataLoader
from dataset import LaneDetectionDataset
from torchvision import transforms

# 获取当前工作目录
current_dir = Path(__file__).resolve().parent

# 设置包含所有子文件夹的主目录路径
DATA_ROOT_DIR = current_dir / 'CULane' / 'driver_24_32frame'

# 定义图像转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640))
])

# 实例化数据集和数据加载器
train_dataset = LaneDetectionDataset(DATA_ROOT_DIR, transform=transform)
BATCH_SIZE = 8
NUM_WORKERS = 4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)  # 使用相同的数据集进行验证
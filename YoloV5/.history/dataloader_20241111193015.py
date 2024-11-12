from pathlib import Path
from torch.utils.data import DataLoader
from dataset import LaneDetectionDataset
from torchvision import transforms

# 获取当前工作目录
current_dir = Path(__file__).resolve().parent

# 设置图像和标注的目录路径
TRAIN_IMAGES_DIR = current_dir / 'CULane' / 'driver_24_32frame'
TRAIN_LABELS_DIR = current_dir / 'CULane' / 'driver_24_32frame'

# 定义图像转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640))
])

# 实例化数据集
train_dataset = LaneDetectionDataset(str(TRAIN_IMAGES_DIR), str(TRAIN_LABELS_DIR), transform=transform)

# 数据加载器配置
BATCH_SIZE = 8
NUM_WORKERS = 4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

if __name__ == "__main__":
    # 测试数据加载器是否工作正常
    for images, masks in train_loader:
        print("Image batch shape:", images.size())
        print("Mask batch shape:", masks.size())
        break
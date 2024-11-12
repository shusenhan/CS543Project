from pathlib import Path
from torch.utils.data import DataLoader
from dataset import LaneDetectionDataset
from torchvision import transforms

# 获取当前工作目录
current_dir = Path(__file__).resolve().parent

# 数据路径配置
TRAIN_IMAGES_DIR = current_dir / 'driver_37_30frame_frames'
TRAIN_LABELS_DIR = current_dir / 'masks'
VAL_IMAGES_DIR = current_dir / 'driver_24_32frame_frames'
VAL_LABELS_DIR = current_dir / 'masks'

# 检查路径是否存在
if not TRAIN_IMAGES_DIR.exists() or not TRAIN_LABELS_DIR.exists():
    print(f"警告：训练图像目录 {TRAIN_IMAGES_DIR} 或标签目录 {TRAIN_LABELS_DIR} 不存在！请检查文件路径。")

if not VAL_IMAGES_DIR.exists() or not VAL_LABELS_DIR.exists():
    print(f"警告：验证图像目录 {VAL_IMAGES_DIR} 或标签目录 {VAL_LABELS_DIR} 不存在！请检查文件路径。")

# 定义图像转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640))
])

# 实例化数据集
train_dataset = LaneDetectionDataset(str(TRAIN_IMAGES_DIR), str(TRAIN_LABELS_DIR), transform=transform)
val_dataset = LaneDetectionDataset(str(VAL_IMAGES_DIR), str(VAL_LABELS_DIR), transform=transform)

# 数据加载器配置
BATCH_SIZE = 8
NUM_WORKERS = 4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

if __name__ == "__main__":
    # 测试数据加载器是否工作正常
    for images, labels in train_loader:
        print("Image batch shape:", images.size())
        print("Label batch shape:", labels.size())
        break
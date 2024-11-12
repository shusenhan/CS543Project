import os
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import glob

class LaneDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # 定义根目录，并使用Pathlib构建图像和掩膜的路径
        self.root_dir = Path(root_dir)
        self.images_path = self.root_dir / 'driver_xx_xxframe'
        self.masks_path = self.root_dir / 'laneseg_label_w16'
        self.image_files = glob.glob(str(self.images_path / '**' / '*.jpg'), recursive=True)  # 递归查找所有.jpg文件
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 获取图像路径和对应的掩膜路径
        image_path = Path(self.image_files[idx])
        mask_path = self.masks_path / image_path.parent.name / image_path.with_suffix('.png').name

        # 读取图像
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取掩膜
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # 应用转换
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# 设置转换（可调整）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640))
])

# 创建数据集和加载器实例
dataset = LaneDetectionDataset(root_dir='path/to/CULane', transform=transform)
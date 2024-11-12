from pathlib import Path
from torch.utils.data import Dataset
import cv2
import glob
import numpy as np
from torchvision import transforms

class LaneDetectionDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        self.images_path = Path(images_path)
        self.masks_path = Path(masks_path)
        self.image_files = glob.glob(str(self.images_path / '**' / '*.jpg'), recursive=True)
        
        # 调试：检查加载的图像文件数量
        if not self.image_files:
            print(f"警告：在 {images_path} 路径中未找到任何图像文件！请检查路径。")
        else:
            print(f"在 {images_path} 路径中找到了 {len(self.image_files)} 个图像文件。")

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

        # 检查掩膜是否存在
        if mask is None:
            print(f"警告：未找到 {mask_path} 对应的掩膜文件！")

        # 应用转换
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
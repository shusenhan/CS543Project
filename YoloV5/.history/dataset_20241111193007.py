from pathlib import Path
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms

class LaneDetectionDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images_path = Path(images_path)
        self.labels_path = Path(labels_path)
        
        # 获取所有图像文件路径，并确保标注文件也存在
        self.image_files = sorted([f for f in self.images_path.glob("*.jpg") if (self.labels_path / f.with_suffix(".lines.txt").name).exists()])
        
        if not self.image_files:
            print(f"警告：在 {images_path} 中未找到任何图像文件，或未找到对应的标注文件。")
        
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 获取图像文件路径
        image_path = self.image_files[idx]
        label_path = self.labels_path / image_path.with_suffix(".lines.txt").name

        # 读取图像
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取标注文件并生成掩膜
        mask = self.create_lane_mask(image.shape[:2], label_path)

        # 应用转换
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def create_lane_mask(self, image_shape, label_path):
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                points = [tuple(map(int, point.split(','))) for point in line.strip().split()]
                for i in range(1, len(points)):
                    cv2.line(mask, points[i - 1], points[i], 255, thickness=5)
        
        return mask
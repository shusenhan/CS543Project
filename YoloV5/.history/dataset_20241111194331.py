from pathlib import Path
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms

class LaneDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        
        # 获取所有子文件夹中的图像文件路径，并确保每个图像文件都有对应的标注文件
        self.image_files = sorted([f for f in self.root_dir.glob("**/*.jpg") if (f.parent / f.with_suffix(".lines.txt").name).exists()])
        
        # 输出找到的图像文件数量和部分示例文件
        if not self.image_files:
            print(f"警告：在 {root_dir} 中未找到任何图像文件，或未找到对应的标注文件。")
        else:
            print(f"在 {root_dir} 找到 {len(self.image_files)} 个图像文件。")
            print("示例文件路径：", self.image_files[:5])  # 输出前5个文件作为示例
        
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = image_path.parent / image_path.with_suffix(".lines.txt").name

        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法加载图像文件 {image_path}")
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
                try:
                    # 将点转换为整数坐标，并使用四舍五入来处理浮点数
                    points = [tuple(map(lambda x: int(round(float(x))), point.split(','))) for point in line.strip().split()]
                    
                    # 检查每个点的格式，确保它是(x, y)的形式
                    for point in points:
                        if len(point) != 2:
                            raise ValueError(f"Invalid point format: {point}")

                    # 绘制线段
                    for i in range(1, len(points)):
                        cv2.line(mask, points[i - 1], points[i], 255, thickness=5)
                except Exception as e:
                    print(f"Error processing line in {label_path}: {line}")
                    print(f"Exception: {e}")
    
        return mask
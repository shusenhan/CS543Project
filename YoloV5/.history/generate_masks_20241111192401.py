from pathlib import Path
import cv2
import numpy as np

# 获取当前工作目录
current_dir = Path(__file__).resolve().parent

# 假设您有车道线标注的坐标文件，定义标注路径和掩膜输出路径
annotation_dir = current_dir / 'annotations'  # 替换为实际标注文件路径
output_mask_dir = current_dir / 'masks'  # 掩膜文件保存目录
output_mask_dir.mkdir(parents=True, exist_ok=True)

# 创建掩膜函数
def create_lane_mask(image_shape, lane_annotations):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for lane in lane_annotations:
        points = [(x, y) for x, y in lane if x >= 0]  # 忽略无效点
        for i in range(1, len(points)):
            cv2.line(mask, points[i - 1], points[i], 255, thickness=5)  # 白色车道线
    return mask

# 示例：处理每个标注文件（假设为.txt文件）
for annotation_file in annotation_dir.glob("*.txt"):
    # 根据您的标注格式读取坐标信息并创建掩膜
    # TODO: 根据具体标注格式填写掩膜生成代码
    pass  # 将此处替换为具体的标注读取和掩膜生成逻辑
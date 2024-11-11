import json
import os

def convert_labelme_to_yolo(labelme_json, image_shape):
    with open(labelme_json) as f:
        data = json.load(f)
    
    img_height, img_width = image_shape[:2]
    yolo_annotations = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':  # 处理多边形标注
            points = np.array(shape['points'])
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            yolo_annotations.append(f"0 {x_center} {y_center} {width} {height}")
    
    return yolo_annotations

# 保存YOLO格式标注文件
def save_yolo_annotations(yolo_annotations, save_path):
    with open(save_path, 'w') as f:
        f.write('\n'.join(yolo_annotations))

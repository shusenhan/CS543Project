# video_to_images.py
import cv2
import os
from pathlib import Path

# 设置视频目录和输出图像目录
video_dir = Path('/Users/liaomaohong/Desktop/CS543Project/YoloV5/CULane/driver_37_30frame')  # 替换为实际路径
output_image_dir = Path('/Users/liaomaohong/Desktop/CS543Project/YoloV5/driver_37_30frame_frames')  # 图像输出目录

# 创建输出图像目录
output_image_dir.mkdir(parents=True, exist_ok=True)

# 视频转图像函数
def video_to_frames(video_path, output_dir, frame_rate=1):
    cap = cv2.VideoCapture(str(video_path))
    count = 0
    success, frame = cap.read()
    while success:
        if count % frame_rate == 0:
            frame_filename = f"{video_path.stem}_frame_{count}.jpg"
            cv2.imwrite(str(output_dir / frame_filename), frame)
        success, frame = cap.read()
        count += 1
    cap.release()

# 转换每个视频文件
for video_file in video_dir.glob("*.MP4"):
    video_to_frames(video_file, output_image_dir)
from pathlib import Path
import cv2

# 获取当前工作目录
current_dir = Path(__file__).resolve().parent

# 设置视频目录和输出图像目录为相对路径
video_dir = current_dir / 'CULane' / 'driver_37_30frame'  # 视频文件路径
output_image_dir = current_dir / 'driver_37_30frame_frames'  # 转换后的图像输出目录

# 创建输出图像目录（如果不存在）
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
from pathlib import Path
import cv2

# 获取当前工作目录
current_dir = Path(__file__).resolve().parent

# 设置视频目录和输出图像目录
video_dir = current_dir / 'CULane' / 'driver_37_30frame'
output_image_dir = current_dir / 'driver_37_30frame_frames'

# 创建输出图像目录
output_image_dir.mkdir(parents=True, exist_ok=True)

# 视频转图像函数
def video_to_frames(video_path, output_dir, frame_rate=1):
    print(f"正在处理视频文件: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return  # 跳过无法读取的视频

    count = 0
    success, frame = cap.read()
    while success:
        if count % frame_rate == 0:
            frame_filename = f"{video_path.stem}_frame_{count}.jpg"
            cv2.imwrite(str(output_dir / frame_filename), frame)
        success, frame = cap.read()
        count += 1
    cap.release()
    print(f"视频文件 {video_path} 处理完成")

# 转换每个视频文件
for video_file in video_dir.glob("*.MP4"):
    video_to_frames(video_file, output_image_dir)
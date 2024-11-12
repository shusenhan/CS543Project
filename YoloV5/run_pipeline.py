import subprocess
from pathlib import Path

# 获取当前脚本所在的目录
current_dir = Path(__file__).resolve().parent

# 定义每个脚本的路径
video_to_images_script = current_dir / 'video_to_images.py'
generate_masks_script = current_dir / 'generate_masks.py'
main_script = current_dir / 'main.py'

def run_script(script_path):
    """运行指定脚本并检查结果"""
    print(f"Running {script_path}...")
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"{script_path} completed successfully.\n")
        print(result.stdout)
    else:
        print(f"Error running {script_path}:\n{result.stderr}")
        raise RuntimeError(f"{script_path} failed")

# 依次运行脚本
try:
    run_script(video_to_images_script)   # 步骤 1：视频转图像
    run_script(generate_masks_script)    # 步骤 2：生成掩膜
    run_script(main_script)              # 步骤 3：运行训练
except RuntimeError as e:
    print(f"Pipeline failed: {e}")
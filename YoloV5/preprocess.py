import cv2
import numpy as np
from torchvision import transforms

def preprocess_image(image):
    # 调整图像大小为640x640
    image = cv2.resize(image, (640, 640))

    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 添加高斯噪声
    noise = np.random.normal(0, 25, gray_image.shape).astype(np.uint8)
    noisy_image = cv2.add(gray_image, noise)

    # 图像旋转（-45 到 45度之间）
    angle = np.random.uniform(-45, 45)
    rotation_matrix = cv2.getRotationMatrix2D((320, 320), angle, 1)
    rotated_image = cv2.warpAffine(noisy_image, rotation_matrix, (640, 640))

    # 将图像转换为Pytorch张量并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
    ])
    processed_image = transform(rotated_image)

    return processed_image


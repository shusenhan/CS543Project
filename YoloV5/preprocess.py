import cv2
import numpy as np
from torchvision import transforms

def preprocess_image(image):
    # Resize the image to 640x640
    image = cv2.resize(image, (640, 640))

    # Add Gaussian noise to the RGB image
    noise = np.random.normal(0, 25, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    # Rotate image (between -45 and 45 degrees)
    angle = np.random.uniform(-45, 45)
    rotation_matrix = cv2.getRotationMatrix2D((320, 320), angle, 1)
    rotated_image = cv2.warpAffine(noisy_image, rotation_matrix, (640, 640))

    # Convert the image to a PyTorch tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalization for 3 channels
    ])
    processed_image = transform(rotated_image)

    return processed_image

from pathlib import Path
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms

class LaneDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        
        # Get image file paths in all subdirectories and ensure each image file has a corresponding annotation file
        self.image_files = sorted([f for f in self.root_dir.glob("**/*.jpg") if (f.parent / f.with_suffix(".lines.txt").name).exists()])
        
        # Print the number of found image files and some sample files
        if not self.image_files:
            print(f"Warning: No image files found in {root_dir}, or corresponding annotation files are missing.")
        else:
            print(f"Found {len(self.image_files)} image files in {root_dir}.")
            print("Sample file paths:", self.image_files[:5])  # Output first 5 files as samples
        
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = image_path.parent / image_path.with_suffix(".lines.txt").name

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to load image file {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load annotation file and create mask
        mask = self.create_lane_mask(image.shape[:2], label_path)

        # Ensure mask size matches the image size
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)

        return image, mask

    def create_lane_mask(self, image_shape, label_path):
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip() == "":
                    continue
                try:
                    # Split line into a list of floats, then create (x, y) coordinate pairs
                    coords = list(map(float, line.strip().split()))
                    points = [(int(round(coords[i])), int(round(coords[i + 1]))) for i in range(0, len(coords), 2)]
                    
                    # Check if each point is in (x, y) format
                    for point in points:
                        if len(point) != 2:
                            raise ValueError(f"Invalid point format: {point}")

                    # Draw line segments
                    for i in range(1, len(points)):
                        cv2.line(mask, points[i - 1], points[i], 255, thickness=5)
                except Exception as e:
                    print(f"Error processing line in {label_path}: {line}")
                    print(f"Exception: {e}")
        
        return mask
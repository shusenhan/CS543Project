from pathlib import Path
from torch.utils.data import DataLoader
from dataset import LaneDetectionDataset
from torchvision import transforms

current_dir = Path(__file__).resolve().parent

TRAIN_ROOT_DIR = current_dir / 'CULane' / 'driver_24_32frame'
VAL_ROOT_DIR = current_dir / 'CULane' / 'driver_37_30frame'

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640))
])

# Instantiate training and validation datasets
train_dataset = LaneDetectionDataset(TRAIN_ROOT_DIR, transform=transform)
val_dataset = LaneDetectionDataset(VAL_ROOT_DIR, transform=transform)

# Data loader configurations
BATCH_SIZE = 8
NUM_WORKERS = 0

# Create training and validation data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
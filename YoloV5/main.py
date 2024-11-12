import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
from dataloader import train_loader, val_loader  # Assumes dataloader.py exists with train_loader and val_loader
from model import YOLOv5WithResNet50, CombinedLoss  # Import model and loss function

# Set up TensorBoard to record training progress
writer = SummaryWriter("runs/YOLOv5_lane_detection")

# Initialize model
num_classes = 3  # Background, left lane line, right lane line
model = YOLOv5WithResNet50(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = CombinedLoss()  # Use a combined loss function, assuming Dice and CrossEntropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training configuration
num_epochs = 20  # Adjust based on training requirements or experimental setup
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Resize labels to match the outputs' spatial dimensions and channel count
        labels_resized = F.interpolate(labels, size=outputs.shape[2:], mode="bilinear", align_corners=False)
        labels_resized = labels_resized.repeat(1, num_classes, 1, 1)
        
        loss = criterion(outputs, labels_resized)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
    writer.add_scalar('Train/Loss', avg_train_loss, epoch)

    # Validation
    model.eval()
    val_loss = 0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            labels_resized = F.interpolate(labels, size=outputs.shape[2:], mode="bilinear", align_corners=False)
            labels_resized = labels_resized.repeat(1, num_classes, 1, 1)
            loss = criterion(outputs, labels_resized)
            val_loss += loss.item()
            
            # Convert to CPU for performance evaluation
            all_targets.extend(labels_resized.cpu().numpy())
            all_outputs.extend(outputs.argmax(dim=1).cpu().numpy())  # Use argmax to get predicted class

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    writer.add_scalar('Validation/Loss', avg_val_loss, epoch)

    # Calculate precision, recall, and F1 score for validation set
    precision = precision_score(all_targets, all_outputs, average='macro')  # Use 'macro' average for multi-class
    recall = recall_score(all_targets, all_outputs, average='macro')
    f1 = f1_score(all_targets, all_outputs, average='macro')
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Write to TensorBoard
    writer.add_scalar('Validation/Precision', precision, epoch)
    writer.add_scalar('Validation/Recall', recall, epoch)
    writer.add_scalar('Validation/F1_Score', f1, epoch)

writer.close()
print("Training complete!")

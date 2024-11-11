import torch
from dataloader import train_loader, val_loader  # 导入数据加载器
from model import YOLOv5WithResNet50, CombinedLoss  # 导入模型和损失函数
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score

# 设置TensorBoard记录训练过程
writer = SummaryWriter("runs/YOLOv5_lane_detection")

# 初始化模型
num_classes = 3  # 背景、左车道线、右车道线
model = YOLOv5WithResNet50(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = CombinedLoss()  # 使用组合损失函数，假设结合了Dice和CrossEntropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练配置
num_epochs = 20  # 根据论文中的训练周期或实验要求调整
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
    writer.add_scalar('Train/Loss', avg_train_loss, epoch)

    # 验证过程
    model.eval()
    val_loss = 0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # 转换为CPU进行性能评估
            all_targets.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.argmax(dim=1).cpu().numpy())  # 使用argmax获取预测的类别

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    writer.add_scalar('Validation/Loss', avg_val_loss, epoch)

    # 计算验证集的精确度、召回率和F1分数
    precision = precision_score(all_targets, all_outputs, average='macro')  # 多分类问题使用'macro'平均
    recall = recall_score(all_targets, all_outputs, average='macro')
    f1 = f1_score(all_targets, all_outputs, average='macro')
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # 写入TensorBoard
    writer.add_scalar('Validation/Precision', precision, epoch)
    writer.add_scalar('Validation/Recall', recall, epoch)
    writer.add_scalar('Validation/F1_Score', f1, epoch)

writer.close()
print("训练完成！")

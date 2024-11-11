from torch.utils.data import Dataset
import glob

class LaneDetectionDataset(Dataset):
    def __init__(self, images_path, annotations_path, transform=None):
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.image_files = glob.glob(f"{images_path}/*.jpg")  # 假设图像为jpg格式
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        annotation_path = os.path.join(self.annotations_path, os.path.basename(image_path).replace('.jpg', '.txt'))
        
        # 读取和预处理图像
        image = cv2.imread(image_path)
        processed_image = preprocess_image(image)
        
        # 读取标注
        with open(annotation_path, 'r') as f:
            annotations = f.readlines()
        labels = []
        for annotation in annotations:
            label = list(map(float, annotation.strip().split()))
            labels.append(label)
        
        return processed_image, labels

import os
import torch
import numpy as np
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from model import create_ConvNext
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
# 使用与训练相同的配置
class Config:
    data_root = "Bra21"
    weight_dir = "weight"
    batch_size = 4  
    img_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['negative', 'positive']



transform = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class QuadModalDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        for label in Config.class_names:
            label_path = os.path.join(self.root, label)
            for case_id in os.listdir(os.path.join(label_path, 'FLAIR')):
                case_id = case_id.replace('FLAIR.jpg', '')
                paths = [
                    os.path.join(label_path, 'FLAIR', f"{case_id}FLAIR.jpg"),
                    os.path.join(label_path, 'T1w', f"{case_id}T1w.jpg"),
                    os.path.join(label_path, 'T1wCE', f"{case_id}T1wCE.jpg"),
                    os.path.join(label_path, 'T2w', f"{case_id}T2w.jpg")
                ]
                if all(os.path.exists(p) for p in paths):
                    self.samples.append({
                        'paths': paths,
                        'label': Config.class_names.index(label)
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        images = [transform(Image.open(p).convert('RGB')) for p in sample['paths']]
        return tuple(images), sample['label']


# 复用collate函数
def collate_fn(batch):
    images = [[] for _ in range(4)]
    labels = []
    for sample in batch:
        for i in range(4):
            images[i].append(sample[0][i])
        labels.append(sample[1])
    return [torch.stack(x) for x in images], torch.LongTensor(labels)


def evaluate():
    
    weight_files = [f for f in os.listdir(Config.weight_dir) if f.startswith("best weight")]#自己输入
    if not weight_files:
        raise FileNotFoundError("No best weight file found in weights directory")

   
    latest_weight = sorted(weight_files, key=lambda x: x.split("_")[2], reverse=True)[0]
    weight_path = os.path.join(Config.weight_dir, latest_weight)

   
    model = create_ConvNext('ConvNeXt_tiny', num_classes=2).to(Config.device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    
    test_dataset = QuadModalDataset(Config.data_root)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = [x.to(Config.device) for x in inputs]
            labels = labels.to(Config.device)

            outputs = model(*inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\nTest Results using {latest_weight}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    evaluate()

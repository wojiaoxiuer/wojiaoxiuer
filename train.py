import os
import torch
import time
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from model import create_ConvNext
from tqdm import tqdm

class Config:
    data_root = "Bra21"
    weight_dir = "weight"
    num_workers = 2
    batch_size =   
    num_epochs = 
    lr = 0.0001
    weight_decay = 0.01
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


def collate_fn(batch):
    images = [[] for _ in range(4)]
    labels = []
    for sample in batch:
        for i in range(4):
            images[i].append(sample[0][i])
        labels.append(sample[1])
    return [torch.stack(x) for x in images], torch.LongTensor(labels)


def train():
    os.makedirs(Config.weight_dir, exist_ok=True)
    torch.manual_seed(42)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    best_weight_file = f"best_{timestamp}.pth"

    
    model = create_ConvNext('ConvNeXt_tiny', num_classes=2).to(Config.device)

    
    full_dataset = QuadModalDataset(Config.data_root)
    loader = DataLoader(
        full_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=collate_fn
    )

    
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs)

    best_train_acc = 0.0

    for epoch in range(Config.num_epochs):
        #train
        model.train()
        train_preds, train_labels = [], []

        for inputs, labels in tqdm(loader, desc=f"Epoch {epoch + 1} [Train]"):
            inputs = [x.to(Config.device) for x in inputs]
            labels = labels.to(Config.device)

            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        
        train_acc = accuracy_score(train_labels, train_preds)
        train_precision = precision_score(train_labels, train_preds)
        train_recall = recall_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)

        #valid
        model.eval()
        valid_preds, valid_labels = [], []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = [x.to(Config.device) for x in inputs]
                labels = labels.to(Config.device)
                outputs = model(*inputs)
                preds = torch.argmax(outputs, dim=1)
                valid_preds.extend(preds.cpu().numpy())
                valid_labels.extend(labels.cpu().numpy())
                break  

        
        valid_acc = accuracy_score(valid_labels, valid_preds)
        valid_precision = precision_score(valid_labels, valid_preds)
        valid_recall = recall_score(valid_labels, valid_preds)
        valid_f1 = f1_score(valid_labels, valid_preds)

       
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(), os.path.join(Config.weight_dir, best_weight_file))
        print(f"new best model saved as:{best_weight_file}")
        
        scheduler.step()

        
        print(f"\nEpoch {epoch + 1}/{Config.num_epochs}")
        print(
            f"Train | Acc: {train_acc:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
        print(
            f"Valid | Acc: {valid_acc:.4f} | Precision: {valid_precision:.4f} | Recall: {valid_recall:.4f} | F1: {valid_f1:.4f}")

    print(f"\nTraining completed! Best model saved as: {best_weight_file}")


if __name__ == "__main__":
    train()

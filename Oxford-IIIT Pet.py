#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 10:09:59 2025

@author: tunahan
"""
# Oxford-IIIT Pet dataset non-transfer learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil


class TransformDataset(Dataset):
    def __init__(self, subset, transform):
        self.transform = transform
        self.subset = subset
    def __getitem__(self, index):
        image, label = self.subset[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.subset)

def filter_image(subset, min_size=200):
    valid_indices = []
    for i in range(len(dataset)):
        img, _ = subset.dataset[subset.indices[i]]
        if min(img.size) >= min_size:
            valid_indices.append(subset.indices[i])
    return torch.utils.data.Subset(subset.dataset, valid_indices)

data_dir = "./data"
for filename in os.listdir(data_dir):
    if not filename.endswith(".jpg"):
        continue
    label = "_".join(filename.split("_")[:-1])  # dosya adındaki sınıf ismi
    class_dir = os.path.join(data_dir, label)
    os.makedirs(class_dir, exist_ok=True)
    shutil.move(os.path.join(data_dir, filename), os.path.join(class_dir, filename))

dataset = datasets.ImageFolder(data_dir)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_subset, val_subset = random_split(dataset, [train_size, val_size])

train_dataset = TransformDataset(train_subset, train_transform)
val_dataset = TransformDataset(val_subset, val_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

class_names = dataset.classes
class_idx = dataset.class_to_idx

class Cnn(nn.Module):
    def __init__(self, num_classes=37):
        super(Cnn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(128 * 14 * 14, 256),
            nn.Dropout(0.3),
            nn.Linear(256, 37)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_model(model, dataloader, criterion, optimizer, scheduler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader):
        images, labels = images.to("cuda"), labels.to("cuda")
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        total += images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (labels == preds).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc
        
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, pred = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

epochs = 25
model = Cnn().to("cuda")
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.5, patience=2,
                                                 verbose=True)

def test_model(model, dataloader, dtype):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels

def test_model_softmax(model, dataloader):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            probs = nn.functional.softmax(outputs, 1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_probs, all_labels

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(epochs):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, scheduler)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    scheduler.step(val_loss)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")            
        

    
# %% TEST
plt.figure(figsize=(8,6))
plt.plot(train_losses, label="Train Loss", color='blue', linewidth=2)
plt.plot(val_losses, label="Validation Loss", color='orange', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(train_accs, label="Train Accs", color='blue', linewidth=2)
plt.plot(val_accs, label="Validation Accs", color='orange', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accs")
plt.title("Train vs Validation Accs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()        

train_preds, train_labels = test_model(model, train_loader, dtype = "train")
test_preds, test_labels = test_model(model, val_loader, dtype = "test")
test_probs, test_labels = test_model_softmax(model, val_loader)

roc_auc_macro = roc_auc_score(test_labels, test_probs, average="macro", multi_class="ovr")
roc_auc_micro = roc_auc_score(test_labels, test_probs, average="micro", multi_class="ovr")

print(f"Macro ROC AUC: {roc_auc_macro:.4f}")
print(f"Micro ROC AUC: {roc_auc_micro:.4f}")  
        
cm = confusion_matrix(test_labels, test_preds)
sns.heatmap(cm,cmap="Purples", xticklabels=class_names, yticklabels=class_names, annot=True)
plt.tight_layout()
plt.show()
        
print(classification_report(test_labels, test_preds))


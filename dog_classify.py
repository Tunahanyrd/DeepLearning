#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 12:44:40 2025

@author: tunahan
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import seaborn as sns
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
class TransformDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.subset[index]
        image = self.transform(image)
        return image,label
    
    def __len__(self):
        return len(self.subset)
    
def filter_image(subset, min_size = 200):
    valid_indices = []
    for i in range(len(subset)):
        img, _ = subset.dataset[subset.indices[i]]
        if min(img.size) >= min_size:
            valid_indices.append(subset.indices[i])
    return torch.utils.data.Subset(subset.dataset, valid_indices)
   
data_dir = "./data"
dataset = datasets.ImageFolder(data_dir)

train_transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(brightness=0.2, 
                                                            contrast=0.2, 
                                                            saturation=0.2),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean = [0.5]*3, std=[0.5]*3)
                                     ])
val_transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
                                    ])

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_subset, val_subset = random_split(dataset, [train_size, val_size])

train_dataset = TransformDataset(train_subset, train_transform)
val_dataset = TransformDataset(val_subset, val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names = dataset.classes
class_names = [name.split("-")[-1] for name in class_names]
print(f"Classes: {class_names[:10]}")
cidx = dataset.class_to_idx = {name.split("-")[-1]: idx for name, idx in dataset.class_to_idx.items()}

class Cnn(nn.Module):
    def __init__(self, num_classes=120):
        super(Cnn, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
    
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
    
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
    
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
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
        _, preds = torch.max(outputs, 1)
        correct = (preds == labels).sum().item()
        total += labels.size(0)
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
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct = (preds == labels).sum().item()
            total += labels.size(0)
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

model = Cnn().to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=3, verbose=True
    )

EPOCHS = 25
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(EPOCHS):
    
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, scheduler)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    scheduler.step(val_loss)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")    

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

train_preds, train_labels = test_model(model, train_loader, dtype = "train")
test_preds, test_labels = test_model(model, val_loader, dtype = "test")

def test_model_softmax(model, dataloader, dtype):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            probs = nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
            all_labels.extend(labels.cpu().numpy())
        all_probs = torch.cat(all_probs).numpy()
        
    return all_probs, all_labels

test_probs, test_labels = test_model_softmax(model, val_loader)

test_labels = label_binarize(test_labels, classes=list(range(120)))

roc_auc_macro = roc_auc_score(test_labels, test_probs, average='macro', multi_class='ovr')
roc_auc_micro = roc_auc_score(test_labels, test_probs, average='micro', multi_class='ovr')


print(f"Macro ROC AUC: {roc_auc_macro:.4f}")
print(f"Micro ROC AUC: {roc_auc_micro:.4f}")

class_index = 17 # totally exemplary
fpr, tpr, thresholds = roc_curve(test_labels[:, class_index], test_probs[:, class_index])

plt.plot(fpr, tpr, label=f"Class {class_index}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - Class {class_index}")
plt.legend()
plt.grid()
plt.show()

precision, recall, thresholds = precision_recall_curve(test_labels[:, class_index], test_probs[:, class_index])

plt.plot(recall, precision, label=f"Class {class_index}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve - Class {class_index}")
plt.grid()
plt.legend()
plt.show()

test_preds, test_labels = test_model(model, val_loader, dtype="test")
print(classification_report(test_labels, test_preds, target_names=class_names))

cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Purples", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# because 120 class is so complicated
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(xticks_rotation=90)

ap_macro = average_precision_score(test_labels, test_probs, average='macro')
ap_micro = average_precision_score(test_labels, test_probs, average='micro')

print(f"Macro Average Precision: {ap_macro:.4f}")
print(f"Micro Average Precision: {ap_micro:.4f}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 18:41:19 2025

@author: tunahan

orvile/digital-knee-x-ray-images
"""
import torch
torch.cuda.empty_cache()	
import torchvision.models as models
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class TransformDataset():
    def __init__(self, transform, subset, target_size = (300, 162)):
        self.transform = transform
        self.data = [(img, lbl) for img, lbl in subset if img.size == target_size]
    def __getitem__(self, index):
        image, label = self.data[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.data)
    
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ColorJitter(0.05,0.05,0.05,0.05),
    transforms.GaussianBlur(3, 0.3),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
    ])

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
    ])

dataset = datasets.ImageFolder("./data/Digital Knee X-ray Images/MedicalExpert-I/MedicalExpert-I")

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size

train_subset, test_subset = random_split(dataset, [train_size, test_size])

train_dataset = TransformDataset(train_transform, train_subset)
test_dataset = TransformDataset(test_transform, test_subset)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128)

model = models.resnet18(pretrained = True)
num_features = model.fc.in_features
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(num_features, 5)


model.to("cuda")
class_names = dataset.classes
class_idx = dataset.class_to_idx



precisions = [0.95, 0.93, 0.76, 0.81, 0.99] # ilk trainden gelen precision değerlerinin biraz değiştirilmesiyle yazıldı
weights = 1 / (np.array(precisions) + 1e-4) # eps
weights = weights / weights.sum()
weights_tensor = torch.tensor(weights, dtype=torch.float).to("cuda")
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

epochs = 25


def train_model(model, dataloader, criterion, optimizer, scheduler):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for images, labels in dataloader:
        images, labels = images.to("cuda"), labels.to("cuda")
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * labels.size(0)
        total += images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (labels == preds).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc        
        
def test_model(model, dataloader, criterion):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return all_preds, all_labels, all_probs

train_losses, val_losses = [], []
train_accs, val_accs = [], []
for epoch in range(epochs):

    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, scheduler)
    eval_loss, eval_acc = evaluate(model, test_loader, criterion)
    scheduler.step(eval_loss)
    
    train_losses.append(train_loss)
    val_losses.append(eval_loss)
    train_accs.append(train_acc)
    val_accs.append(eval_acc)
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {eval_loss:.4f}, Acc: {eval_acc:.4f}")
    print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")            
    
            
# %%
all_preds, all_labels, all_probs = test_model(model, test_loader, criterion)

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

cm, roc_auc, roc_curve = confusion_matrix, roc_auc_score, roc_curve      

cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
plt.figure()
sns.heatmap(cm, annot=True, linewidths=2, xticklabels=class_names, yticklabels=class_names, cmap="Purples")
plt.tight_layout()
plt.show()

report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
df = pd.DataFrame(report).transpose()
class_metrics = df.iloc[:, :4]
plt.figure(figsize=(10, len(class_metrics) * 0.6))
sns.heatmap(class_metrics, annot=True, fmt=".2f", cmap="Purples", linewidths=0.5)
plt.title("Classification Report Heatmap")
plt.ylabel("Classes")
plt.xlabel("Metrics")
plt.tight_layout()
plt.show()  

y_true_bin = label_binarize(all_labels, classes=list(range(len(class_names))))

auc_macro = roc_auc_score(y_true_bin, all_probs, average='macro', multi_class='ovr')
auc_micro = roc_auc_score(y_true_bin, all_probs, average='micro', multi_class='ovr')

print(f"Macro ROC AUC: {auc_macro:.3f}") # her sınıf için auc ortalaması
print(f"Micro ROC AUC: {auc_micro:.3f}") # tüm metrikler tek havuzda toplanır yani genel başarı

def plot_roc_auc(y_true_bin, y_score, class_names):
    n_classes = len(class_names)
    
    plt.figure(figsize=(n_classes * 4, 4))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        plt.subplot(1, n_classes, i + 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.title(f"ROC - {class_names[i]}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

def plot_prc(y_true_bin, y_score, class_names):
    n_classes = len(class_names)
    plt.figure(figsize=(n_classes * 4, 4))
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_score[:, i])
        plt.subplot(1, n_classes, i + 1)
        plt.plot(recall, precision, color='blue', lw=2, label=f"AP = {ap:.2f}")
        plt.title(f"PRC - {class_names[i]}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()
# %% FİNE TUNE
dataset_2 = datasets.ImageFolder("./data/Digital Knee X-ray Images/MedicalExpert-II/MedicalExpert-II")
train_size = int(len(dataset_2) * 0.8)
test_size = len(dataset_2) - train_size

train_subset, test_subset = random_split(dataset_2, [train_size, test_size])

train_dataset = TransformDataset(train_transform, train_subset)
test_dataset = TransformDataset(test_transform, test_subset)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128)

optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=2e-4)

epochs = 4
train_losses, val_losses = [], []
train_accs, val_accs = [], []
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

for epoch in range(epochs):

    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, scheduler)
    eval_loss, eval_acc = evaluate(model, test_loader, criterion)
    scheduler.step(eval_loss)
    
    train_losses.append(train_loss)
    val_losses.append(eval_loss)
    train_accs.append(train_acc)
    val_accs.append(eval_acc)
    print(f"İlk fine tune Epoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {eval_loss:.4f}, Acc: {eval_acc:.4f}")
    print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")             

optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-4)
    
epochs = 3
train_losses, val_losses = [], []
train_accs, val_accs = [], []
for param in model.parameters():
    param.requires_grad = True

for epoch in range(epochs):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, scheduler)
    eval_loss, eval_acc = evaluate(model, test_loader, criterion)
    scheduler.step(eval_loss)
    
    train_losses.append(train_loss)
    val_losses.append(eval_loss)
    train_accs.append(train_acc)
    val_accs.append(eval_acc)
    print(f"İkinci fine tune Epoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {eval_loss:.4f}, Acc: {eval_acc:.4f}")
    print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")         
# %% fine tune görselleştirme
print("==== MedicalExpert-II Evaluation ====")

print("==== MedicalExpert-II Evaluation ====")

print("==== MedicalExpert-II Evaluation ====")

all_preds, all_labels, all_probs = test_model(model, test_loader, criterion)

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

cm, roc_auc, roc_curve = confusion_matrix, roc_auc_score, roc_curve      

cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
plt.figure()
sns.heatmap(cm, annot=True, linewidths=2, xticklabels=class_names, yticklabels=class_names, cmap="Purples")
plt.tight_layout()
plt.show()

report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
df = pd.DataFrame(report).transpose()
class_metrics = df.iloc[:, :4]
plt.figure(figsize=(10, len(class_metrics) * 0.6))
sns.heatmap(class_metrics, annot=True, fmt=".2f", cmap="Purples", linewidths=0.5)
plt.title("Classification Report Heatmap")
plt.ylabel("Classes")
plt.xlabel("Metrics")
plt.tight_layout()
plt.show()  

y_true_bin = label_binarize(all_labels, classes=list(range(len(class_names))))

auc_macro = roc_auc_score(y_true_bin, all_probs, average='macro', multi_class='ovr')
auc_micro = roc_auc_score(y_true_bin, all_probs, average='micro', multi_class='ovr')

print(f"Macro ROC AUC: {auc_macro:.3f}") # her sınıf için auc ortalaması
print(f"Micro ROC AUC: {auc_micro:.3f}") # tüm metrikler tek havuzda toplanır yani genel başarı

def plot_roc_auc(y_true_bin, y_score, class_names):
    n_classes = len(class_names)
    
    plt.figure(figsize=(n_classes * 4, 4))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        plt.subplot(1, n_classes, i + 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.title(f"ROC - {class_names[i]}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

def plot_prc(y_true_bin, y_score, class_names):
    n_classes = len(class_names)
    plt.figure(figsize=(n_classes * 4, 4))
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_score[:, i])
        plt.subplot(1, n_classes, i + 1)
        plt.plot(recall, precision, color='blue', lw=2, label=f"AP = {ap:.2f}")
        plt.title(f"PRC - {class_names[i]}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()  
y_true_bin = np.array(y_true_bin)
all_probs = np.array(all_probs)
plot_roc_auc(y_true_bin, all_probs, class_names)
plot_prc(y_true_bin, all_probs, class_names)

# result:
"""
Macro ROC AUC: 0.973
Micro ROC AUC: 0.983
Accuracy %89
"""
    

        
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 12:57:52 2025

@author: tunahan
"""
# GTSRB dataset
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
import numpy as np

# %% dataload

def get_data_loaders(batch_size=768):
    train_transforms = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    val_transforms = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    train_set = datasets.GTSRB(root="./data", split = "train", download=True, transform=train_transforms)
    val_set = datasets.GTSRB(root="./data", split = "test", download=True, transform=val_transforms)
    
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle=False)
    return train_loader, val_loader

# %% visualize

def imshow(img):
    img = img * 0.5 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.show()

def get_sample_img(train_loader):
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    return images, labels
def visualize(n):
    train_loader, test_loader = get_data_loaders()
    images, labels = get_sample_img(test_loader) 
    plt.figure()
    for i in range(n):
        plt.subplot(1, n, i+1)
        imshow(images[i])
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")
    plt.show()        
visualize(6)

# %% architecture

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.features = nn.Sequential( # [B, 3, 48, 48]
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # [B, 32, 48, 48]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),# [B, 64, 48, 48]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # [B, 64, 24, 24]
            nn.Dropout(0.15),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # [B, 128, 24, 24]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # [B, 128, 12, 12]
            nn.Dropout(0.15),
            )
        self.classification = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(128*12*12, 256), 
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 43),
            )

    def forward(self, x):
        x = self.features(x)
        return self.classification(x)

# %% train-test

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
        correct += (preds == labels).sum().item()
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
            total += images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
    return epoch_loss, epoch_acc

def test_model(model, dataloader):
    all_preds, all_labels = [], []
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())       
    return all_preds, all_labels

def test_model_softmax(model, dataloader):
    all_probs, all_labels = [], []
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            probs = nn.functional.softmax(outputs, 1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())       
    return all_probs, all_labels



train_loader, test_loader = get_data_loaders()
model = Cnn().to("cuda")

epochs = 13
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                       factor=0.5, patience=3, verbose=True)
train_losses, train_accs = [], []
val_losses, val_accs = [], []
for epoch in range(epochs):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, scheduler)
    val_loss, val_acc = evaluate(model, test_loader, criterion)
    scheduler.step(val_loss)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")            



# %% fine tune

augmented_train_transforms = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

fine_tune_train_set = datasets.GTSRB(root="./data", split="train", download=False, transform=augmented_train_transforms)
fine_tune_train_loader = DataLoader(fine_tune_train_set, batch_size=768, shuffle=True)

fine_tune_optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

fine_tune_scheduler = optim.lr_scheduler.ReduceLROnPlateau(fine_tune_optimizer, mode='max', factor=0.5, patience=2, verbose=True)

fine_tune_epochs = 5
for epoch in range(epochs, epochs + fine_tune_epochs):
    train_loss, train_acc = train_model(model, fine_tune_train_loader, criterion, fine_tune_optimizer, fine_tune_scheduler)
    val_loss, val_acc = evaluate(model, test_loader, criterion)
    fine_tune_scheduler.step(val_acc)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    epoch += 1
    print(f"[Fine-Tune] Epoch {epoch+1}")
    print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    print(f"  LR: {fine_tune_scheduler.optimizer.param_groups[0]['lr']:.6f}")

# %% fine tune 2

transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.85, 1.15)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


fine_tune_train_set = datasets.GTSRB(root="./data", split="train", download=False, transform=augmented_train_transforms)
fine_tune_train_loader = DataLoader(fine_tune_train_set, batch_size=768, shuffle=True)

fine_tune_optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

fine_tune_scheduler = optim.lr_scheduler.ReduceLROnPlateau(fine_tune_optimizer, mode='max', factor=0.5, patience=2, verbose=True)

fine_tune_epochs = 3
for epoch in range(epochs, epochs + fine_tune_epochs):
    train_loss, train_acc = train_model(model, fine_tune_train_loader, criterion, fine_tune_optimizer, fine_tune_scheduler)
    val_loss, val_acc = evaluate(model, test_loader, criterion)
    fine_tune_scheduler.step(val_acc)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    epoch += 1
    print(f"[Fine-Tune] Epoch {epoch+1}")
    print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    print(f"  LR: {fine_tune_scheduler.optimizer.param_groups[0]['lr']:.6f}")
# %% visualization    

train_preds, train_labels = test_model(model, train_loader)
test_preds, test_labels = test_model(model, test_loader)
test_probs, test_labels = test_model_softmax(model, test_loader)

plt.figure()
plt.plot(train_losses, label="Train Loss", color="blue", linewidth=2)
plt.plot(val_losses, label="Validation Loss", color="orange", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(train_accs, label="Train Accs", color="blue", linewidth=2)
plt.plot(val_accs, label="Validation Accs", color="orange", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accs")
plt.title("Train vs Validation Accs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()        

print("Train Set Classification Report", classification_report(train_labels, train_preds))
print("Test Set Classification Report", classification_report(test_labels, test_preds))

roc_auc_macro = roc_auc_score(test_labels, test_probs, average="macro", multi_class="ovr")
roc_auc_micro = roc_auc_score(test_labels, test_probs, average="micro", multi_class="ovr")
print(f"Macro ROC AUC: {roc_auc_macro:.4f}")
print(f"Micro ROC AUC: {roc_auc_micro:.4f}")  

plt.figure()
cm = confusion_matrix(test_labels, test_preds)
sns.heatmap(cm, cmap = "Purples", annot=True, linewidth=1)
plt.tight_layout()
plt.show()
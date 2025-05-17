#!/home/tunahan/anaconda3/envs/ml_env/bin/python3
# -*- coding: utf-8 -*-
"""
Created on May 15, 2025 18:10:55

@author: tunahan
"""
# parisrohan/credit-score-classification
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import torch.nn.functional as F

train = pd.read_csv("./data/train.csv", low_memory=False)
test = pd.read_csv("./data/test.csv", low_memory=False)

train.isnull().any()

train.head()

train.drop(columns=['ID', 'Name', 'SSN'], inplace=True)
test.drop(columns=['ID', 'Name', 'SSN'], inplace=True)

train.dtypes

columns = ['Annual_Income', 'Changed_Credit_Limit', 'Amount_invested_monthly', 'Monthly_Balance', "Age", "Num_of_Loan", "Num_of_Delayed_Payment", "Outstanding_Debt"]
for column in columns:
    train[column] = pd.to_numeric(train[column], errors="coerce")
for column in columns:
    test[column] = pd.to_numeric(test[column], errors="coerce")
    
train.dtypes

train.fillna(train.median(numeric_only=True), inplace=True)
test.fillna(test.median(numeric_only=True), inplace=True)

categorical_columns = ['Month', 'Occupation', 'Payment_Behaviour', 'Payment_of_Min_Amount']   
train = pd.get_dummies(train, columns=categorical_columns, drop_first=True)
test = pd.get_dummies(test, columns=categorical_columns, drop_first=True)

train["Type_of_Loan"] = train["Type_of_Loan"].str.split(", ")
test["Type_of_Loan"] = test["Type_of_Loan"].str.split(", ")

train["Type_of_Loan"] = train["Type_of_Loan"].apply(lambda x: [] if isinstance(x, float) else x)
test["Type_of_Loan"] = test["Type_of_Loan"].apply(lambda x: [] if isinstance(x, float) else x)

encoder = MultiLabelBinarizer()
encoder.fit(train["Type_of_Loan"]) 
classes = encoder.classes_

train_loan_encoded = MultiLabelBinarizer(classes=classes).fit_transform(train["Type_of_Loan"])
test_loan_encoded = MultiLabelBinarizer(classes=classes).fit_transform(test["Type_of_Loan"])

train = pd.concat([train, pd.DataFrame(train_loan_encoded, columns=classes, index=train.index)], axis=1)
test = pd.concat([test, pd.DataFrame(test_loan_encoded, columns=classes, index=test.index)], axis=1)

train.drop(columns=["Type_of_Loan"], inplace=True)
test.drop(columns=["Type_of_Loan"], inplace=True)

for column in train.columns:
    if column not in test.columns:
        test[column] = 0  

for column in test.columns:
    if column not in train.columns:
        train[column] = 0  

train = train.reindex(sorted(train.columns), axis=1)
test = test.reindex(sorted(test.columns), axis=1)

col = train.select_dtypes(include=object)
test = test.drop(columns = ["Credit_Score"])
credit = {
    "_": 4, 
    "Poor": 0,
    "Bad": 0,
    "Standard": 1,
    "Good": 2
    }

train["Credit_Mix"] = train["Credit_Mix"].map(credit)
test["Credit_Mix"] = test["Credit_Mix"].map(credit)
train["Credit_Score"] = train["Credit_Score"].map(credit)

def convert_months(value):
    if isinstance(value, str):
        if pd.isna(value): return 0
        years = re.search(r"(\d+)\sYears", value)
        months = re.search(r"(\d+)\sMonths", value)
        total = 0
        if years:
            total += int(years.group(1))* 12
        if months:
            total += int(months.group(1))
        return total
    return 0
train["Credit_History_Age"] = train["Credit_History_Age"].apply(convert_months)
test["Credit_History_Age"] = test["Credit_History_Age"].apply(convert_months)

def outlier_to_nan(df, col, lower=None, upper=None):
    if lower is not None:
        df.loc[df[col] < lower, col] = np.nan
    if upper is not None:
        df.loc[df[col] > upper, col] = np.nan

for col, (low, high) in {
    "Age": (0, 99),
    "Num_Bank_Accounts": (0, 20),
    "Num_of_Loan": (0, 9),
    "Num_Credit_Card": (0, 11),
    "Num_Credit_Inquiries": (0, 17),
    "Num_of_Delayed_Payment": (0, 28),
    "Delay_from_due_date": (0, None),  
    "Interest_Rate": (0, 34)
}.items():
    outlier_to_nan(train, col, lower=low, upper=high)
    outlier_to_nan(test, col, lower=low, upper=high)

for col, upper_iqr in {
    "Annual_Income": 1.5,
    "Total_EMI_per_month": 2.5,
    "Amount_invested_monthly": 1.75
}.items():
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    outlier_to_nan(train, col, lower=Q1 - 1.5*IQR, upper=Q3 + upper_iqr*IQR)
    outlier_to_nan(test, col, lower=Q1 - 1.5*IQR, upper=Q3 + upper_iqr*IQR)

# fill by customer_id (ffill -> bfill -> mode/median)
num_cols = ["Age", "Num_Bank_Accounts", "Num_of_Loan", "Num_Credit_Card", "Num_Credit_Inquiries",
            "Num_of_Delayed_Payment", "Delay_from_due_date", "Interest_Rate",
            "Annual_Income", "Total_EMI_per_month", "Amount_invested_monthly", "Credit_History_Age"]

for col in num_cols:
    train[col] = train.groupby("Customer_ID")[col].transform(lambda x: x.ffill().bfill())
    test[col] = test.groupby("Customer_ID")[col].transform(lambda x: x.ffill().bfill())
    
    if train[col].isnull().any():
        median = train[col].median()
        train[col].fillna(median)
    if test[col].isnull().any():
        median = test[col].median()
        test[col] = test[col].fillna(median)
        
train = train.dropna()
test = test.dropna()

"""
for col in train.select_dtypes(include=["int", "float"]):
    print(f"Sütun: {col}, max: {train[col].max()}")
    print(f"Sütun: {col}, min: {train[col].min()}")
"""
scalers = {}
for col in train.select_dtypes(include = ["int", "float"]):
    scaler = StandardScaler()
    if train[col].nunique() > 2 and col != "Credit_Score":
        train[col] = scaler.fit_transform(train[[col]])
        scalers[col] = scaler
                
for col in train.select_dtypes(include='bool'):
    train[col] = train[col].astype(int)
    
for col in test.select_dtypes(include='bool'):
    test[col] = test[col].astype(int)

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from math import ceil

X = train.drop(columns = ["Customer_ID", "Credit_Score"], axis = 1)
y = train["Credit_Score"].drop(columns = ["Customer_ID"])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]

train_set = CustomDataset(X_train, y_train)
test_set = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

class FFNN(nn.Module):
    def __init__(self, input_size):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 3)
    def forward(self, x):
        x1 = F.relu(self.bn1(self.fc1(x)))
        x2 = F.relu(self.bn2(self.fc2(x1))) + x1
        x2 = self.drop2(x2)            
        out = self.fc3(x2)
        return out
    
class FFNNv2(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.35),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),

            nn.Linear(256, 3)  # output layer
        )

    def forward(self, x):
        return self.model(x)
    
class SoftPrecisionLoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).float()
        tp = torch.sum(probs * targets_one_hot, dim=0)
        fp = torch.sum(probs * (1 - targets_one_hot), dim=0)
        precision = tp / (tp + fp + self.epsilon)
        loss = torch.mean(1.0 - precision)
        return loss
 
    
class FocalLoss(nn.Module):
    def __init__(self, alpha = 1, gamma = 2):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt) ** self.gamma * ce_loss
        return focal_loss.mean()
               
# model = FFNN(input_size=X_train.shape[1]).to("cuda")
model = FFNNv2(input_size=X_train.shape[1]).to("cuda")
def l1_regularization(model, lambda_l1=1e-4):
    l1_norm = sum(param.abs().sum() for param in model.parameters())
    return lambda_l1 * l1_norm

batch_size = 512
epochs = 200
num_train_samples = len(train_set)
steps_per_epoch= ceil(num_train_samples / batch_size)
total_steps = steps_per_epoch * epochs


ce = nn.CrossEntropyLoss()
sp = SoftPrecisionLoss()
focal = FocalLoss(alpha=0.8, gamma=1.5)

def total_loss(logits, targets):
    return (
        0.6 * ce(logits, targets) +
        0.2 * sp(logits, targets) +
        0.2 * focal(logits, targets)
    )

criterion = total_loss

optimizer = optim.RAdam(model.parameters(), lr=1e-2, weight_decay=5e-4)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-2,
    total_steps=total_steps,
    pct_start=0.3,
    anneal_strategy="cos",
    div_factor=25.0,
    final_div_factor=1e4
) # start / stop lr: max_lr / (final_)div_factor


def train_model(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in tqdm(dataloader, desc="Train Model:"):
        batch_X, batch_y = batch_X.to("cuda"), batch_y.long().to("cuda")
        optimizer.zero_grad()
        outputs = model(batch_X)

        loss = criterion(outputs, batch_y) + l1_regularization(model, lambda_l1=1e-4)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate(model, dataloader, criteron):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in tqdm(dataloader, desc="Eval Model:"):

            batch_X, batch_y = batch_X.to("cuda"), batch_y.long().to("cuda")

            outputs = model(batch_X)  # logits
            loss = loss = criterion(outputs, batch_y) + l1_regularization(model, lambda_l1=1e-4)

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)  
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def test_model(model, dataloader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch_X, batch_y in tqdm(dataloader, desc="Test Model:"):
            batch_X, batch_y = batch_X.to("cuda"), batch_y.to("cuda")
            
            outputs = model(batch_X)  # logits
            probs = F.softmax(outputs, dim=1)  
            preds = torch.argmax(outputs, dim=1)  

            all_preds.extend(preds.cpu().numpy())        
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_preds, all_labels, all_probs


train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_val_loss = float("inf")
patience_counter = 0
delta = 1e-4

for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, test_loader, criterion)
    scheduler.step()
    
    if val_loss < best_val_loss - delta:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= 35:
        print("Early stopping triggered at epoch", epoch)
        break
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    print(f"  Epoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
    
model.load_state_dict(torch.load("best_model.pt"))
all_preds, all_labels, all_probs = test_model(model, test_loader)

plt.figure(figsize=(8,6))
plt.plot(train_losses, label="Train Losses", color="red", linewidth=2)
plt.plot(val_losses, label="Validation Losses", color="orange", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(train_accs, label="Train Accuracies", color = "purple", linewidth=2)
plt.plot(val_accs, label="Validation Accuracies", color = "green", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Validation Accuracies")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(all_preds, all_labels, alpha=0.5)
plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], color='red')
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Predictions vs True Values")
plt.show()

unique_labels = sorted(set(all_labels) | set(all_preds))
cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)

cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", linecolor="black", linewidths=1)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

report = classification_report(all_labels, all_preds, output_dict=True)

labels = list(report.keys())[:-3]  # 'accuracy', 'macro avg', 'weighted avg' hariç tut
metrics = ['precision', 'recall', 'f1-score']

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(10,6))

for i, metric in enumerate(metrics):
    values = [report[label][metric] for label in labels]
    ax.bar(x + i*width, values, width, label=metric)

ax.set_xticks(x + width)
ax.set_xticklabels(labels)
ax.set_ylim(0,1)
ax.set_ylabel('Score')
ax.set_title('Classification Report Metrics per Class')
ax.legend()
plt.show()

roc_auc_micro = roc_auc_score(all_labels, all_probs, average='micro', multi_class='ovr')
roc_auc_macro = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')

print(f"Micro ROC AUC: {roc_auc_micro:.4f}")
print(f"Macro ROC AUC: {roc_auc_macro:.4f}")
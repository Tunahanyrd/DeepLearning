#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 20:31:20 2025

@author: tunahan
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import make_regression

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(out_features, in_features) * np.sqrt(2. / in_features)
        self.b = np.zeros((out_features, 1))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
    def forward(self, x):
        self.x = x
        y = self.W @ x + self.b
        return y # Linear transformation Ax + b
    
    def backward(self, grad_output):
        """z = Wx + b so grad_output = dL / dz"""
        batch_size = self.x.shape[1]
        self.dW[:] = (grad_output @ self.x.T) / self.x.shape[1]
        self.db[:] = np.sum(grad_output, axis=1, keepdims=True) / batch_size
        dx = self.W.T @ grad_output
        return dx
    
    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db
        
class ELU:
    def __init__(self, alpha=1.0):
        self.x = None
        self.alpha = alpha
        
    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, self.alpha * ((np.e ** x) - 1))
    
    def backward(self, grad_output):
        return grad_output * np.where(self.x > 0, 1, self.alpha * np.exp(self.x))
         
class ReLU:
    def __init__(self):
        self.x = None
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    def backward(self, grad_output):
        return grad_output * (self.x > 0).astype(np.float32)

class MSELoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true  = y_true 
        
        return np.mean((self.y_true - self.y_pred)**2)
    
    def backward(self):
        N = self.y_pred.shape[1]
        return 2 * (self.y_pred - self.y_true) / N
    
class Dropout:
    def __init__(self, p = 0.2):
        self.p = p
        self.mask = None
        
    def forward(self, x, train=True):
        if train:
            self.mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
            return (x * self.mask) / (1.0 - self.p)
        else: 
            return x
        
    def backward(self, grad_output):
        return grad_output * self.mask / (1.0 - self.p)

class MAELoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
        
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean(np.abs(y_true - y_pred))
    
    def backward(self):
        N = self.y_pred.shape[1]
        grad = np.where(self.y_pred > self.y_true, 1, -1)
        grad = np.where(self.y_pred == self.y_true, 0, grad)
        grad = grad / N  # Mean için
        return grad

class AdamW():
    def __init__(self, parameters, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=5e-4):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas      
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = [np.zeros_like(p["param"]) for p in parameters] # momentum
        self.v = [np.zeros_like(p["param"]) for p in parameters] # squared gradient
        self.t = 0 # time step
           
    def step(self):
        """
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        param -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
        bias ** step --> bias correction
        """     
        self.t += 1
        for i, p in enumerate(self.parameters):
            param, grad = p["param"], p["grad"]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad **2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * param)

class Model:
    def __init__(self, layers):
        self.layers = layers
        self.training =  True
    
    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train=self.training)
            else: 
                x = layer.forward(x)                 
        return x
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
    
    def step(self, lr):
        for layer in self.layers:
            if hasattr(layer, "step"):
                layer.step(lr)
    def __call__(self, x):
        return self.forward(x)
    
class EarlyStopping:
    def __init__(self, model, patience = 5, delta=1e-3):
        self.patience = patience
        self.min_loss = float("inf")
        self.delta = delta
        self.counter = 0
        self.model = model
    def forward(self, val_loss):
        if val_loss < self.min_loss - self.delta:
            self.min_loss = val_loss
            self.counter = 0
            save_model(self.model, "best_model.pkl")
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def __call__(self, val_loss):
        return self.forward(val_loss)
class ReduceLROnPlateau:
    def __init__(self, optimizer, patience = 5, delta=1e-3, factor = 0.5, min_lr = 1e-5):
        self.patience = patience
        self.best_loss = float("inf")
        self.delta = delta
        self.num_bad_epochs = 0
        self.factor = factor
        self.optimizer = optimizer
        self.min_lr = min_lr
    def step(self, val_loss):
        if val_loss > self.best_loss - self.delta:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.optimizer.lr = max(self.optimizer.lr * self.factor, self.min_lr)
                self.num_bad_epochs = 0
        else:
            self.best_loss = val_loss
            self.num_bad_epochs = 0
    def get_last_lr(self):
        return [self.optimizer.lr]       

class DataLoader:
    def __init__(self, X, y, batch_size=512, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[1]
        
    def __iter__(self):
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current = 0
        return self
    
    def __next__(self):
        if self.current >= self.num_samples:
            raise StopIteration
        start = self.current
        end = min(self.current + self.batch_size, self.num_samples)
        batch_idx = self.indices[start:end]
        self.current = end
        return self.X[:, batch_idx], self.y[:, batch_idx]
    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

def save_model(model, path="best_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)
        
def load_model(path="best_model.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) **2)
    return 1 - (ss_res / ss_tot)

def mean_absolute_error(y_true, y_pred): 
    return np.mean(np.abs(y_true - y_pred))

def pearsonr(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    num = np.sum((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
    denom = np.sqrt(np.sum((y_true - y_true.mean())**2) * np.sum((y_pred - y_pred.mean())**2))
    return num / denom
       
X, y = make_regression(
    n_samples=10000,
    n_features=10,
    noise=0.1,
    random_state=42
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(1, 11)])
df['target'] = y

N = df.shape[0]
indices = np.arange(N)
np.random.shuffle(indices)

train_end = int(N * 0.7)
val_end   = int(N * 0.85)

train_idx = indices[:train_end]
val_idx   = indices[train_end:val_end]
test_idx  = indices[val_end:]

X_train = df.iloc[train_idx, :-1].values.T  
y_train = df.iloc[train_idx,  -1].values.reshape(1, -1)  

X_val_raw = df.iloc[val_idx, :-1].values.T
y_val_raw = df.iloc[val_idx,  -1].values.reshape(1, -1)

X_test_raw = df.iloc[test_idx, :-1].values.T
y_test_raw = df.iloc[test_idx,  -1].values.reshape(1, -1)

X_mean = X_train.mean(axis=1, keepdims=True)
X_std  = X_train.std(axis=1, keepdims=True)
X_train = (X_train - X_mean) / X_std
X_val   = (X_val_raw - X_mean) / X_std
X_test  = (X_test_raw - X_mean) / X_std

y_mean = y_train.mean()
y_std  = y_train.std()
y_train = (y_train - y_mean) / y_std
y_val   = (y_val_raw - y_mean) / y_std
y_test  = (y_test_raw - y_mean) / y_std

train_loader = DataLoader(X_train, y_train, batch_size=128, shuffle=True)
val_loader = DataLoader(X_val, y_val, batch_size=128, shuffle=False)
test_loader = DataLoader(X_test, y_test, batch_size=128, shuffle=False)

model = Model([
    Linear(10, 32),  ELU(),
    Linear(32, 32), ELU(),
    Linear(32, 16), ELU(),
    Linear(16, 1)
])

parameters = []
for layer in model.layers:
    if isinstance(layer, Linear):
        parameters.append({"param": layer.W, "grad": layer.dW})       
        parameters.append({"param": layer.b, "grad": layer.db})
        
optimizer = AdamW(parameters, lr=5e-4)     
criterion = MSELoss()
scheduler = ReduceLROnPlateau(optimizer, patience=10, delta=1e-3, factor=0.3)
earlystopping = EarlyStopping(model, patience=30, delta=1e-4)

epochs = 200

def train_model(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in tqdm(dataloader, desc="Train Model:"):
        outputs = model.forward(batch_X)
        loss = criterion.forward(outputs, batch_y)
        grad = criterion.backward()
        model.backward(grad)
        optimizer.step()
        running_loss += loss

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    for batch_X, batch_y in tqdm(dataloader, desc="Eval Model:"):
        outputs = model.forward(batch_X) # I actually added a __call__ object, I just write it like this so that the execution is obvious
        loss = criterion.forward(outputs, batch_y)
        running_loss += loss
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def test_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    for batch_X, batch_y in tqdm(dataloader, desc="Test Model:"):
        outputs = model.forward(batch_X)
        all_preds.append(outputs)
        all_labels.append(batch_y)
    
    all_preds = np.concatenate(all_preds, axis=1)
    all_labels = np.concatenate(all_labels, axis=1)
    return all_preds, all_labels

train_losses, val_losses = [], []
for epoch in tqdm(range(epochs)):
    train_loss = train_model(model, train_loader, criterion, optimizer)
    val_loss = evaluate(model, val_loader, criterion)
    
    scheduler.step(val_loss)
    if earlystopping(val_loss):
        print("Early stopping triggered at epoch", epoch)
        break
    
    first_linear = next(layer for layer in model.layers if isinstance(layer, Linear))
    grad_norm = np.linalg.norm(first_linear.dW)
    if epoch % 50 == 0:  # her 50. batch’te bir yaz
        print(f"[Batch {epoch}] dW norm (1. katman): {grad_norm:.4e}")
       
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"  Epoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val   Loss: {val_loss:.4f}")
    print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")        

model = load_model("best_model.pkl")
y_pred, y_true = test_model(model, test_loader)

y_pred = y_pred * y_std + y_mean
y_true = y_test * y_std + y_mean

print("RMSE:", root_mean_squared_error(y_true, y_pred))
print("MAE:", mean_absolute_error(y_true, y_pred))
print("R2 Score:", r2_score(y_true, y_pred))
print("Pearson Corr:", pearsonr(y_true, y_pred))

plt.figure(figsize=(8,6))
plt.plot(train_losses, label="Train Losses", color="red", linewidth=2)
plt.plot(val_losses, label="Validation Losses", color="orange", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
        
plt.figure(figsize=(8,6))
plt.scatter(y_true, y_pred, alpha=0.5, label="Preds")
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
         color='red', linewidth=2, label="Ideal (y = y_pred)")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs True Values")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()     
# %%

X_new, y_new = make_regression(
    n_samples=10000,
    n_features=10,
    noise=0.1,
    random_state=42
)
df = pd.DataFrame(X_new, columns=[f"feature_{i}" for i in range(1, 11)])
df["target"] = y_new

N = df.shape[0]
indices = np.arange(N)
np.random.shuffle(indices)

train_end = int(N * 0.7)
val_end   = int(N * 0.85)

train_idx = indices[:train_end]
val_idx   = indices[train_end:val_end]
test_idx  = indices[val_end:]

X_train = df.iloc[train_idx, :-1].values.T   # shape = (10, n_train)
y_train = df.iloc[train_idx,  -1].values.reshape(1, -1)  # shape = (1, n_train)

X_val_raw = df.iloc[val_idx, :-1].values.T    # shape = (10, n_val)
y_val_raw = df.iloc[val_idx,  -1].values.reshape(1, -1)

X_test_raw = df.iloc[test_idx, :-1].values.T  # shape = (10, n_test)
y_test_raw = df.iloc[test_idx,  -1].values.reshape(1, -1)

X_mean = X_train.mean(axis=1, keepdims=True)
X_std  = X_train.std(axis=1, keepdims=True)

X_train = (X_train - X_mean) / X_std
X_val   = (X_val_raw - X_mean) / X_std
X_test  = (X_test_raw - X_mean) / X_std

y_mean = y_train.mean()
y_std  = y_train.std()

y_train = (y_train - y_mean) / y_std
y_val   = (y_val_raw - y_mean) / y_std
y_test  = (y_test_raw - y_mean) / y_std

with open("best_model.pkl", "rb") as f:
    model: Model = pickle.load(f)
        
def test_model(model, X, y_true):

    model.eval()
    with np.errstate(all='ignore'):
        y_pred_norm = model.forward(X)         
    y_pred = y_pred_norm * y_std + y_mean     
    
    def rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2))
    def mae(a, b):
        return np.mean(np.abs(a - b))
    def r2_score(a, b):
        ss_res = np.sum((a - b) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        return 1 - (ss_res / ss_tot)
    def pearsonr(a, b):
        a = a.flatten()
        b = b.flatten()
        num = np.sum((a - a.mean()) * (b - b.mean()))
        den = np.sqrt(np.sum((a - a.mean()) ** 2) * np.sum((b - b.mean()) ** 2))
        return num / den
    
    y_true_orig = y_true * y_std + y_mean
    
    rmse_val = rmse(y_true_orig, y_pred)
    mae_val  = mae(y_true_orig, y_pred)
    r2_val   = r2_score(y_true_orig, y_pred)
    pcorr    = pearsonr(y_true_orig, y_pred)
    
    return y_pred, y_true_orig, rmse_val, mae_val, r2_val, pcorr

y_pred, y_true_orig, rmse_val, mae_val, r2_val, pcorr = test_model(model, X_test, y_test)

print(f"Test RMSE:        {rmse_val:.4f}")
print(f"Test MAE:         {mae_val:.4f}")
print(f"Test R2 Score:    {r2_val:.4f}")
print(f"Test Pearson Corr:{pcorr:.4f}")

plt.figure(figsize=(6,6))
plt.scatter(y_true_orig.flatten(), y_pred.flatten(), alpha=0.3, label="Tahminler")
plt.plot(
    [y_true_orig.min(), y_true_orig.max()],
    [y_true_orig.min(), y_true_orig.max()],
    color="red", linewidth=1.5, label="İdeal (y = ŷ)"
)
plt.xlabel("Labels")
plt.ylabel("Preds")
plt.title("Labels vs Preds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
  
        
        
        
        
        
        
        
        
        
        
        
        
        




     
        
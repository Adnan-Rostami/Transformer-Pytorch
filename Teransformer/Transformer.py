import torch
import torch.nn as nn
import numpy as np
import os 
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
import matplotlib.pylab as plt
import pandas as pd
import time
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score


class TransformerClassifier(nn.Module):
    # def __init__(self, input_dim=28, d_model=64, n_heads=4, num_layers=3, d_ff=256, dropout=0.2):
    def __init__(self, input_dim=28, d_model=128, n_heads=8, num_layers=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

 
    def forward(self, x):

        x = self.embedding(x)     
        x = x.unsqueeze(1)        
        x = self.transformer(x)    
        x = x.mean(dim=1)          
        logits = self.fc_out(x)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir  = "results_run"
os.makedirs(base_dir, exist_ok=True)
run_id = len(os.listdir(    base_dir))+1
run_folder = os.path.join(base_dir, f"run_{run_id:03d}")
os.makedirs(run_folder, exist_ok=True)
CSV_PATH = os.path.join(run_folder, f"run_{run_id:03d}.csv")


# ==== بارگذاری مستقیم داده‌های train / test ====
train_data = np.loadtxt('data.csv', delimiter=',', skiprows=1)    
test_data = np.loadtxt('data.csv', delimiter=',', skiprows=1)
 

print("Train unique labels:", np.unique(train_data[:, -1]))
print("Test  unique labels:", np.unique(test_data[:, -1]))

# ==== جداکردن X و Y ====
X_train = torch.tensor(train_data[:, :-1], dtype=torch.float32).to(device)
y_train = torch.tensor(train_data[:, -1],  dtype=torch.float32).unsqueeze(1).to(device)

X_test  = torch.tensor(test_data[:, :-1], dtype=torch.float32).to(device)
y_test  = torch.tensor(test_data[:, -1],  dtype=torch.float32).unsqueeze(1).to(device)

print("Train shape:", X_train.shape, y_train.shape)
print("Test  shape:", X_test.shape, y_test.shape)



model = TransformerClassifier(input_dim=X_train.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)


# epochs = 30
epochs = 10


# --- ساخت DataLoader ---
from torch.utils.data import DataLoader, TensorDataset
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=512)
records = []
start_time = time.time()
# --- حلقهٔ آموزش ---

all_labels = []
all_preds = []
all_probs = []
model.eval()
 
start_time = time.time()
records = []

for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

# --- ارزیابی ---
all_labels, all_preds, all_probs = [], [], []
model.eval()
with torch.no_grad():
    test_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in test_loader:
        logits = model(xb)
        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
        labels = yb.cpu().numpy().flatten()
        all_labels.extend(labels)
        all_preds.extend(preds)
        all_probs.extend(probs)
        test_loss += criterion(logits, yb).item()
        preds_bin = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds_bin == yb).float().sum().item()
        total += yb.size(0)

    acc = correct / total
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Epoch {epoch+1:03d} | Loss={loss.item():.4f} | Accuracy={acc:.4f}")
    print(f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, roc_auc={roc_auc:.4f}, pr_auc={pr_auc:.4f}")

    records.append({
        "epoch": epoch + 1,
        "accuracy": acc,
        "rmse": rmse,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    })

end_time = time.time()
df = pd.DataFrame(records)
df.to_csv(CSV_PATH, index=False)




plt.figure(figsize=(7,5))
plt.plot(df["epoch"], df["accuracy"], marker='o', label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Run {run_id} - Accuracy per Epoch")
plt.grid(True)
plt.legend()
IMG_PATH = os.path.join(run_folder, f"accuracy_run_{run_id:03d}.png")
plt.savefig(IMG_PATH, dpi=200)
plt.close()
print(f"Accuracy plot saved at: {IMG_PATH}")

#pr_auc
plt.figure(figsize=(7,5))
plt.plot(df["epoch"], df["pr_auc"], marker='o', label="pr_auc")
plt.xlabel("Epoch")
plt.ylabel("pr_auc")
plt.title(f"Run {run_id} - pr_auc per Epoch")
plt.grid(True)
plt.legend()
IMG_PATH = os.path.join(run_folder, f"pr_auc_run_{run_id:03d}.png")
plt.savefig(IMG_PATH, dpi=200)
plt.close()
print(f"pr_auc plot saved at: {IMG_PATH}")

#roc_auc
plt.figure(figsize=(7,5))
plt.plot(df["epoch"], df["roc_auc"], marker='o', label="roc_auc")
plt.xlabel("Epoch")
plt.ylabel("roc_auc")
plt.title(f"Run {run_id} - roc_auc per Epoch")
plt.grid(True)
plt.legend()
IMG_PATH = os.path.join(run_folder, f"roc_auc_run_{run_id:03d}.png")
plt.savefig(IMG_PATH, dpi=200)
plt.close()
print(f"roc_auc plot saved at: {IMG_PATH}")


#f1
plt.figure(figsize=(7,5))
plt.plot(df["epoch"], df["f1"], marker='o', label="f1")
plt.xlabel("Epoch")
plt.ylabel("f1")
plt.title(f"Run {run_id} - f1 per Epoch")
plt.grid(True)
plt.legend()
IMG_PATH = os.path.join(run_folder, f"f1_run_{run_id:03d}.png")
plt.savefig(IMG_PATH, dpi=200)
plt.close()
print(f"f1 plot saved at: {IMG_PATH}")





end_time = time.time()

df = pd.DataFrame(records)

df.to_csv(CSV_PATH, index=False)

final_cm = cm

cm_expanded = np.vstack([
    ['', 'Pred_Neg', 'Pred_Pos'],
    ['True_Neg', final_cm[0,0], final_cm[0,1]],
    ['True_Pos', final_cm[1,0], final_cm[1,1]],
    ['', '', ''],
    ['Run_Time_sec', round(end_time - start_time, 4), '']
])
df2 = pd.DataFrame(cm_expanded)
CSV_PATH2 = os.path.join(run_folder, f"confusion_run_{run_id:03d}.csv")
df2.to_csv(CSV_PATH2, index=False)

 

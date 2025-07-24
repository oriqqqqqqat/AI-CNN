import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

# เพิ่ม path เพื่อ import module ได้
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing import ImageOnlyDataset
from cnnmodel.modeldensenet121 import  ImageOnlyDenseNet121

# === CONFIG ===
# ใช้ forward slashes
train_csv = "D:/CNN/data/train.csv"
val_csv = "D:/CNN/data/val.csv"
batch_size = 8
num_epochs = 50
patience = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Data ===
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
class_names = sorted(train_df["disease"].unique().tolist())
label_map = {cls: i for i, cls in enumerate(class_names)}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = ImageOnlyDataset(train_df, transform, label_map)
val_dataset = ImageOnlyDataset(val_df, transform, label_map)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# === Model ===
model =  ImageOnlyDenseNet121(num_classes=len(class_names)).to(device)
optimizer = Adam(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# === Train Loop ===
train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_acc = 0
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_train_loss = train_loss / len(train_loader)
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} Acc={train_acc:.4f} | Val Loss={avg_val_loss:.4f} Acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_densenet_imageonly.pth")
        print("💾 Saved best model")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print("⏹️ Early stopping")
        break

# === Plot ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.title("Accuracy Curve")

plt.tight_layout()
plt.savefig("training_curve_densenet_imageonly.png")
plt.show()

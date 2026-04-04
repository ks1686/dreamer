"""
Homework 4: LeNet-5 for CIFAR-10 Classification
Three variants: Baseline, Dropout, BatchNorm
"""

import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ── Device ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001

# ── Task 1: Data Loading ─────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# ── Task 2: Baseline LeNet-5 (no Dropout, no BatchNorm) ─────────────────────
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ("c1",    nn.Conv2d(3, 6, kernel_size=5)),   # 3→6, 32→28
            ("relu1", nn.ReLU()),
            ("s2",    nn.MaxPool2d(kernel_size=2, stride=2)),  # 28→14
            ("c3",    nn.Conv2d(6, 16, kernel_size=5)),  # 14→10
            ("relu3", nn.ReLU()),
            ("s4",    nn.MaxPool2d(kernel_size=2, stride=2)),  # 10→5
            ("c5",    nn.Conv2d(16, 120, kernel_size=5)), # 5→1
            ("relu5", nn.ReLU()),
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ("f6",    nn.Linear(120, 84)),
            ("relu6", nn.ReLU()),
            ("f7",    nn.Linear(84, 10)),
            ("sig7",  nn.LogSoftmax(dim=-1)),
        ]))

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ── Task 3: LeNet-5 with Dropout ─────────────────────────────────────────────
class LeNet5Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ("c1",    nn.Conv2d(3, 6, kernel_size=5)),
            ("relu1", nn.ReLU()),
            ("s2",    nn.MaxPool2d(kernel_size=2, stride=2)),
            ("c3",    nn.Conv2d(6, 16, kernel_size=5)),
            ("relu3", nn.ReLU()),
            ("s4",    nn.MaxPool2d(kernel_size=2, stride=2)),
            ("c5",    nn.Conv2d(16, 120, kernel_size=5)),
            ("relu5", nn.ReLU()),
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ("f6",      nn.Linear(120, 84)),
            ("relu6",   nn.ReLU()),
            ("drop6",   nn.Dropout(p=p)),
            ("f7",      nn.Linear(84, 10)),
            ("sig7",    nn.LogSoftmax(dim=-1)),
        ]))

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ── Task 4: LeNet-5 with Batch Normalization ─────────────────────────────────
class LeNet5BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ("c1",    nn.Conv2d(3, 6, kernel_size=5)),
            ("bn1",   nn.BatchNorm2d(6)),
            ("relu1", nn.ReLU()),
            ("s2",    nn.MaxPool2d(kernel_size=2, stride=2)),
            ("c3",    nn.Conv2d(6, 16, kernel_size=5)),
            ("bn3",   nn.BatchNorm2d(16)),
            ("relu3", nn.ReLU()),
            ("s4",    nn.MaxPool2d(kernel_size=2, stride=2)),
            ("c5",    nn.Conv2d(16, 120, kernel_size=5)),
            ("bn5",   nn.BatchNorm2d(120)),
            ("relu5", nn.ReLU()),
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ("f6",    nn.Linear(120, 84)),
            ("bn6",   nn.BatchNorm1d(84)),
            ("relu6", nn.ReLU()),
            ("f7",    nn.Linear(84, 10)),
            ("sig7",  nn.LogSoftmax(dim=-1)),
        ]))

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ── Training & Evaluation helpers ────────────────────────────────────────────
criterion = nn.NLLLoss()


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def train_model(model, name):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_accs, test_accs = [], []
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct = total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        test_acc  = evaluate(model, test_loader)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"[{name}] Epoch {epoch:2d}/{EPOCHS} | "
              f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

    elapsed = time.time() - start
    print(f"\n[{name}] Training time: {elapsed:.1f}s | "
          f"Final train acc: {train_accs[-1]:.2f}% | "
          f"Final test acc:  {test_accs[-1]:.2f}%\n")

    return train_accs, test_accs, elapsed


# ── Run all three models ─────────────────────────────────────────────────────
results = {}

for model_cls, label in [
    (LeNet5(),          "Baseline"),
    (LeNet5Dropout(),   "Dropout"),
    (LeNet5BatchNorm(), "BatchNorm"),
]:
    train_accs, test_accs, elapsed = train_model(model_cls, label)
    results[label] = {
        "train_accs": train_accs,
        "test_accs":  test_accs,
        "time":       elapsed,
    }

# ── Task 5: Plot & Compare ───────────────────────────────────────────────────
epochs = range(1, EPOCHS + 1)
colors = {"Baseline": "steelblue", "Dropout": "darkorange", "BatchNorm": "seagreen"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("LeNet-5 on CIFAR-10: Training Curves", fontsize=14)

for label, data in results.items():
    axes[0].plot(epochs, data["train_accs"], label=label, color=colors[label])
    axes[1].plot(epochs, data["test_accs"],  label=label, color=colors[label])

for ax, title in zip(axes, ["Training Accuracy", "Test Accuracy"]):
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("Saved training_curves.png")

# ── Summary table ────────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print(f"{'Model':<12} {'Train Acc':>10} {'Test Acc':>10} {'Time (s)':>10}")
print("=" * 58)
for label, data in results.items():
    print(f"{label:<12} {data['train_accs'][-1]:>9.2f}% "
          f"{data['test_accs'][-1]:>9.2f}% "
          f"{data['time']:>9.1f}s")
print("=" * 58)

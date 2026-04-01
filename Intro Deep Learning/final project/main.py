"""
Final Project --- Type-A Topic 3
Evaluating Dropout (FC layers) and Batch Normalization (Conv layers) on LeNet-5 / MNIST

Runs 4 configurations across NUM_SEEDS random seeds, reports mean +/- std accuracy,
and saves training curves to results.png and raw numbers to results.json.
"""

import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import LeNet5

# -- Hyperparameters ------------------------------------------------------------
LEARNING_RATE = 0.001
BATCH_SIZE    = 128
EPOCHS        = 20
DROPOUT_RATE  = 0.5
NUM_SEEDS     = 5  # seeds 0-4; increase for tighter error bars

CONFIGS = [
    {"use_dropout": True,  "use_bn": True,  "label": "FC-Dropout, CONV-BN"},
    {"use_dropout": True,  "use_bn": False, "label": "FC-Dropout, CONV-noBN"},
    {"use_dropout": False, "use_bn": True,  "label": "FC-noDropout, CONV-BN"},
    {"use_dropout": False, "use_bn": False, "label": "FC-noDropout, CONV-noBN"},
]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- Reproducibility ------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# -- Data -----------------------------------------------------------------------
def get_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_loader = DataLoader(
        datasets.MNIST(root="./data", train=True,  download=True, transform=transform),
        batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        datasets.MNIST(root="./data", train=False, download=True, transform=transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
    )
    return train_loader, test_loader


# -- Training / evaluation ------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    correct, total, total_loss = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return 100.0 * correct / total, total_loss / total


def eval_epoch(model, loader, criterion):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out  = model(images)
            loss = criterion(out, labels)
            total_loss += loss.item() * labels.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += labels.size(0)
    return 100.0 * correct / total, total_loss / total


# -- Single config run across all seeds ----------------------------------------
def run_config(cfg, train_loader, test_loader):
    """Returns train_accs and test_accs arrays of shape (NUM_SEEDS, EPOCHS)."""
    train_accs = np.zeros((NUM_SEEDS, EPOCHS))
    test_accs  = np.zeros((NUM_SEEDS, EPOCHS))

    for s in range(NUM_SEEDS):
        set_seed(s)
        model     = LeNet5(use_dropout=cfg["use_dropout"], use_bn=cfg["use_bn"],
                           dropout_rate=DROPOUT_RATE).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(EPOCHS):
            tr_acc, tr_loss = train_epoch(model, train_loader, optimizer, criterion)
            te_acc, te_loss = eval_epoch(model, test_loader, criterion)
            train_accs[s, epoch] = tr_acc
            test_accs[s,  epoch] = te_acc
            print(f"  [{cfg['label']}] seed={s} epoch={epoch+1:2d} | "
                  f"train={tr_acc:.2f}%  test={te_acc:.2f}%")

    return train_accs, test_accs


# -- Plotting -------------------------------------------------------------------
def plot_results(all_results):
    epochs = np.arange(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LeNet-5 on MNIST: Dropout vs. Batch Normalization Ablation\n"
                 f"(mean +/- std over {NUM_SEEDS} seeds)", fontsize=12)

    for i, (cfg, (train_accs, test_accs)) in enumerate(zip(CONFIGS, all_results)):
        c = COLORS[i]
        for ax, accs, split in zip(axes, [test_accs, train_accs], ["Test", "Train"]):
            mean, std = accs.mean(0), accs.std(0)
            ax.plot(epochs, mean, color=c, label=cfg["label"])
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=c)

    for ax, title in zip(axes, ["Test Accuracy (%)", "Train Accuracy (%)"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nSaved -> results.png")


# -- Main -----------------------------------------------------------------------
def main():
    print(f"Device : {device}")
    print(f"Epochs : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print(f"Seeds  : {NUM_SEEDS}  |  Dropout rate: {DROPOUT_RATE}\n")

    train_loader, test_loader = get_loaders()

    all_results = []
    summary     = []
    total_start = time.time()

    for cfg in CONFIGS:
        print(f"\n{'='*60}")
        print(f"  {cfg['label']}")
        print(f"{'='*60}")
        t0 = time.time()
        train_accs, test_accs = run_config(cfg, train_loader, test_loader)
        elapsed = time.time() - t0

        final_mean = test_accs[:, -1].mean()
        final_std  = test_accs[:, -1].std()
        best_mean  = test_accs.max(axis=1).mean()
        best_std   = test_accs.max(axis=1).std()

        all_results.append((train_accs, test_accs))
        summary.append({
            "label":          cfg["label"],
            "final_acc_mean": round(float(final_mean), 4),
            "final_acc_std":  round(float(final_std),  4),
            "best_acc_mean":  round(float(best_mean),  4),
            "best_acc_std":   round(float(best_std),   4),
            "elapsed_s":      round(elapsed, 1),
        })
        print(f"\n  Final test acc: {final_mean:.2f}% +/- {final_std:.2f}%  "
              f"| Best: {best_mean:.2f}% +/- {best_std:.2f}%  | {elapsed:.1f}s")

    total_elapsed = time.time() - total_start

    # -- Summary table ----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"{'SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Configuration':<35} {'Final Acc':>12} {'Best Acc':>12} {'Time':>8}")
    print(f"{'-'*70}")
    for s in summary:
        print(f"{s['label']:<35} "
              f"{s['final_acc_mean']:.2f}%+/-{s['final_acc_std']:.2f}%  "
              f"{s['best_acc_mean']:.2f}%+/-{s['best_acc_std']:.2f}%  "
              f"{s['elapsed_s']:>6.1f}s")
    print(f"\nTotal experiment time: {total_elapsed:.1f}s")

    # -- Save raw results -------------------------------------------------------
    results_json = {
        cfg["label"]: {
            "train_accs": train_accs.tolist(),
            "test_accs":  test_accs.tolist(),
        }
        for cfg, (train_accs, test_accs) in zip(CONFIGS, all_results)
    }
    results_json["summary"] = summary
    with open("results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print("Saved -> results.json")

    plot_results(all_results)


if __name__ == "__main__":
    main()

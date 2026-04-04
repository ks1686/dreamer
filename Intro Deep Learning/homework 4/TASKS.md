# Homework 4: Build LeNet for CIFAR-10 Colorful Image Classification

## Overview

Train a modified LeNet-5 neural network on the CIFAR-10 dataset using PyTorch. Test three configurations and compare results.

**Due Date:** April 20, 2026  
**Required Test Accuracy:** > 50%

---

## Tasks

### Task 1: Setup & Data Loading

- [x] Load CIFAR-10 dataset using `torchvision.datasets.CIFAR10`
- [x] Create `DataLoader` for training and testing sets
- [x] Apply appropriate transforms (e.g., `ToTensor`, normalization)

### Task 2: Implement Base LeNet-5 (No Dropout / No BatchNorm)

- [x] Adapt the provided `LeNet5` class for CIFAR-10:
  - Change input channels from `1` to `3` (CIFAR-10 is RGB)
  - Keep architecture structure: C1 → S2 → C3 → S4 → C5 → F6 → F7
- [x] Use `nn.ReLU` activations (as shown in sample code)
- [x] Use `nn.LogSoftmax(dim=-1)` on output
- [x] Train the model
- [x] Record: training accuracy, test accuracy, training time
  - Train: 76.67% | Test: 62.24% | Time: 65.1s

### Task 3: Implement LeNet-5 with Dropout

- [x] Add at least one `nn.Dropout` layer to the network
  - Common placement: after fully connected layers
- [x] Train with same/similar hyperparameters
- [x] Record: training accuracy, test accuracy, training time
  - Train: 72.02% | Test: 64.34% | Time: 65.0s

### Task 4: Implement LeNet-5 with Batch Normalization

- [x] Add at least one `nn.BatchNorm2d` (for conv layers) or `nn.BatchNorm1d` (for FC layers)
  - Insert BN layers after conv/linear layers, before activations
- [x] Train with same/similar hyperparameters
- [x] Record: training accuracy, test accuracy, training time
  - Train: 82.69% | Test: 66.47% | Time: 68.8s

### Task 5: Compare & Document Results

- [x] Create a comparison of all three models:

  | Model     | Train Acc | Test Acc | Time  |
  |-----------|-----------|----------|-------|
  | Baseline  | 76.67%    | 62.24%   | 65.1s |
  | Dropout   | 72.02%    | 64.34%   | 65.0s |
  | BatchNorm | 82.69%    | 66.47%   | 68.8s |

- [x] Include snapshots/screenshots of training curves — saved to `training_curves.png`
- [x] Write brief analysis comparing dropout vs batchnorm vs baseline

  **Analysis:** All three models exceed the 50% test accuracy threshold. The **Baseline** achieved 62.24% test accuracy but shows signs of overfitting (train/test gap widening after epoch 10). **Dropout** slightly reduced overfitting — lower training accuracy (72%) but better generalization (64.34% test), with the train/test gap closing. **BatchNorm** was the clear winner: fastest to converge (49% test acc at epoch 1 vs ~48% for others), highest train accuracy (82.69%), and best test accuracy (66.47%), though it also shows some overfitting. BatchNorm's advantage comes from stabilizing gradients and acting as a mild regularizer across all conv layers.

---

## Reference Architecture

```
Input: 3x32x32  (changed from 1 channel to 3 for CIFAR-10)
C1:  Conv2d(3→6, 5x5) → ReLU → 6@28x28
S2:  MaxPool2d(2x2, stride=2) → 6@14x14
C3:  Conv2d(6→16, 5x5) → ReLU → 16@10x10
S4:  MaxPool2d(2x2, stride=2) → 16@5x5
C5:  Conv2d(16→120, 5x5) → ReLU → 120@1x1
F6:  Linear(120→84) → ReLU
F7:  Linear(84→10) → LogSoftmax
```

### For Dropout variant

- Add `nn.Dropout(p=0.x)` — recommended after F6 or between FC layers

### For BatchNorm variant

- Add `nn.BatchNorm2d()` after conv layers, or `nn.BatchNorm1d()` after FC layers
- Order: Conv/Linear → BatchNorm → Activation

---

## Suggested Hyperparameters (adjustable)

| Parameter | Suggested Value |
|-----------|----------------|
| Learning rate | 0.001 – 0.01 |
| Batch size | 64 – 128 |
| Epochs | 20 – 50 |
| Optimizer | Adam or SGD |
| Loss | `nn.NLLLoss()` (since output uses LogSoftmax) |

---

## Deliverables

1. **Source code** — complete, runnable PyTorch implementation
2. **Results for 3 models:**
   - Baseline (no dropout/batchnorm)
   - With dropout
   - With batch normalization
3. **Screenshots** of train/test accuracy curves
4. **Training time** for each model
5. **Comparison analysis**

---

## Base Code Template

```python
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),   # ← Change 1→3
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

---

## Key Notes

- Use `torch.nn.NLLLoss()` as loss function (output is LogSoftmax)
- CIFAR-10 has 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- All images are 32x32 RGB
- Test accuracy MUST exceed 50%
- `torch.nn` has built-in `BatchNorm2d`, `BatchNorm1d`, and `Dropout` layers — use them

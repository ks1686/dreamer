import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        return self.net(x.view(-1, 784))


def train(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    acc = 100.0 * correct / total
    print(f"Epoch {epoch:2d} | Train Loss: {total_loss/total:.4f} | Train Acc: {acc:.2f}%")
    return acc


def test(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    acc = 100.0 * correct / total
    print(f"           Test  Loss: {total_loss/total:.4f} | Test  Acc: {acc:.2f}%")
    return acc


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root="./data", train=True,  download=True, transform=transform),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(
        datasets.MNIST(root="./data", train=False, download=True, transform=transform),
        batch_size=BATCH_SIZE, shuffle=False)

    model     = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nModel: 784 -> 200 (ReLU) -> 50 (ReLU) -> 10 (Softmax)")
    print(f"Optimizer: Adam | LR: {LEARNING_RATE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}\n")

    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, criterion, epoch)
        test(model, test_loader, criterion)

    print(f"\nTotal training time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

import time
import pickle
import numpy as np
from pathlib import Path

MNIST_PKL = Path(__file__).parent / "data" / "mnist_numpy.pkl"
HW1_PKL   = Path(__file__).parent.parent / "homework 1" / "mnist.pkl"


def load_mnist():
    if HW1_PKL.exists():
        with open(HW1_PKL, "rb") as f:
            mnist = pickle.load(f)
        X_train = mnist["training_images"].astype(np.float32) / 255.0
        y_train = mnist["training_labels"].astype(np.int32)
        X_test  = mnist["test_images"].astype(np.float32) / 255.0
        y_test  = mnist["test_labels"].astype(np.int32)
        print("Loaded MNIST from HW1 pickle.")
        return X_train, y_train, X_test, y_test

    import gzip
    from urllib import request

    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        ("train-images-idx3-ubyte.gz", "images", 16),
        ("train-labels-idx1-ubyte.gz", "labels",  8),
        ("t10k-images-idx3-ubyte.gz",  "images", 16),
        ("t10k-labels-idx1-ubyte.gz",  "labels",  8),
    ]
    MNIST_PKL.parent.mkdir(parents=True, exist_ok=True)
    arrays = []
    for fname, kind, offset in files:
        dest = MNIST_PKL.parent / fname
        if not dest.exists():
            print(f"Downloading {fname} ...")
            request.urlretrieve(base + fname, dest)
        with gzip.open(dest, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=offset)
        if kind == "images":
            data = data.reshape(-1, 784)
        arrays.append(data)

    X_train, y_train, X_test, y_test = arrays
    return (X_train.astype(np.float32) / 255.0, y_train.astype(np.int32),
            X_test.astype(np.float32)  / 255.0, y_test.astype(np.int32))


def relu(x):       return np.maximum(0, x)
def relu_grad(x):  return (x > 0).astype(np.float32)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class FCNet:
    def __init__(self, lr=0.01):
        self.lr = lr
        rng = np.random.default_rng(42)
        self.W1 = rng.standard_normal((784, 200)).astype(np.float32) * np.sqrt(2.0 / 784)
        self.b1 = np.zeros((1, 200), dtype=np.float32)
        self.W2 = rng.standard_normal((200,  50)).astype(np.float32) * np.sqrt(2.0 / 200)
        self.b2 = np.zeros((1,  50), dtype=np.float32)
        self.W3 = rng.standard_normal(( 50,  10)).astype(np.float32) * np.sqrt(2.0 /  50)
        self.b3 = np.zeros((1,  10), dtype=np.float32)

    def forward(self, X):
        self.X  = X
        self.z1 = X        @ self.W1 + self.b1;  self.a1 = relu(self.z1)
        self.z2 = self.a1  @ self.W2 + self.b2;  self.a2 = relu(self.z2)
        self.z3 = self.a2  @ self.W3 + self.b3;  self.out = softmax(self.z3)
        return self.out

    def loss(self, probs, y):
        return -np.log(probs[np.arange(y.shape[0]), y] + 1e-9).mean()

    def backward(self, y):
        n = y.shape[0]
        dz3 = self.out.copy();  dz3[np.arange(n), y] -= 1;  dz3 /= n

        dW3 = self.a2.T @ dz3;  db3 = dz3.sum(axis=0, keepdims=True)
        dz2 = (dz3 @ self.W3.T) * relu_grad(self.z2)
        dW2 = self.a1.T @ dz2;  db2 = dz2.sum(axis=0, keepdims=True)
        dz1 = (dz2 @ self.W2.T) * relu_grad(self.z1)
        dW1 = self.X.T  @ dz1;  db1 = dz1.sum(axis=0, keepdims=True)

        self.W3 -= self.lr * dW3;  self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


LEARNING_RATE = 0.01
BATCH_SIZE    = 128
EPOCHS        = 20


def main():
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    net = FCNet(lr=LEARNING_RATE)
    n   = X_train.shape[0]
    rng = np.random.default_rng(0)

    print(f"\nModel: 784 -> 200 (ReLU) -> 50 (ReLU) -> 10 (Softmax)")
    print(f"Optimizer: SGD | LR: {LEARNING_RATE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}\n")

    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        idx = rng.permutation(n)
        X_sh, y_sh = X_train[idx], y_train[idx]

        epoch_loss, steps = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            Xb, yb = X_sh[i:i+BATCH_SIZE], y_sh[i:i+BATCH_SIZE]
            epoch_loss += net.loss(net.forward(Xb), yb)
            net.backward(yb)
            steps += 1

        train_acc = 100.0 * (net.predict(X_train) == y_train).mean()
        test_acc  = 100.0 * (net.predict(X_test)  == y_test).mean()
        print(f"Epoch {epoch:2d} | Loss: {epoch_loss/steps:.4f} "
              f"| Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    print(f"\nTotal training time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

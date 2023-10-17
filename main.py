from MNISTDataset import MNISTDataset, MNISTDatasetReader
from MNISTClassifier import MNISTClassifier

from torch.utils.data import DataLoader

import torch

import time

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'Using "{device}" device')


m = MNISTDataset("./data/train.csv")
t = MNISTDataset("./data/test.csv", False)

train, test = MNISTDatasetReader("./data/train.csv", 0.9).getDatasets()
val = MNISTDatasetReader("./data/test.csv", 1).getDatasets()

train, test, val = (
    MNISTDataset(train, True),
    MNISTDataset(test, True),
    MNISTDataset(val),
)

BATCH_SIZE = 128

train_loader = DataLoader(train, BATCH_SIZE)

test_loader = DataLoader(test, BATCH_SIZE)

val_loader = DataLoader(val, BATCH_SIZE)

model = MNISTClassifier().to(device)

print(model)


def decodeLabel(num: torch.Tensor):
    size = num.size(0)
    labels = torch.zeros((size, 10), dtype=torch.float32)

    for i in range(size):
        n = num[i]
        labels[i][n] = 1
    return labels


loss_fn = torch.nn.CrossEntropyLoss()

optim = torch.optim.Adam(model.parameters(), lr=1e-5)

TRAIN_SIZE = len(train_loader.dataset)
TEST_SIZE = len(test_loader.dataset)


EPOCHS = 100

for currentEpoch in range(EPOCHS):
    print(f"EPOCH: {currentEpoch+1}|{EPOCHS}\n", "-" * 10, sep="")

    model.train()
    for batch, (label, img) in enumerate(train_loader):
        label, img = label.to(device), img.to(device)
        pred = model(img)
        expected = decodeLabel(label).to(device)
        loss: torch.Tensor = loss_fn(pred, expected)

        loss.backward()

        optim.step()
        optim.zero_grad()

        if batch % 100 == 0:
            current = (batch + 1) * BATCH_SIZE
            print(f"Loss: {loss.item():>4f} | Current: [{current}/{TRAIN_SIZE}]")

    model.eval()
    test_loss, correct = 0, 0

    NUM_BATCHES = len(test_loader)

    with torch.no_grad():
        for label, image in test_loader:
            label, image = label.to(device), image.to(device)
            pred = model(image)
            expected = decodeLabel(label).to(device)
            test_loss += loss_fn(pred, expected).item()
            correct += (
                (pred.argmax(1) == expected.argmax(1)).type(torch.float).sum().item()
            )

    test_loss /= NUM_BATCHES
    correct /= TEST_SIZE
    print(
        f"\nTest Error:\nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )

timestamp = time.time()
torch.save(model, f"models/mnist-{timestamp}-{100*correct:.0f}.pth")

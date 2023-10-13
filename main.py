from MNISTDataset import MNISTDataset

from torch.utils.data import DataLoader

import torch


class MNISTClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.IN_SIZE = 784

        self.h1 = torch.nn.Linear(self.IN_SIZE, self.IN_SIZE * 2)
        self.a1 = torch.nn.ReLU()
        self.h2 = torch.nn.Linear(self.IN_SIZE * 2, 512)
        self.a2 = torch.nn.ReLU()
        self.out = torch.nn.Linear(512, 10)
        self.a3 = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = torch.nn.Flatten()(x)
        x = self.h1(x)
        x = self.a1(x)
        x = self.h2(x)
        x = self.a2(x)
        x = self.out(x)
        return self.a3(x)


m = MNISTDataset("./data/train.csv")
t = MNISTDataset("./data/test.csv", False)


train_loader = DataLoader(
    m,
    20,
    True,
)

test_loader = DataLoader(t, 10, True)

model = MNISTClassifier()

print(model)


def decodeLabel(num: torch.Tensor):
    size = num.size(0)
    labels = torch.zeros((size, 10), dtype=torch.float32)

    for i in range(size):
        n = num[i]
        labels[i][n] = 1
    return labels


loss_fn = torch.nn.CrossEntropyLoss()

optim = torch.optim.Adam(model.parameters(), lr=1e-3)

TRAIN_SIZE = len(train_loader.dataset)
TEST_SIZE = len(test_loader.dataset)


EPOCHS = 10

for currentEpoch in range(EPOCHS):
    print(f"EPOCH: {currentEpoch+1}|{EPOCHS}\n", "-" * 10)

    model.train()
    for batch, (label, img) in enumerate(train_loader):
        pred = model(img)
        expected = decodeLabel(label)
        loss: torch.Tensor = loss_fn(pred, expected)

        loss.backward()

        optim.step()
        optim.zero_grad()

        if batch % 100 == 0:
            current = (batch + 1) * len(label)
            print(f"Loss: {loss.item():>4f} | Current: [{current}/{TRAIN_SIZE}]")

    # model.eval()
    # test_loss, correct = 0, 0

    # with torch.no_grad():
    #     for image in test_loader:
    #         pred = model(image)
    #         expected = decodeLabel(label)
    #         test_loss = loss_fn(pred, expected).item()
    #         correct += (pred.argmax(1) == expected).type(torch.float).sum().item()

    # test_loss /= len(test_loader)
    # correct /= TEST_SIZE
    # print(
    #     f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    # )

torch.save(model, "mnist.pth")

print("Size of train:", len(m))
print("Size of test:", len(t))

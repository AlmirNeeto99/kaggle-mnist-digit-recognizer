import torch


class MNISTClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.h1 = torch.nn.Conv2d(1, 128, 3, 1)  # 26
        self.a1 = torch.nn.ReLU()
        self.h2 = torch.nn.Conv2d(128, 256, 3, 1)  # 24
        self.a2 = torch.nn.ReLU()
        self.f = torch.nn.Flatten()
        self.out = torch.nn.Linear(256 * 24 * 24, 10)
        self.outA = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = torch.nn.Unflatten(1, (1, 28, 28))(x)
        x = self.h1(x)
        x = self.a1(x)
        x = self.h2(x)
        x = self.a2(x)
        x = self.f(x)
        x = self.out(x)
        return self.outA(x)

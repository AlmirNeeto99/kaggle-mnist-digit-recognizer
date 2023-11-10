import torch


class MNISTClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # self.h1 = torch.nn.Conv2d(1, 64, 3, 1)  # 26
        # self.a1 = torch.nn.ReLU()
        # self.p1 = torch.nn.MaxPool2d(2, stride=2)  # 13
        # self.h2 = torch.nn.Conv2d(64, 128, 3, 1)  # 11
        # self.a2 = torch.nn.ReLU()
        # self.h3 = torch.nn.Conv2d(128, 256, 3, 1)  # 9
        # self.a3 = torch.nn.ReLU()
        # self.h4 = torch.nn.Conv2d(256, 512, 3, 1) # 7
        # self.a4 = torch.nn.ReLU()
        # self.f = torch.nn.Flatten()
        # self.out = torch.nn.Linear(512 * 7 * 7, 10000)
        # self.outA = torch.nn.ReLU()
        # self.out2 = torch.nn.Linear(10000, 10)
        self.h1 = torch.nn.Linear(784, 512)
        self.a1 = torch.nn.ReLU()
        self.h2 = torch.nn.Linear(512, 256)
        self.a2 = torch.nn.ReLU()
        self.out = torch.nn.Linear(256, 10)
        self.outA = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x = torch.nn.Unflatten(1, (1, 28, 28))(x)
        # x = self.h1(x)
        # x = self.a1(x)
        # x = self.p1(x)
        # x = self.h2(x)
        # x = self.a2(x)
        # x = self.h3(x)
        # x = self.a3(x)
        # x = self.h4(x)
        # x = self.a4(x)
        # x = self.f(x)
        # x = self.out(x)
        # x = self.outA(x)
        # x = self.out2(x)
        x = self.h1(x)
        x = self.a1(x)
        x = self.h2(x)
        x = self.a2(x)
        x = self.out(x)
        return self.outA(x)

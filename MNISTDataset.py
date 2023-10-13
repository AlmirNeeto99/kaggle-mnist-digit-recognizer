from torch import tensor, float32
from torch.utils.data import Dataset
import pandas as pd


class MNISTDataset(Dataset):
    def __init__(self, dataPath, train=True):
        self.data = pd.read_csv(dataPath)
        self.train = train

    def __getitem__(self, idx):
        line = self.data.iloc[idx]
        if self.train:
            return line.iloc[0], tensor(line.iloc[1:].to_list(), dtype=float32)
        else:
            return tensor(line.iloc[0:].to_list(), dtype=float32)

    def __len__(self):
        return self.data.shape[0]

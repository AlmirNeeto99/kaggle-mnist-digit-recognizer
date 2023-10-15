from torch import tensor, float32
from torch.utils.data import Dataset
import pandas as pd


class _PandasDataset(Dataset):
    def __init__(self, dataPath):
        self.data = pd.read_csv(dataPath)


class MNISTDatasetReader:
    def __init__(self, path, division=0.7) -> None:
        self.path = path
        self.division = division
        self.data = _PandasDataset(self.path)

    def getDatasets(self):
        dt = self.data.data

        length = len(dt)
        cutSize = length * self.division

        if self.division == 1:
            return dt

        return dt.loc[:cutSize], dt.loc[cutSize:].reset_index(drop=True)


class MNISTDataset(Dataset):
    def __init__(self, pandas, train=True):
        self.data = pandas
        self.train = train

    def __getitem__(self, idx):
        line = self.data.iloc[idx]
        if self.train:
            return line.iloc[0], tensor(line.iloc[1:].to_list(), dtype=float32)
        else:
            return tensor(line.iloc[0:].to_list(), dtype=float32)

    def __len__(self):
        return self.data.shape[0]

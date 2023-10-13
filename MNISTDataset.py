from torch.utils.data import Dataset
import pandas as pd

class MNISTDataset(Dataset):

    def __init__(self, dataPath):
        self.data = pd.read_csv(dataPath)

    def __getitem__(self, idx):
        line = self.data.iloc[idx]
        return line.iloc[0], line.iloc[1:].to_list()

    def __len__(self):
        return self.data.shape[0]

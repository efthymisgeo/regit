from torch.utils.data import Dataset
import torch
import numpy as np
from mnist import 

class DatasetNACT(Dataset):
    """
        Dataset for loading neural activation patterns
    """
    def __init__(self, x_df, y_df):
        self.x_df = x_df
        self.y_df = y_df

    def __len__(self):
        return len(self.y_df)

    def __getitem__(self, index):
        nacts = torch.tensor(self.x_df.iloc[index, :].astype(np.float32).values)
        label = torch.tensor(self.y_df[index].astype(int))
        return nacts, label

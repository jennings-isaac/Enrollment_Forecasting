import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class EnrollmentDataset(Dataset):
    def __init__(self, csv_files, target_column='ACTUAL_ENROLL'):
        """
        Args:
            csv_files (list of str): List of CSV file paths.
            target_column (str): The name of the target column.
        """
        df_list = [pd.read_csv(file) for file in csv_files]
        self.data = pd.concat(df_list, ignore_index=True)
        self.features = self.data.drop(columns=[target_column]).to_numpy(dtype=np.float32)
        self.targets = self.data[target_column].to_numpy(dtype=np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.targets[idx])
        return x, y

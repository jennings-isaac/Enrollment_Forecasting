# import torch
import pandas as pd
from torch.utils.data import Dataset
# from sklearn.preprocessing import LabelEncoder

class EnrollmentDataset(Dataset):
    def __init__(self, data_paths):
        # Load multiple CSV files if necessary
        if isinstance(data_paths, list):
            self.data = pd.concat([pd.read_csv(f) for f in data_paths], ignore_index=True)
        else:
            self.data = pd.read_csv(data_paths)

        # Extract features and targets
        self.features = self.data.drop(columns=['ACTUAL_ENROLL']).values
        self.targets = self.data["ACTUAL_ENROLL"].values     
 
    def __len__(self):
        # Return the total number of data points
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return one data point and its label
        x = self.features.iloc[idx].values
        y = self.targets.iloc[idx]
        return x, y


# import torch
import pandas as pd
from torch.utils.data import Dataset
# from sklearn.preprocessing import LabelEncoder

class EnrollmentDataset(Dataset):
    def __init__(self, data_path):
        # Load the dataset from a CSV file
        self.data = pd.read_csv(data_path)

        # create features (predictors) and targets (reactors)
        self.features = self.data.drop(columns=['ACTUAL_ENROLL'])
        self.targets = self.data["ACTUAL_ENROLL"]      
 
    def __len__(self):
        # Return the total number of data points
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return one data point and its label
        x = self.features.iloc[idx].values
        y = self.targets.iloc[idx]
        return x, y


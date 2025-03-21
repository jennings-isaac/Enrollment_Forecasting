import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Import your model and EarlyStopping from separate files
from pytorch_model import RegressionNN
from early_stopping import EarlyStopping
from enrollment_dataset import EnrollmentDataset

def r2_score(y_true, y_pred):
    y_true_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def pytorch_load_data():
    test_csv_file = ['data/partition_1.csv']
    test_dataset = EnrollmentDataset(test_csv_file, target_column="ACTUAL_ENROLL")
    test_x = torch.tensor(test_dataset.features)
    test_y = torch.tensor(test_dataset.targets)
    return test_x, test_y

# Trainer class encapsulating training and evaluation
class PyTorchTrainer:
    def __init__(
        self,
        input_size, hidden_sizes,
        learning_rate, batch_size,
        num_epochs=500, patience=100
    ):
        quarters = [
    'data/partition_2.csv','data/partition_3.csv','data/partition_4.csv','data/partition_5.csv',
    'data/partition_6.csv','data/partition_7.csv','data/partition_8.csv','data/partition_9.csv'
    ]

        # For example, leave out the first 4 quarters for validation
        dev_csv_files = quarters[:4]
        train_csv_files = [file for file in quarters if file not in dev_csv_files]
        fold_name = "_".join([q.replace("data/", "").replace(".csv", "") for q in dev_csv_files])

        train_dataset = EnrollmentDataset(train_csv_files, target_column="ACTUAL_ENROLL")
        dev_dataset = EnrollmentDataset(dev_csv_files, target_column="ACTUAL_ENROLL")
        
        # Convert the datasets into tensors
        train_x = torch.tensor(train_dataset.features)
        train_y = torch.tensor(train_dataset.targets)
        dev_x = torch.tensor(dev_dataset.features)
        dev_y = torch.tensor(dev_dataset.targets)
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.fold_name = fold_name

        # Initialize model, loss function, optimizer, and early stopping
        self.model = RegressionNN(input_size, hidden_sizes)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.early_stopping = EarlyStopping(patience=patience, save_path="best_model.pth", verbose=False)

        # Create DataLoaders for training and validation
        self.train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
        self.dev_loader = DataLoader(TensorDataset(dev_x, dev_y), batch_size=batch_size, shuffle=False)

    def train_and_evaluate(self):
        hidden_config_str = "x".join(str(h) for h in self.hidden_sizes)
        self.train_x = train_x
        self.train_y = train_y
        self.dev_x = dev_x
        self.dev_y = dev_y
        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            for batch_x, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x).squeeze(1)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

            # Evaluate on validation set
            self.model.eval()
            with torch.no_grad():
                val_losses = []
                for val_x, val_y in self.dev_loader:
                    val_outputs = self.model(val_x).squeeze(1)
                    val_loss = self.criterion(val_outputs, val_y)
                    val_losses.append(val_loss.item())
                avg_val_loss = sum(val_losses) / len(val_losses)

            

            # Check early stopping condition
            self.early_stopping(avg_val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # Load the best model and evaluate final performance
        self.model.load_state_dict(torch.load("best_model.pth"))
        self.model.eval()
        with torch.no_grad():
            val_losses = []
            all_val_outputs, all_val_targets = [], []
            for val_x, val_y in self.dev_loader:
                val_outputs = self.model(val_x).squeeze(1)
                loss = self.criterion(val_outputs, val_y)
                val_losses.append(loss.item())
                all_val_outputs.append(val_outputs)
                all_val_targets.append(val_y)
            final_val_loss = sum(val_losses) / len(val_losses)
            all_val_outputs = torch.cat(all_val_outputs)
            all_val_targets = torch.cat(all_val_targets)
            best_val_r2 = r2_score(all_val_targets, all_val_outputs).item()
        
        
        return final_val_loss, best_val_r2
    def test(self):
        test_csv_file = ['data/partition_1.csv']
        test_dataset = EnrollmentDataset(test_csv_file, target_column="ACTUAL_ENROLL")
        test_x = torch.tensor(test_dataset.features)
        test_y = torch.tensor(test_dataset.targets)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            test_losses = []
            all_outputs, all_targets = [], []
            for x, y in test_loader:
                outputs = self.model(x).squeeze(1)
                loss = self.criterion(outputs, y)
                test_losses.append(loss.item())
                all_outputs.append(outputs)
                all_targets.append(y)
            final_test_loss = sum(test_losses) / len(test_losses)
            all_outputs = torch.cat(all_outputs)
            all_targets = torch.cat(all_targets)
            test_r2 = r2_score(all_targets, all_outputs).item()
            print("Test Loss:", final_test_loss)
            print("Test R² Score:", test_r2)
        return final_test_loss, test_r2
    def load_model(self, model_path="best_model.pth"):
        
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print(f"Model loaded from {model_path}")
# Example usage:

if __name__ == "__main__":
    # Dummy get_data function (replace with your actual data loading logic)


    # List of CSV files (update paths as needed)
    train_x, train_y, dev_x, dev_y,fold_name = Setup_data()
    input_size = train_x.shape[1]
    


    # Create an instance of the trainer class with desired hyperparameters
    trainer = PyTorchTrainer(       
        input_size, [200, 200, 200, 200],
        learning_rate=0.0001, batch_size=8,
        num_epochs=500, patience=100,

    )
    print("ASDASD")

    final_loss, final_r2 = trainer.train_and_evaluate()
    print("Final Validation Loss:", final_loss)
    print("Final R² Score:", final_r2)

    test_x, test_y = pytorch_load_data()
    trainer.load_model("best_model.pth")
    test_loss, test_r2 = trainer.test(test_x, test_y)
    print("Test Loss:", test_loss)
    print("Test R² Score:", test_r2)

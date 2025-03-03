from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error, mean_squared_error

from dataset import EnrollmentDataset
from sklearn_model import sklearnModel

class Trainer:
    @staticmethod
    def main():
        train_data_path = 'data/inv_fold_1.csv'
        train_dataset = EnrollmentDataset(train_data_path)

        val_data_path = 'data/fold_1.csv'
        val_dataset = EnrollmentDataset(val_data_path)
        
        # Convert the entire dataset from torch tensors to numpy arrays
        X_train, y_train = train_dataset.features, train_dataset.targets
        X_val, y_val = val_dataset.features, val_dataset.targets
        input_dim = X_train.shape[1]
        

        # Initialize the model
        model_type = 'mlp_regressor'
        # model_type = 'random_forest'  
        model = sklearnModel(input_dim, model_type)

        
        
        # Train the model
        model.train(X_train, y_train)
        
        # Evaluate the model on the validation set
        y_pred = model.evaluate(X_val)

        
        
        # Calculate Root Mean Squared Error (RMSE)
        rmse = root_mean_squared_error(y_val, y_pred)
        print(f'Validation RMSE: {rmse:.2f}')
        
        # Calculate R-squared score
        print(f'R² Score: {r2_score(y_val, y_pred):.2f}')

        return val_data_path, y_val, y_pred
      
    if __name__ == '__main__':
        main()


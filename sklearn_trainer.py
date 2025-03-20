from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error, mean_squared_error
import os
import joblib
import pandas as pd
from glob import glob
from dataset import EnrollmentDataset
from sklearn_model import sklearnModel



def load_data():
    # load the data from the joblib file
    data_dir = 'data/'
    file_pattern = os.path.join(data_dir, 'partition_*.csv')
    all_partitions = glob(file_pattern)

    # set the training and validation datasets
    val_data_path = 'data/partition_1.csv'

    train_partitions = [i for i in all_partitions if i != val_data_path]

    train_dataset = EnrollmentDataset(train_partitions)

    val_dataset = EnrollmentDataset(val_data_path)

    X_train, y_train = train_dataset.features, train_dataset.targets
    X_val, y_val = val_dataset.features, val_dataset.targets

    return  val_data_path, X_train, y_train, X_val, y_val




def train_model(model_type, X_train, y_train):
    # train the model
    input_dim = X_train.shape[1]
    model = sklearnModel(input_dim, model_type)
    model.train(X_train, y_train)


    # Save the model
    model_filename = 'sklearn_model.joblib'
    joblib.dump(model, model_filename)

    return model_filename

def load_trained_model(model_filename):
    # fetch model saved in the joblib file
    return joblib.load(model_filename)

def evaluate_model(model, X_val):
    # Evaluate the model on the validation set
    return model.evaluate(X_val)

def report_results(y_val, y_pred):
    # calculate amd output validation scores
    print(f'Validation RMSE: {root_mean_squared_error(y_val, y_pred):.2f}')
    print(f'R² Score: {r2_score(y_val, y_pred):.2f}')

    results = pd.DataFrame({'Actual': y_val, 'Predicted': y_pred})
    print(results)

# "main" method, calls all needed functions
def sklearn_train(model_type):
    val_data_path, X_train, y_train, X_val, y_val = load_data()

    model_filename = train_model(model_type, X_train, y_train)

    model = load_trained_model(model_filename)

    y_pred = evaluate_model(model, X_val)

    report_results(y_val, y_pred)


# Usage
# sklearn_train('mlp_regressor')
# sklearn_train('random_forest')
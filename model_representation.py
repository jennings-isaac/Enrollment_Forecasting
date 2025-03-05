import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from sklearn_trainer import load_data, evaluate_model

def get_category_from_dummy(row, prefix):

    for col in row.index:
        if col.startswith(prefix) and row[col] == 1:
            return col.replace(prefix, "")
    return "Unknown Class"

def plot_results():

    model_filename = 'sklearn_model.joblib'
    model = joblib.load(model_filename)
    
    val_data_path, X_train, y_train, X_val, y_val = load_data()
    y_pred = evaluate_model(model, X_val)
    df = pd.read_csv(val_data_path)

    df['Ground_Truth'] = y_val
    df['Predicted'] = y_pred
    df['Difference'] = abs(df['Ground_Truth'] - df['Predicted'])


    num_classes = 25
    top_differences = df.nlargest(num_classes, 'Difference')

    # Extract quarter and year from dummy-encoded columns
    top_differences['Quarter'] = top_differences.apply(lambda row: get_category_from_dummy(row, "TERM_"), axis=1)
    top_differences['Year'] = top_differences.apply(lambda row: get_category_from_dummy(row, "YEAR_"), axis=1)

    # Construct the x-axis label
    top_differences['Class_Label'] = top_differences['Quarter'] + " " +\
                                    top_differences['Year']+ " " + \
                                    top_differences['COURSE_NUMBER'].astype(str)
                                    

    classes = top_differences['Class_Label'].to_list()
    y_val_subset = top_differences['Ground_Truth'].to_list()
    y_pred_subset = top_differences['Predicted'].to_list()




    x = np.arange(len(classes))

    # Plot the bars
    plt.bar(x - 0.35 / 2, y_val_subset, width=0.35, label='Ground Truth', alpha=0.7)
    plt.bar(x + 0.35 / 2, y_pred_subset, width=0.35, label='Predicted', alpha=0.7)


    plt.xlabel('Classes')
    plt.ylabel('Number Enrolled')
    plt.title('Top Differences between Predicted and Ground Truth Enrollment')
    plt.xticks(x, classes, rotation=45, ha='right') 
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_val, y_pred, color='blue', alpha=0.6, label="Predictions")

    # Plot y = x line in red
    min_val = min(y_val.min(), y_pred.min())
    max_val = max(y_val.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")

    # Labels and title
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()
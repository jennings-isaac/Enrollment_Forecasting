import pandas as pd
import random
import wandb
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
import matplotlib.pyplot as plt

class kfold:
    @staticmethod
    def main(model):
        wandb.init(project="kfold-validation", name="MLP_kfold")

        quarters  = ['data/WI14.csv', 'data/SU14.csv', 'data/FA14.csv', 'data/FA20.csv', 
                'data/WI15.csv', 'data/SP15.csv', 'data/SU15.csv', 'data/FA15.csv', 
                'data/WI16.csv', 'data/SU16.csv', 'data/FA16.csv', 'data/WI17.csv', 
                'data/SP17.csv', 'data/SU17.csv', 'data/FA17.csv', 'data/WI18.csv', 
                'data/SP18.csv', 'data/SU18.csv', 'data/FA18.csv', 'data/WI19.csv', 
                'data/FA19.csv', 'data/WI20.csv', 'data/SP20.csv', 'data/SU20.csv', 
                'data/WI21.csv', 'data/SP21.csv', 'data/SU21.csv', 'data/FA21.csv', 
                'data/WI22.csv', 'data/SP22.csv', 'data/SU22.csv', 'data/FA22.csv', 
                'data/WI23.csv', 'data/SP23.csv', 'data/SU23.csv', 'data/FA23.csv', 
                'data/WI24.csv', 'data/SP24.csv', 'data/SU24.csv', 'data/FA24.csv']
        
        random.shuffle(quarters)

        folds = quarters

        r2_scores = []
        rmse_scores = []
        fold_indices = []

        for i in range(0, len(folds), 4):
            print("Current Test #", (i/4)+1)
            val_set = folds[i: i+4]
            fold_indices.append(" ".join([file.replace("data/", "").replace(".csv", "") for file in val_set]))
            
            train_set = [file for file in folds if file not in val_set]

            merged_train_set = pd.concat([pd.read_csv(file) for file in train_set], ignore_index=True)
            merged_val_set = pd.concat([pd.read_csv(file) for file in val_set], ignore_index=True)
            
            merged_train_set = merged_train_set.drop_duplicates()
            merged_val_set = merged_val_set.drop_duplicates()


            print(f"Total train files: {len(train_set)}")


            print(f"Merged Train Set Size: {merged_train_set.shape}")
            print(f"Merged Validation Set Size: {merged_val_set.shape}")

            # Remove duplicates
            merged_train_set = merged_train_set.drop_duplicates()
            merged_val_set = merged_val_set.drop_duplicates()

            print(f"Train Set After Deduplication: {merged_train_set.shape}")
            print(f"Validation Set After Deduplication: {merged_val_set.shape}")

            merged_train_set.to_csv(f"data/inv_fold_{int((i/4)+1)}.csv", index=False)
            merged_val_set.to_csv(f"data/fold_{int((i/4)+1)}.csv", index=False)


            y_val = merged_val_set['ACTUAL_ENROLL']
            X_val = merged_val_set.drop(columns=['ACTUAL_ENROLL'])
            
            y_train = merged_train_set['ACTUAL_ENROLL']
            X_train = merged_train_set.drop(columns=['ACTUAL_ENROLL'])
            
            print(X_val.shape, "val shape")
            print(X_train.shape, "train shape")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            print(f'Validation RMSE: {rmse:.2f}')
            print(f'R² Score: {r2:.2f}')
            
            rmse_scores.append(rmse)
            r2_scores.append(r2)
            
            wandb.log({"fold": (i/4)+1, "rmse": rmse, "r2": r2})
            print()
        
        print(sum(r2_scores) / len(r2_scores))
        
        fig, ax1 = plt.subplots()
        ax1.plot(fold_indices, rmse_scores, marker='o', label='RMSE', linestyle='-', color='b')
        ax1.set_xlabel('Fold Number')
        ax1.set_ylabel('RMSE', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_title('Validation RMSE and R² Score per Fold')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(fold_indices, r2_scores, marker='o', label='R²', linestyle='-', color='r')
        ax2.set_ylabel('R² Score', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        ax1.set_xticks(fold_indices)
        ax1.set_xticklabels(fold_indices, rotation=45)
        plt.tight_layout()
        plt.show()

        wandb.finish()
    
    model = MLPRegressor(
        hidden_layer_sizes=(100,),
        activation='tanh',
        solver='adam',
        learning_rate='adaptive',
        max_iter=500,
        alpha=0.0001,
        verbose=False
    )
    main(model)


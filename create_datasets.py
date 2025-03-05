import pandas as pd
import random

def Create_Data():
    data = pd.read_csv("data/machine_learning_data.csv")

    # Split by both Quarter and year
    terms = ['Winter', 'Spring', 'Summer', 'Fall']

    # Extract years dynamically
    year_columns = [col for col in data.columns if col.startswith("YEAR_")]
    years = sorted([int(col.split("_")[1]) for col in year_columns])

    quarter_data = {}
    for year in years:
        for term in terms:
            term_key = f'TERM_{term}'
            year_key = f'YEAR_{year}'
            if term_key in data.columns and year_key in data.columns:
                filtered_data = data[
                    (data[term_key] == True) &
                    (data[year_key] == True)
                ]
                if not filtered_data.empty: 
                    quarter_data[f'data/{term[:2].upper()}{year % 100}.csv'] = filtered_data


    # Get the last 4 keys from the quarter_data dictionary
    test_set_keys = set(list(quarter_data.keys())[-4:])

    test_set = pd.concat([quarter_data[key] for key in test_set_keys], ignore_index=True)
    test_set.to_csv("data/test_set.csv", index=False)



    training_quarters = [key for key in quarter_data.keys() if key not in test_set_keys]

    
    random.shuffle(training_quarters)
    fold_indices = []


    for i in range(0, len(training_quarters), 4):
        val_set = training_quarters[i: i+4]
        fold_indices.append(" ".join([file.replace("data/", "").replace(".csv", "") for file in val_set]))             

        merged_val_set = pd.concat([quarter_data[key] for key in val_set], ignore_index=True)

        # Remove duplicates
        merged_val_set = merged_val_set.drop_duplicates()

        merged_val_set.to_csv(f"data/partition_{int((i/4)+1)}.csv", index=False)


        
        

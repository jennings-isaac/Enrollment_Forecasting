import pandas as pd

class CreateData:
    @staticmethod
    def main():
            data = pd.read_csv("data/machine_learning_data.csv")

            # Split by both Quarter and year
            terms = ['Winter', 'Spring', 'Summer', 'Fall']
            years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

            for year in years:
                for term in terms:
                    term_key = f'TERM_{term}'
                    year_key = f'YEAR_{year}'
                    filtered_data = data[
                        (data[term_key] == True) &
                        (data[year_key] == True)
                    ]
                    filtered_data.to_csv(f'data/{term[:2].upper()}{year % 100}.csv', index=False)




            # 4 random quarter, have been removed from train set
            test_set = ['data/SU19.csv', 'data/SP14.csv', 'data/SP19.csv', 'data/SP16.csv'] # Never EVER train on this


            merged_test_set = pd.concat([pd.read_csv(file) for file in test_set], ignore_index=True)

            merged_test_set.to_csv('data/randomized_test_set.csv', index=False)
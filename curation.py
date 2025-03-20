import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
warnings.filterwarnings('ignore')

def curate(data):
    def visualization_curation(enrollment_data):

        # Make time columns readable and datetime objects 
        def format_time(time_float):
            time_str = f"{int(time_float // 100):02d}:{int(time_float % 100):02d}"
            return time_str
        
        enrollment_data['PRIMARY_BEGIN_TIME'] = enrollment_data['PRIMARY_BEGIN_TIME'].apply(lambda x: format_time(x) if not pd.isna(x) else x)
        enrollment_data['PRIMARY_END_TIME'] = enrollment_data['PRIMARY_END_TIME'].apply(lambda x: format_time(x) if not pd.isna(x) else x)

        enrollment_data['PRIMARY_BEGIN_TIME'] = pd.to_datetime(enrollment_data['PRIMARY_BEGIN_TIME'], format='%H:%M').dt.time
        enrollment_data['PRIMARY_END_TIME'] = pd.to_datetime(enrollment_data['PRIMARY_END_TIME'], format='%H:%M').dt.time


        # Make term column readable by mapping from numerical coding to string format
        quarter_map = {
            '10': 'Winter',
            '20': 'Spring',
            '30': 'Summer',
            '40': 'Fall'
        }


        def decode_terms(term):
            year = term[:4]
            quarter = term[4:]
            quarter = quarter_map.get(quarter, 'Unknown')
            return f"{quarter} {year}"


        enrollment_data['TERM'] = enrollment_data['TERM'].astype(str).apply(decode_terms)


        # Add start date column (estimated)
        quarter_start_dates = {
        'Winter': '01-03',
        'Spring': '03-14',
        'Summer': '06-15',
        'Fall': '09-01'
    }
        def term_to_datetime(term):
            try:
                quarter, year = term.split()
                date_str = f"{year}-{quarter_start_dates[quarter]}"
                return pd.to_datetime(date_str)
            except Exception as e:
                print(f"Error converting term '{term}': {e}")
                return pd.NaT

        enrollment_data['Start_Date'] = enrollment_data['TERM'].astype(str).apply(term_to_datetime)
        return enrollment_data

    def ml_curation(enrollment_data):


        # enrollment_data = visualization_curation(enrollment_data)
        date_cols = ['U', 'M', 'T', 'W', 'R', 'F', 'S']
        enrollment_data[date_cols] = enrollment_data[date_cols].fillna(0)
        for column in date_cols:
            enrollment_data[column] = enrollment_data[column].apply(lambda x: 1 if x != 0 else 0)

        # Drop unnecessary columns
        columns_to_drop = ['TITLE', 'PRIMARY_END_TIME', 'Start_Date']
        enrollment_data = enrollment_data.drop(columns=columns_to_drop)
        enrollment_data = enrollment_data[enrollment_data['ACTUAL_ENROLL'] > 10]

        # dummy vector the term column to what term and on what year the class took place
        enrollment_data['YEAR'] = enrollment_data['TERM'].str.split().str[-1]
        enrollment_data['TERM'] = enrollment_data['TERM'].str.split().str[0]

        enrollment_data = pd.get_dummies(enrollment_data, columns=['TERM'])
        enrollment_data = pd.get_dummies(enrollment_data, columns=['YEAR'])
        enrollment_data = pd.get_dummies(enrollment_data, columns=['CAMPUS'])
        enrollment_data = pd.get_dummies(enrollment_data, columns=['PRIMARY_INSTRUCTOR_TENURE_CODE'])
        
        # Create csci_data dataframe
        csci_adj = ['CSCI', 'CISS', 'DATA']
        csci_data = enrollment_data[enrollment_data['SUBJECT'].isin(csci_adj)]


        # Combine days to class times (M, T, W -> MTW)
        def day_combinations(row):
            return ''.join([day for day, present in zip(date_cols, row) if present == 1])
        
        csci_data['DATE_COMBINATION'] = csci_data[date_cols].apply(day_combinations, axis=1)
        unique_combinations = csci_data['DATE_COMBINATION'].unique()

        for combination in unique_combinations:
            csci_data[combination] = csci_data['DATE_COMBINATION'].apply(lambda x: 1 if x == combination else 0)

        csci_data.drop(columns=['DATE_COMBINATION'], inplace=True)
        csci_data = csci_data.dropna(subset=['PRIMARY_BEGIN_TIME'])
        csci_data = pd.get_dummies(csci_data, columns=['CAPENROLL'])

        # Turn start time into a ml readable object (turn into number of minutes since time start)
        csci_data['PRIMARY_BEGIN_TIME'] = pd.to_datetime(
            csci_data['PRIMARY_BEGIN_TIME'], format='%H:%M:%S', errors='coerce'
        )

        csci_data['BEGIN_TIME_MINUTES'] = (
            csci_data['PRIMARY_BEGIN_TIME'].dt.hour * 60 + csci_data['PRIMARY_BEGIN_TIME'].dt.minute
        )

        # Fill na
        mean_minutes = csci_data['BEGIN_TIME_MINUTES'].mean()
        csci_data['BEGIN_TIME_MINUTES'].fillna(mean_minutes, inplace=True)

        csci_data = csci_data.drop(columns=['SUBJECT'])
        csci_data['COURSE_NUMBER'] = pd.to_numeric(csci_data['COURSE_NUMBER'], errors='coerce')
        csci_data_abre = csci_data.drop(columns=['PRIMARY_BEGIN_TIME', 'CRN'])
        csci_data_abre = csci_data_abre.dropna()

        csci_data_abre['BEGIN_TIME_MINUTES'] = csci_data_abre['BEGIN_TIME_MINUTES'] / 60

        # Represent the time by two column, one is sin and one is cos, make it uniquely an ID of time in our data
        csci_data_abre['sin_time'] = np.sin(csci_data_abre['BEGIN_TIME_MINUTES'] * (np.pi / 24))
        csci_data_abre['cos_time'] = np.cos(csci_data_abre['BEGIN_TIME_MINUTES'] * (np.pi / 24))
        # csci_data_abre = csci_data_abre.astype({col: int for col in csci_data_abre.select_dtypes(include='bool').columns})

        # adding more feature 
        # newe column with the average student enroll last year 
        # sum of single year 
        sum = csci_data_abre.groupby(['TERM_Fall', 'YEAR_2014'])['ACTUAL_ENROLL'].sum().reset_index()
        

        return csci_data_abre




    vis_data = visualization_curation(data)
    vis_data.to_csv('data/visualization_data.csv', index=False)


    # ml_base_data = pd.read_csv('data/base_data.csv')

    ml_data = ml_curation(vis_data)
    ml_data.to_csv('data/machine_learning_data.csv', index=False)








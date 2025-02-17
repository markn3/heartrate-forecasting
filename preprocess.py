import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Define ICD-9 ranges and corresponding categories
icd9_ranges = [
    (1.0, 140.0),    # Infectious and parasitic diseases
    (140.0, 240.0),  # Neoplasms
    (240.0, 280.0),  # Endocrine, nutritional, and metabolic diseases, and immunity disorders
    (280.0, 290.0),  # Diseases of the blood and blood-forming organs
    (290.0, 320.0),  # Mental disorders
    (320.0, 390.0),  # Diseases of the nervous system and sense organs
    (390.0, 460.0),  # Diseases of the circulatory system
    (460.0, 520.0),  # Diseases of the respiratory system
    (520.0, 580.0),  # Diseases of the digestive system
    (580.0, 630.0),  # Diseases of the genitourinary system
    (630.0, 680.0),  # Complications of pregnancy, childbirth, and the puerperium
    (680.0, 710.0),  # Diseases of the skin and subcutaneous tissue
    (710.0, 740.0),  # Diseases of the musculoskeletal system and connective tissue
    (740.0, 760.0),  # Congenital anomalies
    (760.0, 780.0),  # Certain conditions originating in the perinatal period
    (780.0, 800.0),  # Symptoms, signs, and ill-defined conditions
    (800.0, 1000.0), # Injury and poisoning
    (1000.0, 2000.0) # Miscellaneous or supplementary classifications
]

categories = [
    'infectious', 'neoplasms', 'endocrine', 'blood', 'mental', 'nervous',
    'circulatory', 'respiratory', 'digestive', 'genitourinary', 'pregnancy',
    'skin', 'muscular', 'congenital', 'perinatal', 'misc', 'injury', 'misc'
]

# Map ICD-10 letters to ICD-9 categories
icd10_ranges = {
    'A': 'infectious',       # Infectious and parasitic diseases
    'B': 'infectious',
    'C': 'neoplasms',        # Neoplasms
    'D': 'blood',            # Diseases of blood
    'E': 'endocrine',        # Endocrine, nutritional, and metabolic diseases
    'F': 'mental',           # Mental disorders
    'G': 'nervous',          # Diseases of the nervous system
    'H': 'misc',             # Eye and Ear diseases
    'I': 'circulatory',      # Diseases of the circulatory system
    'J': 'respiratory',      # Diseases of the respiratory system
    'K': 'digestive',        # Diseases of the digestive system
    'L': 'skin',             # Diseases of the skin and subcutaneous tissue
    'M': 'muscular',         # Musculoskeletal system
    'N': 'genitourinary',    # Genitourinary system
    'O': 'pregnancy',        # Pregnancy, childbirth, and puerperium
    'P': 'perinatal',        # Perinatal period
    'Q': 'congenital',       # Congenital malformations
    'R': 'misc',             # Symptoms and signs
    'S': 'injury',           # Injury and poisoning
    'T': 'injury',           # Injury and poisoning
    'V': 'misc',             # External causes of morbidity
    'W': 'misc',
    'X': 'misc',
    'Y': 'misc',
    'Z': 'misc'              # Factors influencing health status
}

def load_data(data_path):
    """
    Load CSV files from the specified directory into a dictionary of DataFrames.
    """
    data_dict = {}
    for filename in os.listdir(data_path):
        if filename.endswith(".csv"):
            table_name = filename[:-4]  # Remove '.csv' extension
            data_dict[table_name] = pd.read_csv(os.path.join(data_path, filename))
    return data_dict

def normalize_icd9(icd_code):
    """
    Normalize ICD-9 code by adding a decimal point after the third digit.
    """
    code = str(icd_code).zfill(3)
    if len(code) > 3:
        return f"{code[:3]}.{code[3:]}"
    return code

def categorize_normalized_icd9(icd_code):
    """
    Categorize normalized ICD-9 code into categories.
    """
    try:
        code = float(icd_code)
        for i, (start, end) in enumerate(icd9_ranges):
            if start <= code < end:
                return categories[i]
    except ValueError:
        pass
    return 'unknown'

def first_letter_matching(icd_code):
    """
    Categorize ICD-10 code based on the first letter.
    """
    first_letter = str(icd_code)[0]
    return icd10_ranges.get(first_letter, 'unknown')

def process_admissions_icustays_patients(admissions, icustays, patients):
    """
    Process admissions, icustays, and patients DataFrames.
    """
    # Drop unnecessary columns
    admissions = admissions.drop(columns=['admittime', 'dischtime', 'race', 'admit_provider_id', 'language',
                                          'marital_status', 'insurance', 'edregtime', 'edouttime',
                                          'admission_location', 'discharge_location', 'deathtime'])
    icustays = icustays.drop(columns=['first_careunit', 'last_careunit', 'intime', 'outtime'])
    
    # Merge admissions and icustays on 'hadm_id'
    admissions_icustays = pd.merge(admissions, icustays, on='hadm_id', how='inner')
    
    # Drop duplicate 'subject_id' column and rename
    admissions_icustays = admissions_icustays.drop(columns=['subject_id_y']).rename(columns={'subject_id_x': 'subject_id'})
    
    # Merge with patients data
    final_df = pd.merge(admissions_icustays, patients[['subject_id', 'gender', 'anchor_age']], on='subject_id', how='inner')
    
    # Drop 'subject_id' column
    final_df = final_df.drop(columns=['subject_id'])
    
    return final_df

def process_chartevents(chartevents):
    """
    Process chartevents DataFrame.
    """
    # Ensure 'charttime' is datetime
    chartevents['charttime'] = pd.to_datetime(chartevents['charttime'])
    
    # Round 'charttime' to the nearest hour
    chartevents['charttime'] = chartevents['charttime'].dt.floor('h')
    
    # Pivot the DataFrame
    chartevents_pivoted = chartevents.pivot_table(
        index=['stay_id', 'charttime'],
        columns='itemid',
        values='valuenum',
        aggfunc='mean'
    ).reset_index()
    
    # Rename columns
    chartevents_pivoted.columns = [
        f'itemid_{col}' if isinstance(col, int) else col for col in chartevents_pivoted.columns
    ]
    
    # Fill missing values within each stay_id using backfill
    chartevents_pivoted = chartevents_pivoted.groupby('stay_id').apply(lambda group: group.fillna(method='bfill')).reset_index(drop=True)
    
    # Create 'time_idx' column
    chartevents_pivoted['time_idx'] = chartevents_pivoted.groupby('stay_id')['charttime'].transform(
        lambda x: (x - x.min()).dt.total_seconds() / 3600
    ).astype(int)
    
    # Sort the DataFrame
    chartevents_pivoted.sort_values(by=['stay_id', 'time_idx'], inplace=True)
    
    # Drop rows with too many missing values
    threshold = len(chartevents_pivoted.columns) // 2
    chartevents_pivoted.dropna(thresh=threshold, inplace=True)
    
    return chartevents_pivoted

def process_diagnoses_icd(diagnoses_icd):
    """
    Process diagnoses_icd DataFrame.
    """
    # Split into ICD9 and ICD10
    icd9_data = diagnoses_icd[diagnoses_icd['icd_version'] == 9].copy()
    icd10_data = diagnoses_icd[diagnoses_icd['icd_version'] == 10].copy()
    
    # Process ICD9 data
    icd9_data['normalized_icd_code'] = icd9_data['icd_code'].apply(normalize_icd9)
    icd9_data['category'] = icd9_data['normalized_icd_code'].apply(categorize_normalized_icd9)
    icd9_data = icd9_data[icd9_data['category'] != 'unknown']
    icd9_one_hot = pd.get_dummies(icd9_data['category'])
    icd9_one_hot = pd.concat([icd9_data[['hadm_id', 'seq_num']], icd9_one_hot], axis=1)
    icd9_final = icd9_one_hot.groupby('hadm_id').agg(
        {**{col: 'max' for col in icd9_one_hot.columns if col not in ['hadm_id', 'seq_num']}, 'seq_num': 'max'}
    ).reset_index()
    icd9_final = icd9_final.rename(columns={'seq_num': 'max_seq_num'})
    
    # Process ICD10 data
    icd10_data['category'] = icd10_data['icd_code'].apply(first_letter_matching)
    icd10_one_hot = pd.get_dummies(icd10_data['category'])
    icd10_one_hot = pd.concat([icd10_data[['hadm_id', 'seq_num']], icd10_one_hot], axis=1)
    icd10_final = icd10_one_hot.groupby('hadm_id').agg(
        {**{col: 'max' for col in icd10_one_hot.columns if col not in ['hadm_id', 'seq_num']}, 'seq_num': 'max'}
    ).reset_index()
    icd10_final = icd10_final.rename(columns={'seq_num': 'max_seq_num'})
    
    # Remove overlapping hadm_id values
    overlapping_hadm_ids = set(icd9_final['hadm_id']).intersection(set(icd10_final['hadm_id']))
    icd10_final = icd10_final[~icd10_final['hadm_id'].isin(overlapping_hadm_ids)]
    
    # Combine icd9 and icd10 data
    combined_data = pd.concat([icd9_final, icd10_final], ignore_index=True)
    
    # Drop 'unknown' category if present
    if 'unknown' in combined_data.columns:
        combined_data = combined_data.drop(columns=['unknown'])
    
    return combined_data

def fill_gaps(group):
    """
    Fill time gaps within each stay_id group.
    """
    group = group.set_index('charttime').resample('1h').asfreq()
    group = group.ffill().bfill().reset_index()
    return group

def normalize_columns(df, columns, scaler_type="standard"):
    """
    Normalize specified columns in a DataFrame.
    
    Args:
        df (pd.DataFrame): The main dataframe to normalize.
        columns (list): List of columns to normalize.
        scaler_type (str): Type of scaler to use ("standard" or "minmax").
        
    Returns:
        pd.DataFrame: DataFrame with normalized columns.
        scaler_dict (dict): Dictionary of fitted scalers for each column.
    """
    scalers = {}
    
    for col in columns:
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaler_type. Choose 'standard' or 'minmax'.")
        
        # Fit and transform the column
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler  # Store the scaler for later use (e.g., inverse transform)
    
    return df, scalers

def merge_and_clean_data(final_df, chartevents_pivoted, combined_data):
    """
    Merge final_df with chartevents_pivoted and combined_data, and perform cleaning.
    """
    # Merge final_df and chartevents_pivoted on 'stay_id'
    merged_df = pd.merge(final_df, chartevents_pivoted, on='stay_id', how='inner')

    # Merge with combined_data on 'hadm_id'
    merged_df = pd.merge(combined_data, merged_df, on='hadm_id', how='inner')

    # Remove rows with NaN in 'los'
    merged_df = merged_df.dropna(subset=['los'])

    # Drop rows where any 'itemid_' column has NaN
    itemid_cols = [col for col in merged_df.columns if col.startswith('itemid_')]
    merged_df = merged_df.dropna(subset=itemid_cols)

    # Drop duplicates
    merged_df = merged_df.drop_duplicates(subset=['stay_id', 'charttime', 'time_idx'])

    # Filter for stay_ids with maximum time_idx >= desired_length
    desired_length = 30  # Set the desired sequence length
    max_time_idx = merged_df.groupby('stay_id')['time_idx'].max()
    valid_stay_ids = max_time_idx[max_time_idx >= desired_length - 1].index  # Adjusted for zero-based indexing
    merged_df = merged_df[merged_df['stay_id'].isin(valid_stay_ids)]

    # Ensure 'charttime' is datetime
    merged_df['charttime'] = pd.to_datetime(merged_df['charttime'])

    # Calculate time differences and identify gaps > 1 hour
    merged_df['time_diff'] = merged_df.groupby('stay_id')['charttime'].diff()

    # Identify sequences within each stay_id based on time gaps
    merged_df['new_sequence'] = merged_df['time_diff'] > pd.Timedelta(hours=1)
    merged_df['sequence_id'] = merged_df.groupby('stay_id')['new_sequence'].cumsum()

    # Remove rows with gaps > 1 hour
    merged_df = merged_df[merged_df['time_diff'] <= pd.Timedelta(hours=1)]

    # Sort data
    merged_df = merged_df.sort_values(['stay_id', 'charttime'])

    # Fill gaps for each stay_id
    merged_df = merged_df.groupby('stay_id').apply(fill_gaps).reset_index(drop=True)

    # Split sequences longer than 'desired_length' into smaller sequences
    # First, sort the data properly
    merged_df = merged_df.sort_values(['stay_id', 'sequence_id', 'charttime'])

    # Within each (stay_id, sequence_id), create a new 'subsequence_id' that increments every 'desired_length' rows
    merged_df['subsequence_id'] = merged_df.groupby(['stay_id', 'sequence_id']).cumcount() // desired_length

    # Combine 'stay_id', 'sequence_id', and 'subsequence_id' to create a unique sequence identifier
    merged_df['full_sequence_id'] = merged_df.apply(
        lambda x: f"{int(x['stay_id'])}_{int(x['sequence_id'])}_{int(x['subsequence_id'])}", axis=1
    )

    # Keep only sequences that have at least 'desired_length' rows
    sequence_counts = merged_df.groupby('full_sequence_id').size()
    valid_sequences = sequence_counts[sequence_counts >= desired_length].index
    merged_df = merged_df[merged_df['full_sequence_id'].isin(valid_sequences)].reset_index(drop=True)

    # Trim sequences to the desired length
    merged_df = merged_df.groupby('full_sequence_id').head(desired_length).reset_index(drop=True)

    # Recalculate 'time_idx' within each sequence
    merged_df['time_idx'] = merged_df.groupby('full_sequence_id').cumcount()

    # One-hot encode the 'admission_type' column
    merged_df = pd.get_dummies(merged_df, columns=['admission_type'], prefix='admission_type', drop_first=False)

    # Define the column name mapping
    column_mapping = {
        'itemid_220045': 'heart_rate',
        'itemid_220179': 'systolic',
        'itemid_220180': 'diastolic',
        'itemid_220181': 'mean',
        'itemid_220210': 'respiratory_rate',
        'itemid_220277': 'oxygen_saturation',
        'itemid_220546': 'pulse_oximetry',
        'itemid_223761': 'temperature_f',
        "admission_type_AMBULATORY OBSERVATION": 'amb_obs',
        "admission_type_DIRECT EMER.": 'dir_em',
        "admission_type_DIRECT OBSERVATION": 'dir_obs',
        "admission_type_ELECTIVE": 'ele',
        "admission_type_EU OBSERVATION": 'obs',
        "admission_type_EM EMER.": 'em',
        "admission_type_OBSERVATION ADMIT": 'oms_ad',
        "admission_type_SURGICAL SAME DAY ADMISSION": 'sd_adm',
        "admission_type_URGENT": 'urg',
        "admission_type_EW EMER.": 'ew_em'
    }

    # Rename the columns
    merged_df = merged_df.rename(columns=column_mapping)

    # Convert 'gender' column to boolean: True for Male ('M'), False otherwise
    merged_df['is_male'] = merged_df['gender'].apply(lambda x: x == 'M')

    # Convert 'time_idx' column to integers
    merged_df['time_idx'] = merged_df['time_idx'].astype(int)

    # Convert boolean columns to 'category' dtype
    bool_columns = [
        "urg", "sd_adm", "oms_ad", "ew_em", "obs", "ele",
        "dir_obs", "dir_em", "amb_obs",
        # Add other boolean disease category columns here
        "blood", "circulatory", "congenital", "digestive", "endocrine",
        "genitourinary", "infectious", "injury", "mental", "misc",
        "muscular", "neoplasms", "nervous", "pregnancy", "perinatal",
        "respiratory", "skin", "is_male"
    ]
    merged_df[bool_columns] = merged_df[bool_columns].astype('category')

    # Create 'future_heart_rate' by shifting 'heart_rate' within each sequence
    merged_df['future_heart_rate'] = merged_df.groupby('full_sequence_id')['heart_rate'].shift(-6)

    # Drop rows with NaN in 'future_heart_rate'
    merged_df = merged_df.dropna(subset=['future_heart_rate']).reset_index(drop=True)

    # Drop unnecessary columns
    merged_df = merged_df.drop(columns=['sequence_id', 'subsequence_id', 'new_sequence', 'time_diff', 'hospital_expire_flag', 'hadm_id', 'gender', 'stay_id',
                                        'obs','pregnancy', 'misc', 'max_seq_num', 'dir_obs', 'muscular', 'nervous', 'urg','ew_em', 'skin','sd_adm', 'oms_ad',
                                        'mean','diastolic', 'pulse_oximetry'])
    

    # Define variables to normalize
    norm_variables = [
        "heart_rate", "systolic", "diastolic", "mean",
        "respiratory_rate", "oxygen_saturation",
        "pulse_oximetry", "temperature_f", "los", "anchor_age", "future_heart_rate"
    ]

    # # Normalize the main dataframe
    # final_df, scalers = normalize_columns(merged_df, norm_variables, scaler_type="standard")

    # Return the final dataframe
    return merged_df

def main():
    pd.set_option('future.no_silent_downcasting', True)
    data_path = './data'  # Adjust the path as needed
    tables = load_data(data_path)
    
    # Extract the necessary DataFrames
    admissions = tables['admissions']
    diagnoses_icd = tables['diagnoses_icd']
    icustays = tables['icustays']
    patients = tables['patients']
    chartevents = tables['chart_e']
    
    # Process admissions, icustays, and patients
    final_df = process_admissions_icustays_patients(admissions, icustays, patients)
    
    # Process chartevents
    chartevents_pivoted = process_chartevents(chartevents)
    
    # Process diagnoses_icd
    combined_data = process_diagnoses_icd(diagnoses_icd)
    
    # Merge and clean data
    final_cleaned_df = merge_and_clean_data(final_df, chartevents_pivoted, combined_data)

    final_cleaned_df.to_csv('no_norm_df_trim.csv', index=False)

    # Display final data
    print("Final cleaned and processed data:")
    print(final_cleaned_df)

if __name__ == "__main__":
    main()

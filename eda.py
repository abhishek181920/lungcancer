
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set display options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def explore_data(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    print("\nShape of dataset:", df.shape)
    
    print("\n--- Head ---")
    print(df.head())
    
    print("\n--- Info ---")
    print(df.info())
    
    print("\n--- Describe (Numerical) ---")
    print(df.describe())
    
    print("\n--- Describe (Categorical) ---")
    print(df.describe(include=['O']))
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    print("\n--- Target Variable Distribution (survived) ---")
    if 'survived' in df.columns:
        print(df['survived'].value_counts(normalize=True))
    else:
        print("Target column 'survived' not found.")

    # Check for duplicates
    print("\n--- Duplicates ---")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    
    # Unique values for key categorical columns
    categorical_cols = ['cancer_stage', 'smoking_status', 'treatment_type', 'gender', 'country']
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n--- Value Counts for {col} ---")
            print(df[col].value_counts())

if __name__ == "__main__":
    explore_data('dataset_med.csv')

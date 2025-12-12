
import pandas as pd
import numpy as np

df = pd.read_csv('dataset_med.csv')

# Parse dates
df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])
df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'])

# Calculate duration
df['duration_days'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days

print("\n--- Correlation with Survived ---")
# Check if survived is numeric
if df['survived'].dtype == 'object':
    df['survived'] = df['survived'].map({'yes': 1, 'no': 0})

print("Survived values:", df['survived'].unique())
print(df[['duration_days', 'survived']].corr())

print("\n--- Duration Stats by Survival ---")
print(df.groupby('survived')['duration_days'].describe())

# Check sample values for categories
cols = ['cancer_stage', 'smoking_status', 'treatment_type']
for c in cols:
    print(f"\n{c} unique values:", df[c].unique()[:10])

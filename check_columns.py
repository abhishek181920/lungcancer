
import pandas as pd
df = pd.read_csv('dataset_med.csv')
print("COLUMNS:", df.columns.tolist())
print("\nDTYPES:\n", df.dtypes)
print("\nMISSING:\n", df.isnull().sum())
print("\nSURVIVED COUNTS:\n", df['survived'].value_counts())

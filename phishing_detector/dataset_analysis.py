import pandas as pd
import os

DATASET_PATH = os.path.join(
    "data",
    "PhiUSIIL_Phishing_URL_Dataset.csv"
)

print("=" * 60)
print("PHISHING URL DATASET ANALYSIS")
print("=" * 60)

if not os.path.exists(DATASET_PATH):
    print("\nERROR: Dataset file not found!")
    print("Expected location:")
    print(DATASET_PATH)
    exit()

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")

print("\nNumber of rows:")
print(df.shape[0])

print("\nNumber of columns:")
print(df.shape[1])

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

for column in df.columns:
    print(column)

print("\n" + "=" * 60)
print("FIRST 5 RECORDS")
print("=" * 60)

print(df.head())

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

print(df.duplicated().sum())

print("\n" + "=" * 60)
print("LABEL DISTRIBUTION")
print("=" * 60)

if "label" in df.columns:
    print(df["label"].value_counts())
    print("\nPercentage:")
    print(df["label"].value_counts(normalize=True) * 100)
else:
    print("ERROR: 'label' column not found!")

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)

print("\nAnalysis completed.")
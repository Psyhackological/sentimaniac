from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Define the base path relative to the location of the script
base_path = Path(__file__).parent.parent.parent

# Construct the path to the CSV file
csv_path = base_path / 'datasets' / 'training.1600000.processed.noemoticon.csv'

# Read the CSV file
df = pd.read_csv(csv_path, encoding='iso-8859-1')


# Display the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(df.head())

# Inspect the shape of the DataFrame
print("\nShape of the DataFrame:")
print(df.shape)

# Inspect the data types of columns
print("\nData types of columns:")
print(df.dtypes)

# Summary statistics of numeric columns
print("\nSummary statistics of numeric columns:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Inspect unique values in each column
print("\nUnique values in each column:")
for column in df.columns:
    print(f"{column}: {df[column].nunique()} unique values")

# Inspect the first few rows of each column
print("\nFirst few rows of each column:")
for column in df.columns:
    print(f"{column}:")
    print(df[column].head())


# Inspect any specific row or subset of rows
rows_to_inspect = slice(15)  # Change to the specific rows you want to inspect
print(f"\nSubset of rows to inspect:")
print(df.iloc[rows_to_inspect])

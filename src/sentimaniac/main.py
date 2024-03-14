from pathlib import Path
import pandas as pd

# Define the base path relative to the location of the script
base_path = Path(__file__).parent.parent.parent

# Construct the path to the CSV file
csv_path = base_path / 'datasets' / 'training.1600000.processed.noemoticon.csv'

# Read the CSV file
df = pd.read_csv(csv_path, encoding='iso-8859-1')

# Display the first few rows of the DataFrame
print(df.head())

import pandas as pd
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Read the CSV file from the same directory as the script
# The first column is the timestamp (unnamed index column)
df = pd.read_csv(script_dir / 'general_data_csv.csv', index_col=0, parse_dates=True)

# Keep only one data point per hour (first occurrence)
df_hourly = df.resample('H').first().reset_index()

# Save to a new CSV file in the same directory
df_hourly.to_csv(script_dir / 'general_data_csv_hourly.csv', index=False)

print("New file created: general_data_csv_hourly.csv")
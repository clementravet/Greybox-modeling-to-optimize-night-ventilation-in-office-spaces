import pandas as pd
from pathlib import Path

# Read the original CSV file
DATA_DIR = Path(__file__).resolve().parent
input_csv = DATA_DIR / 'data_summer_neighbor.csv'
output_csv = DATA_DIR / 'data_summer_neighbor_15min.csv'

df = pd.read_csv(input_csv)

# Find and normalize the datetime column name.
if 'datetime' in df.columns:
	datetime_col = 'datetime'
elif len(df.columns) > 0 and str(df.columns[0]).startswith('Unnamed'):
	datetime_col = df.columns[0]
else:
	candidates = [
		col for col in df.columns
		if 'date' in str(col).lower() or 'time' in str(col).lower()
	]
	if not candidates:
		raise ValueError(
			f"No datetime-like column found in {input_csv}. Columns: {list(df.columns)}"
		)
	datetime_col = candidates[0]

df['datetime'] = pd.to_datetime(df[datetime_col], errors='coerce')
df = df.drop(columns=[datetime_col], errors='ignore')
df = df.dropna(subset=['datetime'])

# Set datetime as index for resampling
df.set_index('datetime', inplace=True)

# Resample to 15-minute intervals (keeping the mean of values in each interval)
df_15min = df.resample('15min').mean()

# Reset index to have datetime as a column again
df_15min.reset_index(inplace=True)

# Save to new CSV file
df_15min.to_csv(output_csv, index=False)

print(f"File created: {output_csv}")
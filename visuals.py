"""
Launch Datasette visualization for truck sales data.
"""
import glob
import os
import sqlite3
import subprocess

import pandas as pd

# Configuration
CSV_FOLDER = "csv"
DB_NAME = "truck_sales_all.db"


def load_all_csvs(csv_folder):
    """Load and combine all CSV files from the folder."""
    all_data = []

    # Get all CSV files
    csv_pattern = os.path.join(csv_folder, "*.csv")
    csv_files = glob.glob(csv_pattern)

    print(f"Loading {len(csv_files)} CSV files...")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Add source filename column
            df['source_file'] = os.path.basename(csv_file)
            all_data.append(df)
            print(f"  - {os.path.basename(csv_file)}: {len(df):,} records")
        except Exception as e:
            print(f"  - Error reading {os.path.basename(csv_file)}: {e}")

    if all_data:
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal records: {len(combined_df):,}")
        return combined_df
    else:
        print("No data found in CSV files")
        return pd.DataFrame()


# Load all CSV data
df = load_all_csvs(CSV_FOLDER)

if not df.empty:
    # Create SQLite database
    with sqlite3.connect(DB_NAME) as conn:
        df.to_sql("truck_sales", conn, if_exists="replace", index=False)

    print(f"\nCreated database '{DB_NAME}' with {len(df):,} total records")

    # Launch Datasette
    subprocess.run(["datasette", DB_NAME])
else:
    print("No data to visualize")

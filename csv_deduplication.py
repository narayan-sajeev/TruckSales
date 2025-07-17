"""
CSV deduplication utilities for checking against existing records.
"""

import glob
import os

import pandas as pd


def load_existing_records(csv_folder):
    """Load all existing coordinate pairs from CSVs in the folder."""
    existing_coords = set()

    # Get all CSV files in the folder
    csv_pattern = os.path.join(csv_folder, "*.csv")
    csv_files = glob.glob(csv_pattern)

    print(f"\nLoading existing records from {len(csv_files)} CSV files...")

    for csv_file in csv_files:
        try:
            # Load CSV and extract coordinates
            df = pd.read_csv(csv_file)

            # Check if lat/lon columns exist
            if 'lat' in df.columns and 'lon' in df.columns:
                # Add coordinate pairs to set (rounded to 6 decimal places)
                for _, row in df.iterrows():
                    if pd.notna(row['lat']) and pd.notna(row['lon']):
                        coord_pair = (round(row['lat'], 6), round(row['lon'], 6))
                        existing_coords.add(coord_pair)

                print(f"  - {os.path.basename(csv_file)}: {len(df):,} records")
            else:
                print(f"  - {os.path.basename(csv_file)}: Skipped (no lat/lon columns)")

        except Exception as e:
            print(f"  - Error reading {os.path.basename(csv_file)}: {e}")

    return existing_coords


def check_against_existing(df, existing_coords):
    """Remove records that already exist based on coordinates."""
    if not existing_coords:
        return df

    # Create a mask for new records
    new_mask = []
    duplicates = 0

    for _, row in df.iterrows():
        # Round coordinates to 6 decimal places for comparison
        coord_pair = (round(row['lat'], 6), round(row['lon'], 6))

        if coord_pair in existing_coords:
            new_mask.append(False)
            duplicates += 1
        else:
            new_mask.append(True)

    print(f"Found {duplicates:,} businesses that already exist in CSV folder")

    # Return only new records
    return df[new_mask].reset_index(drop=True)


def get_coordinate_summary(csv_folder):
    """Get summary statistics about existing records."""
    existing_coords = load_existing_records(csv_folder)

    if existing_coords:
        # Convert to DataFrame for analysis
        coords_df = pd.DataFrame(list(existing_coords), columns=['lat', 'lon'])

        print("\nExisting records summary:")
        print(f"  Total unique locations: {len(coords_df):,}")
        print(f"  Latitude range: {coords_df['lat'].min():.4f} to {coords_df['lat'].max():.4f}")
        print(f"  Longitude range: {coords_df['lon'].min():.4f} to {coords_df['lon'].max():.4f}")
    else:
        print("\nNo existing records found in CSV folder")

    return existing_coords

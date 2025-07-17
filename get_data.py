"""
Process truck-related businesses from Overture dataset.
"""

import os
from datetime import datetime

import geopandas as gpd
import pandas as pd

from business_filter import filter_truck_businesses, clean_business_names
from business_matcher import deduplicate_businesses
from csv_deduplication import load_existing_records, check_against_existing
from deconflict_hubspot import deconflict_with_hubspot

# Configuration
INPUT_FILE = "places.parquet"
HUBSPOT_FILE = "hubspot.csv"
CSV_FOLDER = "csv"
OUTPUT_TEMPLATE = "targets_{}.csv"


def load_data(filepath):
    """Load parquet file and extract coordinates."""
    print(f"Loading {filepath}...")

    # Try loading with lat/lon columns first (fastest)
    try:
        df = pd.read_parquet(filepath, columns=['names', 'lat', 'lon', 'confidence',
                                                'addresses', 'websites', 'socials', 'emails', 'phones'])
        print(f"Loaded {len(df):,} records with lat/lon")
        return df
    except:
        pass

    # Fall back to loading with geometry
    try:
        gdf = gpd.read_parquet(filepath, columns=['names', 'geometry', 'confidence',
                                                  'addresses', 'websites', 'socials', 'emails', 'phones'])
        print(f"Loaded {len(gdf):,} records with geometry")

        # Extract coordinates
        print("Extracting coordinates...")
        df = pd.DataFrame(gdf)
        df['lat'] = gdf.geometry.y
        df['lon'] = gdf.geometry.x
        return df.drop(columns=['geometry'])
    except Exception as e:
        print(f"Error loading file: {e}")
        exit(1)


def process_business_names(df):
    """Extract business names from names column."""
    print("Processing business names...")
    df['business_name'] = df['names'].apply(
        lambda x: x.get('primary') if isinstance(x, dict) else x
    )
    return df.drop(columns=['names'])


def prepare_output(df):
    """Prepare dataframe for output with required columns."""
    output_cols = ['business_name', 'websites', 'socials', 'emails', 'phones', 'lat', 'lon']

    # Ensure all columns exist
    for col in output_cols:
        if col not in df.columns:
            df[col] = None

    # Select columns and clean empty values
    df = df[output_cols].copy()
    for col in ['websites', 'socials', 'emails', 'phones']:
        df[col] = df[col].replace(['', '[]', "['']"], None)

    return df.sort_values('business_name')


def save_results(df, output_folder):
    """Save results to timestamped CSV file."""
    if len(df) == 0:
        print("\nNo new businesses found. No file created.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = os.path.join(output_folder, OUTPUT_TEMPLATE.format(timestamp))

    # Ensure phones are saved as strings
    if 'phones' in df.columns:
        df['phones'] = df['phones'].astype('string')

    df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"\nSaved {len(df):,} businesses to {output_file}")


# Setup
os.makedirs(CSV_FOLDER, exist_ok=True)

print(f"\n{'=' * 60}")
print("Truck Sales Target Processor")
print(f"{'=' * 60}\n")

# Load and process data
df = load_data(INPUT_FILE)
df = process_business_names(df)

# Filter for truck businesses
df = filter_truck_businesses(df)
print(f"Found {len(df):,} truck-relevant businesses")

df = clean_business_names(df)

# Deduplicate
df = deduplicate_businesses(df)
print(f"After deduplication: {len(df):,} unique businesses")

# Prepare output format
df = prepare_output(df)

# Remove Hubspot conflicts
df = deconflict_with_hubspot(df, HUBSPOT_FILE)
print(f"After Hubspot deconfliction: {len(df):,} businesses")

# Check against existing CSVs
existing_coords = load_existing_records(CSV_FOLDER)
print(f"Loaded {len(existing_coords):,} existing coordinate pairs")

df = check_against_existing(df, existing_coords)
print(f"After checking existing CSVs: {len(df):,} new unique businesses")

# Save results
save_results(df, CSV_FOLDER)

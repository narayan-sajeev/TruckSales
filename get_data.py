"""
Process truck-related businesses from Overture dataset.

This script filters businesses relevant to tractor trailer sales,
deduplicates similar entries, and exports a clean dataset.
"""

import geopandas as gpd
import pandas as pd

from business_filters import filter_truck_businesses, clean_business_names
from business_matcher import deduplicate_businesses

# Configuration
INPUT_FILE = "northeast_places.parquet"
OUTPUT_FILE = "truck_sales_targets.csv"
ENABLE_DEDUPLICATION = True


def extract_coordinates(gdf):
    """Extract lat/lon from geometry."""
    gdf["lat"] = gdf.geometry.y
    gdf["lon"] = gdf.geometry.x
    return gdf


def load_and_process_data(filepath):
    """Load data and perform initial processing."""
    print(f"\n📂 Loading data from {filepath}...")

    # Load and extract names
    gdf = gpd.read_parquet(filepath)
    print(f"   Loaded {len(gdf):,} records")

    # Extract business names (handle dict/string formats)
    gdf["business_name"] = gdf["names"].apply(
        lambda x: x.get("primary") if isinstance(x, dict) else x
    )

    # Extract coordinates
    gdf = extract_coordinates(gdf)

    # Filter for US truck businesses
    gdf = filter_truck_businesses(gdf)
    print(f"   Found {len(gdf):,} US truck-relevant businesses")

    # Clean names
    gdf = clean_business_names(gdf)

    return gdf


def prepare_output(df):
    """Prepare final output with selected columns."""
    output_cols = [
        "business_name", "websites", "socials",
        "emails", "phones", "lat", "lon"
    ]

    # Ensure all columns exist
    for col in output_cols:
        if col not in df.columns:
            df[col] = None

    # Clean empty values
    df = df[output_cols].copy()

    # Replace empty strings and lists with None
    for col in ['websites', 'socials', 'emails', 'phones']:
        df[col] = df[col].replace('', None)
        df[col] = df[col].replace('[]', None)
        df[col] = df[col].replace("['']", None)

    return df.sort_values("business_name")


print("\n🚛 TRACTOR TRAILER SALES TARGET PROCESSOR")

# Load and filter data
gdf = load_and_process_data(INPUT_FILE)

# Convert to DataFrame for deduplication
df = pd.DataFrame(gdf).reset_index(drop=True)

# Deduplicate if enabled
if ENABLE_DEDUPLICATION:
    df = deduplicate_businesses(df)
    print(f"   After deduplication: {len(df):,} unique businesses")

# Prepare and save output
result = prepare_output(df)

# Ensure phone numbers are saved as strings
if 'phones' in result.columns:
    result['phones'] = result['phones'].astype('string')

result.to_csv(OUTPUT_FILE, index=False, float_format='%.6f')

print(f"\n✅ Complete! Saved {len(result):,} businesses to {OUTPUT_FILE}")

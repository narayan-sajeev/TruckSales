"""
Main script for processing truck-related businesses from Overture dataset.

This script:
1. Loads and filters businesses relevant to TRACTOR TRAILER sales
2. Cleans and normalizes business names
3. Performs fuzzy matching to deduplicate similar businesses
4. Exports the final dataset with all information preserved

Requirements:
    pip install geopandas pandas fuzzywuzzy python-Levenshtein tqdm

Note: If python-Levenshtein fails to install, try:
    pip install python-Levenshtein-wheels
"""

import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import time

from business_matcher import group_similar_businesses_fast, merge_business_group, parse_list
from business_filters import is_truck_relevant, clean_business_name, analyze_patterns, is_canadian_business

# Configuration constants
INPUT_FILE = "northeast_places.parquet"
OUTPUT_FILE = "truck_sales_targets.csv"
DEBUG_FILE = "filtered_before_dedup.csv"  # For debugging

# Deduplication settings
ENABLE_DEDUPLICATION = True  # Set to False to skip deduplication entirely
FAST_MODE = True  # Use faster but less thorough deduplication


def get_english_name(name_field):
    """
    Extract English business name from Overture name field.
    
    Args:
        name_field: Name field which can be string, dict, or None
        
    Returns:
        English name string or None
    """
    if isinstance(name_field, str):
        return name_field
    elif isinstance(name_field, dict):
        return name_field.get("primary")
    return None


def load_and_filter_data(filepath: str) -> gpd.GeoDataFrame:
    """
    Load Overture dataset and filter for truck-relevant businesses.
    
    Args:
        filepath: Path to parquet file
        
    Returns:
        Filtered GeoDataFrame
    """
    print(f"\n📂 Loading and filtering data...")
    start_time = time.time()

    # Load data
    gdf = gpd.read_parquet(filepath)
    print(f"   Loaded {len(gdf):,} records")

    # Extract business names with progress bar
    tqdm.pandas(desc="Extracting names")
    gdf["business_name"] = gdf["names"].progress_apply(get_english_name)

    # Extract coordinates
    gdf["lat"] = gdf.geometry.y
    gdf["lon"] = gdf.geometry.x

    # Create a mask with progress bar
    tqdm.pandas(desc="Filtering businesses")
    mask = gdf["business_name"].progress_apply(is_truck_relevant)
    gdf = gdf[mask].copy()

    # Filter out Canadian businesses
    tqdm.pandas(desc="Filtering Canadian businesses")
    canadian_mask = gdf.progress_apply(is_canadian_business, axis=1)
    canadian_count = canadian_mask.sum()
    gdf = gdf[~canadian_mask].copy()

    print(f"   Removed {canadian_count:,} Canadian businesses")

    # Clean business names
    tqdm.pandas(desc="Cleaning names")
    gdf["business_name"] = gdf["business_name"].progress_apply(clean_business_name)

    # Remove rows with empty business names
    gdf = gdf[gdf["business_name"].notna() & (gdf["business_name"] != "")].copy()

    load_time = time.time() - start_time
    print(f"✅ Found {len(gdf):,} US truck-relevant businesses in {load_time:.1f}s")

    return gdf


def deduplicate_businesses(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Perform fuzzy matching and deduplication on businesses.
    
    Args:
        gdf: GeoDataFrame with business data
        
    Returns:
        DataFrame with deduplicated businesses
    """
    # Convert to regular DataFrame and reset index
    df = pd.DataFrame(gdf)
    df = df.reset_index(drop=True)

    # Normalize contact fields first
    for col in ['websites', 'phones', 'emails', 'socials']:
        df[col] = df[col].apply(parse_list)

    # Group similar businesses
    print("\n🔍 Performing fuzzy matching...")
    if FAST_MODE:
        print("   ⚡ Fast mode enabled - using optimized settings")
    start_time = time.time()

    business_groups = group_similar_businesses_fast(df, show_progress=True)
    matching_time = time.time() - start_time

    print(f"✅ Completed in {matching_time:.2f} seconds")

    # Count duplicates
    duplicate_groups = [g for g in business_groups if len(g) > 1]
    total_duplicates = sum(len(g) - 1 for g in duplicate_groups)

    print(f"\n📊 Deduplication statistics:")
    print(f"   • Total groups: {len(business_groups):,}")
    print(f"   • Groups with duplicates: {len(duplicate_groups):,}")
    print(f"   • Total duplicates found: {total_duplicates:,}")

    # Merge similar businesses
    merged_records = []

    for group_indices in tqdm(business_groups, desc="Merging duplicates", unit="groups"):
        merged_record = merge_business_group(df, group_indices)
        merged_records.append(merged_record)

    # Create result DataFrame
    result = pd.DataFrame(merged_records)

    return result


def main():
    """Main processing function."""
    start_total_time = time.time()

    print("\n🚛 TRACTOR TRAILER SALES TARGET PROCESSOR")
    print("   Focusing on US businesses that need heavy trucks")
    print("   Excluding Canadian businesses")

    # Load and filter data
    gdf = load_and_filter_data(INPUT_FILE)

    # Analyze patterns
    analyze_patterns(gdf)

    # Save filtered data before deduplication (for debugging)
    print(f"\n💾 Saving filtered data before deduplication to {DEBUG_FILE}...")
    gdf.to_csv(DEBUG_FILE, index=False)

    # Perform deduplication if enabled
    if ENABLE_DEDUPLICATION:
        result = deduplicate_businesses(gdf)
    else:
        print("\n⚡ Skipping deduplication (ENABLE_DEDUPLICATION = False)")
        result = pd.DataFrame(gdf)
        result = result.reset_index(drop=True)

    # Final processing
    print("\n📊 Finalizing results...")

    # Select and order final columns
    result = result[[
        "business_name", "websites", "socials", "emails", "phones", "lat", "lon"
    ]]

    # Sort by business name
    result = result.sort_values(by="business_name")

    # Save results
    result.to_csv(OUTPUT_FILE, index=False)

    # Calculate total processing time
    total_time = time.time() - start_total_time

    # Print summary
    print(f"\n✅ COMPLETE!")
    print(f"   • Records processed: {len(gdf):,} → {len(result):,}")
    print(f"   • Duplicates removed: {len(gdf) - len(result):,}")
    print(f"   • Reduction: {((len(gdf) - len(result)) / len(gdf) * 100):.1f}%")
    print(f"   • Total time: {total_time:.1f}s")
    print(f"   • Debug file: {DEBUG_FILE}")
    print(f"   • Output: {OUTPUT_FILE}")

    # Print target business types
    print(f"\n🎯 Target business types for tractor trailer sales:")
    print("   • Towing & recovery (heavy duty)")
    print("   • Trucking & freight companies")
    print("   • Transport & logistics companies")
    print("   • Construction & excavation contractors")
    print("   • Demolition companies")
    print("   • Paving & asphalt contractors")
    print("   • Concrete contractors")
    print("   • Heavy hauling specialists")


if __name__ == "__main__":
    main()

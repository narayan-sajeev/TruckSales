"""
Business matching utilities for deduplication.
"""
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from business_filters import normalize_for_matching


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers."""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def parse_list_field(value):
    """Parse list-like fields and return the first valid item or None."""
    if pd.isna(value) or value is None:
        return None

    value_str = str(value).strip()

    # Empty or invalid values
    if not value_str or value_str in ['[]', "['']", 'nan', 'None']:
        return None

    # If it's a string representation of a list
    if value_str.startswith('[') and value_str.endswith(']'):
        try:
            # Parse the list
            import ast
            parsed = ast.literal_eval(value_str)

            if isinstance(parsed, list):
                # Filter out empty strings
                valid_items = [item for item in parsed if item and str(item).strip()]
                return valid_items[0] if valid_items else None
            else:
                return parsed if parsed else None
        except:
            # If parsing fails, try to extract content
            # Remove brackets and quotes
            content = value_str[1:-1].strip("'\"")
            return content if content else None

    # Regular string value
    return value_str


def clean_phone(phone):
    """Clean and format phone number for comparison."""
    if not phone or pd.isna(phone):
        return None

    phone_str = str(phone).strip()

    # Handle list format
    if phone_str.startswith('['):
        phone_str = parse_list_field(phone_str)
        if not phone_str:
            return None

    # Keep only digits
    phone_clean = re.sub(r'[^\d]', '', phone_str)

    # Skip invalid lengths
    if len(phone_clean) < 10:
        return None

    # For US numbers without country code, add 1
    if len(phone_clean) == 10:
        phone_clean = '1' + phone_clean

    return phone_clean


def extract_domain(url):
    """Extract domain from URL for comparison."""
    if not url or pd.isna(url):
        return None

    # First parse if it's in list format
    if str(url).startswith('['):
        url = parse_list_field(url)

    if not url:
        return None

    url = str(url).lower().strip()

    # Remove protocol
    url = re.sub(r'^https?://(www\.)?', '', url)

    # Get domain part
    domain = url.split('/')[0].split(':')[0]

    return domain if domain else None


def are_businesses_similar(row1, row2, max_distance_km=0.5):
    """
    Determine if two businesses are the same entity.
    
    Uses a simple decision tree:
    1. Same website/phone → same business
    2. Very similar name + close location → same business
    3. Otherwise → different businesses
    """
    # Calculate distance
    distance = haversine_distance(
        row1['lat'], row1['lon'],
        row2['lat'], row2['lon']
    )

    # Check website match (strong signal)
    if pd.notna(row1.get('websites')) and pd.notna(row2.get('websites')):
        domain1 = extract_domain(row1['websites'])
        domain2 = extract_domain(row2['websites'])

        if domain1 and domain2 and domain1 == domain2:
            # Same website - check if reasonable distance (within 10km for same business)
            if distance <= 10:
                return True

    # Check phone match (strong signal)
    if pd.notna(row1.get('phones')) and pd.notna(row2.get('phones')):
        # Compare cleaned phone numbers
        phone1_clean = clean_phone(row1['phones'])
        phone2_clean = clean_phone(row2['phones'])
        
        if phone1_clean and phone2_clean and phone1_clean == phone2_clean:
            # Same phone - check if reasonable distance (within 10km for same business)
            if distance <= 10:
                return True

    # If businesses are too far apart and don't share website/phone, they're different
    if distance > 50:
        return False

    # Name similarity check
    name1 = str(row1['business_name']) if pd.notna(row1['business_name']) else ""
    name2 = str(row2['business_name']) if pd.notna(row2['business_name']) else ""

    # Exact match
    if name1 == name2 and distance <= 1:
        return True

    # Fuzzy match for close businesses
    if distance <= max_distance_km:
        norm1 = normalize_for_matching(name1)
        norm2 = normalize_for_matching(name2)

        if norm1 == norm2:
            return True

        # Token-based similarity
        tokens1 = set(norm1.split()) - {'&', 'and', 'the', 'a', 'of'}
        tokens2 = set(norm2.split()) - {'&', 'and', 'the', 'a', 'of'}

        if tokens1 and tokens2:
            overlap = len(tokens1 & tokens2)
            total = max(len(tokens1), len(tokens2))

            if overlap / total >= 0.8:  # 80% token overlap
                return True

    return False


def deduplicate_businesses(df):
    """
    Deduplicate businesses using geographic blocking.
    """
    print("\n🔍 Deduplicating businesses...")

    # First clean the data
    print("   Cleaning contact fields...")

    # Clean all list fields but keep phones as cleaned numbers for comparison
    for field in ['websites', 'socials', 'emails', 'phones']:
        if field in df.columns:
            if field != 'phones':
                df[field] = df[field].apply(parse_list_field)

    n = len(df)
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Group by geographic cells for efficient comparison
    df['grid_key'] = (
            (df['lat'] / 0.01).astype(int).astype(str) + '_' +
            (df['lon'] / 0.01).astype(int).astype(str)
    )

    comparisons = 0
    matches = 0

    # Compare within geographic cells
    for grid_key, group in tqdm(df.groupby('grid_key'), desc="Processing grid cells"):
        indices = group.index.tolist()

        if len(indices) < 2 or len(indices) > 50:  # Skip single or huge cells
            continue

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                comparisons += 1

                if are_businesses_similar(df.loc[idx1], df.loc[idx2]):
                    union(idx1, idx2)
                    matches += 1

    print(f"   Made {comparisons:,} comparisons, found {matches:,} matches")

    # Group businesses by parent
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    # Merge groups
    merged_records = []
    for indices in groups.values():
        if len(indices) == 1:
            record = df.loc[indices[0]].to_dict()
            # Format phone back with quotes if it exists
            if pd.notna(record.get('phones')):
                cleaned = clean_phone(record['phones'])
                if cleaned:
                    record['phones'] = "'" + cleaned + "'"
            merged_records.append(record)
        else:
            # Take the record with the longest name
            group = df.loc[indices]
            best_idx = group['business_name'].str.len().idxmax()

            # Use the best record but average the location
            record = df.loc[best_idx].to_dict()
            record['lat'] = group['lat'].mean()
            record['lon'] = group['lon'].mean()
            
            # Format phone back with quotes if it exists
            if pd.notna(record.get('phones')):
                cleaned = clean_phone(record['phones'])
                if cleaned:
                    record['phones'] = "'" + cleaned + "'"

            merged_records.append(record)

    # Clean up temporary column
    result = pd.DataFrame(merged_records)
    if 'grid_key' in result.columns:
        result = result.drop('grid_key', axis=1)

    return result
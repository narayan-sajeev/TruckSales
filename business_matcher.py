"""
Business matching utilities for deduplication.
"""
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from business_filter import normalize_for_matching


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers."""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def parse_list_field(value, return_all=False):
    """Parse list-like fields and return items."""
    if pd.isna(value) or not str(value).strip():
        return [] if return_all else None
    
    value_str = str(value).strip()
    
    # Try to parse as list
    if value_str.startswith('[') and value_str.endswith(']'):
        try:
            import ast
            items = [str(item).strip() for item in ast.literal_eval(value_str) 
                    if item and str(item).strip()]
            return items if return_all else (items[0] if items else None)
        except:
            pass
    
    # Single value
    clean_val = value_str.strip("[]'\"")
    if return_all:
        return [clean_val] if clean_val else []
    return clean_val if clean_val else None


def clean_phone(phone):
    """Clean and format phone number for comparison."""
    if not phone:
        return None
    
    # Extract digits only
    digits = re.sub(r'[^\d]', '', str(phone))
    
    # Validate and format
    if len(digits) == 10:
        return '1' + digits
    elif len(digits) >= 11:
        return digits
    return None


def extract_domain(url):
    """Extract domain from URL for comparison."""
    if not url:
        return None
    
    url = str(url).lower().strip()
    url = re.sub(r'^https?://(www\.)?', '', url)
    return url.split('/')[0].split(':')[0] or None


def check_strong_match(row1, row2, distance):
    """Check for strong matching signals (website/phone)."""
    # Website match
    if row1.get('websites') and row2.get('websites'):
        domain1 = extract_domain(parse_list_field(row1['websites']))
        domain2 = extract_domain(parse_list_field(row2['websites']))
        if domain1 and domain1 == domain2 and distance <= 10:
            return True
    
    # Phone match
    if row1.get('phones') and row2.get('phones'):
        phone1 = clean_phone(parse_list_field(row1['phones']))
        phone2 = clean_phone(parse_list_field(row2['phones']))
        if phone1 and phone1 == phone2 and distance <= 10:
            return True
    
    return False


def check_name_similarity(name1, name2, threshold=0.8):
    """Check if business names are similar."""
    if name1 == name2:
        return True
    
    # Normalize and tokenize
    norm1 = normalize_for_matching(name1)
    norm2 = normalize_for_matching(name2)
    
    if norm1 == norm2:
        return True
    
    # Token overlap
    stop_words = {'&', 'and', 'the', 'a', 'of'}
    tokens1 = set(norm1.split()) - stop_words
    tokens2 = set(norm2.split()) - stop_words
    
    if not tokens1 or not tokens2:
        return False
    
    overlap = len(tokens1 & tokens2)
    return overlap / max(len(tokens1), len(tokens2)) >= threshold


def are_businesses_similar(row1, row2, max_distance_km=0.5):
    """Determine if two businesses are the same entity."""
    distance = haversine_distance(row1['lat'], row1['lon'], row2['lat'], row2['lon'])
    
    # Check strong signals first
    if check_strong_match(row1, row2, distance):
        return True
    
    # Too far apart without strong match
    if distance > 50:
        return False
    
    # Name + location check
    if distance <= max_distance_km:
        name1 = str(row1.get('business_name', ''))
        name2 = str(row2.get('business_name', ''))
        return check_name_similarity(name1, name2)
    
    return False


def merge_duplicates(group_df):
    """Merge duplicate business records."""
    # Use record with longest name
    best_idx = group_df['business_name'].str.len().idxmax()
    record = group_df.loc[best_idx].to_dict()
    
    # Average location
    record['lat'] = group_df['lat'].mean()
    record['lon'] = group_df['lon'].mean()
    
    # Merge phone numbers
    all_phones = set()
    for _, row in group_df.iterrows():
        phones = [clean_phone(p) for p in parse_list_field(row.get('phones'), return_all=True)]
        all_phones.update(p for p in phones if p)
    
    if all_phones:
        sorted_phones = sorted(all_phones)
        record['phones'] = (f"{sorted_phones[0]}" if len(sorted_phones) == 1
                          else "[" + ", ".join(f"{p}" for p in sorted_phones) + "]")
    
    return record


def deduplicate_businesses(df):
    """Deduplicate businesses using geographic blocking."""
    print("\n🔍 Deduplicating businesses...")
    
    # Clean non-phone fields
    for field in ['websites', 'socials', 'emails']:
        if field in df.columns:
            df[field] = df[field].apply(parse_list_field)
    
    # Union-Find setup
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
    
    # Geographic grouping
    df['grid_key'] = ((df['lat'] / 0.01).astype(int).astype(str) + '_' + 
                      (df['lon'] / 0.01).astype(int).astype(str))
    
    # Compare within cells
    comparisons = matches = 0
    for _, group in df.groupby('grid_key'):
        indices = group.index.tolist()
        if 2 <= len(indices) <= 50:
            for i, idx1 in enumerate(indices):
                for idx2 in indices[i+1:]:
                    comparisons += 1
                    if are_businesses_similar(df.loc[idx1], df.loc[idx2]):
                        union(idx1, idx2)
                        matches += 1
    
    print(f"   Made {comparisons:,} comparisons, found {matches:,} matches")
    
    # Group by parent
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    
    # Merge records
    merged_records = []
    for indices in groups.values():
        if len(indices) == 1:
            record = df.loc[indices[0]].to_dict()
            if phone := clean_phone(parse_list_field(record.get('phones'))):
                record['phones'] = f"{phone}"
        else:
            record = merge_duplicates(df.loc[indices])
        merged_records.append(record)
    
    return pd.DataFrame(merged_records).drop('grid_key', axis=1)
"""
Business matching utilities for fuzzy deduplication of truck-related businesses.

This module contains functions for:
- Calculating geographic distances between businesses
- Performing fuzzy name matching with blocking for efficiency
- Merging duplicate business records while preserving all information
"""

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from collections import defaultdict
from tqdm import tqdm
import re

from business_filters import normalize_for_matching


def names_are_very_similar(name1: str, name2: str):
    """
    Check if two names are very similar using multiple strategies.
    Optimized for speed with early exits.
    
    Args:
        name1: First business name
        name2: Second business name
        
    Returns:
        Tuple of (is_similar, similarity_score)
    """
    if not name1 or not name2:
        return False, 0
    
    # Quick exact match check
    if name1 == name2:
        return True, 100
    
    # Normalize names
    norm1 = normalize_for_matching(name1)
    norm2 = normalize_for_matching(name2)
    
    # Exact match after normalization
    if norm1 == norm2:
        return True, 100
    
    # Quick length check - if too different, skip expensive checks
    len_diff = abs(len(norm1) - len(norm2))
    if len_diff > max(len(norm1), len(norm2)) * 0.5:
        return False, 0
    
    # Token-based comparison (fast and effective)
    tokens1 = set(norm1.split())
    tokens2 = set(norm2.split())
    
    # Remove common words
    common_words = {'&', 'and', 'the', 'a', 'of', 'in', 'at'}
    tokens1_clean = tokens1 - common_words
    tokens2_clean = tokens2 - common_words
    
    # Need at least some meaningful tokens
    if not tokens1_clean or not tokens2_clean:
        return False, 0
    
    # Calculate token overlap
    if tokens1_clean == tokens2_clean:
        return True, 95
    
    overlap = len(tokens1_clean & tokens2_clean)
    total = max(len(tokens1_clean), len(tokens2_clean))
    token_score = (overlap / total) * 100 if total > 0 else 0
    
    if token_score >= 80:
        return True, token_score
    
    # Only do expensive fuzzy matching if token overlap is promising
    if token_score >= 50:
        # Use token sort ratio (good for reordered words)
        fuzzy_score = fuzz.token_sort_ratio(norm1, norm2)
        if fuzzy_score >= 85:
            return True, fuzzy_score
        
        return False, max(token_score, fuzzy_score)
    
    return False, token_score


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point
        lat2: Latitude of second point
        lon2: Longitude of second point
        
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def get_grid_key(lat: float, lon: float, grid_size: float = 0.01):
    """
    Get grid cell key for geographic blocking.
    
    Args:
        lat: Latitude
        lon: Longitude
        grid_size: Size of grid cells in degrees (default ~1km)
        
    Returns:
        Tuple of (lat_cell, lon_cell)
    """
    return (int(lat / grid_size), int(lon / grid_size))


def get_name_blocks(name: str) -> str:
    """
    Get blocking keys from business name for efficient matching.
    
    Args:
        name: Business name
        
    Returns:
        Set of blocking keys
    """
    if not name:
        return set()
    
    blocks = set()
    
    # Normalize the name first
    norm_name = normalize_for_matching(name)
    name_lower = name.lower().strip()
    
    # First word of normalized name
    words = norm_name.split()
    if words and len(words[0]) >= 3:
        blocks.add(f"word:{words[0]}")
    
    # First two words (for cases like "5 Star")
    if len(words) >= 2:
        blocks.add(f"two:{' '.join(words[:2])}")
    
    # First 4-5 characters of normalized name
    if len(norm_name) >= 4:
        blocks.add(f"prefix:{norm_name[:4]}")
        if len(norm_name) >= 5:
            blocks.add(f"prefix:{norm_name[:5]}")
    
    # Numbers in name (important for "1st Priority", "5 Star", etc.)
    numbers = re.findall(r'\d+', name_lower)
    for num in numbers[:1]:  # Just first number
        blocks.add(f"num:{num}")
    
    return blocks


def merge_values(values):
    """
    Merge multiple values into a single list, removing duplicates and None values.
    
    Args:
        values: List of values to merge (can include lists, None, NaN)
        
    Returns:
        List of unique string values or None if empty
    """
    flat_values = []
    
    # Flatten nested lists and filter out None/NaN values
    for v in values:
        if isinstance(v, list):
            flat_values.extend(v)
        elif v is not None and not (isinstance(v, float) and np.isnan(v)):
            flat_values.append(v)
    
    # Convert to strings and remove empty/invalid values
    str_values = [str(v) for v in flat_values if str(v) not in ['nan', 'None', '']]
    
    # Remove duplicates while preserving order
    unique_values = list(dict.fromkeys(str_values))
    
    return unique_values if unique_values else None


def check_matching_field(row1: pd.Series, row2: pd.Series, field: str) -> bool:
    """
    Check if a specific field matches between two business records.
    
    Args:
        row1: First business record
        row2: Second business record
        field: Field name to compare
        
    Returns:
        True if fields match, False otherwise
    """
    if pd.notna(row1[field]) and pd.notna(row2[field]):
        val1 = str(row1[field]).strip()
        val2 = str(row2[field]).strip()
        
        # For websites, normalize and check similarity
        if field == 'websites':
            # Remove http/https and www
            norm1 = re.sub(r'^https?://(www\.)?', '', val1.lower())
            norm2 = re.sub(r'^https?://(www\.)?', '', val2.lower())
            
            # Remove trailing slashes
            norm1 = norm1.rstrip('/')
            norm2 = norm2.rstrip('/')
            
            # Check if they're the same domain
            if norm1 == norm2:
                return True
            
            # Check if one is a subdomain of the other
            if norm1.startswith(norm2) or norm2.startswith(norm1):
                return True
            
            # Check if they share the same base domain (e.g., sjs-construction.ca)
            base1 = norm1.split('/')[0]
            base2 = norm2.split('/')[0]
            if base1 == base2:
                return True
                
        else:
            # For other fields, exact match
            return val1 == val2
    
    return False


def are_businesses_similar(
        row1: pd.Series,
        row2: pd.Series,
        name_threshold: int = 80,
        distance_km: float = 0.5
) -> bool:
    """
    Determine if two businesses are likely the same entity.

    Comparison is based on:
    - Name similarity (fuzzy matching with multiple strategies)
    - Geographic proximity
    - Matching contact information

    Args:
        row1: First business record
        row2: Second business record
        name_threshold: Minimum name similarity score (0-100)
        distance_km: Maximum distance in kilometers

    Returns:
        True if businesses are likely the same entity
    """
    # Extract business names
    name1 = str(row1['business_name']) if pd.notna(row1['business_name']) else ""
    name2 = str(row2['business_name']) if pd.notna(row2['business_name']) else ""

    # Quick exact match check (cheap)
    if name1 == name2:
        # For exact name matches, be more lenient with distance
        distance = haversine_distance(row1['lat'], row1['lon'], row2['lat'], row2['lon'])

        # If exact name and any matching contact info, consider them the same
        has_matching_phone = check_matching_field(row1, row2, 'phones')
        has_matching_website = check_matching_field(row1, row2, 'websites')
        has_matching_email = check_matching_field(row1, row2, 'emails')

        if has_matching_phone or has_matching_website or has_matching_email:
            return True  # Same name + same contact = same business, regardless of distance

        # If exact name but no matching contact info, only merge if very close
        if distance <= 5:  # 5km for exact name matches
            return True

    # Quick distance check first (cheapest operation)
    distance = haversine_distance(row1['lat'], row1['lon'], row2['lat'], row2['lon'])

    # If too far apart and not exact name match, skip
    if distance > 50:  # 50km absolute max
        return False

    # Check for matching contact info (relatively cheap)
    has_matching_website = check_matching_field(row1, row2, 'websites')
    has_matching_phone = check_matching_field(row1, row2, 'phones')
    has_matching_email = check_matching_field(row1, row2, 'emails')
    has_matching_social = check_matching_field(row1, row2, 'socials')

    # Same website or phone = same business (very strong signal)
    if has_matching_website or has_matching_phone:
        return True

    # Now do the expensive name similarity check
    names_similar, name_score = names_are_very_similar(name1, name2)

    # Very similar names + close location
    if names_similar and distance <= distance_km:
        return True

    # High name similarity even at moderate distance
    if name_score >= 90 and distance <= 25:
        return True

    # Only check other fields if name similarity is promising
    if name_score >= 70:
        has_matching_info = any([
            has_matching_website,
            has_matching_email,
            has_matching_phone,
            has_matching_social
        ])

        # High name similarity + any matching contact info
        if name_score >= 75 and has_matching_info:
            return True

        # Very close location + matching contact info
        if distance <= 0.1 and has_matching_info:
            return True

        # Moderate name similarity + very close + any matching field
        if distance <= 0.2 and has_matching_info:
            return True

    return False


def group_similar_businesses_fast(df: pd.DataFrame, show_progress: bool = True):
    """
    Group similar businesses using blocking for efficiency.
    
    This uses geographic and name-based blocking to dramatically reduce comparisons.
    Instead of O(n²) comparisons, we only compare businesses that share blocking keys.
    
    Args:
        df: DataFrame containing business records
        show_progress: Whether to show progress bar
        
    Returns:
        List of groups, where each group is a list of row indices
    """
    n = len(df)
    parent = list(range(n))
    
    def find(x: int) -> int:
        """Find root parent of element x with path compression."""
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x: int, y: int) -> None:
        """Unite two elements into the same group."""
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Create blocking indices
    geo_blocks = defaultdict(list)
    name_blocks = defaultdict(list)
    
    # Build blocking indices
    for idx in tqdm(range(len(df)), desc="Building indices", disable=not show_progress):
        row = df.iloc[idx]
        
        # Geographic blocking - use larger grid cells for speed
        lat, lon = row['lat'], row['lon']
        base_key = get_grid_key(lat, lon, grid_size=0.01)  # ~1km cells
        
        # Only add to current cell (not neighbors - too many comparisons)
        geo_blocks[base_key].append(idx)
        
        # Name blocking - be more selective
        name = row['business_name']
        if pd.notna(name):
            name_lower = str(name).lower()
            
            # Only use first significant word for blocking
            words = normalize_for_matching(name).split()
            if words:
                # Skip very common words
                first_word = words[0]
                if len(first_word) >= 3 and first_word not in {'the', 'new', 'and'}:
                    name_blocks[f"first:{first_word}"].append(idx)
            
            # Use first 3 chars if name is short enough
            if len(name_lower) >= 3:
                name_blocks[f"pre3:{name_lower[:3]}"].append(idx)
    
    # Track which pairs we've already compared
    compared_pairs = set()
    total_comparisons = 0
    matches_found = 0
    
    # Strategy: Check geographic blocks first (fewer comparisons)
    if show_progress:
        print("   Checking geographic proximity...")
    
    geo_comparisons = 0
    for indices in geo_blocks.values():
        if len(indices) >= 2 and len(indices) <= 20:  # Skip very large blocks
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    pair = (min(idx1, idx2), max(idx1, idx2))
                    compared_pairs.add(pair)
                    geo_comparisons += 1
                    
                    if are_businesses_similar(df.iloc[idx1], df.iloc[idx2]):
                        union(idx1, idx2)
                        matches_found += 1
    
    if show_progress:
        print(f"      Geographic comparisons: {geo_comparisons:,}")
    
    # Name-based comparisons (only for pairs not already compared)
    if show_progress:
        print("   Checking name similarity...")
    
    name_comparisons = 0
    # Only process smaller name blocks to avoid explosion
    name_block_items = [(k, v) for k, v in name_blocks.items() 
                        if 2 <= len(v) <= 10]  # Much smaller limit
    
    for block_key, indices in tqdm(name_block_items, 
                                   desc="   Name blocks", 
                                   disable=not show_progress):
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                pair = (min(idx1, idx2), max(idx1, idx2))
                
                if pair not in compared_pairs:
                    compared_pairs.add(pair)
                    name_comparisons += 1
                    
                    # Quick distance check first (cheaper than name similarity)
                    row1, row2 = df.iloc[idx1], df.iloc[idx2]
                    distance = haversine_distance(
                        row1['lat'], row1['lon'], 
                        row2['lat'], row2['lon']
                    )
                    
                    # Only do expensive name comparison if reasonably close
                    if distance <= 50:  # 50km - generous for name matches
                        if are_businesses_similar(row1, row2):
                            union(idx1, idx2)
                            matches_found += 1
    
    total_comparisons = geo_comparisons + name_comparisons
    
    if show_progress:
        print(f"      Name comparisons: {name_comparisons:,}")
        print(f"\n   Total comparisons: {total_comparisons:,} (vs {n*(n-1)//2:,} brute force)")
        print(f"   Reduction: {(1 - total_comparisons/(n*(n-1)//2))*100:.1f}%")
        print(f"   Matches found: {matches_found:,}")
    
    # Group businesses by their root parent
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    
    return list(groups.values())


def merge_business_group(df: pd.DataFrame, indices):
    """
    Merge a group of similar businesses, retaining all information.
    
    Args:
        df: DataFrame containing business records
        indices: List of row indices to merge
        
    Returns:
        Dictionary containing merged business information
    """
    group = df.iloc[indices]
    
    # Select the longest/most complete business name
    names = group['business_name'].dropna().tolist()
    merged_name = max(names, key=len) if names else None
    
    # Collect all contact information
    contact_fields = {
        'websites': [],
        'socials': [],
        'emails': [],
        'phones': []
    }
    
    for _, row in group.iterrows():
        for field in contact_fields:
            if pd.notna(row[field]):
                contact_fields[field].append(row[field])
    
    # Calculate average location (centroid)
    avg_lat = group['lat'].mean()
    avg_lon = group['lon'].mean()
    
    # Create merged record
    merged_record = {
        'business_name': merged_name,
        'lat': avg_lat,
        'lon': avg_lon
    }
    
    # Process contact fields
    for field, values in contact_fields.items():
        merged_values = merge_values(values)
        if isinstance(merged_values, list):
            merged_record[field] = '; '.join(merged_values)
        else:
            merged_record[field] = np.nan
    
    return merged_record
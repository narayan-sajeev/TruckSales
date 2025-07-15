"""
Business filtering utilities for identifying truck-relevant businesses.
"""

import re

# Threshold for info confidence
CONFIDENCE_THRESHOLD = 0.9

# Core patterns for truck-relevant businesses
TRUCK_KEYWORDS = [
    "towing", "tow", "tows", "trucking", "transport", "freight", "logistics",
    "hauling", "excavation", "construction", "paving", "asphalt",
    "concrete", "demolition", "garage", "repair"
]

# Exclusions (false positives and small contractors)
EXCLUDE_KEYWORDS = [
    # False positives for "tow"
    "tower", "towel", "stow", "towne", "town", "township",
    # Small contractors unlikely to need tractor trailers
    "plumber", "electrician", "hvac", "handyman",
    # Non-business entities
    "town of", "city of", "school", "church", "apartment",
    "hospital", "restaurant", "bank", "salon"
]


def is_truck_relevant(name):
    """Check if business name indicates truck relevance."""
    if not isinstance(name, str) or not name:
        return False

    name_lower = name.lower()

    # Quick exclusion check
    if any(exclude in name_lower for exclude in EXCLUDE_KEYWORDS):
        return False

    # Check for truck keywords
    return any(keyword in name_lower for keyword in TRUCK_KEYWORDS)


def filter_truck_businesses(gdf):
    """Filter GeoDataFrame for truck-relevant US businesses."""
    # Confidence threshold
    gdf = gdf[gdf["confidence"] >= CONFIDENCE_THRESHOLD]

    # Filter by business name
    mask = gdf["business_name"].apply(is_truck_relevant)
    gdf = gdf[mask].copy()

    # Filter out Canadian businesses
    if "addresses" in gdf.columns:
        us_mask = gdf["addresses"].apply(
            lambda x: x[0]["country"] != "CA" if x else True
        )
        gdf = gdf[us_mask]

    return gdf


def clean_name(name):
    if not isinstance(name, str):
        return None

    # Basic cleaning
    name = re.sub(r'["\']', '', name)  # Remove quotes
    name = re.sub(r'\s+', ' ', name)  # Normalize whitespace
    name = name.strip()

    # Title case unless already uppercase
    return name if name.isupper() else name.title()


def clean_business_names(gdf):
    """Clean business names in GeoDataFrame."""

    gdf["business_name"] = gdf["business_name"].apply(clean_name)

    # Remove empty names
    return gdf[gdf["business_name"].notna() & (gdf["business_name"] != "")]


def normalize_for_matching(name):
    """Normalize name for fuzzy matching comparisons."""
    if not name:
        return ""

    # Lowercase and remove business suffixes
    name = name.lower().strip()

    # Common suffixes to remove
    for suffix in [' llc', ' inc', ' corp', ' ltd', ' co']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    # Remove punctuation and normalize whitespace
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()

    return name

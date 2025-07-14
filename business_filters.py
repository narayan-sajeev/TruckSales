"""
Business filtering and classification utilities for identifying truck-relevant businesses.

This module contains:
- Pattern definitions for identifying businesses that buy tractor trailers
- Exclusion logic to filter out irrelevant businesses
- Business name cleaning and normalization functions
- Pattern analysis utilities
"""
import re
import pandas as pd

# Tractor trailer-relevant search pattern
# Focus on businesses that actually need/use tractor trailers
TRUCK_PATTERN = re.compile(
    r"\b(towing|tow\b|trucking|transport|freight|logistics|hauling|"
    r"excavation|construction|paving|asphalt|concrete|demolition)\b",
    re.IGNORECASE
)

# Exclude patterns that look like tow but aren't
# Also exclude small contractors that don't need tractor trailers
EXCLUDE_PATTERN = re.compile(
    r"\b(tower|towers|towel|towels|stow|stowage|towne|township|"
    r"plumber|plumbing|electrician|electrical|electric\b|hvac|"
    r"heating|cooling|air\s+conditioning|handyman|repair\s+service)\b",
    re.IGNORECASE
)

# Terms that indicate non-relevant businesses
EXCLUSION_KEYWORDS = [
    "town of", "city of", "village of",
    "school", "university", "college",
    "church", "mosque", "temple", "synagogue",
    "apartment", "condo", "housing",
    "hospital", "clinic", "medical", "dental", "health",
    "restaurant", "cafe", "diner", "pizza", "food",
    "law office", "attorney", "lawyer", "legal",
    "bank", "credit union", "insurance",
    "hair", "salon", "spa", "nails",
    "daycare", "child care"
]

# Strong indicators that override exclusions
STRONG_TRUCK_INDICATORS = [
    "towing", "trucking", "transport", "construction",
    "excavation", "hauling", "freight"
]

# Pattern categories for analysis
PATTERN_CATEGORIES = [
    ("Towing services", r"\b(towing|tow\b)"),
    ("Trucking/Transport", r"\b(trucking|transport|freight|logistics|hauling)"),
    ("Construction/Excavation", r"\b(construction|excavation|demolition)"),
    ("Paving/Concrete", r"\b(paving|asphalt|concrete)"),
]


def is_truck_relevant(name: str) -> bool:
    """
    Determine if a business name is relevant to tractor trailer sales.
    
    Args:
        name: Business name to check
        
    Returns:
        True if business is truck-relevant, False otherwise
    """
    if not isinstance(name, str):
        return False

    name_lower = name.lower()

    # First, check if it contains any excluded patterns (false positives for "tow")
    if EXCLUDE_PATTERN.search(name_lower):
        # But allow if it also has strong truck indicators
        if not any(indicator in name_lower for indicator in STRONG_TRUCK_INDICATORS):
            return False

    # Check for exclusion keywords
    for ex in EXCLUSION_KEYWORDS:
        if ex in name_lower:
            return False

    # Check for truck-relevant patterns
    return bool(TRUCK_PATTERN.search(name_lower))


def clean_business_name(name: str) -> str:
    """
    Clean and normalize business name for display.
    
    Args:
        name: Raw business name
        
    Returns:
        Cleaned business name or None
    """
    if not isinstance(name, str):
        return None

    # Remove quotes and extra whitespace
    name = name.replace('"', '').replace("'", "").strip()
    name = re.sub(r"\s+", " ", name)

    # Convert to title case unless already all uppercase
    return name if name.isupper() else name.title()


def normalize_for_matching(name: str) -> str:
    """
    Normalize business name for comparison by removing common suffixes and punctuation.
    This is more aggressive than clean_business_name and is used for matching.
    
    Args:
        name: Business name
        
    Returns:
        Normalized name for matching
    """
    if not name:
        return ""

    name_lower = name.lower().strip()

    # Remove common business suffixes
    suffixes = [
        ' llc', ' inc', ' incorporated', ' corp', ' corporation', ' ltd', ' limited',
        ' co', ' company', ' enterprises', ' enterprise', ' services', ' service',
        ' and sons', ' & sons', ' and son', ' & son', ' son', ' sons',
        ' and associates', ' & associates', ' associate', ' associates'
    ]

    for suffix in suffixes:
        if name_lower.endswith(suffix):
            name_lower = name_lower[:-len(suffix)].strip()

    # Remove punctuation and extra spaces
    name_lower = re.sub(r'[^\w\s]', ' ', name_lower)
    name_lower = re.sub(r'\s+', ' ', name_lower).strip()

    # Replace 'and' with '&' for consistency
    name_lower = name_lower.replace(' and ', ' & ')

    return name_lower


def analyze_patterns(df: pd.DataFrame) -> None:
    """
    Analyze which patterns matched the businesses in the dataframe.
    
    Args:
        df: DataFrame with 'business_name' column
    """
    print("\n📈 Pattern Analysis:")

    total = len(df)
    if total == 0:
        print("   No businesses to analyze")
        return

    for pattern_name, pattern in PATTERN_CATEGORIES:
        regex = re.compile(pattern, re.IGNORECASE)
        matches = df["business_name"].apply(
            lambda x: bool(regex.search(x)) if pd.notna(x) else False
        ).sum()
        percentage = (matches / total * 100)
        print(f"   • {pattern_name}: {matches:,} ({percentage:.1f}%)")


def is_canadian_business(row: pd.Series) -> bool:
    """
    Determine if a business is Canadian based on address.

    Args:
        row: Business record with name, website, and other fields

    Returns:
        True if business appears to be Canadian
    """
    return row["addresses"][0]["country"] == "CA"


def get_business_category(name: str) -> str:
    """
    Get the primary category for a business based on its name.
    
    Args:
        name: Business name
        
    Returns:
        Primary category string
    """
    if not name:
        return "Unknown"

    name_lower = name.lower()

    # Check each category pattern
    for category_name, pattern in PATTERN_CATEGORIES:
        if re.search(pattern, name_lower, re.IGNORECASE):
            return category_name

    return "Other"

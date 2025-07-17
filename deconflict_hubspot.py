"""
De-conflict truck sales targets with existing Hubspot contacts.

Removes businesses that already exist in Hubspot based on matching
phone numbers and normalized company names. Domain matching is skipped
due to unreliable data (many businesses incorrectly list franchise/partner domains).
"""

import re

import pandas as pd

from business_filter import normalize_for_matching


def normalize_phone(phone):
    """Normalize phone number to digits only."""
    if pd.isna(phone) or not phone:
        return None

    # Convert to string and extract digits only
    phone_str = str(phone).strip()
    digits = re.sub(r'[^\d]', '', phone_str)

    # Handle different lengths
    if len(digits) == 10:
        return '1' + digits  # Add US country code
    elif len(digits) == 11 and digits.startswith('1'):
        return digits
    elif len(digits) == 7:
        return None  # Too short, ignore
    else:
        return digits  # Return as is for other cases


def parse_phone_list(phone_value):
    """Parse phone field which might be a single number or a list."""
    if pd.isna(phone_value) or not str(phone_value).strip():
        return []

    phone_str = str(phone_value).strip()

    # Check if it's a list format
    if phone_str.startswith('[') and phone_str.endswith(']'):
        # Extract numbers from list format
        phone_str = phone_str.strip('[]')
        phones = []
        for phone in phone_str.split(','):
            phone = phone.strip().strip('"').strip("'")
            if phone:
                phones.append(phone)
        return phones
    else:
        # Single phone number
        return [phone_str]


def load_hubspot_data(filepath):
    """Load and extract matching criteria from Hubspot data."""
    df = pd.read_csv(filepath)

    # Extract and normalize phone numbers
    phones = set()
    for phone in df['Phone Number'].dropna():
        normalized = normalize_phone(phone)
        if normalized:
            phones.add(normalized)

    # Collect company names - normalize them for better matching
    companies = set()

    for company in df['Associated Company'].dropna():
        normalized = normalize_for_matching(company)
        if normalized and len(normalized) > 3:  # Skip very short company names
            companies.add(normalized)

    print(f"Hubspot data: {len(phones):,} phones, {len(companies):,} companies")

    return phones, companies


def check_conflicts(row, hubspot_phones, hubspot_companies):
    """Check if a business conflicts with Hubspot data."""
    conflict_reasons = []

    # Check phone numbers
    if row.get('phones'):
        phones = parse_phone_list(row['phones'])
        for phone in phones:
            normalized = normalize_phone(phone)
            if normalized and normalized in hubspot_phones:
                conflict_reasons.append(f'Phone match: {phone}')
                break

    # Check company name - EXACT MATCH ONLY on normalized names
    if row['business_name']:
        business_name = str(row['business_name']).strip()
        normalized_business = normalize_for_matching(business_name)

        # Only do exact match on normalized names
        if normalized_business in hubspot_companies:
            conflict_reasons.append('Company name match')

    return conflict_reasons


def deconflict_with_hubspot(truck_df, hubspot_file):
    """Remove businesses that exist in Hubspot."""
    # Load Hubspot data
    hubspot_phones, hubspot_companies = load_hubspot_data(hubspot_file)

    # Find conflicts
    conflicts = {}  # Use dict to prevent duplicates

    for idx, row in truck_df.iterrows():
        business_name = row['business_name']
        conflict_reasons = check_conflicts(row, hubspot_phones, hubspot_companies)

        if conflict_reasons:
            # Use business name as key to prevent duplicates
            if business_name not in conflicts:
                conflicts[business_name] = {
                    'business_name': business_name,
                    'reason': conflict_reasons,
                    'indices': [idx]
                }
            else:
                # Add index to existing conflict
                conflicts[business_name]['indices'].append(idx)
                # Merge reasons if different
                for reason in conflict_reasons:
                    if reason not in conflicts[business_name]['reason']:
                        conflicts[business_name]['reason'].append(reason)

    # Get indices to remove
    indices_to_remove = []
    for conflict in conflicts.values():
        indices_to_remove.extend(conflict['indices'])
    indices_to_remove = list(set(indices_to_remove))

    # Remove conflicts
    clean_df = truck_df.drop(indices_to_remove).reset_index(drop=True)

    print(f"Found {len(conflicts):,} businesses with Hubspot conflicts")

    return clean_df

# Truck Sales Target Processor with Hubspot Deconfliction

A Python tool for extracting and processing truck-relevant businesses from Overture Maps data to identify potential tractor trailer sales targets, with automatic removal of existing Hubspot contacts.

## Overview

This project processes businesses from Overture Maps datasets through three stages:
1. **Filters** businesses likely to need tractor trailers (towing, trucking, construction, etc.)
2. **Deduplicates** similar entries based on location and contact info
3. **Removes** businesses that already exist in your Hubspot CRM

The result is a clean list of new sales leads that don't exist in your current CRM.

## Features

- **Smart Filtering**: Identifies truck-relevant businesses using keyword matching while excluding false positives
- **Geographic Deduplication**: Merges duplicate businesses based on name similarity, location proximity, and matching contact information
- **Hubspot Deconfliction**: Automatically removes businesses that already exist in your Hubspot database by matching:
  - Phone numbers (with normalization)
  - Company names (with normalization and exact matching)
- **Single Output**: Produces one clean CSV file with all processing stages applied
- **Conflict Reporting**: Generates a detailed report of which businesses were removed due to Hubspot conflicts

## Installation

```bash
# Clone the repository
git clone https://github.com/narayan-sajeev/TruckSales.git
cd TruckSales

# Install dependencies
pip install geopandas pandas numpy

# For visualization (optional)
pip install datasette
```

## Required Files

Before running, ensure you have:
1. `northeast_places.parquet` - Overture Maps data file
2. `hubspot.csv` - Export from your Hubspot CRM with at least:
   - `Phone Number` column (for phone matching)
   - `Associated Company` column (for company name matching)

## Usage

Run the processor:
```bash
python get_data.py
```

The script will:
1. Load and filter businesses for truck relevance
2. Remove geographic duplicates
3. Remove businesses that exist in Hubspot
4. Save the final clean list to `truck_sales_targets.csv`

### Processing Stages

The script shows progress through three stages:

```
Loaded 1,679,803 records
Found 9,481 truck-relevant businesses
After deduplication: 9,444 unique businesses
Hubspot data: 4,224 phones, 859 companies
Found 87 businesses with Hubspot conflicts
Final: 9,357 businesses saved to truck_sales_targets.csv
```

## Output Files

### truck_sales_targets.csv
The main output file containing new leads not in Hubspot:
- `business_name`: Cleaned company name
- `websites`: Company website
- `socials`: Social media profiles
- `emails`: Contact email addresses (typically null in Overture data)
- `phones`: Phone numbers (formatted as strings)
- `lat`, `lon`: Geographic coordinates

### hubspot_conflicts.csv
Report of businesses removed because they exist in Hubspot:
- `business_name`: Company that was removed
- `reason`: Why it was removed (phone match or company name match)

## Deconfliction Process

The Hubspot deconfliction works by:

1. **Phone Number Matching**:
   - Normalizes phone numbers (removes formatting, adds country code)
   - Compares normalized numbers between datasets
   - Handles various formats: "555-1234", "(555) 123-4567", "5551234567"

2. **Company Name Matching**:
   - Normalizes company names (lowercase, removes LLC/Inc/Corp suffixes)
   - Performs exact matching only (no substring matching)
   - Prevents false positives from partial name matches

3. **Why No Domain Matching?**:
   - Many businesses incorrectly list franchise/partner domains (e.g., uhaul.com, ryder.com)
   - Email fields in Overture data are typically null
   - Over 200 domains are shared by multiple unrelated businesses
   - Domain matching led to many false positives

## Configuration

Edit variables in `get_data.py`:
- `INPUT_FILE`: Path to Overture parquet file (default: `northeast_places.parquet`)
- `HUBSPOT_FILE`: Path to Hubspot export (default: `hubspot.csv`)
- `OUTPUT_FILE`: Name for final output (default: `truck_sales_targets.csv`)
- `CONFLICTS_FILE`: Name for conflicts report (default: `hubspot_conflicts.csv`)

## Visualization

To explore the results interactively:
```bash
python visuals.py
```

This launches Datasette for browsing and querying the final dataset.

## Module Structure

- `get_data.py`: Main processing script that orchestrates all three stages
- `business_filter.py`: Business filtering and name cleaning utilities
- `business_matcher.py`: Deduplication logic using geographic blocking
- `deconflict_hubspot.py`: Hubspot deconfliction using phone and company name matching
- `visuals.py`: Datasette visualization launcher

## Performance

- Processes ~1.7M businesses down to ~9,000 relevant targets in under a minute
- Efficiently handles deduplication using geographic blocking
- Fast phone number and company name matching for Hubspot deconfliction
- Typically removes 50-150 businesses due to Hubspot conflicts

## Troubleshooting

If you get an error about missing Hubspot file:
- Ensure `hubspot.csv` is in the same directory as the script
- Check that the CSV has `Phone Number` and `Associated Company` columns
- Verify the file path in the `HUBSPOT_FILE` configuration

If conflicts seem incorrect:
- Review the `hubspot_conflicts.csv` file
- Phone matches are based on normalized numbers (digits only)
- Company matches require exact normalized name matches
- No domain matching is performed due to data quality issues
# Truck Sales Target Processor with CSV Deduplication

A Python tool for extracting and processing truck-relevant businesses from Overture Maps data to identify potential tractor trailer sales targets, with automatic deduplication against existing records and Hubspot contacts.

## Overview

This project processes businesses from Overture Maps datasets through four stages:
1. **Filters** businesses likely to need tractor trailers (towing, trucking, construction, etc.)
2. **Deduplicates** similar entries based on location and contact info
3. **Removes** businesses that already exist in your Hubspot CRM
4. **Checks** against all previously processed records in the CSV folder to ensure no duplicates

The result is a clean list of NEW sales leads that don't exist in your current CRM or any previous exports.

## Features

- **Smart Filtering**: Identifies truck-relevant businesses using keyword matching while excluding false positives
- **Geographic Deduplication**: Merges duplicate businesses based on name similarity, location proximity, and matching contact information
- **Hubspot Deconfliction**: Automatically removes businesses that already exist in your Hubspot database
- **CSV History Checking**: Prevents duplicate records across multiple runs by checking coordinates against all existing CSV files
- **Timestamped Outputs**: Each run creates a uniquely named file with timestamp to prevent overwrites
- **Zero Overlap Guarantee**: Ensures each new output file contains only businesses not in any previous file

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

## Required Files and Folders

Before running, ensure you have:
1. `places.parquet` - Overture Maps data file
2. `hubspot.csv` - Export from your Hubspot CRM with at least:
   - `Phone Number` column (for phone matching)
   - `Associated Company` column (for company name matching)
3. `csv/` folder - Will be created automatically if it doesn't exist. This stores all output files.

## Usage

Run the processor:
```bash
python get_data.py
```

The script will:
1. Load and filter businesses for truck relevance
2. Remove geographic duplicates
3. Remove businesses that exist in Hubspot
4. Check against ALL existing records in the CSV folder (based on lat/lon coordinates)
5. Save only NEW businesses to a timestamped file in the CSV folder

### Output Example

```
Loaded 1,679,803 records
Found 9,481 truck-relevant businesses
After deduplication: 9,444 unique businesses
After Hubspot deconfliction: 9,357 businesses

Loading existing records from 3 CSV files...
  - targets_20240115_1430.csv: 9,357 records
  - targets_20240116_0915.csv: 8,234 records
  - targets_20240117_1530.csv: 7,891 records
Loaded 25,482 existing coordinate pairs from CSV folder
Found 9,357 businesses that already exist in CSV folder
After checking existing CSVs: 0 new unique businesses

Final: 0 businesses saved to csv/targets_20240118_1023.csv
```

## Output Files

### csv/targets_YYYYMMDD_HHMM.csv
Each run creates a timestamped file containing only NEW leads:
- `business_name`: Cleaned company name
- `websites`: Company website
- `socials`: Social media profiles
- `emails`: Contact email addresses (typically null in Overture data)
- `phones`: Phone numbers (formatted as strings)
- `lat`, `lon`: Geographic coordinates (used for deduplication)

## Deduplication Process

### Coordinate-Based Checking
The system uses latitude and longitude (rounded to 6 decimal places) as the unique identifier for businesses:
- Ensures no business at the same location appears in multiple files
- Handles variations in phone/website formatting that might differ between data sources
- Provides ~0.1 meter precision for location matching

### Why Coordinates?
- Phone numbers and websites can have varying formats across different data sources
- Business names might be slightly different in various datasets
- Coordinates provide a consistent, reliable way to identify unique physical locations

## Working with Multiple Geographic Areas

The system is designed for processing different geographic regions:

1. **First Run**: Process your first area (e.g., places.parquet from northeast region)
   - Creates `csv/targets_20240118_0930.csv`
   - All businesses are new

2. **Second Run**: Process a different area (e.g., places.parquet from midwest region)
   - Checks against the first file
   - Only saves businesses not in the northeast
   - Creates `csv/targets_20240118_1415.csv`

3. **Subsequent Runs**: Continue with other regions
   - Each run checks against ALL previous files
   - Ensures zero overlap across all outputs

## Visualization

To explore all collected data across multiple files:
```bash
python visuals.py
```

This:
- Loads all CSV files from the csv folder
- Combines them into a single database
- Launches Datasette for browsing and querying the complete dataset
- Includes source filename for tracking which file each record came from

## Module Structure

- `get_data.py`: Main processing script with CSV checking integration
- `csv_deduplication.py`: New module for loading and checking against existing CSV records
- `business_filter.py`: Business filtering and name cleaning utilities
- `business_matcher.py`: Deduplication logic using geographic blocking
- `deconflict_hubspot.py`: Hubspot deconfliction using phone and company name matching
- `visuals.py`: Updated to visualize all CSV files combined

## SharePoint Integration

The timestamped output files make it easy to upload to SharePoint:
- Each file has a unique name preventing overwrites
- Files contain only new businesses not in previous uploads
- Can upload files sequentially as you process different regions
- No manual file renaming required

## Performance

- Processes ~1.7M businesses efficiently
- Coordinate checking is fast using Python sets
- Handles thousands of existing records without significant slowdown
- Memory efficient for large datasets

## Troubleshooting

If all businesses show as duplicates:
- Check if you're processing the same geographic area twice
- Verify the input parquet file is from a new region
- Ensure the CSV folder contains the expected files

If the CSV folder is missing:
- The script creates it automatically
- Check you have write permissions in the current directory

To reset and start fresh:
- Move or delete all files from the csv folder
- The next run will treat all businesses as new
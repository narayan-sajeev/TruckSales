# Truck Sales Target Processor

A Python tool for extracting and processing truck-relevant businesses from Overture Maps data to identify potential tractor trailer sales targets.

## Overview

This project filters businesses from Overture Maps datasets to identify companies likely to need tractor trailers, such as:
- Towing companies
- Trucking and transport businesses
- Construction companies
- Freight and logistics operations
- Excavation and paving contractors

The tool performs intelligent deduplication to merge duplicate business entries and exports clean data suitable for sales prospecting.

## Features

- **Smart Filtering**: Identifies truck-relevant businesses using keyword matching while excluding false positives
- **Geographic Deduplication**: Merges duplicate businesses based on name similarity, location proximity, and matching contact information
- **Data Cleaning**: Normalizes business names and formats contact information
- **US-Only Focus**: Filters out Canadian and other international businesses
- **Visualization Support**: Includes Datasette integration for interactive data exploration

## Installation

```bash
# Clone the repository
git clone https://github.com/narayan-sajeev/NETC.git
cd NETC

# Install dependencies
pip install geopandas pandas numpy

# For visualization (optional)
pip install datasette
```

## Usage

1. **Prepare Input Data**: Place your Overture Maps parquet file in the project directory as `northeast_places.parquet`

2. **Run the Processor**:
```bash
python get_data.py
```

This will:
- Load and filter businesses for truck relevance
- Remove duplicates using geographic blocking
- Export results to `truck_sales_targets.csv`

3. **Visualize Results** (optional):
```bash
python visuals.py
```

This launches Datasette for interactive exploration of the processed data.

## Configuration

Edit `get_data.py` to adjust:
- `INPUT_FILE`: Path to your Overture parquet file
- `OUTPUT_FILE`: Name for the exported CSV

## Output Format

The exported CSV includes:
- `business_name`: Cleaned company name
- `websites`: Company website
- `socials`: Social media profiles
- `emails`: Contact email addresses
- `phones`: Phone numbers (formatted as strings)
- `lat`, `lon`: Geographic coordinates

## Module Structure

- `get_data.py`: Main processing script
- `business_filter.py`: Business filtering and name cleaning utilities
- `business_matcher.py`: Deduplication logic using geographic blocking
- `visuals.py`: Datasette visualization launcher

## Deduplication Strategy

The tool uses a sophisticated approach to identify duplicate businesses:

1. **Geographic Blocking**: Groups businesses into geographic grid cells for efficient comparison
2. **Multi-Signal Matching**: Considers multiple factors:
   - Name similarity (token overlap, normalized comparison)
   - Geographic proximity (typically within 0.5km)
   - Matching websites or phone numbers
3. **Smart Merging**: Preserves the best data from duplicate records

## Performance

- Processes ~100,000 businesses in under a minute
- Deduplication scales efficiently using geographic blocking
- Typically reduces dataset size by 10-20% through duplicate removal
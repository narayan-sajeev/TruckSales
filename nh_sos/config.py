# config.py
"""Configuration for NH scraper"""

# File names
STATE_FILE = "nh_scraper_state.json"
PROGRESS_FILE = "nh_progress_all.csv"
FINAL_FILE = "nh_final_data.csv"
ACTIVE_FILE = "nh_active_only.csv"

# Search terms
SEARCH_TERMS = [
    "truck", "freight", "transport", "excavation", "trailer",
    "ltl", "haul", "grading", "sitework", "aggregates",
    "paving", "asphalt", "concrete", "diesel", "towing"
]

# Timing
MIN_WAIT = 15
MAX_WAIT = 25
PAGES_PER_BATCH = 6
MAX_PAGES_PER_SESSION = 11

# Browser
VIEWPORT = {'width': 1920, 'height': 1080}
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
BASE_URL = "https://quickstart.sos.nh.gov/online/BusinessInquire"

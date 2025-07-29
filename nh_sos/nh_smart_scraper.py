"""
NH SOS scraper with configurable search terms and optimized wait times
"""
import asyncio
import json
import os
import random
import re
from datetime import datetime

import pandas as pd
from playwright.async_api import async_playwright

# ============== CONFIGURATION ==============
# File names - easy to change
STATE_FILE = "nh_scraper_state.json"
PROGRESS_FILE = "nh_progress_all.csv"
FINAL_FILE = "nh_final_data.csv"
ACTIVE_FILE = "nh_active_only.csv"

# Search terms - will process in order
SEARCH_TERMS = [
    "truck",  # Currently running
    "freight",  # High-value term
    "transport",  # Broad coverage
    "excavation"  # Construction sector
]

# Timing configuration
MIN_WAIT = 45  # Reduced from 90 - evidence shows no blocks at higher pages
MAX_WAIT = 60  # Reduced from 110 - saves ~2 hours total
PAGES_PER_BATCH = 3  # Pages per session
MAX_PAGES_PER_SESSION = 11  # Site limit for sequential clicking


# ==========================================


class NHSmartScraper:
    def __init__(self):
        self.all_businesses = []
        self.completed_pages = {}  # Changed to dict: {search_term: set(pages)}
        self.failed_pages = {}  # Changed to dict: {search_term: set(pages)}
        self.total_pages = {}  # NEW: Track total pages per search term
        self.current_search_term = None
        self.load_state()

    def load_state(self):
        """Load previous progress if exists"""
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)

                self.completed_pages = {term: set(pages) for term, pages in
                                        state.get('completed_pages_by_term', {}).items()}
                self.failed_pages = {term: set(pages) for term, pages in state.get('failed_pages_by_term', {}).items()}
                self.total_pages = state.get('total_pages_by_term')

                self.current_search_term = state.get('current_search_term', SEARCH_TERMS[0])

                print(f"Loaded previous state:")
                for term in self.completed_pages:
                    completed = len(self.completed_pages.get(term, set()))
                    total = self.total_pages.get(term, "?")
                    print(f"  {term}: {completed} completed pages" + (f" (of {total})" if total != "?" else ""))

                # Load existing businesses
                if os.path.exists(PROGRESS_FILE):
                    df = pd.read_csv(PROGRESS_FILE)
                    self.all_businesses = df.to_dict('records')
                    print(f"Loaded {len(self.all_businesses)} existing businesses")

    def save_state(self):
        """Save current progress"""
        state = {
            'completed_pages_by_term': {term: list(pages) for term, pages in self.completed_pages.items()},
            'failed_pages_by_term': {term: list(pages) for term, pages in self.failed_pages.items()},
            'total_pages_by_term': self.total_pages,
            'current_search_term': self.current_search_term,
            'timestamp': datetime.now().isoformat()
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

    def get_current_search_term(self):
        """Get the search term to work on"""
        for term in SEARCH_TERMS:
            completed = len(self.completed_pages.get(term, set()))
            total = self.total_pages.get(term)

            # If we don't know total pages yet, or haven't completed all pages
            if total is None or completed < total:
                return term
        return None

    def get_next_batch(self):
        """Get next batch of pages for current search term"""
        if not self.current_search_term:
            return []

        completed = self.completed_pages.get(self.current_search_term, set())
        total = self.total_pages.get(self.current_search_term)

        # If we don't know total pages yet, assume a large number
        max_page = total if total else 999

        # Find the lowest uncompleted page
        start_page = None
        for page in range(1, max_page + 1):
            if page not in completed:
                start_page = page
                break

        if start_page is None:
            return []

        # Get consecutive pages up to limit
        batch = []
        for page in range(start_page, min(start_page + MAX_PAGES_PER_SESSION, max_page + 1)):
            if page not in completed:
                batch.append(page)
            if len(batch) >= PAGES_PER_BATCH:
                break

        return batch

    async def scrape_consecutive_pages(self, start_page, end_page):
        """Scrape a range of consecutive pages in one session"""
        businesses = []
        pages_completed = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,
                args=['--disable-blink-features=AutomationControlled']
            )

            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            )

            page = await context.new_page()

            try:
                # Navigate to site
                print("Opening NH SOS website...")
                await page.goto("https://quickstart.sos.nh.gov/online/BusinessInquire")
                await asyncio.sleep(3)

                # Search with current term
                await page.click('input[value="All"]')
                await asyncio.sleep(1)
                await page.fill('input[name="txtBusinessName"]', self.current_search_term)
                await asyncio.sleep(1)
                await page.click('input[type="submit"]')
                print(f"Searching for '{self.current_search_term}'...")

                # Wait for results
                try:
                    await page.wait_for_selector('table', timeout=30000)
                    await asyncio.sleep(2)

                    # Detect total pages for this search term if we don't know yet
                    if self.current_search_term not in self.total_pages:
                        page_text = await page.inner_text('body')
                        match = re.search(r'Page \d+ of (\d+)', page_text)
                        if match:
                            self.total_pages[self.current_search_term] = int(match.group(1))
                            print(
                                f"Detected {self.total_pages[self.current_search_term]} total pages for '{self.current_search_term}'")

                except:
                    print("Search failed")
                    return businesses, pages_completed

                # Navigate to start page if needed
                current_page = 1
                if start_page > 1:
                    print(f"Navigating to page {start_page}...")

                    # For pages beyond 11, use direct jump
                    if start_page > 11:
                        success = await self.jump_to_page(page, start_page)
                        if success:
                            current_page = start_page
                        else:
                            print(f"Could not jump to page {start_page}")
                            return businesses, pages_completed
                    else:
                        # Click through pages
                        for target in range(2, start_page + 1):
                            success = await self.click_next_page(page)
                            if not success:
                                print(f"Navigation failed at page {target}")
                                return businesses, pages_completed
                            current_page = target

                            # Check for browser block
                            page_text = await page.inner_text('body')
                            if 'One moment, while we check your browser' in page_text:
                                print(f"Hit browser check at page {target}")
                                return businesses, pages_completed

                # Now scrape from start_page to end_page
                for target_page in range(start_page, end_page + 1):
                    if target_page != current_page:
                        success = await self.click_next_page(page)
                        if not success:
                            break
                        current_page = target_page

                    print(f"\nProcessing page {target_page}...")

                    # Check for browser check
                    page_text = await page.inner_text('body')
                    if 'One moment, while we check your browser' in page_text:
                        print("Browser check detected!")
                        await asyncio.sleep(10)
                        page_text = await page.inner_text('body')
                        if 'One moment, while we check your browser' in page_text:
                            print("Still blocked by browser check")
                            break

                    # Check if we've gone past the last page
                    if self.current_search_term in self.total_pages:
                        if f"Page {target_page} of" not in page_text:
                            print(f"Page {target_page} doesn't exist - reached end of results")
                            # Mark this search term as complete
                            self.total_pages[self.current_search_term] = target_page - 1
                            break

                    # Extract businesses
                    page_businesses = await self.extract_businesses(page)
                    if page_businesses:
                        # Add search term to each business
                        for business in page_businesses:
                            business['search_term'] = self.current_search_term
                        businesses.extend(page_businesses)
                        pages_completed.append(target_page)
                        print(f"✓ Found {len(page_businesses)} businesses")
                    else:
                        print(f"✗ No businesses found")

            except Exception as e:
                print(f"Session error: {e}")

            finally:
                await browser.close()

        return businesses, pages_completed

    async def jump_to_page(self, page, target_page):
        """Try to jump directly to a page using Go to Page feature"""
        try:
            goto_elements = await page.query_selector_all('a')
            goto_link = None

            for elem in goto_elements:
                text = await elem.inner_text()
                if 'Go to Page' in text:
                    goto_link = elem
                    break

            if goto_link:
                await goto_link.click()
                await asyncio.sleep(1)

                await page.keyboard.type(str(target_page))
                await page.keyboard.press('Enter')
                await page.wait_for_load_state('networkidle')
                await asyncio.sleep(3)

                page_text = await page.inner_text('body')
                if f"Page {target_page} of" in page_text:
                    return True

        except Exception as e:
            print(f"Jump to page error: {e}")

        return False

    async def click_next_page(self, page):
        """Click to next page"""
        try:
            next_link = await page.query_selector('a:has-text("Next >")')
            if next_link:
                await next_link.click()
                await page.wait_for_load_state('networkidle')
                await asyncio.sleep(2)
                return True
        except:
            pass
        return False

    async def extract_businesses(self, page):
        """Extract businesses from current page"""
        businesses = []
        try:
            rows = await page.query_selector_all('table tr')

            for row in rows:
                cells = await row.query_selector_all('td')

                if len(cells) == 8:
                    first_text = await cells[0].inner_text()

                    if not any(x in first_text for x in ['Previous', 'Page', '<', '>', 'moment']):
                        business = {
                            'business_name': (await cells[0].inner_text()).strip(),
                            'business_id': (await cells[1].inner_text()).strip(),
                            'homestate_name': (await cells[2].inner_text()).strip(),
                            'previous_name': (await cells[3].inner_text()).strip(),
                            'business_type': (await cells[4].inner_text()).strip(),
                            'address': (await cells[5].inner_text()).strip(),
                            'agent': (await cells[6].inner_text()).strip(),
                            'status': (await cells[7].inner_text()).strip()
                        }

                        if business['business_id'] and business['business_id'].isdigit():
                            businesses.append(business)

        except Exception as e:
            print(f"Extraction error: {e}")

        return businesses

    async def run(self):
        """Main scraping loop"""

        while True:
            # Get current search term
            self.current_search_term = self.get_current_search_term()
            if not self.current_search_term:
                print("\nAll search terms completed!")
                break

            # Initialize tracking for new search term
            if self.current_search_term not in self.completed_pages:
                self.completed_pages[self.current_search_term] = set()
                self.failed_pages[self.current_search_term] = set()
                print(f"\n{'=' * 60}")
                print(f"Starting new search term: '{self.current_search_term}'")
                print(f"{'=' * 60}")

            # Get next batch for current term
            batch = self.get_next_batch()
            if not batch:
                # This term is done, move to next
                completed = len(self.completed_pages.get(self.current_search_term, set()))
                total = self.total_pages.get(self.current_search_term, "?")
                print(f"\nCompleted '{self.current_search_term}': {completed}/{total} pages")
                continue

            start_page = min(batch)
            end_page = max(batch)

            print(f"\n{'=' * 60}")
            print(f"SESSION: '{self.current_search_term}' - Pages {start_page} to {end_page}")
            total_for_term = self.total_pages.get(self.current_search_term, "?")
            print(f"Progress: {len(self.completed_pages.get(self.current_search_term, set()))}/{total_for_term} pages")
            print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'=' * 60}")

            # Scrape pages
            businesses, completed = await self.scrape_consecutive_pages(start_page, end_page)

            # Update state
            if businesses:
                self.all_businesses.extend(businesses)
                for page_num in completed:
                    self.completed_pages[self.current_search_term].add(page_num)
                    self.failed_pages[self.current_search_term].discard(page_num)

                # Mark uncompleted pages in range as failed
                for page_num in range(start_page, end_page + 1):
                    if page_num not in self.completed_pages[self.current_search_term]:
                        self.failed_pages[self.current_search_term].add(page_num)

                # Save progress
                df = pd.DataFrame(self.all_businesses)
                df = df.drop_duplicates(subset=['business_id'])
                df.to_csv(PROGRESS_FILE, index=False)

                print(f"\n✓ Session complete: {len(businesses)} businesses from {len(completed)} pages")
                print(f"Total unique businesses: {len(df)}")
            else:
                # Mark all pages in batch as failed
                for page_num in batch:
                    self.failed_pages[self.current_search_term].add(page_num)
                print("\n✗ Session failed - no data retrieved")

            # Save state
            self.save_state()

            # Calculate wait time (reduced, with small random variation)
            wait_time = random.randint(MIN_WAIT, MAX_WAIT)

            # Check if more work to do
            next_term = self.get_current_search_term()
            if next_term:
                next_time = datetime.now()
                next_time = datetime.fromtimestamp(next_time.timestamp() + wait_time)
                print(f"\n⏳ Waiting {wait_time} seconds...")
                print(f"Next session at: {next_time.strftime('%H:%M:%S')}")

                await asyncio.sleep(wait_time)


async def main():
    print("\n" + "=" * 80)
    print("NH SOS Multi-Term Scraper v3")
    print(f"Search terms: {', '.join(SEARCH_TERMS)}")
    print(f"Wait time: {MIN_WAIT}-{MAX_WAIT} seconds between sessions")
    print("=" * 80 + "\n")

    scraper = NHSmartScraper()
    await scraper.run()

    # Final summary
    print("\n" + "=" * 80)
    print("SCRAPING COMPLETE")
    print("=" * 80)

    # Calculate total completed across all terms
    total_completed = 0
    total_expected = 0

    print("\nResults by search term:")
    for term in SEARCH_TERMS:
        completed = len(scraper.completed_pages.get(term, set()))
        total = scraper.total_pages.get(term, "unknown")
        failed = len(scraper.failed_pages.get(term, set()))

        print(f"\n'{term}':")
        print(f"  Completed: {completed}" + (f"/{total}" if total != "unknown" else "") + " pages")
        if failed:
            print(f"  Failed: {failed} pages")

        if isinstance(total, int):
            total_completed += completed
            total_expected += total

    if total_expected > 0:
        print(
            f"\nOverall completion: {total_completed}/{total_expected} pages ({total_completed / total_expected * 100:.1f}%)")

    # Final data
    if os.path.exists(PROGRESS_FILE):
        df = pd.read_csv(PROGRESS_FILE)
        print(f"\nTotal unique businesses: {len(df)}")

        # Breakdown by search term
        if 'search_term' in df.columns:
            print("\nBusinesses by search term:")
            print(df['search_term'].value_counts())

        # Save final version
        df.to_csv(FINAL_FILE, index=False)
        print(f"\nFinal data saved to {FINAL_FILE}")

        # Active businesses
        active_df = df[df['status'].isin(['Good Standing', 'Active'])]
        if len(active_df) > 0:
            active_df.to_csv(ACTIVE_FILE, index=False)
            print(f"{len(active_df)} active businesses saved to {ACTIVE_FILE}")


if __name__ == "__main__":
    asyncio.run(main())

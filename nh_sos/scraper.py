# scraper.py
"""Main scraper logic"""
import asyncio
import random
from datetime import datetime
import pandas as pd

import config
import browser
from state import StateManager


def get_current_search_term(state):
    for term in config.SEARCH_TERMS:
        completed = len(state.completed_pages.get(term, set()))
        total = state.total_pages.get(term)
        
        if total is None or completed < total:
            return term
    return None


def get_next_batch(state, search_term):
    completed = state.completed_pages.get(search_term, set())
    total = state.total_pages.get(search_term)
    max_page = total if total else 999
    
    # Find first uncompleted page
    start_page = None
    for page_num in range(1, max_page + 1):
        if page_num not in completed:
            start_page = page_num
            break
    
    if not start_page:
        return []
    
    # Get consecutive pages
    batch = []
    for page_num in range(start_page, min(start_page + config.MAX_PAGES_PER_SESSION, max_page + 1)):
        if page_num not in completed:
            batch.append(page_num)
        if len(batch) >= config.PAGES_PER_BATCH:
            break
    
    return batch


async def scrape_session(state, search_term, start_page, end_page):
    businesses = []
    pages_completed = []
    
    p, browser_obj, page = await browser.create_browser()
    
    try:
        # Search
        if not await browser.search_term(page, search_term):
            return businesses, pages_completed
        
        # Detect total pages if needed
        if search_term not in state.total_pages:
            total = await browser.detect_total_pages(page)
            if total:
                state.total_pages[search_term] = total
                print(f"Detected {total} total pages for '{search_term}'")
        
        # Navigate to start page
        current_page = 1
        if start_page > 1:
            print(f"Navigating to page {start_page}...")
            current_page = await browser.navigate_to_page(page, start_page, current_page)
            if not current_page or current_page != start_page:
                print(f"Could not navigate to page {start_page}")
                return businesses, pages_completed
        
        # Scrape pages
        for target_page in range(start_page, end_page + 1):
            if target_page != current_page:
                current_page = await browser.navigate_to_page(page, target_page, current_page)
                if not current_page:
                    break
            
            print(f"\nProcessing page {target_page}...")
            
            page_text = await page.inner_text('body')
            if not browser.validate_page(page_text, target_page, state, search_term):
                break
            
            # Extract businesses
            page_businesses = await browser.extract_businesses(page)
            if page_businesses:
                businesses.extend(page_businesses)
                pages_completed.append(target_page)
                print(f"✓ Found {len(page_businesses)} businesses")
            else:
                print(f"✗ No businesses found")
                
    except Exception as e:
        print(f"Session error: {e}")
    finally:
        await browser_obj.close()
        await p.stop()
    
    return businesses, pages_completed


def print_session_header(search_term, start_page, end_page, state):
    print(f"\n{'=' * 60}")
    print(f"SESSION: '{search_term}' - Pages {start_page} to {end_page}")
    
    total = state.total_pages.get(search_term, "?")
    completed = len(state.completed_pages.get(search_term, set()))
    print(f"Progress: {completed}/{total} pages")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'=' * 60}")


def print_final_summary(state):
    print("\n" + "=" * 80)
    print("SCRAPING COMPLETE")
    print("=" * 80)
    
    total_completed = 0
    total_expected = 0
    known_terms = []
    unknown_terms = []
    incomplete_terms = []
    
    # Gather data
    for term in config.SEARCH_TERMS:
        completed = len(state.completed_pages.get(term, set()))
        total = state.total_pages.get(term)
        failed = len(state.failed_pages.get(term, set()))
        
        if isinstance(total, int):
            total_completed += completed
            total_expected += total
            known_terms.append(total)
            
            # Only show incomplete terms
            if completed < total or failed > 0:
                incomplete_terms.append((term, completed, total, failed))
        else:
            if completed > 0:  # Started but total unknown
                incomplete_terms.append((term, completed, None, failed))
            else:
                unknown_terms.append(term)
    
    # Only print incomplete terms
    if incomplete_terms:
        print("\nIncomplete terms:")
        for term, completed, total, failed in incomplete_terms:
            total_str = f"/{total}" if total else "/?"
            print(f"  '{term}': {completed}{total_str} pages", end="")
            if failed:
                print(f" ({failed} failed)", end="")
            print()
    
    # Print remaining terms count
    if unknown_terms:
        print(f"\nRemaining terms: {len(unknown_terms)} ({', '.join(unknown_terms)})")
    
    # Smart estimation
    if known_terms and unknown_terms:
        sorted_pages = sorted(known_terms)
        median_pages = sorted_pages[len(sorted_pages) // 2]
        estimated_per_unknown = int(median_pages * 0.75)
        estimated_unknown_total = estimated_per_unknown * len(unknown_terms)
        
        total_with_estimate = total_expected + estimated_unknown_total
        overall_percent = (total_completed / total_with_estimate * 100)
        
        print(f"\nProgress: {total_completed}/{total_expected} known + ~{estimated_unknown_total} estimated = {overall_percent:.1f}% complete")
    else:
        print(f"\nProgress: {total_completed}/{total_expected} pages ({total_completed/total_expected*100:.1f}% complete)")


async def run():
    state = StateManager()
    
    try:
        while True:
            # Get current search term
            search_term = get_current_search_term(state)
            if not search_term:
                print("\nAll search terms completed!")
                break
            
            state.current_search_term = search_term
            
            # Initialize tracking for new term
            if search_term not in state.completed_pages:
                state.completed_pages[search_term] = set()
                state.failed_pages[search_term] = set()
                print(f"\n{'=' * 60}")
                print(f"Starting new search term: '{search_term}'")
                print(f"{'=' * 60}")
            
            # Get next batch
            batch = get_next_batch(state, search_term)
            if not batch:
                completed = len(state.completed_pages.get(search_term, set()))
                total = state.total_pages.get(search_term, "?")
                print(f"\nCompleted '{search_term}': {completed}/{total} pages")
                continue
            
            start_page = min(batch)
            end_page = max(batch)
            
            print_session_header(search_term, start_page, end_page, state)
            
            # Scrape pages
            businesses, completed = await scrape_session(state, search_term, start_page, end_page)
            
            # Update state
            if businesses:
                state.add_businesses(businesses, search_term)
                
                for page_num in completed:
                    state.mark_page_complete(search_term, page_num)
                
                # Only mark pages as failed if they exist but couldn't be scraped
                total_pages = state.total_pages.get(search_term)
                for page_num in range(start_page, end_page + 1):
                    if page_num not in state.completed_pages[search_term]:
                        # Only mark as failed if within known page range
                        if total_pages is None or page_num <= total_pages:
                            if page_num not in completed:
                                state.mark_page_failed(search_term, page_num)
                
                unique_count = state.save_businesses()
                
                print(f"\n✓ Session complete: {len(businesses)} businesses from {len(completed)} pages")
                print(f"Total unique businesses: {unique_count}")
            else:
                # No businesses found - check if pages should exist
                total_pages = state.total_pages.get(search_term)
                for page_num in batch:
                    if total_pages is None or page_num <= total_pages:
                        state.mark_page_failed(search_term, page_num)
                print("\n✗ Session failed - no data retrieved")
            
            state.save_state()
            
            # Wait between sessions
            if get_current_search_term(state):
                wait_time = random.randint(config.MIN_WAIT, config.MAX_WAIT)
                next_time = datetime.fromtimestamp(datetime.now().timestamp() + wait_time)
                
                print(f"\n⏳ Waiting {wait_time} seconds...")
                print(f"Next session at: {next_time.strftime('%H:%M:%S')}")
                
                await asyncio.sleep(wait_time)
    
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n\nScraping interrupted by user")
        print("Progress has been saved - you can resume anytime")
        state.save_state()
    
    return state

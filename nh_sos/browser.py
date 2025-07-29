# browser.py
"""Browser automation and page interactions"""
import asyncio
import re
from playwright.async_api import async_playwright
import config


async def create_browser():
    p = await async_playwright().start()
    browser = await p.chromium.launch(
        headless=False,
        args=['--disable-blink-features=AutomationControlled']
    )
    context = await browser.new_context(
        viewport=config.VIEWPORT,
        user_agent=config.USER_AGENT
    )
    page = await context.new_page()
    return p, browser, page


async def search_term(page, term):
    try:
        await page.goto(config.BASE_URL)
        await asyncio.sleep(3)
        
        await page.click('input[value="All"]')
        await asyncio.sleep(1)
        await page.fill('input[name="txtBusinessName"]', term)
        await asyncio.sleep(1)
        await page.click('input[type="submit"]')
        print(f"Searching for '{term}'...")
        
        await page.wait_for_selector('table', timeout=30000)
        await asyncio.sleep(2)
        return True
    except:
        print("Search failed")
        return False


async def navigate_to_page(page, target_page, current_page=1):
    if target_page == current_page:
        return current_page
    
    # Try direct jump for pages > 11
    if target_page > 11 and current_page == 1:
        if await jump_to_page(page, target_page):
            return target_page
    
    # Click through pages
    while current_page < target_page:
        if not await click_next_page(page):
            return None
        current_page += 1
        
        if await is_browser_blocked(page):
            print(f"Hit browser check at page {current_page}")
            return None
    
    return current_page


async def jump_to_page(page, target_page):
    try:
        goto_elements = await page.query_selector_all('a')
        for elem in goto_elements:
            text = await elem.inner_text()
            if 'Go to Page' in text:
                await elem.click()
                await asyncio.sleep(1)
                await page.keyboard.type(str(target_page))
                await page.keyboard.press('Enter')
                await page.wait_for_load_state('networkidle')
                await asyncio.sleep(3)
                
                page_text = await page.inner_text('body')
                return f"Page {target_page} of" in page_text
    except:
        pass
    return False


async def click_next_page(page):
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


async def is_browser_blocked(page):
    page_text = await page.inner_text('body')
    return 'One moment, while we check your browser' in page_text


async def detect_total_pages(page):
    try:
        page_text = await page.inner_text('body')
        match = re.search(r'Page \d+ of (\d+)', page_text)
        if match:
            return int(match.group(1))
    except:
        pass
    return None


async def extract_businesses(page):
    businesses = []
    try:
        rows = await page.query_selector_all('table tr')
        
        for row in rows:
            cells = await row.query_selector_all('td')
            
            if len(cells) == 8:
                first_text = await cells[0].inner_text()
                
                # Skip navigation rows
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


def validate_page(page_text, expected_page, state, search_term):
    if 'One moment, while we check your browser' in page_text:
        print("Browser check detected!")
        return False
    
    # Check if we've gone past the last page
    if search_term in state.total_pages:
        if f"Page {expected_page} of" not in page_text:
            print(f"Page {expected_page} doesn't exist - reached end of results")
            state.total_pages[search_term] = expected_page - 1
            return False
    
    return True

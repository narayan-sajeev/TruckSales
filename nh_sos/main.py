# main.py
"""NH SOS Scraper - Main entry point"""
import asyncio
import os
import pandas as pd

import config
import scraper


async def main():
    print("\n" + "=" * 80)
    print("NH SOS Multi-Term Scraper v4")
    print(f"Search terms: {', '.join(config.SEARCH_TERMS)}")
    print(f"Wait time: {config.MIN_WAIT}-{config.MAX_WAIT} seconds between sessions")
    print("=" * 80 + "\n")
    
    state = None
    try:
        state = await scraper.run()
    except KeyboardInterrupt:
        # This is handled in scraper.run(), but just in case
        pass
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        print("Progress has been saved - you can resume after fixing the issue")
    
    # Final summary - only show if we have state
    if state:
        scraper.print_final_summary(state)
        
        # Save final data
        if os.path.exists(config.PROGRESS_FILE):
            df = pd.read_csv(config.PROGRESS_FILE)
            print(f"\nTotal unique businesses: {len(df)}")
            
            if 'search_term' in df.columns:
                print("\nBusinesses by search term:")
                print(df['search_term'].value_counts())
            
            # Save final version
            df.to_csv(config.FINAL_FILE, index=False)
            print(f"\nFinal data saved to {config.FINAL_FILE}")
            
            # Active businesses
            active_df = df[df['status'].isin(['Good Standing', 'Active'])]
            if len(active_df) > 0:
                active_df.to_csv(config.ACTIVE_FILE, index=False)
                print(f"{len(active_df)} active businesses saved to {config.ACTIVE_FILE}")


if __name__ == "__main__":
    asyncio.run(main())

# state.py
"""State persistence and management"""
import json
import os
import pandas as pd
import config


class StateManager:
    def __init__(self):
        self.completed_pages = {}  # {search_term: set(pages)}
        self.failed_pages = {}
        self.total_pages = {}
        self.current_search_term = None
        self.all_businesses = []
        self.load_state()
    
    def load_state(self):
        if os.path.exists(config.STATE_FILE):
            with open(config.STATE_FILE, 'r') as f:
                state = json.load(f)
                
                self.completed_pages = {
                    term: set(pages) 
                    for term, pages in state.get('completed_pages_by_term', {}).items()
                }
                self.failed_pages = {
                    term: set(pages) 
                    for term, pages in state.get('failed_pages_by_term', {}).items()
                }
                self.total_pages = state.get('total_pages_by_term', {})
                self.current_search_term = state.get('current_search_term', config.SEARCH_TERMS[0])
                
                print(f"Loaded previous state:")
                incomplete_found = False
                for term in self.completed_pages:
                    completed = len(self.completed_pages.get(term, set()))
                    total = self.total_pages.get(term)
                    # Only show incomplete terms
                    if total and completed < total:
                        if not incomplete_found:
                            print("  Incomplete:")
                            incomplete_found = True
                        print(f"    {term}: {completed}/{total} pages")
                
                if not incomplete_found:
                    # Just show a summary if everything is complete
                    completed_count = len([t for t in self.completed_pages if self.total_pages.get(t) and 
                                         len(self.completed_pages[t]) >= self.total_pages[t]])
                    print(f"  {completed_count} terms completed")
                
                # Clean up invalid failed pages on load
                self.clear_invalid_failed_pages()
        
        if os.path.exists(config.PROGRESS_FILE):
            df = pd.read_csv(config.PROGRESS_FILE)
            self.all_businesses = df.to_dict('records')
            print(f"Loaded {len(self.all_businesses)} existing businesses")
    
    def save_state(self):
        from datetime import datetime
        state = {
            'completed_pages_by_term': {
                term: list(pages) for term, pages in self.completed_pages.items()
            },
            'failed_pages_by_term': {
                term: list(pages) for term, pages in self.failed_pages.items()
            },
            'total_pages_by_term': self.total_pages,
            'current_search_term': self.current_search_term,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(config.STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    
    def save_businesses(self):
        if self.all_businesses:
            df = pd.DataFrame(self.all_businesses)
            df = df.drop_duplicates(subset=['business_id'])
            df.to_csv(config.PROGRESS_FILE, index=False)
            return len(df)
        return 0
    
    def add_businesses(self, businesses, search_term):
        for business in businesses:
            business['search_term'] = search_term
        self.all_businesses.extend(businesses)
    
    def mark_page_complete(self, search_term, page_num):
        if search_term not in self.completed_pages:
            self.completed_pages[search_term] = set()
        self.completed_pages[search_term].add(page_num)
        
        if search_term in self.failed_pages:
            self.failed_pages[search_term].discard(page_num)
    
    def mark_page_failed(self, search_term, page_num):
        if search_term not in self.failed_pages:
            self.failed_pages[search_term] = set()
        self.failed_pages[search_term].add(page_num)
    
    def clear_invalid_failed_pages(self):
        """Remove failed pages that are beyond the total page count"""
        for term in list(self.failed_pages.keys()):
            if term in self.total_pages:
                total = self.total_pages[term]
                # Remove any failed pages beyond the total
                self.failed_pages[term] = {
                    page for page in self.failed_pages[term] 
                    if page <= total
                }
                # Remove empty sets
                if not self.failed_pages[term]:
                    del self.failed_pages[term]

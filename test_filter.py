"""
Test script to verify the filtering logic works correctly for truck-relevant businesses.

Run this to ensure:
- Towing businesses are properly included
- Small contractors (plumbing, electrical) are excluded
- Edge cases are handled correctly
"""

from business_filters import is_truck_relevant, get_business_category, is_canadian_business
from business_matcher import names_are_very_similar
import pandas as pd


def test_filtering():
    """Test the business filtering logic."""
    
    # Test cases: (business_name, should_be_included)
    test_cases = [
        # TOWING - Should all be INCLUDED
        ("1St Priority Automotive And Towing", True),
        ("1St Priority Towing & Automotive", True),
        ("Cbk Towing & Recovery", True),
        ("Bills Towing & Recovery", True),
        ("Integrity Towing Inc.", True),
        ("AAA Towing Services", True),
        ("24/7 Emergency Towing", True),
        ("Bob's Tow Truck Service", True),
        
        # TRUCKING/TRANSPORT - Should be INCLUDED
        ("ABC Trucking Company", True),
        ("Johnson Transport LLC", True),
        ("Express Freight Services", True),
        ("Heavy Hauling Specialists", True),
        ("Regional Logistics Inc", True),
        
        # CONSTRUCTION/EXCAVATION - Should be INCLUDED
        ("Smith Construction LLC", True),
        ("Premier Excavation Services", True),
        ("ABC Demolition Company", True),
        ("Foundation Specialists Inc", True),
        ("Heavy Construction Co", True),
        
        # PAVING/CONCRETE - Should be INCLUDED
        ("Asphalt Paving Specialists", True),
        ("ABC Concrete Company", True),
        ("Road Construction Inc", True),
        
        # EXCLUDED - Not tractor trailer customers
        ("Johnson Farm Equipment", False),
        ("Valley Agricultural Services", False),
        ("Waste Management Corp", False),
        ("Green Valley Recycling", False),
        ("City Refuse Services", False),
        ("ABC Moving & Storage", False),
        ("Snow Removal Services", False),
        ("Green Landscaping LLC", False),
        ("Tree Service Company", False),
        ("Mining Operations Inc", False),
        
        # EXCLUDED - False positives for "tow"
        ("Tower Records", False),
        ("Towel Supply Company", False),
        ("Stowage Solutions", False),
        ("Township Building", False),
        ("Towne Center Mall", False),
        
        # EXCLUDED - Small contractors (don't need tractor trailers)
        ("Joe's Plumbing", False),
        ("ABC Electric", False),
        ("Quick Fix Electrical Services", False),
        ("HVAC Repair Services", False),
        ("Handyman Services LLC", False),
        ("Air Conditioning Specialists", False),
        
        # EXCLUDED - Non-business entities
        ("City of Boston", False),
        ("First Baptist Church", False),
        ("University Medical Center", False),
        ("Pizza Palace Restaurant", False),
        ("Hair Salon & Spa", False),
        
        # EDGE CASES - Mixed signals
        ("Tower Construction", True),  # Has construction, overrides "tower"
        ("Electric Transport Inc", True),  # Has transport, overrides "electric"
        ("Plumbing & Excavation", True),  # Has excavation, overrides "plumbing"
        ("Township Trucking", True),  # Has trucking, overrides "township"
    ]
    
    print("TESTING BUSINESS FILTERING LOGIC")
    print("=" * 80)
    print(f"{'Business Name':<45} {'Expected':<10} {'Result':<10} {'Status'}")
    print("-" * 80)
    
    passed = 0
    failed = 0
    
    for business_name, expected in test_cases:
        result = is_truck_relevant(business_name)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        expected_str = "Include" if expected else "Exclude"
        result_str = "Include" if result else "Exclude"
        print(f"{business_name:<45} {expected_str:<10} {result_str:<10} {status}")
    
    print("-" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    
    if failed == 0:
        print("\n✅ All filtering tests passed!")
    else:
        print(f"\n❌ {failed} filtering tests failed.")
    
    return failed == 0


def test_name_matching():
    """Test the name matching logic for deduplication."""
    
    print("\n\nTESTING NAME MATCHING LOGIC")
    print("=" * 80)
    
    # Test cases: (name1, name2, should_match)
    match_cases = [
        # Should match - same business with variations
        ("1St Priority Automotive And Towing", "1St Priority Towing & Automotive", True),
        ("ABC Trucking Inc", "ABC Trucking Incorporated", True),
        ("Smith Construction LLC", "Smith Construction", True),
        ("Johnson & Sons Excavation", "Johnson and Sons Excavation", True),
        ("5 Star Towing", "5 Star Towing Service", True),
        
        # Should NOT match - different businesses
        ("ABC Towing", "XYZ Towing", False),
        ("Smith Construction", "Jones Construction", False),
        ("North Trucking", "South Trucking", False),
        ("First Priority Towing", "Second Priority Towing", False),
    ]
    
    print(f"{'Name 1':<35} {'Name 2':<35} {'Should Match':<12} {'Result'}")
    print("-" * 95)
    
    passed = 0
    failed = 0
    
    for name1, name2, should_match in match_cases:
        is_similar, score = names_are_very_similar(name1, name2)
        result = "Match" if is_similar else "No Match"
        expected = "Match" if should_match else "No Match"
        status = "✓" if is_similar == should_match else "✗"
        
        if is_similar == should_match:
            passed += 1
        else:
            failed += 1
        
        print(f"{name1:<35} {name2:<35} {expected:<12} {result} ({score:.0f}) {status}")
    
    print("-" * 95)
    print(f"Results: {passed} passed, {failed} failed out of {len(match_cases)} tests")
    
    if failed == 0:
        print("\n✅ All name matching tests passed!")
    else:
        print(f"\n❌ {failed} name matching tests failed.")
    
    return failed == 0


def test_categorization():
    """Test business categorization."""
    
    print("\n\nTESTING BUSINESS CATEGORIZATION")
    print("=" * 80)
    
    test_businesses = [
        "ABC Towing & Recovery",
        "Smith Trucking Company",
        "Valley Construction LLC",
        "Johnson Transport Services",
        "Express Freight Logistics",
        "Heavy Hauling Inc",
        "Premier Paving Services",
        "Concrete Solutions LLC",
        "Demo Experts Inc",
        "Regional Excavation Co",
    ]
    
    print(f"{'Business Name':<35} {'Category'}")
    print("-" * 60)
    
    for business in test_businesses:
        category = get_business_category(business)
        print(f"{business:<35} {category}")


def test_canadian_detection():
    """Test the Canadian business detection logic."""
    
    print("\n\nTESTING CANADIAN BUSINESS DETECTION")
    print("=" * 80)
    
    # Test cases with mock business data
    test_cases = [
        # Canadian businesses (should be detected)
        ({"business_name": "ABC Trucking Ltd", "websites": "www.abctrucking.ca"}, True),
        ({"business_name": "Québec Transport", "websites": None}, True),
        ({"business_name": "Centre de Construction", "websites": None}, True),
        ({"business_name": "Toronto Towing ON", "websites": None}, True),
        ({"business_name": "BC Heavy Hauling", "websites": None}, True),
        ({"business_name": "Déménagement Rapide", "websites": None}, True),
        ({"business_name": "ABC Transport", "phones": "+1-416-555-1234"}, True),
        ({"business_name": "XYZ Trucking", "websites": "trucking.company.ca"}, True),
        
        # US businesses (should NOT be detected)
        ({"business_name": "ABC Trucking Inc", "websites": "www.abctrucking.com"}, False),
        ({"business_name": "California Transport", "websites": None}, False),
        ({"business_name": "Center Construction", "websites": None}, False),
        ({"business_name": "Boston Towing MA", "websites": None}, False),
        ({"business_name": "Heavy Hauling LLC", "websites": "www.heavyhaul.net"}, False),
        ({"business_name": "Quick Move", "websites": None}, False),
        ({"business_name": "XYZ Transport", "phones": "+1-212-555-1234"}, False),
    ]
    
    print(f"{'Business Info':<50} {'Expected':<10} {'Result':<10} {'Status'}")
    print("-" * 80)
    
    passed = 0
    failed = 0
    
    for business_data, expected in test_cases:
        # Create a pandas Series from the business data
        row = pd.Series(business_data)
        result = is_canadian_business(row)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        # Create display string
        display = business_data['business_name']
        if business_data.get('websites'):
            display += f" ({business_data['websites']})"
        elif business_data.get('phones'):
            display += f" ({business_data['phones']})"
            
        expected_str = "Canadian" if expected else "US"
        result_str = "Canadian" if result else "US"
        print(f"{display:<50} {expected_str:<10} {result_str:<10} {status}")
    
    print("-" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    
    if failed == 0:
        print("\n✅ All Canadian detection tests passed!")
    else:
        print(f"\n❌ {failed} Canadian detection tests failed.")
    
    return failed == 0


def main():
    """Run all tests."""
    print("\n🧪 TRUCK SALES TARGET FILTER TEST SUITE\n")
    
    # Run tests
    filter_ok = test_filtering()
    matching_ok = test_name_matching()
    canadian_ok = test_canadian_detection()
    test_categorization()
    
    # Summary
    print("\n" + "=" * 80)
    print("OVERALL TEST SUMMARY")
    print("=" * 80)
    
    if filter_ok and matching_ok and canadian_ok:
        print("✅ All tests passed! The filtering and matching logic is working correctly.")
    else:
        print("❌ Some tests failed. Please review the logic.")
        if not filter_ok:
            print("   - Filtering logic needs attention")
        if not matching_ok:
            print("   - Name matching logic needs attention")
        if not canadian_ok:
            print("   - Canadian detection logic needs attention")


if __name__ == "__main__":
    main()
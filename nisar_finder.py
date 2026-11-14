import earthaccess

def search_nisar_box():
    # 1. Authenticate with NASA Earthdata interactively
    print("Authenticating with NASA Earthdata...")
    auth = earthaccess.login(strategy="environment", username='marco_mars', password='U/CQp7RcWGyq3iY', persist=True)
    
    # Fix: .authenticated is a boolean property (no parentheses)
    if not auth.authenticated:
        print("Authentication failed. Please check your credentials.")
        return

    # 2. Define your bounding box (lon_min, lat_min, lon_max, lat_max)
    box_bounds = (117.5, 13.5, 118.2, 14.2)
    
    print(f"\nSearching for NISAR data inside bounding box: {box_bounds}")
    
    # 3. Search for NISAR products
    results = earthaccess.search_data(
        short_name='NISAR_L2_GCOV_PROVISIONAL_V1',
        bounding_box=box_bounds,
        temporal=("2024-01-01", "2026-12-31")
    )
    
    print(f"\nFound {len(results)} matching granules:")
    print("-" * 50)
    
    for i, granule in enumerate(results, 1):
        print(f"{i}. Granule ID: {granule.get('meta', {}).get('concept-id', 'N/A')}")
        print(f"   Timestamp: {granule.get('time-start', 'N/A')}")
        print("-" * 50)

if __name__ == "__main__":
    search_nisar_box()
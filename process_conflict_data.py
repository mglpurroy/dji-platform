#!/usr/bin/env python3
"""
Process ACLED conflict data to sub-prefecture (admin3) level using spatial joins.
This script matches ACLED event coordinates to admin3 boundaries and creates
a preprocessed ward_conflict_data.csv file for the dashboard.
Note: In Djibouti, admin3 is actually admin2 (sub-prefectures) from GADM.
"""

import sys
import geopandas as gpd
import pandas as pd
from pathlib import Path
import zipfile
import tempfile
import shutil
import warnings
warnings.filterwarnings('ignore')

# Check if we're in the virtual environment
def check_environment():
    """Check if required packages are available"""
    try:
        import geopandas
        import pandas
        import shapely
        return True
    except ImportError as e:
        print("=" * 60)
        print("ERROR: Required packages not found in current Python environment")
        print("=" * 60)
        print(f"Missing: {e}")
        print("\nPlease activate the virtual environment first:")
        print("  source venv/bin/activate")
        print("\nOr run with venv Python directly:")
        print(f"  {Path(__file__).parent / 'venv/bin/python'} {__file__}")
        print("=" * 60)
        return False

if not check_environment():
    sys.exit(1)

print("=" * 60)
print("Processing ACLED Conflict Data to Sub-prefecture Level")
print("=" * 60)

# Paths
DATA_PATH = Path("data/")
PROCESSED_PATH = DATA_PATH / "processed"
ACLED_FILE = Path("acled_Djibouti.csv")
GADM_ZIP = Path("gadm41_DJI_shp.zip")
OUTPUT_FILE = PROCESSED_PATH / "ward_conflict_data.csv"

# Ensure output directory exists
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# Step 1: Load ACLED data
print("\n1. Loading ACLED conflict data...")
if not ACLED_FILE.exists():
    print(f"❌ Error: ACLED data file not found: {ACLED_FILE}")
    sys.exit(1)

acled_df = pd.read_csv(ACLED_FILE)
print(f"   ✓ Loaded {len(acled_df):,} events")

# Step 2: Filter for events with fatalities (BRD events)
print("\n2. Filtering for events with fatalities...")
brd_events = acled_df[
    (~acled_df['event_type'].isin(['Protests', 'Riots'])) &
    (acled_df['fatalities'] > 0)
].copy()
print(f"   ✓ Filtered to {len(brd_events):,} events with fatalities")

# Step 3: Convert event_date to datetime and extract year/month
print("\n3. Processing dates...")
brd_events['event_date'] = pd.to_datetime(brd_events['event_date'])
brd_events['month'] = brd_events['event_date'].dt.month
brd_events['year'] = brd_events['event_date'].dt.year
print(f"   ✓ Date range: {brd_events['event_date'].min()} to {brd_events['event_date'].max()}")

# Step 4: Categorize violence type
print("\n4. Categorizing violence types...")
def categorize_violence(interaction):
    if pd.isna(interaction):
        return 'unknown'
    interaction_lower = str(interaction).lower()
    if 'state forces' in interaction_lower:
        return 'state'
    else:
        return 'nonstate'

brd_events['violence_type'] = brd_events['interaction'].apply(categorize_violence)
print(f"   ✓ State violence: {(brd_events['violence_type'] == 'state').sum():,} events")
print(f"   ✓ Non-state violence: {(brd_events['violence_type'] == 'nonstate').sum():,} events")

# Step 5: Convert to GeoDataFrame
print("\n5. Converting events to geographic points...")
# Drop rows with missing coordinates
brd_events_geo = brd_events.dropna(subset=['latitude', 'longitude']).copy()
print(f"   ✓ Events with coordinates: {len(brd_events_geo):,}")

# Create GeoDataFrame
events_gdf = gpd.GeoDataFrame(
    brd_events_geo,
    geometry=gpd.points_from_xy(brd_events_geo.longitude, brd_events_geo.latitude),
    crs="EPSG:4326"
)
print(f"   ✓ Created GeoDataFrame with {len(events_gdf):,} points")

# Step 6: Load admin3 (sub-prefecture) boundaries from GADM
print("\n6. Loading admin boundaries from GADM shapefiles...")
if not GADM_ZIP.exists():
    print(f"❌ Error: GADM file not found: {GADM_ZIP}")
    sys.exit(1)

# Extract GADM from zip
temp_dir = tempfile.mkdtemp()
try:
    with zipfile.ZipFile(GADM_ZIP, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # GADM structure: gadm41_DJI_0.shp (country), gadm41_DJI_1.shp (admin1), gadm41_DJI_2.shp (admin2)
    # Note: GADM for Djibouti doesn't have admin3, so we'll use admin2 as the lowest level
    admin2_shp = Path(temp_dir) / "gadm41_DJI_2.shp"
    
    if not admin2_shp.exists():
        print(f"❌ Error: {admin2_shp.name} not found in GADM zip")
        sys.exit(1)
    
    print(f"   ✓ Extracted GADM shapefiles")
    print(f"   ℹ Note: GADM for Djibouti only has admin2. Using admin2 as the lowest level (admin3).")
    
    # Load admin2 (we'll treat it as admin3 for compatibility)
    admin3_gdf = gpd.read_file(str(admin2_shp))
    print(f"   ✓ Loaded {len(admin3_gdf):,} admin2 units (treated as admin3)")
    
    # Map GADM columns to standard format
    if 'GID_2' in admin3_gdf.columns:
        admin3_gdf['ADM3_PCODE'] = admin3_gdf['GID_2']
        admin3_gdf['ADM2_PCODE'] = admin3_gdf['GID_2']  # Also set as ADM2 for compatibility
    if 'NAME_2' in admin3_gdf.columns:
        admin3_gdf['ADM3_EN'] = admin3_gdf['NAME_2']
        admin3_gdf['ADM2_EN'] = admin3_gdf['NAME_2']  # Also set as ADM2 for compatibility
    if 'GID_1' in admin3_gdf.columns:
        admin3_gdf['ADM1_PCODE'] = admin3_gdf['GID_1']
    if 'NAME_1' in admin3_gdf.columns:
        admin3_gdf['ADM1_EN'] = admin3_gdf['NAME_1']
    
    # Ensure CRS match
    if admin3_gdf.crs != events_gdf.crs:
        print(f"   ℹ Reprojecting boundaries from {admin3_gdf.crs} to {events_gdf.crs}")
        admin3_gdf = admin3_gdf.to_crs(events_gdf.crs)
    
    # Keep only necessary columns
    keep_cols = ['ADM3_PCODE', 'ADM3_EN', 'ADM2_PCODE', 'ADM2_EN', 'ADM1_PCODE', 'ADM1_EN', 'geometry']
    admin3_gdf = admin3_gdf[[col for col in keep_cols if col in admin3_gdf.columns]]
    
finally:
    # Clean up temp directory
    shutil.rmtree(temp_dir)

# Step 7: Spatial join - assign each event to a sub-prefecture
print("\n7. Performing spatial join (this may take a moment)...")
print("   ℹ Matching", len(events_gdf), "events to", len(admin3_gdf), "sub-prefectures...")

events_with_payam = gpd.sjoin(
    events_gdf,
    admin3_gdf,
    how='left',
    predicate='within'
)

# Count how many events were matched
matched = events_with_payam['ADM3_PCODE'].notna().sum()
unmatched = len(events_with_payam) - matched
print(f"   ✓ Matched: {matched:,} events ({matched/len(events_with_payam)*100:.1f}%)")
print(f"   ℹ Unmatched: {unmatched:,} events ({unmatched/len(events_with_payam)*100:.1f}%)")

# Step 8: Aggregate by sub-prefecture, year, month, and violence type
print("\n8. Aggregating events by sub-prefecture, time, and violence type...")

# Keep only matched events
events_matched = events_with_payam[events_with_payam['ADM3_PCODE'].notna()].copy()

# Aggregate
aggregated = events_matched.groupby(
    ['ADM3_PCODE', 'ADM3_EN', 'ADM2_PCODE', 'ADM2_EN', 'ADM1_PCODE', 'ADM1_EN', 'year', 'month', 'violence_type'],
    as_index=False
).agg({'fatalities': 'sum'})

print(f"   ✓ Created {len(aggregated):,} aggregated records")

# Step 9: Pivot to create state and nonstate columns
print("\n9. Pivoting violence types...")
pivoted = aggregated.pivot_table(
    index=['ADM3_PCODE', 'ADM3_EN', 'ADM2_PCODE', 'ADM2_EN', 'ADM1_PCODE', 'ADM1_EN', 'year', 'month'],
    columns='violence_type',
    values='fatalities',
    fill_value=0
).reset_index()

# Flatten column names
pivoted.columns.name = None

# Rename violence type columns to match expected format
rename_map = {}
if 'state' in pivoted.columns:
    rename_map['state'] = 'ACLED_BRD_state'
if 'nonstate' in pivoted.columns:
    rename_map['nonstate'] = 'ACLED_BRD_nonstate'
if 'unknown' in pivoted.columns:
    rename_map['unknown'] = 'ACLED_BRD_unknown'

pivoted = pivoted.rename(columns=rename_map)

# Ensure both state and nonstate columns exist
if 'ACLED_BRD_state' not in pivoted.columns:
    pivoted['ACLED_BRD_state'] = 0
if 'ACLED_BRD_nonstate' not in pivoted.columns:
    pivoted['ACLED_BRD_nonstate'] = 0

# Calculate total
pivoted['ACLED_BRD_total'] = pivoted['ACLED_BRD_state'] + pivoted['ACLED_BRD_nonstate']

# Rename columns to match old format (wardcode, wardname, etc.)
final_df = pivoted.rename(columns={
    'ADM3_PCODE': 'wardcode',
    'ADM3_EN': 'wardname',
    'ADM2_EN': 'countyname',
    'ADM1_EN': 'statename'
})

print(f"   ✓ Final dataset: {len(final_df):,} records")
print(f"   ✓ Unique sub-prefectures: {final_df['wardcode'].nunique():,}")
print(f"   ✓ Total fatalities: {final_df['ACLED_BRD_total'].sum():,.0f}")

# Step 10: Save to CSV
print(f"\n10. Saving to {OUTPUT_FILE}...")
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"   ✓ Saved successfully")

# Summary
print("\n" + "=" * 60)
print("PROCESSING COMPLETE!")
print("=" * 60)
print(f"Output file: {OUTPUT_FILE}")
print(f"Records: {len(final_df):,}")
print(f"Sub-prefectures with conflict: {final_df['wardcode'].nunique():,}")
print(f"Date range: {final_df['year'].min()}-{final_df['month'].min():02d} to {final_df['year'].max()}-{final_df['month'].max():02d}")
print(f"Total fatalities: {final_df['ACLED_BRD_total'].sum():,.0f}")
print("\nYou can now refresh the Streamlit dashboard to see sub-prefecture-level conflict data.")
print("=" * 60)


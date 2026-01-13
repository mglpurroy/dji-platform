"""
Script to pre-extract population data from raster for each admin level
Run this once to create GeoJSON files with population data attached

Usage:
    source venv/bin/activate
    python extract_population.py
"""

import sys
from pathlib import Path

# Check if we're in a virtual environment or if packages are available
try:
    import geopandas as gpd
    import rasterio
    from rasterstats import zonal_stats
    import zipfile
    import tempfile
    import shutil
    import fiona
    import pandas as pd
except ImportError as e:
    venv_python = Path(__file__).parent / "venv" / "bin" / "python"
    if venv_python.exists():
        print("=" * 60)
        print("ERROR: Required packages not found in current Python environment")
        print("=" * 60)
        print(f"Missing: {e}")
        print(f"\nPlease activate the virtual environment first:")
        print(f"  source venv/bin/activate")
        print(f"\nOr run with venv Python directly:")
        print(f"  {venv_python} {__file__}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("ERROR: Required packages not found. Please install:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

# Data paths
DATA_PATH = Path("data/")
PROCESSED_PATH = DATA_PATH / "processed"
POPULATION_RASTER = Path("dji_pop_2025_CN_100m_R2025A_v1.tif")
GADM_ZIP = Path("gadm41_DJI_shp.zip")

# Create processed directory
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

def load_boundaries_from_gadm():
    """Load admin boundaries from GADM shapefile zip"""
    boundaries = {}
    temp_dir = None
    
    try:
        if not GADM_ZIP.exists():
            print(f"Error: {GADM_ZIP} not found")
            return boundaries
        
        print("Extracting GADM shapefiles from zip...")
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(GADM_ZIP, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # GADM structure: gadm41_DJI_0.shp (country), gadm41_DJI_1.shp (admin1), gadm41_DJI_2.shp (admin2)
        # Note: GADM for Djibouti only has admin0, admin1, admin2 - no admin3
        
        # Load admin1 (states/regions)
        admin1_shp = Path(temp_dir) / "gadm41_DJI_1.shp"
        if admin1_shp.exists():
            print(f"Loading admin1 from: {admin1_shp.name}")
            boundaries[1] = gpd.read_file(str(admin1_shp))
            boundaries[1] = boundaries[1].to_crs('EPSG:4326')
            # Map GADM columns to standard format
            if 'GID_1' in boundaries[1].columns:
                boundaries[1]['ADM1_PCODE'] = boundaries[1]['GID_1']
            if 'NAME_1' in boundaries[1].columns:
                boundaries[1]['ADM1_EN'] = boundaries[1]['NAME_1']
            print(f"  Loaded {len(boundaries[1])} features")
        else:
            print(f"Warning: {admin1_shp.name} not found in zip")
        
        # Load admin2 (counties/districts)
        admin2_shp = Path(temp_dir) / "gadm41_DJI_2.shp"
        if admin2_shp.exists():
            print(f"Loading admin2 from: {admin2_shp.name}")
            boundaries[2] = gpd.read_file(str(admin2_shp))
            boundaries[2] = boundaries[2].to_crs('EPSG:4326')
            # Map GADM columns to standard format
            if 'GID_2' in boundaries[2].columns:
                boundaries[2]['ADM2_PCODE'] = boundaries[2]['GID_2']
            if 'NAME_2' in boundaries[2].columns:
                boundaries[2]['ADM2_EN'] = boundaries[2]['NAME_2']
            if 'GID_1' in boundaries[2].columns:
                boundaries[2]['ADM1_PCODE'] = boundaries[2]['GID_1']
            if 'NAME_1' in boundaries[2].columns:
                boundaries[2]['ADM1_EN'] = boundaries[2]['NAME_1']
            print(f"  Loaded {len(boundaries[2])} features")
        else:
            print(f"Warning: {admin2_shp.name} not found in zip")
        
        # Note: GADM for Djibouti doesn't have admin3, so we'll use admin2 as the lowest level
        # For compatibility, we can create admin3 from admin2 (each admin2 becomes an admin3)
        if 2 in boundaries and not boundaries[2].empty:
            print("Note: GADM doesn't include admin3. Using admin2 as the lowest administrative level.")
            # Create admin3 from admin2 for compatibility
            boundaries[3] = boundaries[2].copy()
            if 'ADM2_PCODE' in boundaries[3].columns:
                boundaries[3]['ADM3_PCODE'] = boundaries[3]['ADM2_PCODE']
            if 'ADM2_EN' in boundaries[3].columns:
                boundaries[3]['ADM3_EN'] = boundaries[3]['ADM2_EN']
            print(f"  Created admin3 from admin2: {len(boundaries[3])} features")
        
        return boundaries
        
    except Exception as e:
        print(f"Error loading boundaries: {e}")
        import traceback
        traceback.print_exc()
        return boundaries
    finally:
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

def standardize_admin_columns(gdf, level):
    """Standardize column names for admin boundaries"""
    if gdf.empty:
        return gdf
    
    gdf = gdf.copy()
    
    # Try to find and map columns
    for col in gdf.columns:
        col_upper = col.upper()
        if level >= 1:
            if ('ADM1' in col_upper and 'PCODE' in col_upper) or col_upper == 'ADM1_PCODE':
                gdf = gdf.rename(columns={col: 'ADM1_PCODE'})
            elif ('ADM1' in col_upper and ('EN' in col_upper or 'NAME' in col_upper)) or col_upper in ['ADM1_EN', 'ADM1_NAME']:
                gdf = gdf.rename(columns={col: 'ADM1_EN'})
        
        if level >= 2:
            if ('ADM2' in col_upper and 'PCODE' in col_upper) or col_upper == 'ADM2_PCODE':
                gdf = gdf.rename(columns={col: 'ADM2_PCODE'})
            elif ('ADM2' in col_upper and ('EN' in col_upper or 'NAME' in col_upper)) or col_upper in ['ADM2_EN', 'ADM2_NAME']:
                gdf = gdf.rename(columns={col: 'ADM2_EN'})
        
        if level >= 3:
            if ('ADM3' in col_upper and 'PCODE' in col_upper) or col_upper == 'ADM3_PCODE':
                gdf = gdf.rename(columns={col: 'ADM3_PCODE'})
            elif ('ADM3' in col_upper and ('EN' in col_upper or 'NAME' in col_upper)) or col_upper in ['ADM3_EN', 'ADM3_NAME']:
                gdf = gdf.rename(columns={col: 'ADM3_EN'})
    
    return gdf

def extract_population_from_raster(gdf, raster_path, level_name):
    """Extract population statistics from raster for each geometry"""
    print(f"\nExtracting population for {level_name}...")
    
    if gdf.empty:
        print(f"  No geometries for {level_name}")
        return gdf
    
    # Ensure CRS matches raster (usually EPSG:4326 or projected)
    # Raster is likely in a projected CRS, so we may need to reproject
    try:
        # Read raster to get its CRS
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            print(f"  Raster CRS: {raster_crs}")
        
        # Reproject boundaries to match raster CRS if needed
        if gdf.crs != raster_crs:
            print(f"  Reprojecting from {gdf.crs} to {raster_crs}")
            gdf_proj = gdf.to_crs(raster_crs)
        else:
            gdf_proj = gdf.copy()
        
        # Get nodata value from raster
        with rasterio.open(raster_path) as src:
            nodata_val = src.nodata if src.nodata is not None else -99999.0
            print(f"  Raster nodata value: {nodata_val}")
        
        # Extract zonal statistics
        print(f"  Calculating zonal statistics for {len(gdf_proj)} features...")
        stats = zonal_stats(
            gdf_proj.geometry,
            str(raster_path),
            stats=['sum', 'mean', 'count'],
            nodata=nodata_val,
            all_touched=False
        )
        
        # Add population data to GeoDataFrame
        gdf['pop_count'] = [s.get('sum', 0) if s else 0 for s in stats]
        gdf['pop_mean'] = [s.get('mean', 0) if s else 0 for s in stats]
        gdf['pop_pixel_count'] = [s.get('count', 0) if s else 0 for s in stats]
        
        # Convert to integers
        gdf['pop_count'] = gdf['pop_count'].fillna(0).astype(int)
        gdf['pop_count_millions'] = gdf['pop_count'] / 1e6
        
        total_pop = gdf['pop_count'].sum()
        print(f"  Total population extracted: {total_pop:,.0f}")
        print(f"  Average per feature: {gdf['pop_count'].mean():,.0f}")
        
    except Exception as e:
        print(f"  Error extracting population: {e}")
        import traceback
        traceback.print_exc()
        # Add zero population as fallback
        gdf['pop_count'] = 0
        gdf['pop_count_millions'] = 0.0
    
    return gdf

def main():
    """Main function to extract population for all admin levels"""
    print("=" * 60)
    print("Population Extraction Script")
    print("=" * 60)
    
    # Check if raster exists
    if not POPULATION_RASTER.exists():
        print(f"Error: Population raster not found at {POPULATION_RASTER}")
        return
    
    print(f"Using population raster: {POPULATION_RASTER}")
    
    # Load boundaries
    print("\nLoading administrative boundaries...")
    boundaries = load_boundaries_from_gadm()
    
    if not boundaries:
        print("Error: Could not load boundaries")
        return
    
    # Process each admin level
    for level in [3, 2, 1]:  # Start with sub-prefectures (most detailed), then aggregate
        if level not in boundaries or boundaries[level].empty:
            print(f"\nSkipping admin level {level} (no data)")
            continue
        
        level_name = {1: 'admin1_states', 2: 'admin2_counties', 3: 'admin3_payams'}[level]  # Note: file names kept for compatibility
        gdf = boundaries[level].copy()
        
        # Standardize columns
        gdf = standardize_admin_columns(gdf, level)
        
        # Extract population
        gdf = extract_population_from_raster(gdf, POPULATION_RASTER, level_name)
        
        # Save as GeoJSON
        output_file = PROCESSED_PATH / f"{level_name}_with_population.geojson"
        print(f"\nSaving to {output_file}...")
        gdf.to_file(output_file, driver='GeoJSON')
        print(f"  Saved {len(gdf)} features")
        print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 60)
    print("Population extraction complete!")
    print("=" * 60)
    print(f"\nOutput files saved to: {PROCESSED_PATH}")
    print("  - admin3_payams_with_population.geojson (sub-prefectures)")
    print("  - admin2_counties_with_population.geojson (sub-prefectures)")
    print("  - admin1_states_with_population.geojson (regions)")

if __name__ == "__main__":
    main()



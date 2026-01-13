"""
Extract GADM shapefiles to GeoJSON format for easier loading
Run this once to create GeoJSON files from GADM shapefiles
"""

import sys
from pathlib import Path
import geopandas as gpd
import zipfile
import tempfile
import shutil

# Check if we're in a virtual environment or if packages are available
try:
    import geopandas as gpd
    import fiona
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
BOUNDARIES_PATH = DATA_PATH / "boundaries"
GADM_ZIP = Path("gadm41_DJI_shp.zip")

# Create boundaries directory
BOUNDARIES_PATH.mkdir(parents=True, exist_ok=True)

def extract_gadm_to_geojson():
    """Extract GADM shapefiles and convert to GeoJSON"""
    print("=" * 60)
    print("Extracting GADM Boundaries to GeoJSON")
    print("=" * 60)
    
    if not GADM_ZIP.exists():
        print(f"Error: {GADM_ZIP} not found")
        return
    
    temp_dir = None
    try:
        print(f"\nExtracting {GADM_ZIP}...")
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(GADM_ZIP, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # GADM structure: gadm41_DJI_0.shp (country), gadm41_DJI_1.shp (admin1), gadm41_DJI_2.shp (admin2)
        
        # Extract admin1 (regions)
        admin1_shp = Path(temp_dir) / "gadm41_DJI_1.shp"
        if admin1_shp.exists():
            print(f"\nLoading admin1 (regions) from {admin1_shp.name}...")
            admin1_gdf = gpd.read_file(str(admin1_shp))
            admin1_gdf = admin1_gdf.to_crs('EPSG:4326')
            
            # Map GADM columns to standard format
            if 'GID_1' in admin1_gdf.columns:
                admin1_gdf['ADM1_PCODE'] = admin1_gdf['GID_1']
            if 'NAME_1' in admin1_gdf.columns:
                admin1_gdf['ADM1_EN'] = admin1_gdf['NAME_1']
            
            # Save as GeoJSON
            output_file = BOUNDARIES_PATH / "admin1_regions.geojson"
            admin1_gdf.to_file(output_file, driver='GeoJSON')
            print(f"  ✓ Saved {len(admin1_gdf)} regions to {output_file}")
            print(f"  File size: {output_file.stat().st_size / 1024:.2f} KB")
        
        # Extract admin2 (sub-prefectures)
        admin2_shp = Path(temp_dir) / "gadm41_DJI_2.shp"
        if admin2_shp.exists():
            print(f"\nLoading admin2 (sub-prefectures) from {admin2_shp.name}...")
            admin2_gdf = gpd.read_file(str(admin2_shp))
            admin2_gdf = admin2_gdf.to_crs('EPSG:4326')
            
            # Map GADM columns to standard format
            if 'GID_2' in admin2_gdf.columns:
                admin2_gdf['ADM2_PCODE'] = admin2_gdf['GID_2']
                admin2_gdf['ADM3_PCODE'] = admin2_gdf['GID_2']  # Also set as ADM3 for compatibility
            if 'NAME_2' in admin2_gdf.columns:
                admin2_gdf['ADM2_EN'] = admin2_gdf['NAME_2']
                admin2_gdf['ADM3_EN'] = admin2_gdf['NAME_2']  # Also set as ADM3 for compatibility
            if 'GID_1' in admin2_gdf.columns:
                admin2_gdf['ADM1_PCODE'] = admin2_gdf['GID_1']
            if 'NAME_1' in admin2_gdf.columns:
                admin2_gdf['ADM1_EN'] = admin2_gdf['NAME_1']
            
            # Save as GeoJSON for admin2
            output_file = BOUNDARIES_PATH / "admin2_subprefectures.geojson"
            admin2_gdf.to_file(output_file, driver='GeoJSON')
            print(f"  ✓ Saved {len(admin2_gdf)} sub-prefectures to {output_file}")
            print(f"  File size: {output_file.stat().st_size / 1024:.2f} KB")
            
            # Also save as admin3 (for compatibility)
            output_file = BOUNDARIES_PATH / "admin3_subprefectures.geojson"
            admin2_gdf.to_file(output_file, driver='GeoJSON')
            print(f"  ✓ Saved as admin3 (for compatibility) to {output_file}")
        
        print("\n" + "=" * 60)
        print("Boundary extraction complete!")
        print("=" * 60)
        print(f"\nOutput files saved to: {BOUNDARIES_PATH}")
        print("  - admin1_regions.geojson")
        print("  - admin2_subprefectures.geojson")
        print("  - admin3_subprefectures.geojson")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

if __name__ == "__main__":
    extract_gadm_to_geojson()

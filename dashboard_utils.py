"""Shared utilities for Djibouti Violence Dashboard multi-page app"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import pickle
import hashlib
import time
import warnings
import requests
import zipfile
import tempfile
import shutil
import fiona
warnings.filterwarnings('ignore')

# Data paths
DATA_PATH = Path("data/")
PROCESSED_PATH = DATA_PATH / "processed"
CACHE_PATH = Path("cache/")

# Create cache directory only if we have write permissions
try:
    CACHE_PATH.mkdir(exist_ok=True)
    CACHE_ENABLED = True
except (PermissionError, OSError):
    CACHE_ENABLED = False

POPULATION_RASTER = Path("dji_pop_2025_CN_100m_R2025A_v1.tif")
ACLED_DATA = Path("acled_Djibouti.csv")

START_YEAR = 1997
END_YEAR = 2025

# Initialize session state for performance tracking
def init_session_state():
    """Initialize session state variables"""
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = {}
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.periods = None
        st.session_state.pop_data = None
        st.session_state.admin_data = None
        st.session_state.conflict_data = None
        st.session_state.boundaries = None
        st.session_state.subpref_timeseries_loaded = False

def log_performance(func_name, duration):
    """Log performance metrics for monitoring"""
    if func_name not in st.session_state.performance_metrics:
        st.session_state.performance_metrics[func_name] = []
    st.session_state.performance_metrics[func_name].append(duration)

# Cache utilities
def get_cache_key(*args):
    """Generate cache key from arguments"""
    return hashlib.md5(str(args).encode()).hexdigest()

def save_to_cache(key, data):
    """Save data to cache file"""
    if not CACHE_ENABLED:
        return
    try:
        cache_file = CACHE_PATH / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception:
        pass

def load_from_cache(key):
    """Load data from cache file"""
    if not CACHE_ENABLED:
        return None
    try:
        cache_file = CACHE_PATH / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    except Exception:
        pass
    return None

# Data loading functions
@st.cache_data(ttl=3600)
def generate_12_month_periods():
    """Generate 12-month periods every 6 months - optimized"""
    periods = []
    
    # Calendar year periods (Jan-Dec)
    for year in range(START_YEAR, END_YEAR + 1):
        periods.append({
            'label': f'Jan {year} - Dec {year}',
            'start_month': 1,
            'start_year': year,
            'end_month': 12,
            'end_year': year,
            'type': 'calendar'
        })
    
    # Mid-year periods (Jul-Jun)
    for year in range(START_YEAR, END_YEAR):
        periods.append({
            'label': f'Jul {year} - Jun {year+1}',
            'start_month': 7,
            'start_year': year,
            'end_month': 6,
            'end_year': year + 1,
            'type': 'mid_year'
        })
    
    return periods

@st.cache_data(ttl=3600, show_spinner=False)
def load_population_data():
    """Load and cache population data from pre-extracted GeoJSON files"""
    start_time = time.time()
    
    # Check cache first (but use new key to force reload after extraction)
    cache_key = get_cache_key("djibouti_population_data", "v7_extracted_geojson_fixed")
    cached_data = load_from_cache(cache_key)
    if cached_data is not None and not cached_data.empty:
        log_performance("load_population_data", time.time() - start_time)
        return cached_data
    
    # Initialize result_df to ensure it's always defined
    result_df = pd.DataFrame(columns=['ADM3_PCODE', 'ADM1_PCODE', 'ADM2_PCODE', 'ADM3_EN', 
                                      'ADM0_PCODE', 'ADM1_EN', 'ADM2_EN', 'pop_count', 'pop_count_millions'])
    
    try:
        # Try to load from pre-extracted GeoJSON file (preferred method)
        # Note: File is named "payams" for compatibility but contains sub-prefectures
        payam_pop_file = PROCESSED_PATH / "admin3_payams_with_population.geojson"
        
        # Ensure processed directory exists
        PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists and show debug info
        file_exists = payam_pop_file.exists()
        if not file_exists:
            st.warning(f"⚠️ Population file not found at: {payam_pop_file.absolute()}")
            st.info("💡 Run 'python extract_population.py' to create the file")
        
        if file_exists:
            try:
                payam_gdf = gpd.read_file(payam_pop_file)
                
                # Extract population data and standardize column names
                # Ensure required columns exist
                required_cols = ['ADM3_PCODE', 'ADM1_PCODE', 'ADM2_PCODE', 'ADM3_EN', 'pop_count']
                missing_cols = [col for col in required_cols if col not in payam_gdf.columns]
                
                if missing_cols:
                    st.warning(f"⚠️ Missing columns in population file: {missing_cols}")
                    # Try to infer from available columns
                    if 'pop_count' not in payam_gdf.columns:
                        payam_gdf['pop_count'] = 0
                        payam_gdf['pop_count_millions'] = 0.0
                    else:
                        payam_gdf['pop_count_millions'] = payam_gdf['pop_count'] / 1e6
                else:
                    # Ensure pop_count_millions exists
                    if 'pop_count_millions' not in payam_gdf.columns:
                        payam_gdf['pop_count_millions'] = payam_gdf['pop_count'] / 1e6
                
                # Create result DataFrame - use direct column access since we know they exist
                result_df = pd.DataFrame({
                    'ADM3_PCODE': payam_gdf['ADM3_PCODE'].astype(str),
                    'ADM1_PCODE': payam_gdf['ADM1_PCODE'].astype(str),
                    'ADM2_PCODE': payam_gdf['ADM2_PCODE'].astype(str),
                    'ADM3_EN': payam_gdf['ADM3_EN'].astype(str),
                    'ADM0_PCODE': 'DJI',
                    'pop_count': payam_gdf['pop_count'].fillna(0).astype(int),
                    'pop_count_millions': payam_gdf['pop_count_millions'].fillna(0)
                })
                
                # Add state and county names if available
                if 'ADM1_EN' in payam_gdf.columns:
                    result_df['ADM1_EN'] = payam_gdf['ADM1_EN'].astype(str)
                else:
                    result_df['ADM1_EN'] = result_df['ADM1_PCODE']
                
                if 'ADM2_EN' in payam_gdf.columns:
                    result_df['ADM2_EN'] = payam_gdf['ADM2_EN'].astype(str)
                else:
                    result_df['ADM2_EN'] = result_df['ADM2_PCODE']
                
                total_pop = result_df['pop_count'].sum()
                st.success(f"✅ Loaded population data for {len(result_df)} sub-prefectures (total population: {total_pop:,.0f})")
                
            except Exception as e:
                st.error(f"❌ Error loading population file: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                result_df = pd.DataFrame(columns=['ADM3_PCODE', 'ADM1_PCODE', 'ADM2_PCODE', 'ADM3_EN', 
                                                  'ADM0_PCODE', 'ADM1_EN', 'ADM2_EN', 'pop_count', 'pop_count_millions'])
            
        else:
            # Fallback to legacy shapefile method
            st.info("Pre-extracted population file not found. Using legacy shapefile method...")
            payam_file = DATA_PATH / "wards" / "wards.shp"  # Legacy path name
            
            if payam_file.exists():
                payam_gdf = gpd.read_file(payam_file)
                
                result_df = pd.DataFrame({
                    'ADM3_PCODE': payam_gdf.get('ward_cd', payam_gdf.index.astype(str)),
                    'ADM1_PCODE': payam_gdf.get('stat_cd', ''),
                    'ADM2_PCODE': payam_gdf.get('lga_cod', ''),
                    'ADM3_EN': payam_gdf.get('wrd_nm_x', ''),
                    'ADM0_PCODE': 'DJI',
                    'pop_count': payam_gdf.get('total_pop', 0).fillna(0).astype(int),
                    'pop_count_millions': payam_gdf.get('total_pop', 0).fillna(0) / 1e6
                })
                
                # Add state and county names using mapping from county boundaries
                county_file = DATA_PATH / "dji_county_boundaries.geojson"
                if county_file.exists():
                    county_gdf = gpd.read_file(county_file)
                    state_mapping = dict(zip(county_gdf['statecode'], county_gdf['statename']))
                    county_mapping = dict(zip(county_gdf['countycode'], county_gdf['countyname']))
                    
                    result_df['ADM1_EN'] = result_df['ADM1_PCODE'].map(state_mapping).fillna(result_df['ADM1_PCODE'])
                    result_df['ADM2_EN'] = result_df['ADM2_PCODE'].map(county_mapping).fillna(result_df['ADM2_PCODE'])
                else:
                    result_df['ADM1_EN'] = result_df['ADM1_PCODE']
                    result_df['ADM2_EN'] = result_df['ADM2_PCODE']
            else:
                # Last resort: Extract population from raster on-the-fly using boundaries
                st.warning("⚠️ No pre-extracted population files found. Extracting from raster on-the-fly (this may take 2-5 minutes)...")
                
                # Get boundaries - try session state first, then load if needed
                boundaries = st.session_state.get('boundaries')
                if not boundaries:
                    # Try to load boundaries if not in session state (this will be cached)
                    st.info("Loading boundaries for population extraction...")
                    boundaries = load_admin_boundaries()
                
                if not boundaries:
                    st.error("❌ Could not load boundaries for population extraction")
                    result_df = pd.DataFrame(columns=['ADM3_PCODE', 'ADM1_PCODE', 'ADM2_PCODE', 'ADM3_EN', 
                                                      'ADM0_PCODE', 'ADM1_EN', 'ADM2_EN', 'pop_count', 'pop_count_millions'])
                elif 3 not in boundaries or boundaries[3].empty:
                    st.error(f"❌ No admin3 boundaries available. Boundaries keys: {list(boundaries.keys()) if boundaries else 'None'}")
                    result_df = pd.DataFrame(columns=['ADM3_PCODE', 'ADM1_PCODE', 'ADM2_PCODE', 'ADM3_EN', 
                                                      'ADM0_PCODE', 'ADM1_EN', 'ADM2_EN', 'pop_count', 'pop_count_millions'])
                else:
                    try:
                        from rasterstats import zonal_stats
                        import rasterio
                        
                        raster_path = Path("dji_pop_2025_CN_100m_R2025A_v1.tif")
                        st.info(f"🔍 Checking for raster at: {raster_path.absolute()}")
                        if raster_path.exists():
                            st.info("✅ Raster file found. Starting extraction...")
                            payam_gdf = boundaries[3].copy()
                            
                            # Standardize columns using the helper function
                            payam_gdf = standardize_admin_columns(payam_gdf, level=3)
                            
                            # Debug: show what columns we have after standardization
                            st.info(f"Boundaries loaded: {len(payam_gdf)} sub-prefectures. Columns: {list(payam_gdf.columns)[:10]}")
                            
                            # Ensure we have required columns - check and standardize if needed
                            if 'ADM3_PCODE' not in payam_gdf.columns:
                                # Try to find PCODE column
                                pcode_cols = [c for c in payam_gdf.columns if 'PCODE' in c.upper() and 'ADM3' in c.upper()]
                                if pcode_cols:
                                    payam_gdf['ADM3_PCODE'] = payam_gdf[pcode_cols[0]]
                                    st.info(f"Using {pcode_cols[0]} as ADM3_PCODE")
                                else:
                                    st.error(f"Boundaries missing ADM3_PCODE column. Available columns: {list(payam_gdf.columns)}")
                                    result_df = pd.DataFrame(columns=['ADM3_PCODE', 'ADM1_PCODE', 'ADM2_PCODE', 'ADM3_EN', 
                                                                      'ADM0_PCODE', 'ADM1_EN', 'ADM2_EN', 'pop_count', 'pop_count_millions'])
                            else:
                                # We have ADM3_PCODE, proceed with extraction
                                # Get raster CRS and reproject if needed
                                with rasterio.open(raster_path) as src:
                                    raster_crs = src.crs
                                
                                if payam_gdf.crs != raster_crs:
                                    payam_gdf_proj = payam_gdf.to_crs(raster_crs)
                                else:
                                    payam_gdf_proj = payam_gdf.copy()
                                
                                # Extract population with progress
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                status_text.text(f"Extracting population from raster for {len(payam_gdf_proj)} sub-prefectures... This may take 2-5 minutes.")
                                progress_bar.progress(10)
                                
                                try:
                                    # Get nodata value from raster
                                    with rasterio.open(raster_path) as src:
                                        nodata_val = src.nodata if src.nodata is not None else -99999.0
                                    
                                    stats = zonal_stats(
                                        payam_gdf_proj.geometry,
                                        str(raster_path),
                                        stats=['sum'],
                                        nodata=nodata_val,
                                        all_touched=False
                                    )
                                    progress_bar.progress(90)
                                    
                                    # Add population to GeoDataFrame
                                    payam_gdf['pop_count'] = [s.get('sum', 0) if s else 0 for s in stats]
                                    payam_gdf['pop_count'] = payam_gdf['pop_count'].fillna(0).astype(int)
                                    payam_gdf['pop_count_millions'] = payam_gdf['pop_count'] / 1e6
                                    
                                    total_pop = payam_gdf['pop_count'].sum()
                                    st.info(f"Extracted total population: {total_pop:,.0f}")
                                    
                                    progress_bar.progress(100)
                                    status_text.text("Population extraction complete!")
                                    
                                    # Create result DataFrame - ensure all columns exist
                                    result_df = pd.DataFrame({
                                        'ADM3_PCODE': payam_gdf['ADM3_PCODE'].astype(str),
                                        'ADM1_PCODE': payam_gdf.get('ADM1_PCODE', '').astype(str) if 'ADM1_PCODE' in payam_gdf.columns else '',
                                        'ADM2_PCODE': payam_gdf.get('ADM2_PCODE', '').astype(str) if 'ADM2_PCODE' in payam_gdf.columns else '',
                                        'ADM3_EN': payam_gdf.get('ADM3_EN', '').astype(str) if 'ADM3_EN' in payam_gdf.columns else '',
                                        'ADM0_PCODE': 'DJI',
                                        'pop_count': payam_gdf['pop_count'],
                                        'pop_count_millions': payam_gdf['pop_count_millions']
                                    })
                                    
                                    # Add names
                                    if 'ADM1_EN' in payam_gdf.columns:
                                        result_df['ADM1_EN'] = payam_gdf['ADM1_EN'].astype(str)
                                    else:
                                        result_df['ADM1_EN'] = result_df['ADM1_PCODE']
                                    
                                    if 'ADM2_EN' in payam_gdf.columns:
                                        result_df['ADM2_EN'] = payam_gdf['ADM2_EN'].astype(str)
                                    else:
                                        result_df['ADM2_EN'] = result_df['ADM2_PCODE']
                                    
                                except Exception as e:
                                    st.error(f"Error during zonal statistics: {str(e)}")
                                    import traceback
                                    st.error(traceback.format_exc())
                                    result_df = pd.DataFrame(columns=['ADM3_PCODE', 'ADM1_PCODE', 'ADM2_PCODE', 'ADM3_EN', 
                                                                      'ADM0_PCODE', 'ADM1_EN', 'ADM2_EN', 'pop_count', 'pop_count_millions'])
                                
                                # Save for future use (only if extraction succeeded)
                                if 'result_df' in locals() and not result_df.empty and 'pop_count' in payam_gdf.columns:
                                    output_file = PROCESSED_PATH / "admin3_payams_with_population.geojson"
                                    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
                                    payam_gdf.to_file(output_file, driver='GeoJSON')
                                    st.success(f"✅ Extracted population for {len(result_df)} sub-prefectures (total: {result_df['pop_count'].sum():,.0f}) and saved to {output_file}")
                                    
                                    time.sleep(1)
                                    progress_bar.empty()
                                    status_text.empty()
                                else:
                                    st.error("Population extraction failed. Check errors above.")
                                    if 'result_df' not in locals():
                                        result_df = pd.DataFrame(columns=['ADM3_PCODE', 'ADM1_PCODE', 'ADM2_PCODE', 'ADM3_EN', 
                                                                          'ADM0_PCODE', 'ADM1_EN', 'ADM2_EN', 'pop_count', 'pop_count_millions'])
                        else:
                            st.error(f"Population raster not found at {raster_path}")
                            result_df = pd.DataFrame(columns=['ADM3_PCODE', 'ADM1_PCODE', 'ADM2_PCODE', 'ADM3_EN', 
                                                              'ADM0_PCODE', 'ADM1_EN', 'ADM2_EN', 'pop_count', 'pop_count_millions'])
                    except ImportError:
                        st.error("rasterstats not installed. Please run: pip install rasterstats")
                        result_df = pd.DataFrame(columns=['ADM3_PCODE', 'ADM1_PCODE', 'ADM2_PCODE', 'ADM3_EN', 
                                                          'ADM0_PCODE', 'ADM1_EN', 'ADM2_EN', 'pop_count', 'pop_count_millions'])
                    except Exception as e:
                        st.error(f"Error extracting population from raster: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
                        result_df = pd.DataFrame(columns=['ADM3_PCODE', 'ADM1_PCODE', 'ADM2_PCODE', 'ADM3_EN', 
                                                          'ADM0_PCODE', 'ADM1_EN', 'ADM2_EN', 'pop_count', 'pop_count_millions'])
        
        # Ensure result_df is defined (should always be, but safety check)
        if 'result_df' not in locals():
            st.error("⚠️ Internal error: result_df was not created. This should not happen.")
            result_df = pd.DataFrame(columns=['ADM3_PCODE', 'ADM1_PCODE', 'ADM2_PCODE', 'ADM3_EN', 
                                              'ADM0_PCODE', 'ADM1_EN', 'ADM2_EN', 'pop_count', 'pop_count_millions'])
        
        # Cache the result (only if we have data)
        if not result_df.empty:
            save_to_cache(cache_key, result_df)
            st.info(f"📊 Population data loaded: {len(result_df)} sub-prefectures")
        else:
            st.warning("⚠️ No population data was loaded. Check errors above.")
        
        log_performance("load_population_data", time.time() - start_time)
        return result_df
        
    except Exception as e:
        st.error(f"Error loading population data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def create_admin_levels(pop_data):
    """Create admin level aggregations from population data - optimized"""
    start_time = time.time()
    
    if pop_data.empty:
        return {'admin1': pd.DataFrame(), 'admin2': pd.DataFrame(), 'admin3': pop_data}
    
    # Use vectorized operations for better performance
    admin2_agg = pop_data.groupby(['ADM2_PCODE', 'ADM2_EN', 'ADM1_PCODE', 'ADM1_EN', 'ADM0_PCODE'], 
                                   as_index=False).agg({
        'pop_count': 'sum',
        'pop_count_millions': 'sum'
    })
    
    admin1_agg = pop_data.groupby(['ADM1_PCODE', 'ADM1_EN', 'ADM0_PCODE'], as_index=False).agg({
        'pop_count': 'sum',
        'pop_count_millions': 'sum'
    })
    
    log_performance("create_admin_levels", time.time() - start_time)
    
    return {
        'admin3': pop_data,
        'admin2': admin2_agg,
        'admin1': admin1_agg
    }

@st.cache_data(ttl=3600, show_spinner=False)
def load_conflict_data():
    """Load and cache conflict data with optimized processing"""
    start_time = time.time()
    
    # Check cache first
    cache_key = get_cache_key("djibouti_conflict_data", "v2")
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        log_performance("load_conflict_data", time.time() - start_time)
        return cached_data
    
    try:
        if not ACLED_DATA.exists():
            st.error(f"Conflict data not found: {ACLED_DATA}")
            return pd.DataFrame()
        
        # Load ACLED data directly
        djibouti_acled = pd.read_csv(ACLED_DATA)
        
        if djibouti_acled.empty:
            return pd.DataFrame()
        
        # Convert event_date to datetime and extract month/year
        djibouti_acled['event_date'] = pd.to_datetime(djibouti_acled['event_date'])
        djibouti_acled['month'] = djibouti_acled['event_date'].dt.month
        djibouti_acled['year'] = djibouti_acled['event_date'].dt.year
        
        # Process the data to match our format
        brd_events = djibouti_acled[
            (~djibouti_acled['event_type'].isin(['Protests', 'Riots'])) &
            (djibouti_acled['fatalities'] > 0)
        ].copy()
        
        # Categorize violence
        def categorize_violence(interaction):
            if pd.isna(interaction):
                return 'unknown', 'unknown'
            interaction_lower = str(interaction).lower()
            if 'state forces' in interaction_lower:
                return 'state', 'state'
            else:
                return 'nonstate', 'nonstate'
        
        brd_events[['violence_type', 'actor_type']] = brd_events['interaction'].apply(
            lambda x: pd.Series(categorize_violence(x))
        )
        
        # Load preprocessed ward-level conflict data
        ward_conflict_file = PROCESSED_PATH / "ward_conflict_data.csv"
        if ward_conflict_file.exists():
            conflict_processed = pd.read_csv(ward_conflict_file)
            
            conflict_processed = conflict_processed.rename(columns={
                'wardcode': 'ADM3_PCODE',
                'wardname': 'ADM3_EN',
                'countyname': 'ADM2_EN',
                'statename': 'ADM1_EN'
            })
            
            conflict_processed['ADM1_PCODE'] = conflict_processed['ADM1_EN'].astype(str)
            conflict_processed['ADM2_PCODE'] = conflict_processed['ADM2_EN'].astype(str)
        else:
            # Fallback: Perform spatial join using lat/lon to match events to sub-prefectures
            st.info("⚠️ Preprocessed conflict file not found. Performing spatial matching using lat/lon...")
            
            # Load admin3 boundaries for spatial join
            boundaries = load_admin_boundaries()
            if boundaries and 3 in boundaries and not boundaries[3].empty:
                admin3_gdf = boundaries[3].copy()
                
                # Filter events with valid coordinates
                brd_events_geo = brd_events.dropna(subset=['latitude', 'longitude']).copy()
                
                if len(brd_events_geo) > 0:
                    # Create GeoDataFrame from events
                    events_gdf = gpd.GeoDataFrame(
                        brd_events_geo,
                        geometry=gpd.points_from_xy(brd_events_geo.longitude, brd_events_geo.latitude),
                        crs="EPSG:4326"
                    )
                    
                    # Ensure CRS match
                    if admin3_gdf.crs != events_gdf.crs:
                        admin3_gdf = admin3_gdf.to_crs(events_gdf.crs)
                    
                    # Perform spatial join
                    events_with_payam = gpd.sjoin(
                        events_gdf,
                        admin3_gdf[['ADM3_PCODE', 'ADM3_EN', 'ADM2_PCODE', 'ADM2_EN', 'ADM1_PCODE', 'ADM1_EN', 'geometry']],
                        how='left',
                        predicate='within'
                    )
                    
                    # Keep only matched events
                    events_matched = events_with_payam[events_with_payam['ADM3_PCODE'].notna()].copy()
                    
                    if len(events_matched) > 0:
                        # Aggregate by sub-prefecture, year, month, and violence type
                        aggregated = events_matched.groupby(
                            ['ADM3_PCODE', 'ADM3_EN', 'ADM2_PCODE', 'ADM2_EN', 'ADM1_PCODE', 'ADM1_EN', 'year', 'month', 'violence_type'],
                            as_index=False
                        ).agg({'fatalities': 'sum'})
                        
                        # Pivot to create state and nonstate columns
                        conflict_pivot = aggregated.pivot_table(
                            index=['ADM3_PCODE', 'ADM3_EN', 'ADM2_PCODE', 'ADM2_EN', 'ADM1_PCODE', 'ADM1_EN', 'year', 'month'],
                            columns='violence_type',
                            values='fatalities',
                            fill_value=0
                        ).reset_index()
                        
                        conflict_pivot.columns.name = None
                        
                        # Rename violence type columns
                        if 'state' in conflict_pivot.columns:
                            conflict_pivot = conflict_pivot.rename(columns={'state': 'ACLED_BRD_state'})
                        else:
                            conflict_pivot['ACLED_BRD_state'] = 0
                        
                        if 'nonstate' in conflict_pivot.columns:
                            conflict_pivot = conflict_pivot.rename(columns={'nonstate': 'ACLED_BRD_nonstate'})
                        else:
                            conflict_pivot['ACLED_BRD_nonstate'] = 0
                        
                        conflict_processed = conflict_pivot.copy()
                        conflict_processed['ACLED_BRD_total'] = conflict_processed['ACLED_BRD_state'] + conflict_processed['ACLED_BRD_nonstate']
                        
                        st.success(f"✅ Spatially matched {len(events_matched):,} events to sub-prefectures")
                    else:
                        st.warning("⚠️ No events could be matched to sub-prefectures. Falling back to region-level aggregation.")
                        # Fall through to county-level aggregation below
                        conflict_processed = None
                else:
                    st.warning("⚠️ No events with valid coordinates. Falling back to county-level aggregation.")
                    conflict_processed = None
            else:
                st.warning("⚠️ Admin3 boundaries not available. Falling back to county-level aggregation.")
                conflict_processed = None
            
            # Fallback to county-level aggregation if spatial join failed
            if conflict_processed is None:
                conflict_processed = brd_events.groupby(['year', 'month', 'admin1', 'admin2', 'violence_type'], 
                                                       as_index=False).agg({'fatalities': 'sum'})
                
                conflict_pivot = conflict_processed.pivot_table(
                    index=['year', 'month', 'admin1', 'admin2'],
                    columns='violence_type',
                    values='fatalities',
                    fill_value=0
                ).reset_index()
                
                conflict_pivot['ADM1_PCODE'] = conflict_pivot['admin1'].astype(str)
                conflict_pivot['ADM2_PCODE'] = conflict_pivot['admin2'].astype(str)
                
                conflict_processed = conflict_pivot.rename(columns={
                    'admin1': 'ADM1_EN',
                    'admin2': 'ADM2_EN'
                })
                conflict_processed['ADM3_PCODE'] = ''
                conflict_processed['ADM3_EN'] = ''
        
        # Ensure required columns exist
        if 'ACLED_BRD_total' not in conflict_processed.columns:
            # Handle case where we need to create ACLED_BRD columns from violence_type columns
            if 'state' in conflict_processed.columns:
                conflict_processed['ACLED_BRD_state'] = conflict_processed['state']
            elif 'ACLED_BRD_state' not in conflict_processed.columns:
                conflict_processed['ACLED_BRD_state'] = 0
            
            if 'nonstate' in conflict_processed.columns:
                conflict_processed['ACLED_BRD_nonstate'] = conflict_processed['nonstate']
            elif 'ACLED_BRD_nonstate' not in conflict_processed.columns:
                conflict_processed['ACLED_BRD_nonstate'] = 0
            
            if 'ACLED_BRD_total' not in conflict_processed.columns:
                conflict_processed['ACLED_BRD_total'] = conflict_processed['ACLED_BRD_state'] + conflict_processed['ACLED_BRD_nonstate']
            
            # Drop old violence_type columns if they exist
            conflict_processed = conflict_processed.drop(columns=['state', 'nonstate'], errors='ignore')
        
        # Remove rows with zero total BRD
        conflict_processed = conflict_processed[conflict_processed['ACLED_BRD_total'] > 0]
        
        # Cache the result
        save_to_cache(cache_key, conflict_processed)
        
        log_performance("load_conflict_data", time.time() - start_time)
        return conflict_processed
        
    except Exception as e:
        st.error(f"Error loading conflict data: {str(e)}")
        return pd.DataFrame()

def standardize_admin_columns(gdf, level):
    """Standardize column names for administrative boundaries to COD-AB format
    
    Args:
        gdf: GeoDataFrame with administrative boundaries
        level: Admin level (1, 2, or 3)
    
    Returns:
        GeoDataFrame with standardized column names
    """
    if gdf.empty:
        return gdf
    
    gdf = gdf.copy()
    column_mapping = {}
    
    # Debug: show available columns (commented out - st.debug doesn't exist in Streamlit)
    # Columns available: {list(gdf.columns)}
    
    # Map columns for the specified level and parent levels
    # Handle various naming conventions: ADM1_PCODE, ADM1_Pcode, adm1_pcode, etc.
    for col in gdf.columns:
        col_upper = col.upper()
        col_lower = col.lower()
        
        # Admin level specific mappings - be more flexible
        if level >= 1:
            # PCODE variations
            if (col_upper == 'ADM1_PCODE' or 
                col_upper.endswith('ADM1_PCODE') or 
                ('ADM1' in col_upper and 'PCODE' in col_upper) or
                (col_upper.endswith('_ADM1') and 'CODE' in col_upper) or
                col_upper == 'ADM1_CODE'):
                column_mapping[col] = 'ADM1_PCODE'
            # Name variations
            elif (col_upper == 'ADM1_EN' or 
                  col_upper == 'ADM1_NAME' or 
                  ('ADM1' in col_upper and ('EN' in col_upper or 'NAME' in col_upper)) or
                  col_upper.endswith('_ADM1_NAME') or
                  col_upper.endswith('_ADM1_EN')):
                column_mapping[col] = 'ADM1_EN'
        
        if level >= 2:
            # PCODE variations
            if (col_upper == 'ADM2_PCODE' or 
                col_upper.endswith('ADM2_PCODE') or 
                ('ADM2' in col_upper and 'PCODE' in col_upper) or
                (col_upper.endswith('_ADM2') and 'CODE' in col_upper) or
                col_upper == 'ADM2_CODE'):
                column_mapping[col] = 'ADM2_PCODE'
            # Name variations
            elif (col_upper == 'ADM2_EN' or 
                  col_upper == 'ADM2_NAME' or 
                  ('ADM2' in col_upper and ('EN' in col_upper or 'NAME' in col_upper)) or
                  col_upper.endswith('_ADM2_NAME') or
                  col_upper.endswith('_ADM2_EN')):
                column_mapping[col] = 'ADM2_EN'
        
        if level >= 3:
            # PCODE variations
            if (col_upper == 'ADM3_PCODE' or 
                col_upper.endswith('ADM3_PCODE') or 
                ('ADM3' in col_upper and 'PCODE' in col_upper) or
                (col_upper.endswith('_ADM3') and 'CODE' in col_upper) or
                col_upper == 'ADM3_CODE'):
                column_mapping[col] = 'ADM3_PCODE'
            # Name variations
            elif (col_upper == 'ADM3_EN' or 
                  col_upper == 'ADM3_NAME' or 
                  ('ADM3' in col_upper and ('EN' in col_upper or 'NAME' in col_upper)) or
                  col_upper.endswith('_ADM3_NAME') or
                  col_upper.endswith('_ADM3_EN')):
                column_mapping[col] = 'ADM3_EN'
    
    # Apply column mapping
    if column_mapping:
        gdf = gdf.rename(columns=column_mapping)
    
    # Ensure required columns exist for the level
    if level >= 1:
        if 'ADM1_PCODE' not in gdf.columns:
            pcode_cols = [c for c in gdf.columns if 'ADM1' in c.upper() and ('PCODE' in c.upper() or 'CODE' in c.upper())]
            if pcode_cols:
                gdf['ADM1_PCODE'] = gdf[pcode_cols[0]].astype(str)
            else:
                gdf['ADM1_PCODE'] = gdf.index.astype(str)
        
        if 'ADM1_EN' not in gdf.columns:
            name_cols = [c for c in gdf.columns if 'ADM1' in c.upper() and ('NAME' in c.upper() or 'EN' in c.upper())]
            if name_cols:
                gdf['ADM1_EN'] = gdf[name_cols[0]].astype(str)
            else:
                gdf['ADM1_EN'] = gdf['ADM1_PCODE']
    
    if level >= 2:
        if 'ADM2_PCODE' not in gdf.columns:
            pcode_cols = [c for c in gdf.columns if 'ADM2' in c.upper() and ('PCODE' in c.upper() or 'CODE' in c.upper())]
            if pcode_cols:
                gdf['ADM2_PCODE'] = gdf[pcode_cols[0]].astype(str)
            elif 'ADM1_PCODE' in gdf.columns:
                gdf['ADM2_PCODE'] = gdf['ADM1_PCODE']  # Fallback
            else:
                gdf['ADM2_PCODE'] = gdf.index.astype(str)
        
        if 'ADM2_EN' not in gdf.columns:
            name_cols = [c for c in gdf.columns if 'ADM2' in c.upper() and ('NAME' in c.upper() or 'EN' in c.upper())]
            if name_cols:
                gdf['ADM2_EN'] = gdf[name_cols[0]].astype(str)
            elif 'ADM2_PCODE' in gdf.columns:
                gdf['ADM2_EN'] = gdf['ADM2_PCODE']
            else:
                gdf['ADM2_EN'] = gdf.index.astype(str)
    
    if level >= 3:
        if 'ADM3_PCODE' not in gdf.columns:
            pcode_cols = [c for c in gdf.columns if 'ADM3' in c.upper() and ('PCODE' in c.upper() or 'CODE' in c.upper())]
            if pcode_cols:
                gdf['ADM3_PCODE'] = gdf[pcode_cols[0]].astype(str)
            else:
                gdf['ADM3_PCODE'] = gdf.index.astype(str)
        
        if 'ADM3_EN' not in gdf.columns:
            name_cols = [c for c in gdf.columns if 'ADM3' in c.upper() and ('NAME' in c.upper() or 'EN' in c.upper())]
            if name_cols:
                gdf['ADM3_EN'] = gdf[name_cols[0]].astype(str)
            elif 'ADM3_PCODE' in gdf.columns:
                gdf['ADM3_EN'] = gdf['ADM3_PCODE']
            else:
                gdf['ADM3_EN'] = gdf.index.astype(str)
    
    return gdf

@st.cache_data(ttl=3600, show_spinner=False)
def load_admin_boundaries():
    """Load administrative boundaries from GADM shapefiles
    
    Primary source: gadm41_DJI_shp.zip (local file with admin0-2)
    Note: GADM for Djibouti only includes admin0, admin1, and admin2
    Admin2 will be used as admin3 for compatibility
    
    Administrative levels:
    - Admin Level 0: Country (Djibouti)
    - Admin Level 1: States/Regions
    - Admin Level 2: Sub-prefectures/Districts (used as admin3)
    """
    start_time = time.time()
    
    # Check cache first
    cache_key = get_cache_key("djibouti_admin_boundaries", "v7_gadm_local")
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        log_performance("load_admin_boundaries", time.time() - start_time)
        return cached_data
    
    boundaries = {}
    temp_dir = None
    
    try:
        # First, try to load from local GADM shapefile zip
        gadm_zip = Path("gadm41_DJI_shp.zip")
        if gadm_zip.exists():
            st.info("Loading administrative boundaries from GADM shapefiles...")
            
            # Extract GADM from zip
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(gadm_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # GADM structure: gadm41_DJI_0.shp (country), gadm41_DJI_1.shp (admin1), gadm41_DJI_2.shp (admin2)
            # Note: GADM for Djibouti only has admin0, admin1, admin2 - no admin3
            
            # Load admin1 (states/regions)
            admin1_shp = Path(temp_dir) / "gadm41_DJI_1.shp"
            if admin1_shp.exists():
                try:
                    admin1_gdf = gpd.read_file(str(admin1_shp))
                    admin1_gdf = admin1_gdf.to_crs('EPSG:4326')
                    # Map GADM columns to standard format
                    if 'GID_1' in admin1_gdf.columns:
                        admin1_gdf['ADM1_PCODE'] = admin1_gdf['GID_1']
                    if 'NAME_1' in admin1_gdf.columns:
                        admin1_gdf['ADM1_EN'] = admin1_gdf['NAME_1']
                    boundaries[1] = standardize_admin_columns(admin1_gdf, level=1)
                    st.success(f"✓ Loaded {len(boundaries[1])} admin1 units")
                except Exception as e:
                    st.warning(f"Could not load admin1: {e}")
                    import traceback
                    st.error(traceback.format_exc())
            
            # Load admin2 (counties/districts)
            admin2_shp = Path(temp_dir) / "gadm41_DJI_2.shp"
            if admin2_shp.exists():
                try:
                    admin2_gdf = gpd.read_file(str(admin2_shp))
                    admin2_gdf = admin2_gdf.to_crs('EPSG:4326')
                    # Map GADM columns to standard format
                    if 'GID_2' in admin2_gdf.columns:
                        admin2_gdf['ADM2_PCODE'] = admin2_gdf['GID_2']
                    if 'NAME_2' in admin2_gdf.columns:
                        admin2_gdf['ADM2_EN'] = admin2_gdf['NAME_2']
                    if 'GID_1' in admin2_gdf.columns:
                        admin2_gdf['ADM1_PCODE'] = admin2_gdf['GID_1']
                    if 'NAME_1' in admin2_gdf.columns:
                        admin2_gdf['ADM1_EN'] = admin2_gdf['NAME_1']
                    boundaries[2] = standardize_admin_columns(admin2_gdf, level=2)
                    st.success(f"✓ Loaded {len(boundaries[2])} admin2 units")
                except Exception as e:
                    st.warning(f"Could not load admin2: {e}")
                    import traceback
                    st.error(traceback.format_exc())
            
            # GADM doesn't have admin3, so create it from admin2 for compatibility
            if 2 in boundaries and not boundaries[2].empty:
                boundaries[3] = boundaries[2].copy()
                # Ensure admin3 columns exist
                if 'ADM2_PCODE' in boundaries[3].columns:
                    boundaries[3]['ADM3_PCODE'] = boundaries[3]['ADM2_PCODE']
                if 'ADM2_EN' in boundaries[3].columns:
                    boundaries[3]['ADM3_EN'] = boundaries[3]['ADM2_EN']
                st.info(f"ℹ Note: GADM doesn't include admin3. Using admin2 as admin3 ({len(boundaries[3])} units)")
            
            # Build admin1 from admin2 if admin1 is missing
            if (1 not in boundaries or boundaries[1].empty) and (2 in boundaries and not boundaries[2].empty):
                boundaries[1] = boundaries[2].dissolve(by=['ADM1_PCODE', 'ADM1_EN'], aggfunc='first').reset_index()
                st.info(f"✓ Created admin1 from admin2 ({len(boundaries[1])} units)")
            
            # If we successfully loaded boundaries, cache and return
            if (boundaries.get(1) is not None and not boundaries[1].empty) and \
               (boundaries.get(2) is not None and not boundaries[2].empty) and \
               (boundaries.get(3) is not None and not boundaries[3].empty):
                save_to_cache(cache_key, boundaries)
                log_performance("load_admin_boundaries", time.time() - start_time)
                return boundaries
        
        # Fallback to FeatureServer if GADM not available or failed
        st.info("Loading admin2 (county) boundaries from ArcGIS FeatureServer...")
        admin2_url = "https://services.arcgis.com/nzS0F0zdNLvs7nc8/arcgis/rest/services/djibouti_admin2_boundaries_from_HDX/FeatureServer/0/query"
        params = {
            'where': '1=1',
            'outFields': '*',
            'f': 'geojson',
            'outSR': '4326'
        }
        
        response = requests.get(admin2_url, params=params, timeout=60)
        response.raise_for_status()
        
        # Load as GeoDataFrame
        admin2_gdf = gpd.read_file(response.text, driver='GeoJSON')
        admin2_gdf = admin2_gdf.to_crs('EPSG:4326')
        
        # Standardize column names - COD-AB standard uses ADM2_PCODE, ADM2_EN, ADM1_PCODE, ADM1_EN
        # Check for exact matches first, then pattern match
        column_mapping = {}
        for col in admin2_gdf.columns:
            col_upper = col.upper()
            # Exact matches first
            if col_upper == 'ADM2_PCODE':
                column_mapping[col] = 'ADM2_PCODE'
            elif col_upper == 'ADM2_EN' or col_upper == 'ADM2_NAME':
                column_mapping[col] = 'ADM2_EN'
            elif col_upper == 'ADM1_PCODE':
                column_mapping[col] = 'ADM1_PCODE'
            elif col_upper == 'ADM1_EN' or col_upper == 'ADM1_NAME':
                column_mapping[col] = 'ADM1_EN'
            # Pattern matches (more specific)
            elif 'ADM2' in col_upper and 'PCODE' in col_upper:
                column_mapping[col] = 'ADM2_PCODE'
            elif 'ADM2' in col_upper and ('EN' in col_upper or 'NAME' in col_upper):
                column_mapping[col] = 'ADM2_EN'
            elif 'ADM1' in col_upper and 'PCODE' in col_upper:
                column_mapping[col] = 'ADM1_PCODE'
            elif 'ADM1' in col_upper and ('EN' in col_upper or 'NAME' in col_upper):
                column_mapping[col] = 'ADM1_EN'
        
        # Rename columns
        if column_mapping:
            admin2_gdf = admin2_gdf.rename(columns=column_mapping)
        
        # Ensure required columns exist - create if missing
        if 'ADM2_PCODE' not in admin2_gdf.columns:
            # Try to find any PCODE column (prefer ADM2, but take any)
            pcode_cols = [c for c in admin2_gdf.columns if 'PCODE' in c.upper() and 'ADM2' in c.upper()]
            if not pcode_cols:
                pcode_cols = [c for c in admin2_gdf.columns if 'PCODE' in c.upper() or ('CODE' in c.upper() and 'ADM2' in c.upper())]
            if pcode_cols:
                admin2_gdf['ADM2_PCODE'] = admin2_gdf[pcode_cols[0]].astype(str)
            else:
                admin2_gdf['ADM2_PCODE'] = admin2_gdf.index.astype(str)
        
        if 'ADM2_EN' not in admin2_gdf.columns:
            name_cols = [c for c in admin2_gdf.columns if ('ADM2' in c.upper() and ('NAME' in c.upper() or 'EN' in c.upper()))]
            if not name_cols:
                name_cols = [c for c in admin2_gdf.columns if 'NAME' in c.upper() or 'EN' in c.upper()]
            if name_cols:
                admin2_gdf['ADM2_EN'] = admin2_gdf[name_cols[0]].astype(str)
            else:
                admin2_gdf['ADM2_EN'] = admin2_gdf['ADM2_PCODE']
        
        if 'ADM1_PCODE' not in admin2_gdf.columns:
            # Try to find ADM1/state code column
            adm1_cols = [c for c in admin2_gdf.columns if 'ADM1' in c.upper() and 'PCODE' in c.upper()]
            if not adm1_cols:
                adm1_cols = [c for c in admin2_gdf.columns if 'ADM1' in c.upper() or ('STATE' in c.upper() and 'CODE' in c.upper())]
            if adm1_cols:
                admin2_gdf['ADM1_PCODE'] = admin2_gdf[adm1_cols[0]].astype(str)
            else:
                # Extract from ADM2_PCODE if it contains state code (first 3 chars often)
                if 'ADM2_PCODE' in admin2_gdf.columns:
                    admin2_gdf['ADM1_PCODE'] = admin2_gdf['ADM2_PCODE'].astype(str).str[:3]
                else:
                    admin2_gdf['ADM1_PCODE'] = admin2_gdf.index.astype(str)
        
        if 'ADM1_EN' not in admin2_gdf.columns:
            adm1_name_cols = [c for c in admin2_gdf.columns if 'ADM1' in c.upper() and ('NAME' in c.upper() or 'EN' in c.upper())]
            if not adm1_name_cols:
                adm1_name_cols = [c for c in admin2_gdf.columns if 'STATENAME' in c.upper() or ('STATE' in c.upper() and 'NAME' in c.upper())]
            if adm1_name_cols:
                admin2_gdf['ADM1_EN'] = admin2_gdf[adm1_name_cols[0]].astype(str)
            else:
                admin2_gdf['ADM1_EN'] = admin2_gdf['ADM1_PCODE']
        
        boundaries[2] = admin2_gdf
        
        # Build State boundaries by dissolving admin2 (10 states according to COD-AB metadata)
        state_gdf = admin2_gdf.dissolve(by=['ADM1_PCODE', 'ADM1_EN'], aggfunc='first')
        state_gdf = state_gdf.reset_index()
        boundaries[1] = state_gdf
        
        # Try to load admin3 (payam) boundaries from FeatureServer
        # 512 payams according to COD-AB metadata
        # Try different layer numbers (common patterns: layer 1 for admin1, layer 2 for admin3, etc.)
        admin3_gdf = None
        st.info("Checking for admin3 (sub-prefecture) boundaries in FeatureServer...")
        for layer_id in [1, 2, 3, 4, 5]:
            try:
                admin3_url = f"https://services.arcgis.com/nzS0F0zdNLvs7nc8/arcgis/rest/services/djibouti_admin2_boundaries_from_HDX/FeatureServer/{layer_id}/query"
                response = requests.get(admin3_url, params=params, timeout=30)
                if response.status_code == 200:
                    temp_gdf = gpd.read_file(response.text, driver='GeoJSON')
                    if not temp_gdf.empty:
                        # Check if this looks like admin3 data (has ADM3 columns or many features ~512)
                        col_str = str(temp_gdf.columns).upper()
                        if 'ADM3' in col_str or (len(temp_gdf) > 400 and len(temp_gdf) < 600):
                            admin3_gdf = temp_gdf.to_crs('EPSG:4326')
                            break
            except Exception:
                continue
        
        # If admin3 not found in FeatureServer, try local shapefile
        if admin3_gdf is None or admin3_gdf.empty:
            ward_file = DATA_PATH / "wards" / "wards.shp"
            if ward_file.exists():
                st.info("Loading admin3 (sub-prefecture) boundaries from local shapefile...")
                admin3_gdf = gpd.read_file(ward_file)
                admin3_gdf = admin3_gdf.to_crs('EPSG:4326')
                
                admin3_gdf = admin3_gdf.rename(columns={
                    'ward_cd': 'ADM3_PCODE',
                    'stat_cd': 'ADM1_PCODE', 
                    'lga_cod': 'ADM2_PCODE',
                    'wrd_nm_x': 'ADM3_EN'
                })
            else:
                admin3_gdf = gpd.GeoDataFrame()
        
        # Standardize admin3 column names if loaded from FeatureServer
        if admin3_gdf is not None and not admin3_gdf.empty and 'ADM3_PCODE' not in admin3_gdf.columns:
            column_mapping = {}
            for col in admin3_gdf.columns:
                col_upper = col.upper()
                if col_upper == 'ADM3_PCODE' or ('ADM3' in col_upper and 'PCODE' in col_upper):
                    column_mapping[col] = 'ADM3_PCODE'
                elif col_upper == 'ADM3_EN' or col_upper == 'ADM3_NAME' or ('ADM3' in col_upper and ('EN' in col_upper or 'NAME' in col_upper)):
                    column_mapping[col] = 'ADM3_EN'
                elif col_upper == 'ADM2_PCODE' or ('ADM2' in col_upper and 'PCODE' in col_upper):
                    column_mapping[col] = 'ADM2_PCODE'
                elif col_upper == 'ADM2_EN' or col_upper == 'ADM2_NAME' or ('ADM2' in col_upper and ('EN' in col_upper or 'NAME' in col_upper)):
                    column_mapping[col] = 'ADM2_EN'
                elif col_upper == 'ADM1_PCODE' or ('ADM1' in col_upper and 'PCODE' in col_upper):
                    column_mapping[col] = 'ADM1_PCODE'
                elif col_upper == 'ADM1_EN' or col_upper == 'ADM1_NAME' or ('ADM1' in col_upper and ('EN' in col_upper or 'NAME' in col_upper)):
                    column_mapping[col] = 'ADM1_EN'
            
            if column_mapping:
                admin3_gdf = admin3_gdf.rename(columns=column_mapping)
            
            # Ensure required columns exist
            if 'ADM3_PCODE' not in admin3_gdf.columns:
                pcode_cols = [c for c in admin3_gdf.columns if 'PCODE' in c.upper() and 'ADM3' in c.upper()]
                if pcode_cols:
                    admin3_gdf['ADM3_PCODE'] = admin3_gdf[pcode_cols[0]].astype(str)
                else:
                    admin3_gdf['ADM3_PCODE'] = admin3_gdf.index.astype(str)
            
            if 'ADM3_EN' not in admin3_gdf.columns:
                name_cols = [c for c in admin3_gdf.columns if 'ADM3' in c.upper() and ('NAME' in c.upper() or 'EN' in c.upper())]
                if name_cols:
                    admin3_gdf['ADM3_EN'] = admin3_gdf[name_cols[0]].astype(str)
                elif 'ADM3_PCODE' in admin3_gdf.columns:
                    admin3_gdf['ADM3_EN'] = admin3_gdf['ADM3_PCODE']
                else:
                    admin3_gdf['ADM3_EN'] = admin3_gdf.index.astype(str)
        
        # Map state and county names from admin2 boundaries to admin3
        if admin3_gdf is not None and not admin3_gdf.empty and not admin2_gdf.empty:
            if 'ADM1_EN' not in admin3_gdf.columns or (admin3_gdf['ADM1_EN'].isna().any() if 'ADM1_EN' in admin3_gdf.columns else True):
                if 'ADM1_PCODE' in admin3_gdf.columns:
                    state_mapping = dict(zip(admin2_gdf['ADM1_PCODE'], admin2_gdf['ADM1_EN']))
                    admin3_gdf['ADM1_EN'] = admin3_gdf['ADM1_PCODE'].map(state_mapping).fillna(admin3_gdf['ADM1_PCODE'])
            
            if 'ADM2_EN' not in admin3_gdf.columns or (admin3_gdf['ADM2_EN'].isna().any() if 'ADM2_EN' in admin3_gdf.columns else True):
                if 'ADM2_PCODE' in admin3_gdf.columns:
                    county_mapping = dict(zip(admin2_gdf['ADM2_PCODE'], admin2_gdf['ADM2_EN']))
                    admin3_gdf['ADM2_EN'] = admin3_gdf['ADM2_PCODE'].map(county_mapping).fillna(admin3_gdf['ADM2_PCODE'])
        
        boundaries[3] = admin3_gdf if admin3_gdf is not None else gpd.GeoDataFrame()
            
    except Exception as e:
        st.error(f"Error loading administrative boundaries: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        boundaries = {1: gpd.GeoDataFrame(), 2: gpd.GeoDataFrame(), 3: gpd.GeoDataFrame()}
    finally:
        # Clean up temporary directory if created
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass  # Ignore cleanup errors
    
    # Cache the results
    save_to_cache(cache_key, boundaries)
    log_performance("load_admin_boundaries", time.time() - start_time)
    return boundaries

@st.cache_data(ttl=3600, show_spinner=False)
def load_neighboring_country_events(period_info, country='somalia', border_distance_km=200):
    """Load ACLED events from neighboring countries (Somalia/Ethiopia) near Djibouti borders
    
    Args:
        period_info: Period information dict with start/end year/month
        country: 'somalia' or 'ethiopia'
        border_distance_km: Maximum distance from Djibouti border in km (default 200km)
    
    Returns:
        GeoDataFrame with events as points, filtered by period and proximity to Djibouti
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        
        # Load ACLED data
        if country.lower() == 'somalia':
            acled_file = Path("acled_Somalia.csv")
        elif country.lower() == 'ethiopia':
            acled_file = Path("acled_Ethiopia.csv")
        else:
            return gpd.GeoDataFrame()
        
        if not acled_file.exists():
            return gpd.GeoDataFrame()
        
        # Load and filter ACLED data
        acled_df = pd.read_csv(acled_file)
        
        # Filter for events with fatalities (BRD events)
        brd_events = acled_df[
            (~acled_df['event_type'].isin(['Protests', 'Riots'])) &
            (acled_df['fatalities'] > 0)
        ].copy()
        
        # Convert event_date to datetime and extract year/month
        brd_events['event_date'] = pd.to_datetime(brd_events['event_date'])
        brd_events['month'] = brd_events['event_date'].dt.month
        brd_events['year'] = brd_events['event_date'].dt.year
        
        # Filter by period
        start_year = period_info['start_year']
        end_year = period_info['end_year']
        start_month = period_info['start_month']
        end_month = period_info['end_month']
        
        if start_year == end_year:
            period_mask = (brd_events['year'] == start_year) & (brd_events['month'] >= start_month) & (brd_events['month'] <= end_month)
        else:
            period_mask = (
                ((brd_events['year'] == start_year) & (brd_events['month'] >= start_month)) |
                ((brd_events['year'] > start_year) & (brd_events['year'] < end_year)) |
                ((brd_events['year'] == end_year) & (brd_events['month'] <= end_month))
            )
        
        period_filtered = brd_events[period_mask].copy()
        
        # Filter events with valid coordinates
        events_geo = period_filtered.dropna(subset=['latitude', 'longitude']).copy()
        
        if len(events_geo) == 0:
            return gpd.GeoDataFrame()
        
        # Create GeoDataFrame
        events_gdf = gpd.GeoDataFrame(
            events_geo,
            geometry=gpd.points_from_xy(events_geo.longitude, events_geo.latitude),
            crs="EPSG:4326"
        )
        
        # Get Djibouti boundaries to filter by proximity
        boundaries = load_admin_boundaries()
        if boundaries and 1 in boundaries and not boundaries[1].empty:
            # Use actual Djibouti boundaries and buffer
            djibouti_proj = boundaries[1].to_crs('EPSG:3857')  # Web Mercator for accurate buffering
            djibouti_buffered = djibouti_proj.geometry.unary_union.buffer(border_distance_km * 1000)  # Convert km to meters
            djibouti_buffered_wgs84 = gpd.GeoSeries([djibouti_buffered], crs='EPSG:3857').to_crs('EPSG:4326').iloc[0]
            
            # Filter events within buffered boundary
            events_gdf = events_gdf[events_gdf.geometry.within(djibouti_buffered_wgs84)].copy()
        else:
            # Fallback: Use a bounding box around Djibouti
            # Djibouti approximate bounds: lat 10.9-12.7, lon 41.7-43.7
            lat_expand = border_distance_km / 111.0  # ~111 km per degree latitude
            lon_expand = border_distance_km / (111.0 * abs(np.cos(np.radians(11.8))))  # Adjust for longitude
            
            mask = (
                (events_gdf.geometry.y >= 10.9 - lat_expand) &
                (events_gdf.geometry.y <= 12.7 + lat_expand) &
                (events_gdf.geometry.x >= 41.7 - lon_expand) &
                (events_gdf.geometry.x <= 43.7 + lon_expand)
            )
            events_gdf = events_gdf[mask].copy()
        
        # Add country column
        events_gdf['neighbor_country'] = country.capitalize()
        
        return events_gdf
        
    except Exception as e:
        # Silently return empty GeoDataFrame on error (don't show warning in UI)
        return gpd.GeoDataFrame()

def filter_data_by_period_impl(data, period_info):
    """Filter data based on custom date range - optimized implementation"""
    if len(data) == 0:
        return data
    
    start_year = period_info['start_year']
    end_year = period_info['end_year']
    start_month = period_info['start_month']
    end_month = period_info['end_month']
    
    if start_year == end_year:
        mask = (data['year'] == start_year) & (data['month'] >= start_month) & (data['month'] <= end_month)
    else:
        mask = (
            ((data['year'] == start_year) & (data['month'] >= start_month)) |
            ((data['year'] > start_year) & (data['year'] < end_year)) |
            ((data['year'] == end_year) & (data['month'] <= end_month))
        )
    
    return data[mask]
def classify_and_aggregate_data(pop_data, admin_data, conflict_data, period_info, rate_thresh, abs_thresh, agg_thresh, agg_level):
    """Classify sub-prefectures (admin3) and aggregate to selected administrative level - optimized"""
    start_time = time.time()
    
    # Filter conflict data for selected period
    period_conflict = filter_data_by_period_impl(conflict_data, period_info)
    
    # Check if we have sub-prefecture-level (admin3) conflict data
    if len(period_conflict) > 0 and 'ADM3_PCODE' in period_conflict.columns and period_conflict['ADM3_PCODE'].notna().any():
        conflict_payam = period_conflict.groupby(['ADM3_PCODE'], as_index=False).agg({
            'ACLED_BRD_state': 'sum',
            'ACLED_BRD_nonstate': 'sum',
            'ACLED_BRD_total': 'sum'
        })
        
        merged = pd.merge(pop_data, conflict_payam, on='ADM3_PCODE', how='left')
        
        conflict_cols = ['ACLED_BRD_state', 'ACLED_BRD_nonstate', 'ACLED_BRD_total']
        merged[conflict_cols] = merged[conflict_cols].fillna(0)
    else:
        merged = pop_data.copy()
        merged['ACLED_BRD_state'] = 0
        merged['ACLED_BRD_nonstate'] = 0
        merged['ACLED_BRD_total'] = 0
    
    # Calculate death rates
    merged['acled_total_death_rate'] = (merged['ACLED_BRD_total'] / (merged['pop_count_millions'] * 1e6)) * 1e5
    
    # Classify sub-prefectures as violence-affected
    merged['violence_affected'] = (
        (merged['acled_total_death_rate'] > rate_thresh) & 
        (merged['ACLED_BRD_total'] > abs_thresh)
    )
    
    # Aggregate to selected level
    if agg_level == 'ADM1':
        group_cols = ['ADM1_PCODE', 'ADM1_EN']
    else:  # ADM2
        group_cols = ['ADM2_PCODE', 'ADM2_EN', 'ADM1_PCODE', 'ADM1_EN']
    
    aggregated = merged.groupby(group_cols, as_index=False).agg({
        'pop_count': 'sum',
        'violence_affected': 'sum',
        'ADM3_PCODE': 'count',
        'ACLED_BRD_total': 'sum'
    })
    
    aggregated.rename(columns={'ADM3_PCODE': 'total_payams'}, inplace=True)  # Internal column name kept for compatibility
    
    # Calculate shares
    aggregated['share_payams_affected'] = aggregated['violence_affected'] / aggregated['total_payams']
    
    # Calculate population share
    affected_pop = merged[merged['violence_affected']].groupby(group_cols[0], as_index=False)['pop_count'].sum()
    affected_pop.rename(columns={'pop_count': 'affected_population'}, inplace=True)
    aggregated = pd.merge(aggregated, affected_pop, on=group_cols[0], how='left')
    aggregated['affected_population'] = aggregated['affected_population'].fillna(0)
    aggregated['share_population_affected'] = aggregated['affected_population'] / aggregated['pop_count']
    
    # Mark units above threshold
    aggregated['above_threshold'] = aggregated['share_payams_affected'] > agg_thresh
    
    log_performance("classify_and_aggregate_data", time.time() - start_time)
    
    return aggregated, merged

# Custom CSS that can be reused across pages
def load_custom_css():
    """Load custom CSS for all pages"""
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .status-info {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 6px;
            border-left: 4px solid #28a745;
            margin: 1rem 0;
            font-family: monospace;
            font-size: 0.9rem;
        }
        .performance-info {
            background: #e3f2fd;
            padding: 0.5rem;
            border-radius: 4px;
            border-left: 3px solid #2196f3;
            margin: 0.5rem 0;
            font-size: 0.8rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0px 24px;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff;
        }
        .element-container iframe {
            width: 100% !important;
        }
        .stSpinner > div {
            border-top-color: #667eea !important;
        }
    </style>
    """, unsafe_allow_html=True)

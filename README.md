# Djibouti Violence Analysis Dashboard

Interactive dashboard for analyzing conflict data in Djibouti with spatial visualization and administrative level analysis.

## Setup

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Extract population data (recommended for faster loading):**
   ```bash
   python extract_population.py
   ```
   This will create pre-extracted GeoJSON files with population data in `data/processed/`.

4. **Run the app:**
   ```bash
   streamlit run Home.py
   ```

## Data Files Required

- `gadm41_DJI_shp.zip` - Administrative boundaries from GADM (admin0-2)
  - Note: GADM for Djibouti includes admin0 (country), admin1 (states), and admin2 (counties)
  - Admin2 is used as admin3 (payams) for compatibility with the app
- `dji_pop_2025_CN_100m_R2025A_v1.tif` - Population raster
- `acled_Djibouti.csv` - ACLED conflict event data

## Features

- **Spatial Analysis**: Interactive maps showing violence-affected payams, counties, and states
- **Payam Analysis**: Detailed time series analysis for individual payams
- **Data Export**: Download processed data and visualizations

## Administrative Levels

Djibouti uses:
- **Admin Level 1**: States/Regions
- **Admin Level 2**: Counties/Districts
- **Admin Level 3**: Payams/Sub-districts

## Notes

- First load may take time if population data needs to be extracted from raster
- Pre-extracted population files significantly improve load times
- The app automatically extracts population on-the-fly if pre-extracted files are not available


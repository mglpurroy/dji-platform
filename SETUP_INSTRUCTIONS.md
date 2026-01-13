# Setup and Data Generation Instructions

## Step 1: Set Up Python Environment

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Regenerate Population Data

The old processed files contain South Sudan data. Regenerate for Djibouti:

```bash
python extract_population.py
```

This will:
- Extract GADM shapefiles from `gadm41_DJI_shp.zip`
- Load admin1 (states) and admin2 (counties) boundaries
- Use admin2 as admin3 (payams) since GADM doesn't include admin3
- Extract population from `dji_pop_2025_CN_100m_R2025A_v1.tif`
- Create GeoJSON files in `data/processed/`:
  - `admin3_payams_with_population.geojson`
  - `admin2_counties_with_population.geojson`
  - `admin1_states_with_population.geojson`

## Step 4: Process Conflict Data

```bash
python process_conflict_data.py
```

This will:
- Load ACLED data from `acled_Djibouti.csv`
- Match events to admin boundaries using spatial joins
- Create `data/processed/ward_conflict_data.csv`

## Step 5: Run the Dashboard

```bash
streamlit run Home.py
```

The dashboard will be available at `http://localhost:8501`

## Verification

After running the scripts, verify the output:

- Check that `data/processed/` contains new GeoJSON files with Djibouti data
- Verify the conflict data CSV has been updated
- The dashboard should load Djibouti-specific data

## Troubleshooting

If you encounter issues:

1. **Missing packages**: Make sure the virtual environment is activated and all packages from `requirements.txt` are installed
2. **File not found errors**: Ensure `gadm41_DJI_shp.zip` and `dji_pop_2025_CN_100m_R2025A_v1.tif` are in the project root
3. **Population extraction takes time**: This is normal - extracting from raster can take 2-5 minutes depending on your system

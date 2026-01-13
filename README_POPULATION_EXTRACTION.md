# Population Data Extraction

This script pre-extracts population data from the raster file for each administrative level to improve performance.

## Prerequisites

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure you have:
   - `dji_pop_2025_CN_100m_R2025A_v1.tif` - Population raster file
   - `gadm41_DJI_shp.zip` - Administrative boundaries from GADM

## Running the Extraction

Run the extraction script once to create pre-processed GeoJSON files:

```bash
python extract_population.py
```

This will:
1. Extract the GADM shapefiles from the zip file
2. Load admin1 (states) and admin2 (counties) boundaries
3. Use admin2 as admin3 (payams) since GADM for Djibouti doesn't include admin3
4. Extract population statistics from the raster for each administrative unit
5. Save the results as GeoJSON files in `data/processed/`:
   - `admin3_payams_with_population.geojson` (created from admin2)
   - `admin2_counties_with_population.geojson`
   - `admin1_states_with_population.geojson`

## Output Files

The GeoJSON files contain:
- Administrative codes and names (ADM1_PCODE, ADM2_PCODE, ADM3_PCODE, etc.)
- Population count (`pop_count`)
- Population in millions (`pop_count_millions`)
- Geometry data for mapping

## Performance

- **Before**: Population extraction happens on-the-fly (slow, 30+ seconds)
- **After**: Population data is pre-loaded from GeoJSON (fast, <1 second)

The app will automatically use the pre-extracted files if they exist, otherwise it falls back to the legacy shapefile method.

## Notes

- The extraction script handles CRS reprojection automatically
- Population values are summed from all raster pixels within each administrative boundary
- The script uses `rasterstats` library for efficient zonal statistics calculation



import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import datetime
import calendar
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import utilities
from dashboard_utils import (
    init_session_state, load_custom_css,
    load_population_data, create_admin_levels, load_conflict_data,
    load_admin_boundaries, classify_and_aggregate_data, load_neighboring_country_events,
    get_latest_commit_date, load_unhcr_refugee_data
)
from mapping_functions import create_admin_map, create_payam_map
from streamlit_folium import st_folium

# Page configuration
st.set_page_config(
    page_title="Spatial Analysis - Djibouti Violence Dashboard",
    page_icon="🗺️",
    layout="wide"
)

# Initialize
init_session_state()
load_custom_css()

# Header
st.markdown("""
<div class="main-header">
    <h1>🗺️ Spatial Analysis</h1>
</div>
""", unsafe_allow_html=True)

# Load data with progress indicators
if not st.session_state.data_loaded:
    with st.spinner("Loading data... This may take a moment on first load."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Loading population data...")
            progress_bar.progress(20)
            st.session_state.pop_data = load_population_data()
            
            status_text.text("Creating administrative levels...")
            progress_bar.progress(50)
            st.session_state.admin_data = create_admin_levels(st.session_state.pop_data)
            
            status_text.text("Loading conflict data...")
            progress_bar.progress(70)
            st.session_state.conflict_data = load_conflict_data()
            
            status_text.text("Loading administrative boundaries...")
            progress_bar.progress(90)
            st.session_state.boundaries = load_admin_boundaries()
            
            progress_bar.progress(100)
            status_text.text("✅ Data loaded successfully!")
            st.session_state.data_loaded = True
            
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()

# Sidebar controls
st.sidebar.header("📊 Analysis Parameters")

# Custom date range selection
st.sidebar.subheader("📅 Time Period")

col1, col2 = st.sidebar.columns(2)

with col1:
    start_year = st.selectbox(
        "Start Year",
        options=list(range(1997, 2026)),
        index=27,  # Default to 2024
        key="start_year"
    )
    start_month = st.selectbox(
        "Start Month",
        options=list(range(1, 13)),
        format_func=lambda x: datetime.date(2020, x, 1).strftime('%B'),
        index=0,  # Default to January
        key="start_month"
    )

with col2:
    # Get latest commit date for default end date
    latest_year, latest_month = get_latest_commit_date()
    # Ensure year is within valid range
    if latest_year < 1997:
        latest_year = 2025
    if latest_year > 2025:
        latest_year = 2025
    
    # Calculate index for year (year - 1997)
    year_index = min(latest_year - 1997, 28)  # Max index is 28 (2025)
    
    # Calculate index for month (month - 1, since months are 1-12 but index is 0-11)
    month_index = min(latest_month - 1, 11)  # Max index is 11 (December)
    
    end_year = st.selectbox(
        "End Year",
        options=list(range(1997, 2026)),
        index=year_index,  # Default to latest commit year
        key="end_year"
    )
    end_month = st.selectbox(
        "End Month",
        options=list(range(1, 13)),
        format_func=lambda x: datetime.date(2020, x, 1).strftime('%B'),
        index=month_index,  # Default to latest commit month
        key="end_month"
    )

# Create period_info from custom selection
period_info = {
    'start_year': start_year,
    'start_month': start_month,
    'end_year': end_year,
    'end_month': end_month,
    'start_date': datetime.date(start_year, start_month, 1),
    'end_date': datetime.date(end_year, end_month, calendar.monthrange(end_year, end_month)[1]),
    'label': f"{datetime.date(2020, start_month, 1).strftime('%b')} {start_year} - {datetime.date(2020, end_month, 1).strftime('%b')} {end_year}"
}

# Calculate number of months
if start_year == end_year:
    period_info['months'] = end_month - start_month + 1
else:
    period_info['months'] = (end_year - start_year - 1) * 12 + (12 - start_month + 1) + end_month

# Validate date range
if (start_year > end_year) or (start_year == end_year and start_month > end_month):
    st.sidebar.error("⚠️ Start date must be before end date")
    st.stop()

# Display selected period info
st.sidebar.info(f"**Period:** {period_info['label']}\n\n**Duration:** {period_info['months']} months")

# Thresholds
st.sidebar.subheader("🎯 Violence Thresholds")

rate_thresh = st.sidebar.slider(
    "Death Rate (per 100k)",
    min_value=0.0,
    max_value=50.0,
    value=10.0,
    step=0.5,
    help="Minimum death rate per 100,000 population"
)

abs_thresh = st.sidebar.slider(
    "Absolute Deaths",
    min_value=0,
    max_value=100,
    value=5,
    step=1,
    help="Minimum number of deaths in absolute terms"
)

# Aggregation settings
st.sidebar.subheader("📍 Aggregation Settings")

agg_level = st.sidebar.radio(
    "Administrative Level",
    ["ADM1 (Region)", "ADM2 (Sub-prefecture)"],
    help="Level for administrative aggregation"
)
agg_level = agg_level.split()[0]

agg_thresh = st.sidebar.slider(
    "Share Threshold (%)",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=1.0,
    help="Minimum percentage of sub-prefectures affected to highlight administrative unit"
) / 100

map_var = st.sidebar.selectbox(
    "Map Variable",
    ["share_payams_affected", "share_population_affected"],
    format_func=lambda x: "Share of Sub-prefectures Affected" if x == "share_payams_affected" else "Share of Population Affected",
    help="Variable to display on administrative map"
)

# Sub-prefecture display options
st.sidebar.subheader("🗺️ Sub-prefecture Map Options")

show_all_payams = st.sidebar.checkbox(
    "Show All Sub-prefectures",
    value=True,
    help="If checked, shows all sub-prefectures. If unchecked, shows only violence-affected sub-prefectures (faster rendering)"
)

# Additional layers info
st.sidebar.subheader("🌍 Additional Map Layers")

st.sidebar.info("💡 **Note:** Additional layers (neighboring country events and UNHCR refugee data) are automatically loaded. Toggle visibility using the map's layer control (top-left of map).")

# Process data
with st.spinner("Processing data for selected period..."):
    pop_data = st.session_state.pop_data
    admin_data = st.session_state.admin_data
    conflict_data = st.session_state.conflict_data
    boundaries = st.session_state.boundaries
    
    # Check if we have population data
    if pop_data.empty or admin_data['admin3'].empty:
        st.warning("⚠️ No population data available. Please ensure the population data file exists and matches the boundary data.")
        st.info(f"Population data: {len(pop_data)} rows, Admin3 data: {len(admin_data.get('admin3', pd.DataFrame()))} rows")
        if boundaries and 3 in boundaries and not boundaries[3].empty:
            st.info(f"Boundaries admin3: {len(boundaries[3])} features")
            if 'ADM3_PCODE' in boundaries[3].columns:
                st.info(f"Sample boundary PCODEs: {boundaries[3]['ADM3_PCODE'].head(3).tolist()}")
        st.stop()
    
    # Classify and aggregate
    aggregated, merged = classify_and_aggregate_data(
        admin_data['admin3'], admin_data, conflict_data, period_info,
        rate_thresh, abs_thresh, agg_thresh, agg_level
    )
    
    if merged.empty:
        st.error("❌ No sub-prefecture data available after processing.")
        st.stop()

# Key metrics
st.header("📊 Overview Metrics")

total_subprefs = len(merged)
affected_subprefs = merged['violence_affected'].sum()
total_population = merged['pop_count'].sum()
affected_population = merged[merged['violence_affected']]['pop_count'].sum()
total_deaths = merged['ACLED_BRD_total'].sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>📍 Total Sub-prefectures</h4>
        <div style="font-size: 24px; font-weight: bold;">{total_subprefs:,}</div>
        <div>analyzed in {period_info['label']}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    affected_pct = (affected_subprefs/total_subprefs*100) if total_subprefs > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <h4>⚠️ Affected Sub-prefectures</h4>
        <div style="font-size: 24px; font-weight: bold;">{affected_subprefs:,}</div>
        <div>out of {total_subprefs:,} ({affected_pct:.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    affected_pop_pct = (affected_population/total_population*100) if total_population > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <h4>👥 Affected Population</h4>
        <div style="font-size: 24px; font-weight: bold;">{affected_population:,.0f}</div>
        <div>out of {total_population:,.0f} ({affected_pop_pct:.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Total Deaths</h4>
        <div style="font-size: 24px; font-weight: bold;">{total_deaths:,}</div>
        <div>in {period_info['label']}</div>
    </div>
    """, unsafe_allow_html=True)

# Maps section
tab1, tab2 = st.tabs(["🏘️ Sub-prefectures", "📍 Regions"])

with tab1:
    if len(merged) > 0 and boundaries and isinstance(boundaries, dict) and 3 in boundaries and not boundaries[3].empty:
        with st.spinner("Generating sub-prefecture map... This may take a moment."):
            # Always load neighboring country events (filtered by selected date range)
            somalia_events = load_neighboring_country_events(period_info, country='somalia', border_distance_km=200)
            if somalia_events is not None and not somalia_events.empty:
                st.info(f"🇸🇴 Loaded {len(somalia_events)} Somalia events for {period_info['label']}")
            
            ethiopia_events = load_neighboring_country_events(period_info, country='ethiopia', border_distance_km=200)
            if ethiopia_events is not None and not ethiopia_events.empty:
                st.info(f"🇪🇹 Loaded {len(ethiopia_events)} Ethiopia events for {period_info['label']}")
            
            yemen_events = load_neighboring_country_events(period_info, country='yemen', border_distance_km=200)
            if yemen_events is not None and not yemen_events.empty:
                st.info(f"🇾🇪 Loaded {len(yemen_events)} Yemen events for {period_info['label']}")
            
            # Always load UNHCR refugee data
            refugee_data = load_unhcr_refugee_data()
            if refugee_data is not None and not refugee_data.empty:
                st.success(f"🏕️ Loaded {len(refugee_data)} UNHCR refugee locations")
            
            payam_map = create_payam_map(
                merged, boundaries, period_info, rate_thresh, abs_thresh, show_all_payams,
                somalia_events=somalia_events, ethiopia_events=ethiopia_events, yemen_events=yemen_events,
                refugee_data=refugee_data, show_refugee_layer=True  # Always show refugee layer if data exists
            )
            if payam_map:
                st_folium(payam_map, width=None, height=600, returned_objects=["last_object_clicked"])
            else:
                st.error("Could not create sub-prefecture map due to missing boundary data.")
    else:
        st.error("No sub-prefecture data available for the selected period.")

with tab2:
    if len(aggregated) > 0 and agg_level in ['ADM1', 'ADM2']:
        admin_level_num = 1 if agg_level == 'ADM1' else 2
        if boundaries and isinstance(boundaries, dict) and admin_level_num in boundaries and not boundaries[admin_level_num].empty:
            with st.spinner("Generating administrative map..."):
                try:
                    admin_map = create_admin_map(
                        aggregated, boundaries, agg_level, map_var, agg_thresh, period_info, rate_thresh, abs_thresh
                    )
                    if admin_map:
                        st_folium(admin_map, width=None, height=600, returned_objects=["last_object_clicked"])
                    else:
                        st.error("Could not create administrative map due to missing boundary data.")
                except Exception as e:
                    st.error(f"Error creating administrative map: {str(e)}")
        else:
            st.error("No administrative boundary data available.")
    else:
        st.error("No administrative data available for the selected period.")


import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import datetime
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import utilities
from dashboard_utils import (
    init_session_state, load_custom_css, generate_12_month_periods,
    load_population_data, create_admin_levels, load_conflict_data,
    classify_and_aggregate_data
)

# Page configuration
st.set_page_config(
    page_title="Data Export - Djibouti Violence Dashboard",
    page_icon="📥",
    layout="wide"
)

# Initialize
init_session_state()
load_custom_css()

# Header
st.markdown("""
<div class="main-header">
    <h1>📥 Data Export</h1>
</div>
""", unsafe_allow_html=True)

# Load data
if not st.session_state.data_loaded:
    with st.spinner("Loading data..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Loading time periods...")
            progress_bar.progress(20)
            st.session_state.periods = generate_12_month_periods()
            
            status_text.text("Loading population data...")
            progress_bar.progress(40)
            st.session_state.pop_data = load_population_data()
            
            status_text.text("Creating administrative levels...")
            progress_bar.progress(60)
            st.session_state.admin_data = create_admin_levels(st.session_state.pop_data)
            
            status_text.text("Loading conflict data...")
            progress_bar.progress(80)
            st.session_state.conflict_data = load_conflict_data()
            
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
st.sidebar.header("📊 Export Parameters")

# Period selection
periods = st.session_state.periods
if periods is None or not isinstance(periods, list) or len(periods) == 0:
    st.error("Periods data not loaded. Please refresh the page.")
    st.stop()

period_labels = [p['label'] for p in periods]
default_period = next((i for i, p in enumerate(periods) if p['label'] == 'Jan 2020 - Dec 2020'), -1)

selected_period_label = st.sidebar.selectbox(
    "Time Period",
    period_labels,
    index=default_period if default_period >= 0 else len(period_labels)-1,
    help="Select a 12-month period for analysis"
)

period_info = next(p for p in periods if p['label'] == selected_period_label)
period_info['start_date'] = datetime.date(period_info['start_year'], period_info['start_month'], 1)
end_month = period_info['end_month']
end_year = period_info['end_year']
if end_month == 12:
    period_info['end_date'] = datetime.date(end_year, 12, 31)
else:
    import calendar
    last_day = calendar.monthrange(end_year, end_month)[1]
    period_info['end_date'] = datetime.date(end_year, end_month, last_day)
period_info['months'] = 12

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
    help="Minimum percentage of sub-prefectures affected"
) / 100

map_var = st.sidebar.selectbox(
    "Map Variable",
    ["share_payams_affected", "share_population_affected"],
    format_func=lambda x: "Share of Sub-prefectures Affected" if x == "share_payams_affected" else "Share of Population Affected"
)

# Process data
with st.spinner("Processing data for export..."):
    pop_data = st.session_state.pop_data
    admin_data = st.session_state.admin_data
    conflict_data = st.session_state.conflict_data
    
    # Classify and aggregate
    aggregated, merged = classify_and_aggregate_data(
        admin_data['admin3'], admin_data, conflict_data, period_info,
        rate_thresh, abs_thresh, agg_thresh, agg_level
    )

# Create comprehensive export data
def create_export_data(merged, aggregated, period_info, rate_thresh, abs_thresh, agg_thresh, agg_level, map_var):
    """Create comprehensive export datasets with metadata"""
    
    # Sub-prefecture-level export (admin3, which is actually admin2 in Djibouti)
    payam_export = merged.copy()
    
    # Add metadata columns
    payam_export['export_timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    payam_export['analysis_period'] = period_info['label']
    payam_export['period_start'] = period_info['start_date'].strftime('%Y-%m-%d')
    payam_export['period_end'] = period_info['end_date'].strftime('%Y-%m-%d')
    payam_export['period_months'] = period_info['months']
    payam_export['rate_threshold_per_100k'] = rate_thresh
    payam_export['absolute_threshold_deaths'] = abs_thresh
    
    # Rename columns for clarity
    payam_export = payam_export.rename(columns={
        'ADM3_PCODE': 'subpref_code',
        'ADM3_EN': 'subpref_name',
        'ADM2_PCODE': 'subpref_code_alt',
        'ADM2_EN': 'subpref_name_alt',
        'ADM1_PCODE': 'region_code',
        'ADM1_EN': 'region_name',
        'pop_count': 'population',
        'ACLED_BRD_total': 'total_deaths',
        'ACLED_BRD_state': 'state_violence_deaths',
        'ACLED_BRD_nonstate': 'nonstate_violence_deaths',
        'acled_total_death_rate': 'death_rate_per_100k',
        'violence_affected': 'is_violence_affected'
    })
    
    # Add violence status
    payam_export['violence_status'] = payam_export['is_violence_affected'].map({
        True: 'Violence Affected',
        False: 'Not Affected'
    })
    
    # Aggregated export
    if len(aggregated) > 0:
        agg_export = aggregated.copy()
        
        # Add metadata
        agg_export['export_timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        agg_export['analysis_period'] = period_info['label']
        agg_export['period_start'] = period_info['start_date'].strftime('%Y-%m-%d')
        agg_export['period_end'] = period_info['end_date'].strftime('%Y-%m-%d')
        agg_export['aggregation_level'] = agg_level
        agg_export['aggregation_threshold'] = agg_thresh
        agg_export['map_variable'] = map_var
        
        # Rename columns
        agg_export = agg_export.rename(columns={
            'ADM1_PCODE': 'region_code',
            'ADM1_EN': 'region_name',
            'ADM2_PCODE': 'subpref_code',
            'ADM2_EN': 'subpref_name',
            'pop_count': 'total_population',
            'violence_affected': 'number_subprefs_affected',
            'ACLED_BRD_total': 'total_deaths',
            'share_payams_affected': 'percentage_subprefs_affected',
            'share_population_affected': 'percentage_population_affected',
            'above_threshold': 'is_above_threshold'
        })
    else:
        agg_export = pd.DataFrame()
    
    return payam_export, agg_export

payam_export, agg_export = create_export_data(
    merged, aggregated, period_info, rate_thresh, abs_thresh, agg_thresh, agg_level, map_var
)

# Summary
total_subprefs = len(merged)
affected_subprefs = merged['violence_affected'].sum()
total_population = merged['pop_count'].sum()
affected_population = merged[merged['violence_affected']]['pop_count'].sum()
total_deaths = merged['ACLED_BRD_total'].sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Sub-prefectures", f"{total_subprefs:,}")

with col2:
    st.metric("Affected", f"{affected_subprefs:,}", 
              f"{(affected_subprefs/total_subprefs*100) if total_subprefs > 0 else 0:.1f}%")

with col3:
    st.metric("Population", f"{affected_population/1e6:.1f}M",
              f"{affected_population/total_population*100:.1f}%")

with col4:
    st.metric("Deaths", f"{total_deaths:,}")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"**🏘️ Sub-prefecture Data** ({len(payam_export):,} sub-prefectures)")

    if len(payam_export) > 0:
        csv = payam_export.to_csv(index=False)
        filename = f"djibouti_subprefectures_{period_info['label'].replace(' ', '_').replace('-', '_')}.csv"
        st.download_button(
            label="📥 Download Sub-prefecture Data (CSV)",
            data=csv,
            file_name=filename,
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.error("No sub-prefecture data to export.")

with col2:
    st.markdown(f"**📊 Aggregated** ({len(agg_export):,} {agg_level})")
    
    if len(agg_export) > 0:
        csv = agg_export.to_csv(index=False)
        filename = f"djibouti_aggregated_{agg_level}_{period_info['label'].replace(' ', '_').replace('-', '_')}.csv"
        st.download_button(
            label="📥 Download Aggregated Data (CSV)",
            data=csv,
            file_name=filename,
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.error("No aggregated data to export.")

with col3:
    st.markdown("**📈 Summary** (metadata)")
    
    # Create summary
    summary_data = {
        'export_timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'analysis_period': period_info['label'],
        'period_start': period_info['start_date'].strftime('%Y-%m-%d'),
        'period_end': period_info['end_date'].strftime('%Y-%m-%d'),
        'period_months': period_info['months'],
        'rate_threshold_per_100k': rate_thresh,
        'absolute_threshold_deaths': abs_thresh,
        'aggregation_threshold': agg_thresh,
        'aggregation_level': agg_level,
        'map_variable': map_var,
        'total_subprefs_analyzed': total_subprefs,
        'affected_subprefs_count': affected_subprefs,
        'affected_subprefs_percentage': f"{(affected_subprefs/total_subprefs*100) if total_subprefs > 0 else 0:.1f}%",
        'total_population': total_population,
        'affected_population': affected_population,
        'affected_population_percentage': f"{affected_population/total_population*100:.1f}%",
        'total_battle_related_deaths': total_deaths,
        'average_death_rate_per_100k': f"{total_deaths/(total_population/1e5):.1f}",
        'data_source': 'ACLED + Djibouti Administrative Boundaries',
        'analysis_method': 'Spatial intersection with sub-prefecture-level aggregation'
    }
    
    summary_df = pd.DataFrame([summary_data])
    csv = summary_df.to_csv(index=False)
    filename = f"analysis_summary_{period_info['label'].replace(' ', '_').replace('-', '_')}.csv"
    st.download_button(
        label="📥 Download Summary (CSV)",
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

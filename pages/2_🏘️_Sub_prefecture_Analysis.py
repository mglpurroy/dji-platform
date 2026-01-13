import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import datetime
import time
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import utilities
from dashboard_utils import (
    init_session_state, load_custom_css, generate_12_month_periods,
    load_population_data, create_admin_levels, load_conflict_data,
    classify_and_aggregate_data, DATA_PATH
)

# Page configuration
st.set_page_config(
    page_title="Sub-prefecture Analysis - Djibouti Violence Dashboard",
    page_icon="🏘️",
    layout="wide"
)

# Initialize
init_session_state()
load_custom_css()

# Header
st.markdown("""
<div class="main-header">
    <h1>🏘️ Sub-prefecture Analysis</h1>
</div>
""", unsafe_allow_html=True)

# Load data
# Ensure periods are always loaded (needed for this page)
if 'periods' not in st.session_state or st.session_state.periods is None:
    st.session_state.periods = generate_12_month_periods()

if not st.session_state.data_loaded:
    with st.spinner("Loading data..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Loading population data...")
            progress_bar.progress(30)
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
st.sidebar.header("📊 Analysis Parameters")

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

# Process data
with st.spinner("Processing data for selected period..."):
    pop_data = st.session_state.pop_data
    admin_data = st.session_state.admin_data
    conflict_data = st.session_state.conflict_data
    
    # Check data availability
    if pop_data.empty:
        st.error("❌ No population data loaded. Please check data files.")
        st.stop()
    
    if admin_data['admin3'].empty:
        st.error("❌ No admin3 (sub-prefecture) data available.")
        st.stop()
    
    # Classify and aggregate
    aggregated, merged = classify_and_aggregate_data(
        admin_data['admin3'], admin_data, conflict_data, period_info,
        rate_thresh, abs_thresh, 0.1, 'ADM2'
    )
    
    if merged.empty:
        st.error("❌ No sub-prefecture data available after processing.")
        st.stop()

if len(merged) > 0:
    # Create sub-prefecture selection dropdown
    if 'payam_options' not in st.session_state:
        payam_options = merged[['ADM3_PCODE', 'ADM3_EN', 'ADM2_EN', 'ADM1_EN']].copy()
        payam_options['display_name'] = payam_options['ADM3_EN'] + ' (' + payam_options['ADM2_EN'] + ', ' + payam_options['ADM1_EN'] + ')'
        payam_options = payam_options.sort_values('display_name')
        st.session_state.payam_options = payam_options
    else:
        payam_options = st.session_state.payam_options
    
    # Sub-prefecture selection
    selected_payam_display = st.selectbox(
        "🔍 Select Sub-prefecture:",
        options=payam_options['display_name'].tolist(),
        index=None,
        placeholder="Type to search...",
        help="Search by sub-prefecture name"
    )
    
    # Display analysis for selected sub-prefecture
    if selected_payam_display:
        selected_payam_code = payam_options[payam_options['display_name'] == selected_payam_display]['ADM3_PCODE'].iloc[0]
        
        # Get sub-prefecture info from the merged data
        payam_info = merged[merged['ADM3_PCODE'] == selected_payam_code].iloc[0]
        
        # Display sub-prefecture info compactly
        st.markdown(f"**{payam_info['ADM3_EN']}** · {payam_info['ADM2_EN']} · {payam_info['ADM1_EN']}")
        
        # Key metrics for selected period
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Population",
                f"{payam_info['pop_count']:,.0f}"
            )
        
        with col2:
            st.metric(
                "Total Deaths",
                f"{payam_info['ACLED_BRD_total']:,.0f}"
            )
        
        with col3:
            st.metric(
                "Death Rate",
                f"{payam_info['acled_total_death_rate']:.1f} per 100k"
            )
        
        with col4:
            violence_status = "⚠️ Affected" if payam_info['violence_affected'] else "✅ Not Affected"
            st.metric(
                "Violence Status",
                violence_status
            )
        
        # Static charts section
        st.markdown("---")
        st.subheader("📊 Detailed Analysis")
        
        # Check if static chart exists
        payam_name_clean = payam_info['ADM3_EN'].replace(' ', '_').replace('/', '_')
        png_path = DATA_PATH / f"processed/static_charts/{payam_name_clean}_{payam_info['ADM3_PCODE']}_timeseries.png"
        pdf_path = DATA_PATH / f"processed/static_charts/{payam_name_clean}_{payam_info['ADM3_PCODE']}_timeseries.pdf"
        
        if png_path.exists():
            # Show the static chart
            st.image(str(png_path), caption=f"Complete Violence Analysis - {payam_info['ADM3_EN']}", use_container_width=True)
            
            # Download buttons
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                with open(png_path, "rb") as file:
                    st.download_button(
                        label="📊 Download PNG",
                        data=file.read(),
                        file_name=f"{payam_info['ADM3_EN'].replace(' ', '_')}_violence_analysis.png",
                        mime="image/png",
                        use_container_width=True
                    )
            
            with col_dl2:
                if pdf_path.exists():
                    with open(pdf_path, "rb") as file:
                        st.download_button(
                            label="📄 Download PDF",
                            data=file.read(),
                            file_name=f"{payam_info['ADM3_EN'].replace(' ', '_')}_violence_analysis.pdf",
                            mime="application/pdf",
                        use_container_width=True
                    )
        else:
            # No chart exists, but show summary of available data
            if payam_info['ACLED_BRD_total'] > 0:
                st.info(f"📊 Violence data available: {payam_info['ACLED_BRD_total']:.0f} total deaths reported for this sub-prefecture. Detailed time series chart not yet generated.")
            else:
                st.info("📊 No violence events with fatalities reported for this sub-prefecture in the selected period.")
    else:
        st.info("👆 Select a sub-prefecture from the dropdown above.")
else:
    st.error("No sub-prefecture data available.")

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="Djibouti Violence Analysis Dashboard",
    page_icon="🇩🇯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🇩🇯 Djibouti Violence Analysis</h1>
    <p style="font-size: 1rem; margin-top: 0.5rem; opacity: 0.9;">
        Interactive conflict data visualization & analysis
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>🗺️ Maps</h3>
        <p>Interactive spatial analysis with sub-prefecture & region-level visualizations</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🏘️ Sub-Prefectures</h3>
        <p>Detailed sub-prefecture-level time series & violence analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>📥 Export</h3>
        <p>Download data & charts for further analysis</p>
    </div>
    """, unsafe_allow_html=True)

# Quick info
st.markdown("---")
with st.expander("ℹ️ About the Data"):
    st.markdown("""
    **Sources:** ACLED conflict events | Djibouti admin boundaries | Population data (2025)
    
    **Coverage:** All Djiboutian sub-prefectures and regions from 1997-2025
    """)

with st.expander("🚀 Quick Start Guide"):
    st.markdown("""
    1. **Choose a page** from the sidebar
    2. **Set your parameters** (time period, thresholds)
    3. **Explore** interactive visualizations
    4. **Download** data and charts as needed
    """)

with st.expander("⚡ Performance Tips"):
    st.markdown("""
    - First load may take 10-30 seconds (data caching)
    - Subsequent loads are much faster
    - Use smaller time ranges for faster processing
    """)

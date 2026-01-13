import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
import time
from datetime import datetime

def clean_gdf_for_folium(gdf):
    """Remove non-serializable columns (Timestamps, etc.) from GeoDataFrame for Folium"""
    if gdf.empty:
        return gdf
    
    gdf_clean = gdf.copy()
    
    # Drop columns that contain Timestamp or other non-serializable types
    cols_to_drop = []
    for col in gdf_clean.columns:
        if col == 'geometry':
            continue
        # Check for datetime64 dtypes
        if pd.api.types.is_datetime64_any_dtype(gdf_clean[col]):
            cols_to_drop.append(col)
        # Check if column contains Timestamp objects (object dtype)
        elif gdf_clean[col].dtype == 'object':
            sample = gdf_clean[col].dropna()
            if len(sample) > 0 and isinstance(sample.iloc[0], (pd.Timestamp, datetime)):
                cols_to_drop.append(col)
        # Also drop known date columns by name
        elif 'date' in col.lower() or 'valid' in col.lower():
            cols_to_drop.append(col)
    
    if cols_to_drop:
        gdf_clean = gdf_clean.drop(columns=cols_to_drop)
    
    # Keep only essential columns for boundaries (geometry + identifiers)
    essential_cols = ['geometry']
    for col in ['ADM1_PCODE', 'ADM1_EN', 'ADM2_PCODE', 'ADM2_EN', 'ADM3_PCODE', 'ADM3_EN']:
        if col in gdf_clean.columns:
            essential_cols.append(col)
    
    # Keep only essential columns
    available_cols = [col for col in essential_cols if col in gdf_clean.columns]
    gdf_clean = gdf_clean[available_cols]
    
    return gdf_clean

def create_admin_map(aggregated, boundaries, agg_level, map_var, agg_thresh, period_info, rate_thresh, abs_thresh):
    """Create administrative units map with optimized performance"""
    import time
    start_time = time.time()
    
    # Determine columns based on boundary structure
    if agg_level == 'ADM1':
        # Region level
        pcode_col = 'ADM1_PCODE'  # From boundary file
        name_col = 'ADM1_EN'      # From boundary file
        agg_pcode_col = 'ADM1_PCODE'  # From aggregated data
        agg_name_col = 'ADM1_EN'      # From aggregated data
    else:
        # Sub-prefecture level  
        pcode_col = 'ADM2_PCODE'  # From boundary file
        name_col = 'ADM2_EN'      # From boundary file
        agg_pcode_col = 'ADM2_PCODE'  # From aggregated data
        agg_name_col = 'ADM2_EN'      # From aggregated data
    
    if map_var == 'share_payams':
        value_col = 'share_payams_affected'
        value_label = 'Share of Sub-prefectures Affected'
    else:
        value_col = 'share_population_affected'
        value_label = 'Share of Population Affected'
    
    # Get appropriate boundary data
    map_level_num = 1 if agg_level == 'ADM1' else 2
    gdf = boundaries[map_level_num]
    
    if gdf.empty:
        st.error(f"No boundary data available for {agg_level}")
        return None
    
    # Merge data with boundaries using optimized merge
    merge_cols = [agg_pcode_col, value_col, 'above_threshold', 'violence_affected', 'total_payams', 'pop_count', 'ACLED_BRD_total']
    merged_gdf = gdf.merge(aggregated[merge_cols], left_on=pcode_col, right_on=agg_pcode_col, how='left')
    
    # Use vectorized fillna
    fill_values = {
        value_col: 0, 
        'above_threshold': False, 
        'violence_affected': 0, 
        'total_payams': 0,
        'pop_count': 0,
        'ACLED_BRD_total': 0
    }
    merged_gdf = merged_gdf.fillna(fill_values)
    
    # Create map with optimized settings - centered on Djibouti
    m = folium.Map(
        location=[11.8, 42.6],  # Djibouti center coordinates
        zoom_start=6, 
        tiles='OpenStreetMap',
        prefer_canvas=True
    )
    
    # Pre-calculate colors and status for better performance
    def get_color_status(value):
        if value > agg_thresh:
            return '#d73027', 0.8, "HIGH VIOLENCE"
        elif value > 0:
            return '#fd8d3c', 0.7, "Some Violence"
        else:
            return '#2c7fb8', 0.4, "Low/No Violence"
    
    # Add choropleth layer with optimized rendering
    for _, row in merged_gdf.iterrows():
        value = row[value_col]
        color, opacity, status = get_color_status(value)
        
        # Simplified popup content for better performance
        popup_content = f"""
        <div style="width: 280px; font-family: Arial, sans-serif;">
            <h4 style="color: {color}; margin: 0;">{row.get(name_col, 'Unknown')}</h4>
            <div style="background: {color}; color: white; padding: 3px; border-radius: 2px; text-align: center; margin: 5px 0;">
                <strong>{status}</strong>
            </div>
            <p><strong>{value_label}:</strong> {value:.1%}</p>
            <p><strong>Affected Sub-prefectures:</strong> {row['violence_affected']}/{row['total_payams']}</p>
            <p><strong>Total Deaths:</strong> {row['ACLED_BRD_total']:,.0f}</p>
        </div>
        """
        
        folium.GeoJson(
            row.geometry,
            style_function=lambda x, color=color, opacity=opacity: {
                'fillColor': color,
                'color': 'black',
                'weight': 0.8,
                'fillOpacity': opacity
            },
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=f"{row.get(name_col, 'Unknown')}: {value:.1%}"
        ).add_to(m)
    
    # Simplified legend
    legend_html = f'''
    <div style="position: fixed; top: 10px; right: 10px; width: 250px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:11px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                border-radius: 4px;">
    <h4 style="margin: 0 0 6px 0; color: #333;">{value_label}</h4>
    <div style="margin-bottom: 6px;">
        <div style="margin: 2px 0;"><span style="background:#d73027; color:white; padding:1px 3px; border-radius:1px; font-size:9px;">HIGH</span> >{agg_thresh:.1%}</div>
        <div style="margin: 2px 0;"><span style="background:#fd8d3c; color:white; padding:1px 3px; border-radius:1px; font-size:9px;">SOME</span> >0%</div>
        <div style="margin: 2px 0;"><span style="background:#2c7fb8; color:white; padding:1px 3px; border-radius:1px; font-size:9px;">LOW</span> 0%</div>
    </div>
    <div style="font-size:9px; color:#666;">
        <strong>Period:</strong> {period_info['label']}<br>
        <strong>Criteria:</strong> >{rate_thresh:.1f}/100k & >{abs_thresh} deaths<br>
        <strong>Black borders:</strong> Region boundaries
    </div>
    </div>
    '''
    
    # Add Region borders on top of admin units (non-interactive reference layer)
    admin1_gdf = boundaries[1]
    if not admin1_gdf.empty:
        admin1_gdf_clean = clean_gdf_for_folium(admin1_gdf)
        folium.GeoJson(
            admin1_gdf_clean,
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': '#000000',
                'weight': 2,
                'fillOpacity': 0,
                'opacity': 0.8,
                'interactive': False
            },
            interactive=False
        ).add_to(m)
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def create_payam_map(payam_data, boundaries, period_info, rate_thresh, abs_thresh, show_all_payams=False, somalia_events=None, ethiopia_events=None):
    """Create sub-prefecture (admin3) classification map with highly optimized performance"""
    import time
    import json
    start_time = time.time()
    
    # Get sub-prefecture boundaries (admin3, which is actually admin2 in Djibouti)
    payam_gdf = boundaries[3].copy()
    
    if payam_gdf.empty:
        st.error("No sub-prefecture boundary data available")
        return None
    
    # Filter out null or invalid geometries before processing
    valid_geom_mask = payam_gdf.geometry.notna() & payam_gdf.geometry.is_valid
    payam_gdf = payam_gdf[valid_geom_mask].copy()
    
    if payam_gdf.empty:
        st.error("No valid sub-prefecture geometries available")
        return None
    
    # Simplify geometries for faster rendering (tolerance in degrees, ~1km)
    payam_gdf['geometry'] = payam_gdf['geometry'].simplify(tolerance=0.01, preserve_topology=True)
    
    # Merge with classification data using optimized merge
    # Only merge on ADM3_PCODE to avoid column name conflicts
    merge_cols = ['ADM3_PCODE', 'ADM3_EN', 'ADM2_EN', 'ADM1_EN', 'pop_count', 'violence_affected', 'ACLED_BRD_total', 'acled_total_death_rate']
    
    # Check which columns actually exist in payam_data
    available_cols = ['ADM3_PCODE'] + [col for col in merge_cols[1:] if col in payam_data.columns]
    
    # Drop columns from payam_gdf that will cause conflicts (keep only ADM3_PCODE and geometry)
    payam_gdf_clean = payam_gdf[['ADM3_PCODE', 'geometry']].copy()
    
    merged_payam = payam_gdf_clean.merge(payam_data[available_cols], on='ADM3_PCODE', how='left')
    
    # Use vectorized fillna
    fill_values = {
        'ADM3_EN': 'Unknown',
        'ADM2_EN': 'Unknown',
        'ADM1_EN': 'Unknown',
        'pop_count': 0,
        'ACLED_BRD_total': 0,
        'acled_total_death_rate': 0.0,
        'violence_affected': False
    }
    
    # Only fill values for columns that exist
    fill_values_filtered = {k: v for k, v in fill_values.items() if k in merged_payam.columns}
    merged_payam = merged_payam.fillna(fill_values_filtered)
    
    # Ensure all required columns exist with defaults
    for col in ['ADM3_EN', 'ADM2_EN', 'ADM1_EN']:
        if col not in merged_payam.columns:
            merged_payam[col] = 'Unknown'
    for col in ['pop_count', 'ACLED_BRD_total', 'acled_total_death_rate']:
        if col not in merged_payam.columns:
            merged_payam[col] = 0
    if 'violence_affected' not in merged_payam.columns:
        merged_payam['violence_affected'] = False
    
    # Filter to only affected sub-prefectures if requested (default for performance)
    if not show_all_payams:
        merged_payam = merged_payam[merged_payam['violence_affected'] == True].copy()
    
    # If no affected sub-prefectures, return None with message
    if len(merged_payam) == 0:
        st.warning("No violence-affected sub-prefectures to display in the selected period. Try selecting 'Show All Sub-prefectures' or a different time period.")
        return None
    
    # Pre-calculate statistics for legend
    total_payams = len(payam_data)
    affected_payams = sum(payam_data['violence_affected'])
    affected_percentage = (affected_payams / total_payams * 100) if total_payams > 0 else 0
    
    # Clean the GeoDataFrame to remove any non-serializable columns (Timestamps, etc.)
    # Keep only the columns we need for the map
    essential_cols = ['geometry', 'ADM3_PCODE', 'ADM3_EN', 'ADM2_EN', 'ADM1_EN', 
                      'pop_count', 'violence_affected', 'ACLED_BRD_total', 'acled_total_death_rate']
    available_cols = [col for col in essential_cols if col in merged_payam.columns]
    merged_payam = merged_payam[available_cols].copy()
    
    # Add color column for choropleth-style rendering
    merged_payam['color'] = merged_payam.apply(
        lambda x: '#d73027' if x['violence_affected'] else (
            '#fd8d3c' if x['ACLED_BRD_total'] > 0 else '#2c7fb8'
        ), axis=1
    )
    merged_payam['status'] = merged_payam.apply(
        lambda x: 'AFFECTED' if x['violence_affected'] else (
            'Below Threshold' if x['ACLED_BRD_total'] > 0 else 'No Violence'
        ), axis=1
    )
    
    # Create map with optimized settings - centered on Djibouti
    m = folium.Map(
        location=[11.8, 42.6],  # Djibouti center coordinates
        zoom_start=8, 
        tiles='CartoDB positron',  # Lighter, faster tiles
        prefer_canvas=True,
        max_zoom=12,
        zoom_control=True
    )
    
    # Create a single GeoJson layer with all sub-prefectures (much faster than individual layers)
    # Prepare fields for tooltip and popup
    merged_payam['popup_html'] = merged_payam.apply(
        lambda row: f"""
        <b>{row['ADM3_EN']}</b><br>
        <b>Status:</b> {row['status']}<br>
        <b>Sub-prefecture:</b> {row['ADM2_EN']}<br>
        <b>Region:</b> {row['ADM1_EN']}<br>
        <b>Deaths:</b> {int(row['ACLED_BRD_total'])}<br>
        <b>Rate:</b> {row['acled_total_death_rate']:.1f}/100k<br>
        <b>Pop:</b> {int(row['pop_count']):,}
        """, axis=1
    )
    
    # Use style_function for dynamic coloring
    def style_function(feature):
        return {
            'fillColor': feature['properties']['color'],
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7
        }
    
    def highlight_function(feature):
        return {
            'fillColor': feature['properties']['color'],
            'color': 'yellow',
            'weight': 3,
            'fillOpacity': 0.9
        }
    
    # Convert to GeoJSON for better performance
    payam_geojson = folium.GeoJson(
        merged_payam[['geometry', 'ADM3_EN', 'ADM2_EN', 'ADM1_EN', 'status', 'ACLED_BRD_total', 
                     'acled_total_death_rate', 'pop_count', 'color', 'popup_html']],
        name='Sub-prefectures',
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['ADM3_EN', 'status', 'ACLED_BRD_total'],
            aliases=['Sub-prefecture:', 'Status:', 'Deaths:'],
            localize=True,
            sticky=False,
            labels=True,
            style="""
                background-color: white;
                border: 2px solid black;
                border-radius: 3px;
                box-shadow: 3px;
            """,
        ),
        popup=folium.GeoJsonPopup(
            fields=['popup_html'],
            labels=False,
            localize=True,
            style="background-color: white;",
        ),
        zoom_on_click=True,
    )
    
    payam_geojson.add_to(m)
    
    # Add neighboring country events as point layers
    if somalia_events is not None and not somalia_events.empty:
        for idx, event in somalia_events.iterrows():
            # Create popup content
            event_date = event.get('event_date', 'N/A')
            if pd.notna(event_date) and hasattr(event_date, 'strftime'):
                event_date = event_date.strftime('%Y-%m-%d')
            elif pd.notna(event_date):
                event_date = str(event_date)[:10]  # Take first 10 chars for date
            
            notes = event.get('notes', '')
            notes_html = f"<p><strong>Notes:</strong> {str(notes)[:100]}...</p>" if pd.notna(notes) and str(notes) != '' else ''
            
            popup_html = f"""
            <div style="width: 250px; font-family: Arial, sans-serif;">
                <h4 style="color: #e31a1c; margin: 0;">🇸🇴 Somalia Event</h4>
                <p><strong>Date:</strong> {event_date}</p>
                <p><strong>Type:</strong> {event.get('event_type', 'N/A')}</p>
                <p><strong>Location:</strong> {event.get('location', 'N/A')}</p>
                <p><strong>Fatalities:</strong> {int(event.get('fatalities', 0))}</p>
                <p><strong>Admin1:</strong> {event.get('admin1', 'N/A')}</p>
                {notes_html}
            </div>
            """
            
            folium.CircleMarker(
                location=[event.geometry.y, event.geometry.x],
                radius=5 + min(int(event.get('fatalities', 0)) / 5, 15),  # Size based on fatalities
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Somalia: {int(event.get('fatalities', 0))} deaths",
                color='#e31a1c',
                fillColor='#e31a1c',
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
    
    if ethiopia_events is not None and not ethiopia_events.empty:
        for idx, event in ethiopia_events.iterrows():
            # Create popup content
            event_date = event.get('event_date', 'N/A')
            if pd.notna(event_date) and hasattr(event_date, 'strftime'):
                event_date = event_date.strftime('%Y-%m-%d')
            elif pd.notna(event_date):
                event_date = str(event_date)[:10]  # Take first 10 chars for date
            
            notes = event.get('notes', '')
            notes_html = f"<p><strong>Notes:</strong> {str(notes)[:100]}...</p>" if pd.notna(notes) and str(notes) != '' else ''
            
            popup_html = f"""
            <div style="width: 250px; font-family: Arial, sans-serif;">
                <h4 style="color: #238b45; margin: 0;">🇪🇹 Ethiopia Event</h4>
                <p><strong>Date:</strong> {event_date}</p>
                <p><strong>Type:</strong> {event.get('event_type', 'N/A')}</p>
                <p><strong>Location:</strong> {event.get('location', 'N/A')}</p>
                <p><strong>Fatalities:</strong> {int(event.get('fatalities', 0))}</p>
                <p><strong>Admin1:</strong> {event.get('admin1', 'N/A')}</p>
                {notes_html}
            </div>
            """
            
            folium.CircleMarker(
                location=[event.geometry.y, event.geometry.x],
                radius=5 + min(int(event.get('fatalities', 0)) / 5, 15),  # Size based on fatalities
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Ethiopia: {int(event.get('fatalities', 0))} deaths",
                color='#238b45',
                fillColor='#238b45',
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
    
    # Add Region borders on top of sub-prefectures (non-interactive to allow sub-prefecture clicks)
    admin1_gdf = boundaries[1]
    if not admin1_gdf.empty:
        admin1_gdf_clean = clean_gdf_for_folium(admin1_gdf)
        folium.GeoJson(
            admin1_gdf_clean,
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': '#000000',
                'weight': 2,
                'fillOpacity': 0,
                'opacity': 0.8,
                'interactive': False  # Makes the layer non-interactive
            },
            interactive=False  # Disable all interactivity for this layer
        ).add_to(m)
    
    # Simplified legend
    legend_html = f'''
    <div style="position: fixed; top: 10px; right: 10px; width: 240px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:11px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                border-radius: 4px;">
    <h4 style="margin: 0 0 6px 0; color: #333;">Sub-prefecture Classification</h4>
    <div style="margin-bottom: 6px;">
        <div style="margin: 2px 0;"><span style="background:#d73027; color:white; padding:1px 3px; border-radius:1px; font-size:9px;">AFFECTED</span> Violence Affected</div>
        <div style="margin: 2px 0;"><span style="background:#fd8d3c; color:white; padding:1px 3px; border-radius:1px; font-size:9px;">BELOW</span> Below Threshold</div>
        <div style="margin: 2px 0;"><span style="background:#2c7fb8; color:white; padding:1px 3px; border-radius:1px; font-size:9px;">NONE</span> No Violence</div>
    </div>
    <div style="font-size:9px; color:#666;">
        <strong>Period:</strong> {period_info['label']}<br>
        <strong>Criteria:</strong> >{rate_thresh:.1f}/100k & >{abs_thresh} deaths<br>
        <strong>Affected:</strong> {affected_payams}/{total_payams} ({affected_percentage:.1f}%)<br>
        <strong>Black borders:</strong> Region boundaries
        {f"<br><strong>🇸🇴 Somalia events:</strong> {len(somalia_events) if somalia_events is not None and not somalia_events.empty else 0}" if somalia_events is not None else ""}
        {f"<br><strong>🇪🇹 Ethiopia events:</strong> {len(ethiopia_events) if ethiopia_events is not None and not ethiopia_events.empty else 0}" if ethiopia_events is not None else ""}
    </div>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    
    return m

import warnings
import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import BabWrangle
import UZeppWrangle
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress warnings
warnings.simplefilter("ignore", UserWarning)

# Input data paths
bab_path = "/home/blueaz/Downloads/SensorDownload/Feb2025/log_21.02.25_230217.db"
uzepp_path = "/home/blueaz/Downloads/SensorDownload/Feb2025/ztennis.db"

def format_metric_name(metric, sensor):
    """Format the metric names for display purposes based on sensor."""
    return f"{metric} ({sensor})"

@st.cache_data
def normalize_column(dfa, dfb, ref_col, norm_col, new_col_name):
    """Normalize dfb's column (norm_col) to match the range of dfa's reference column (ref_col)."""
    min_A = dfa[ref_col].min()
    max_A = dfa[ref_col].max()
    min_B = dfb[norm_col].min()
    max_B = dfb[norm_col].max()

    if max_B == min_B:
        dfb[new_col_name] = 0
    else:
        def normalize(x, min_B, max_B, min_A, max_A):
            return ((x - min_B) * (max_A - min_A) / (max_B - min_B)) + min_A
        
        dfb[new_col_name] = dfb[norm_col].apply(normalize, args=(min_B, max_B, min_A, max_A))

@st.cache_data
def load_data(start_date, end_date):
    """Load and process the merged sensor data for the specified date range."""
    try:
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        
        dfb = BabWrangle.BabWrangle(bab_path)
        dfu = UZeppWrangle.UZeppWrangle(uzepp_path)
        
        dfb['sensor'] = 'Babolat'
        dfu['sensor'] = 'Zepp'
        dfb['stroke'] = dfb['type']
        
        dfu_norm = dfu.copy()
        normalize_column(dfb, dfu_norm, 'EffectScore', 'ball_spin', 'ZIQspin')
        normalize_column(dfb, dfu_norm, 'SpeedScore', 'racket_speed', 'ZIQspeed')
        
        absx = 0 - dfu_norm['impact_position_x'].abs()
        absy = 0 - dfu_norm['impact_position_y'].abs()
        dfu_norm['abs_imp'] = 0 + (absx + absy)
        normalize_column(dfb, dfu_norm, 'StyleScore', 'abs_imp', 'ZIQpos')
        
        dfu_norm.loc[dfu_norm['stroke'] != 'SERVEFH', 'ZIQspin'] *= 2
        dfu_norm.loc[dfu_norm['stroke'] != 'SERVEFH', 'ZIQspeed'] *= 1.6
        dfu_norm['ZIQ'] = dfu_norm['ZIQspeed'] + dfu_norm['ZIQspin'] + dfu_norm['ZIQpos']
        dfu_norm.loc[dfu_norm['stroke'] == 'SERVEFH', 'ZIQ'] *= 0.9
        
        dfu = dfu_norm
        dfb['time'] -= pd.Timedelta(seconds=5)
        
        bab_cols = ['time', 'type', 'spin', 'StyleScore', 'StyleValue', 'EffectScore',
                    'EffectValue', 'SpeedScore', 'SpeedValue', 'stroke_counter', 'PIQ', 
                    'sensor', 'stroke']
        
        zepp_cols = ['time', 'ball_spin', 'racket_speed', 'power', 'impact_position_x',
                 'impact_position_y', 'backswing_time', 'stroke',
                 'ZIQspin', 'ZIQspeed', 'ZIQpos', 'ZIQ', 'sensor',
                 'dbg_acc_1', 'dbg_acc_2', 'dbg_acc_3', 'dbg_gyro_1', 'dbg_gyro_2',
                 'dbg_var_1', 'dbg_var_2', 'dbg_var_3', 'dbg_var_4', 'dbg_sum_gx',
                 'dbg_sum_gy', 'dbg_sv_ax', 'dbg_sv_ay', 'dbg_max_ax', 'dbg_max_ay',
                 'dbg_min_az', 'dbg_max_az']
        
        dfb_slim = dfb[bab_cols].copy()
        dfu_slim = dfu[zepp_cols].copy()
        
        dfb_slim = dfb_slim.sort_values('time')
        dfu_slim = dfu_slim.sort_values('time')
        
        # Store the raw merged data before additional processing
        raw_merged = pd.merge_asof(
            dfu_slim, 
            dfb_slim,
            on='time',
            tolerance=pd.Timedelta('15s'),
            direction='nearest',
            suffixes=('_zepp', '_bab')
        )
        
        # Create processed version for visualization
        df_merged = raw_merged.copy()
        
        df_merged['sensor'] = df_merged['sensor_zepp'].fillna(df_merged['sensor_bab'])
        df_merged['stroke'] = df_merged['stroke_zepp'].fillna(df_merged['stroke_bab'])
        df_merged.drop(columns=['sensor_zepp', 'sensor_bab', 'stroke_zepp', 'stroke_bab'], inplace=True)

        def categorize_stroke(stroke):
            stroke_lower = str(stroke).lower()
            if 'serve' in stroke_lower:
                return 'Serve'
            elif 'forehand' in stroke_lower or 'fh' in stroke_lower:
                return 'Forehand'
            elif 'backhand' in stroke_lower or 'bh' in stroke_lower:
                return 'Backhand'
            else:
                return 'Other'
        
        df_merged['stroke_category'] = df_merged['stroke'].apply(categorize_stroke)
        return raw_merged, df_merged
        
    except Exception as e:
        st.error(f"Error in load_data: {str(e)}")
        raise e

if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.merged_data = None  # Add this line to store merged data

st.title("MergeDash")

st.sidebar.header("Controls")

default_start = datetime(2025, 1, 31)
default_end = datetime(2025, 1, 31)

start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", default_end)

if start_date <= end_date:
    try:
        if st.session_state.data is None:
            raw_merged, processed_df = load_data(start_date, end_date)
            st.session_state.raw_merged = raw_merged  # Store the raw merged data
            st.session_state.data = processed_df  # Store the processed data for visualization

        df = st.session_state.data

        if df.empty:
            st.warning("No data found for the selected date range")
        else:
            # Sidebar filters
            types = st.sidebar.multiselect("Select Types", df['type'].unique(), default=df['type'].unique())
            spins = st.sidebar.multiselect("Select Spins", df['spin'].unique(), default=df['spin'].unique())
            stroke_categories = st.sidebar.multiselect(
                "Select Stroke Categories",
                ['Serve', 'Forehand', 'Backhand', 'Other'],
                default=['Serve', 'Forehand', 'Backhand', 'Other']
            )

                # Add CSV export button to sidebar
            if st.sidebar.button("Write to CSV"):
                csv_filename = f"merged_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                # Load fresh data
                dfb = BabWrangle.BabWrangle(bab_path)
                dfu = UZeppWrangle.UZeppWrangle(uzepp_path)
                
                # Get the slim versions
                dfb_slim = dfb[bab_cols].copy()
                dfu_slim = dfu[zepp_cols].copy()
                
                # Merge them
                raw_merged = pd.merge_asof(
                    dfu_slim, 
                    dfb_slim,
                    on='time',
                    tolerance=pd.Timedelta('15s'),
                    direction='nearest'
                )
                
                raw_merged.to_csv(csv_filename, index=False)
                st.sidebar.success(f"Data written to {csv_filename}")

            # Apply all filters
            filtered_df = df[
                (df['stroke_category'].isin(stroke_categories)) &
                (df['type'].isin(types)) &
                (df['spin'].isin(spins))
            ]

            # Generate correlation plot
            st.header("Correlation Plot")
            color_options = ["None", "sensor", "stroke", "stroke_category"]
            color_by = st.selectbox("Color by", color_options, key='scatter_color')
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis metric", filtered_df.select_dtypes(include=[np.number]).columns.tolist(), index=0)
            with col2:
                y_axis = st.selectbox("Y-axis metric", filtered_df.select_dtypes(include=[np.number]).columns.tolist(), index=1)

            if color_by == "None":
                scatter_fig = px.scatter(
                    filtered_df, x=x_axis, y=y_axis,
                    title=f"Correlation plot of {x_axis} vs {y_axis}"
                )
            else:
                scatter_fig = px.scatter(
                    filtered_df, x=x_axis, y=y_axis, color=color_by,
                    title=f"Correlation plot of {x_axis} vs {y_axis}"
                )
            st.plotly_chart(scatter_fig)

            # Generate normalized line plot
            st.header("Normalized Line Plot")
            
            # First create normalized versions of all numeric columns
            norm_df = filtered_df.copy()
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            normalized_cols = []
            
            # Create reference dataframe with range [0, 100]
            ref_df = pd.DataFrame({'ref': [0, 100]})
            
            # Generate all normalized columns first
            for col in numeric_cols:
                normalized_name = f'{col}_normalized'
                normalize_column(ref_df, norm_df, 'ref', col, normalized_name)
                normalized_cols.append(normalized_name)
            
            # Add color by option with None
            color_options_line = ["None", "type", "spin", "stroke_category"]
            color_by_line = st.selectbox("Color by", color_options_line, key='line_color')

            # Debug: Write processed_df to a CSV for verification
            norm_df.to_csv("debug_processed_df.csv", index=False)

            # Let user select from normalized columns
            selected_metrics = st.multiselect(
                "Select metrics to plot",
                normalized_cols,
                default=normalized_cols[:2] if normalized_cols else []
            )
            
            if selected_metrics:
                fig = go.Figure()
                
                if color_by_line == "None":
                    # Create a single trace for each metric
                    for metric in selected_metrics:
                        original_name = metric.replace('_normalized', '')
                        fig.add_trace(go.Scatter(
                            x=norm_df['time'],
                            y=norm_df[metric],
                            mode='lines+markers',
                            name=original_name,
                            showlegend=True
                        ))
                else:
                    # Get unique categories for coloring
                    categories = norm_df[color_by_line].unique()
                    
                    for metric in selected_metrics:
                        original_name = metric.replace('_normalized', '')
                        
                        # Create a trace for each category
                        for category in categories:
                            mask = norm_df[color_by_line] == category
                            fig.add_trace(go.Scatter(
                                x=norm_df[mask]['time'],
                                y=norm_df[mask][metric],
                                mode='lines+markers',
                                name=f'{original_name} - {category}',  # Combine metric name and category
                                showlegend=True
                            ))
                
                fig.update_layout(
                    title="Normalized Metrics Over Time",
                    yaxis=dict(range=[0, 100]),
                    xaxis_title="Time",
                    yaxis_title="Normalized Value",
                    legend_title="Metrics" if color_by_line == "None" else f"Metrics by {color_by_line.replace('_', ' ').title()}"
                )
                st.plotly_chart(fig)
            else:
                st.warning("Please select at least one metric to plot")

            # Summary statistics
            st.header("Summary Statistics")
            st.dataframe(filtered_df.describe())

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
else:
    st.error("Error: End date must be after start date")

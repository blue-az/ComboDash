import streamlit as st
import pandas as pd
import BabWrangle
import UZeppWrangle
import plotly.express as px

# Modular paths
bab_db_path = "BabPopExt.db"
uzepp_db_path = "ztennis.db"

@st.cache_data
def load_and_merge_data():
    df_bab = BabWrangle.BabWrangle(bab_db_path)
    df_zepp = UZeppWrangle.UZeppWrangle(bab_db_path)

    dfb_slim = dfb[['time', 'type', 'spin', 'StyleScore', 'EffectScore', 'SpeedScore', 'PIQ']].copy()
    dfu_slim = dfu[['time', 'ball_spin', 'racket_speed', 'stroke', 'ZIQspin', 'ZIQspeed', 'ZIQpos', 'ZIQ']].copy()

    merged_df = pd.merge_asof(
        dfb_slim.sort_values('time'),
        dfu_slim.sort_values('time'),
        on='time',
        tolerance=pd.Timedelta('15s'),
        direction='nearest',
        suffixes=('_bab', '_zepp')
    ).dropna()

    return merged_df

st.title("ComboDash Streamlit App")

if 'data' not in st.session_state:
    st.session_state.data = load_and_merge_data()

# Data loaded
merged_df = st.session_state.data

# Sidebar filters
st.sidebar.header("Filters")
selected_stroke = st.sidebar.multiselect("Select Stroke Type", merged_df['stroke_zepp'].unique())

# Apply Filters
if selected_stroke:
    filtered_data = merged_df[merged_df['stroke'].isin(selected_stroke)]
else:
    filtered_data = merged_df

# Visualization
fig = px.scatter(filtered_data, 
                 x='time', 
                 y='PIQ', 
                 color='stroke_zepp', 
                 title='PIQ Over Time by Stroke Type',
                 labels={'PIQ': 'Performance IQ'})

st.plotly_chart(fig)

# Data Display
st.dataframe(filtered_data)


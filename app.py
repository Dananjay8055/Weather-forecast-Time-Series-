import streamlit as st
import pandas as pd
import subprocess
import os
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Real-Time Weather Forecasting",
    layout="wide",
)

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.title("⚙️ Controls")

auto_refresh = st.sidebar.checkbox("🔁 Auto-refresh (30 sec)")
show_raw_data = st.sidebar.checkbox("📄 Show raw dataset")
show_metrics = st.sidebar.checkbox("📊 Show model metrics", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Time Series Forecasting App")

# -----------------------------
# AUTO REFRESH
# -----------------------------
if auto_refresh:
    time.sleep(30)
    st.experimental_rerun()

# -----------------------------
# MAIN TITLE
# -----------------------------
st.title("🌦️ Real-Time Weather Forecasting System")
st.subheader("Time Series Prediction using NOAA Data")

st.markdown("""
This interactive application predicts **tomorrow’s maximum temperature**
using historical weather data and machine learning.
""")

st.divider()

# -----------------------------
# REFRESH PIPELINE
# -----------------------------
if st.button("🔄 Fetch Latest Data & Predict"):
    with st.spinner("Running real-time forecasting pipeline..."):
        subprocess.run(["python", "run_pipeline.py"])
    st.success("Prediction updated successfully!")

# -----------------------------
# LOAD DATA
# -----------------------------
if not os.path.exists("data/daily_temperature.csv"):
    st.error("Weather data not found. Run the pipeline first.")
    st.stop()

df = pd.read_csv("data/daily_temperature.csv")
df["DATE"] = pd.to_datetime(df["DATE"])

# -----------------------------
# TOMORROW PREDICTION DISPLAY
# -----------------------------
st.divider()
st.subheader("🔮 Tomorrow’s Temperature Prediction")

if os.path.exists("output/latest_prediction.txt"):
    with open("output/latest_prediction.txt") as f:
        tomorrow_temp = float(f.read())

    st.metric(
        label="Predicted Maximum Temperature",
        value=f"{tomorrow_temp:.2f} °C",
        delta=f"{tomorrow_temp - df['TMAX'].iloc[-1]:.2f} °C vs today"
    )
else:
    st.warning("Prediction not available yet.")

# -----------------------------
# INTERACTIVE TIME SERIES
# -----------------------------
st.divider()
st.subheader("📈 Historical Temperature Trend")

st.line_chart(
    df.set_index("DATE")["TMAX"],
    height=300
)

# -----------------------------
# MODEL METRICS
# -----------------------------
if show_metrics:
    st.divider()
    st.subheader("📊 Model Insights")

    col1, col2, col3 = st.columns(3)

    col1.metric("Records Used", len(df))
    col2.metric("Latest Temp", f"{df['TMAX'].iloc[-1]:.2f} °C")
    col3.metric("7-Day Avg", f"{df['TMAX'].iloc[-7:].mean():.2f} °C")

# -----------------------------
# OUTPUT VISUALIZATIONS
# -----------------------------
st.divider()
st.subheader("🖼️ Model Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**2D Actual vs Predicted**")
    if os.path.exists("output/weather_forecasting_output.png"):
        st.image("output/weather_forecasting_output.png", use_column_width=True)

with col2:
    st.markdown("**3D Time Series Visualization**")
    if os.path.exists("output/weather_forecasting_3d.png"):
        st.image("output/weather_forecasting_3d.png", use_column_width=True)

# -----------------------------
# RAW DATA VIEW
# -----------------------------
if show_raw_data:
    st.divider()
    st.subheader("📄 Raw Weather Data")
    st.dataframe(df.tail(100))

# -----------------------------
# FOOTER
# -----------------------------
st.divider()
st.caption("Real-Time Time Series Forecasting")
st.caption("Data Source: NOAA National Centers for Environmental Information (NCEI)")
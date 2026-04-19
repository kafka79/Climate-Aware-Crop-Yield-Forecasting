import streamlit as st
import json
import requests
import os
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Configurations
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="Crop Yield Dashboard", page_icon="🌾", layout="wide")

st.title("🌾 Climate-Aware Crop Yield Forecasting")
st.markdown("Upload your satellite, weather, and soil data in CSV or JSON format to get probabilistic yield predictions with full explainability.")

# ─── Sidebar: Upload ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📤 Upload Your Data")
    upload_format = st.radio("Select Input Format", ["CSV (3 files)", "JSON"], horizontal=True, index=0)

    st.divider()
    
    if upload_format == "CSV (3 files)":
        st.markdown("### 📁 Upload CSV Files")
        st.caption("Click the buttons below to browse and upload files")
        sat_file = st.file_uploader("Satellite CSV", type=["csv"], key="sat")
        weather_file = st.file_uploader("Weather CSV", type=["csv"], key="weather")
        soil_file = st.file_uploader("Soil CSV", type=["csv"], key="soil")
        
        if sat_file and weather_file and soil_file:
            st.success("✅ All files uploaded!")
        else:
            missing = []
            if not sat_file: missing.append("Satellite")
            if not weather_file: missing.append("Weather")
            if not soil_file: missing.append("Soil")
            if missing:
                st.info(f"Waiting for: {', '.join(missing)}")
    else:
        st.markdown("### 📄 Upload JSON File")
        uploaded_file = st.file_uploader("JSON Input", type=["json"], key="json")
        st.caption("JSON must have keys: sat, weather, soil")

# ─── Parse uploaded data ───────────────────────────────────────────────────────
data = None

if upload_format == "CSV (3 files)" and sat_file and weather_file and soil_file:
    try:
        sat_df     = pd.read_csv(sat_file)
        weather_df = pd.read_csv(weather_file)
        soil_df    = pd.read_csv(soil_file)
        data = {
            "sat":     sat_df.values.tolist(),
            "weather": weather_df.values.tolist(),
            "soil":    soil_df.values.tolist()[0]
        }
        st.success("✅ Files parsed successfully! Processing prediction...")
    except Exception as e:
        st.error(f"Failed to parse CSV files: {e}")

elif upload_format == "JSON" and uploaded_file:
    try:
        data = json.load(uploaded_file)
        st.success("✅ JSON loaded successfully! Processing prediction...")
    except json.JSONDecodeError:
        st.error("Invalid JSON file. Please upload a valid JSON.")

# ─── Auto-predict when data is ready ─────────────────────────────────────────
if data:
    st.divider()
    st.subheader("📊 Prediction Results")
    
    with st.spinner("Running Multi-Modal Transformer inference..."):
        try:
            response = requests.post(API_URL, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()

            yield_val   = result.get("yield_prediction", 0)
            explanation = result.get("explanation", {})

            st.success("✅ Inference Complete!")
            st.divider()

            r_col1, r_col2 = st.columns([1, 2])

            with r_col1:
                st.subheader("📈 Predicted Yield")
                st.metric("Yield", f"{yield_val:.3f} t/ha")
                st.caption("Inferred via Multi-modal Transformer + MDN head.")

            with r_col2:
                st.subheader("📊 Feature Contributions (XAI)")
                if explanation:
                    labels = ['Satellite', 'Weather', 'Soil']
                    values = [
                        explanation.get('satellite_overall', 0),
                        explanation.get('weather_overall', 0),
                        explanation.get('soil_overall', 0)
                    ]
                    if sum(values) > 0:
                        fig = go.Figure(data=[go.Pie(
                            labels=labels, values=values,
                            hole=0.4,
                            marker_colors=['#00b4d8', '#90e0ef', '#0077b6']
                        )])
                        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Model loaded but attribution scores are zero")
                        st.info("This is normal - the model is working but couldn't generate feature attributions.")
                else:
                    st.info("No explanation data returned from API")

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Please start the API server first:")
            st.code("python -m uvicorn deployment.api.app:app --reload", language="bash")
        except Exception as req_e:
            st.error(f"❌ API Error: {req_e}")
else:
    st.info("👈 Upload your data files in the sidebar to get started.")

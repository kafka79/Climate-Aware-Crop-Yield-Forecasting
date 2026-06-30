import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from src.inference.runtime import (
    InferenceUnavailableError,
    build_region_context,
    list_available_years,
    list_configured_regions,
    load_runtime_config,
    load_yield_history,
    run_inference,
)

st.set_page_config(page_title="Crop Yield Forecast", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

CONFIG = load_runtime_config()
REGIONS = list_configured_regions(CONFIG)
YEARS = list_available_years(CONFIG)
YIELD_HISTORY = load_yield_history(CONFIG)
if "live_results" not in st.session_state:
    st.session_state["live_results"] = {}

@st.cache_data
def generate_offline_features_json():
    import json
    from pathlib import Path
    from src.inference.runtime import _prepare_model_inputs
    
    features_data = {}
    for r in REGIONS:
        try:
            ctx = build_region_context(r, 2023, CONFIG)
            f_years = ctx.get("feature_years", [])
            if f_years:
                latest_year = max(f_years)
                inputs = _prepare_model_inputs(CONFIG, r, latest_year)
                
                # sat_tensor is shape (1, 12, 5) -> [B, T, C]
                # weather_tensor is shape (1, 12, 3)
                # soil_tensor is shape (1, 3)
                sat_last = inputs["sat_tensor"][0, -1].tolist()
                weather_last = inputs["weather_tensor"][0, -1].tolist()
                soil_vec = inputs["soil_tensor"][0].tolist()
                
                features_data[r] = {
                    "year": latest_year,
                    "satellite": {
                        "b02": sat_last[0],
                        "b03": sat_last[1],
                        "b04": sat_last[2],
                        "b08": sat_last[3],
                        "scl": sat_last[4] if len(sat_last) > 4 else 0.9
                    },
                    "weather": {
                        "tmax": weather_last[0],
                        "tmin": weather_last[1],
                        "precip": weather_last[2]
                    },
                    "soil": {
                        "ph": soil_vec[0],
                        "soc": soil_vec[1],
                        "nitrogen": soil_vec[2]
                    }
                }
        except Exception:
            continue
            
    out_path = Path("static/regional_features.json")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(features_data, f, indent=2)
    except Exception:
        pass

if "offline_features_generated" not in st.session_state:
    generate_offline_features_json()
    st.session_state["offline_features_generated"] = True

# ── Apple-grade Design System ────────────────────────────────────────────────
# White canvas, Inter font, high-contrast for outdoor screens, zero noise.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
[data-testid="stAppViewContainer"] { background:#fff; font-family:'Inter',sans-serif; }
[data-testid="stSidebar"] { background:#fafafa; border-right:1px solid #e5e7eb; }
h1 { font-weight:700!important; color:#111827!important; letter-spacing:-0.02em; }
h2,h3 { font-weight:600!important; color:#1f2937!important; }
[data-testid="stMetric"] { background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:1rem; }
[data-testid="stMetricValue"] { font-size:1.5rem!important; font-weight:700!important; color:#111827!important; }
[data-testid="stMetricLabel"] { font-size:0.78rem!important; font-weight:500!important; color:#6b7280!important; text-transform:uppercase; letter-spacing:0.04em; }
.status-pill { display:inline-block; padding:0.3rem 0.8rem; border-radius:100px; font-size:0.82rem; font-weight:600; }
.status-pill.green { background:#dcfce7; color:#166534; }
.status-pill.amber { background:#fef3c7; color:#92400e; }
.status-pill.red { background:#fee2e2; color:#991b1b; }
.info-card { background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:1rem; margin:0.5rem 0; font-size:0.88rem; line-height:1.6; color:#374151; }
.info-card strong { color:#111827; }
.advice-item { background:#f0fdf4; border-left:3px solid #22c55e; border-radius:0 8px 8px 0; padding:0.7rem 1rem; margin:0.4rem 0; font-size:0.9rem; color:#1f2937; }
.advice-item.warning { background:#fffbeb; border-left-color:#f59e0b; }
.advice-item.critical { background:#fef2f2; border-left-color:#ef4444; }
.stButton>button { min-height:48px; font-weight:600; border-radius:10px; }
hr { border:none; border-top:1px solid #e5e7eb; margin:1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Theme-aware color palette (Flaw 9 fix) ──────────────────────────────────
# Centralized design tokens instead of ad-hoc hex strings scattered across the file.
THEME = {
    "primary": "#16a34a",       # Green — healthy / low-risk / positive
    "primary_dark": "#14532d",  # Dark green — chart gradient endpoint
    "primary_light": "#d1fae5", # Light green — chart gradient start / backgrounds
    "primary_fill": "rgba(22,163,74,0.08)",  # Transparent green for area fills
    "warning": "#d97706",       # Amber — warnings / forecast markers
    "warning_bg": "#fef3c7",   # Amber background
    "danger": "#ef4444",        # Red — critical alerts / high-risk
    "danger_bg": "#fee2e2",    # Red background
    "success_bg": "#dcfce7",   # Green background for status pills
    "text": "#111827",          # Primary text
    "text_secondary": "#6b7280",  # Muted labels
    "border": "#e5e7eb",       # Borders / dividers
    "surface": "#f9fafb",      # Card backgrounds
    "marker_selected": "#1f2937",  # Map marker — selected region
    "marker_default": "#9ca3af",   # Map marker — unselected
}

# ── User-friendly label mapping (Flaw 8 fix) ─────────────────────────────────
# Maps internal system keys to human-readable labels so the UI never exposes
# raw model/config terminology to end users.
LABEL_MAP = {
    # Attribution modalities
    "Weather": "🌦️ Weather Influence",
    "Satellite": "🛰️ Vegetation & Biomass",
    "Soil": "🌱 Soil Properties",
    # Status keys
    "live_ready": "Forecast Ready",
    "feature_store_ready": "Data Available",
    "model_ready": "Model Loaded",
    # Risk levels
    "LOW": "✅ Low Risk",
    "MODERATE": "⚠️ Moderate Risk",
    "HIGH": "🚨 High Risk",
}

def _friendly(key: str) -> str:
    """Return the user-friendly label for an internal key, or the key itself."""
    return LABEL_MAP.get(key, key)

# ── Audience view modes (Flaw 7 fix) ─────────────────────────────────────────
# Different stakeholders need different information density:
#   Farmer    → simplified metrics, plain-language advice, no internals
#   Analyst   → full technical output, attribution charts, GMM diagnostics
#   Planner   → aggregate trends, risk heatmap, policy-level recommendations
VIEW_MODES = ["🌾 Farmer", "📊 Analyst", "🏛️ Policy-Maker"]

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 Crop Intelligence")
    view_mode = st.selectbox("View Mode", VIEW_MODES, help="Choose your role to see the most relevant information.")
    region = st.selectbox("Region", REGIONS)
    year = st.selectbox("Year", YEARS if YEARS else [2023])
    
    st.markdown("---")
    st.markdown("🌍 **Offline Workspace**")
    st.markdown("To compute yield forecasts locally and offline, use the standalone edge PWA:")
    st.markdown("[**Open Offline Standalone App**](http://localhost:8000/)")

context = build_region_context(region, int(year), CONFIG)
result_key = f"{region}:{year}"

with st.sidebar:
    st.markdown("---")
    if context["live_ready"]:
        sc, sl = "green", "Ready"
    elif context["feature_store_ready"]:
        sc, sl = "amber", "Partial"
    else:
        sc, sl = "red", "Unavailable"
    st.markdown(f'<span class="status-pill {sc}">{sl}</span>', unsafe_allow_html=True)
    yrs = ", ".join(str(y) for y in context["feature_years"]) or "none"
    st.markdown(f'<div class="info-card"><strong>Features:</strong> {"Yes" if context["feature_store_ready"] else "No"}<br><strong>Checkpoint:</strong> {"Yes" if context.get("model_ready") else "No"}<br><strong>Years:</strong> {yrs}</div>', unsafe_allow_html=True)
    run_live = st.button("Run Forecast", use_container_width=True, type="primary", disabled=not context["live_ready"])

if run_live:
    try:
        st.session_state["live_results"][result_key] = run_inference(region=region, year=int(year))
        # Clear the streamlit cache and dynamically regenerate to capture any new inputs
        generate_offline_features_json.clear()
        generate_offline_features_json()
    except InferenceUnavailableError as exc:
        st.session_state["live_results"].pop(result_key, None)
        st.error(str(exc))

prediction = st.session_state["live_results"].get(result_key)
active_ndvi = prediction["ndvi_series"] if prediction else context["ndvi_series"]

# ── Header ───────────────────────────────────────────────────────────────────
st.title(f"{region}")
st.caption(f"Yield forecast workspace · {year}")

# ── Metrics (THE focal point) ────────────────────────────────────────────────
if prediction:
    st.success("Live forecast from checkpoint + processed feature store.")
    if prediction.get("modality_warnings"):
        for warn_msg in prediction["modality_warnings"]:
            st.warning(f"⚠️ {warn_msg}")
    mc = st.columns(4)
    mc[0].metric("Predicted Yield", f"{prediction['predicted_yield']:.2f} t/ha")
    mc[1].metric("95% Confidence", f"{prediction['lower_bound']:.2f} – {prediction['upper_bound']:.2f}")
    mc[2].metric("Risk Level", prediction["risk"])
    mc[3].metric("vs. Historical", f"{context['historical_average']:.2f} t/ha" if context["historical_average"] else "n/a")
elif context["live_ready"]:
    st.info("Data ready. Press **Run Forecast** to generate a prediction.")
    mc = st.columns(3)
    mc[0].metric("Historical Avg", f"{context['historical_average']:.2f} t/ha" if context["historical_average"] else "n/a")
    mc[1].metric(f"Observed ({year})", f"{context['observed_yield']:.2f} t/ha" if context["observed_yield"] else "n/a")
    mc[2].metric("Status", "Ready")
else:
    st.warning(context["status"])
    mc = st.columns(3)
    mc[0].metric("Historical Avg", f"{context['historical_average']:.2f} t/ha" if context["historical_average"] else "n/a")
    mc[1].metric(f"Observed ({year})", f"{context['observed_yield']:.2f} t/ha" if context["observed_yield"] else "n/a")
    mc[2].metric("Status", "Unavailable")

# ── Bimodal Distribution Alert ──────────────────────────────────────────────
if prediction:
    br = prediction.get("bimodality_report", {})
    if br.get("is_bimodal"):
        modes_text = " vs. ".join(
            f"**{m:.2f} t/ha** ({w:.0%} probability)"
            for w, m in br.get("modes", [])
        )
        st.markdown(
            f'<div class="advice-item warning">'
            f'<strong>⚠ Two Distinct Scenarios Detected (valley depth={br["valley_depth"]:.2f})</strong><br>'
            f'The model sees two plausible but very different outcomes: {modes_text}. '
            f'The displayed forecast uses the dominant scenario. '
            f'Investigate satellite and weather signals separately before acting.'
            f'</div>',
            unsafe_allow_html=True,
        )
        
        # Actionable GMM PDF Diagnostic Visualization
        gmm = prediction.get("gmm_params")
        if gmm:
            import math
            import numpy as np
            pi_list = gmm["pi"]
            sigma_list = gmm["sigma"]
            mu_list = gmm["mu"]
            
            # Create a fine grid for plotting from 0.0 to 12.0 t/ha
            grid_y = np.linspace(0.0, 12.0, 200)
            grid_pdf = []
            for y in grid_y:
                pdf_val = 0.0
                for w, s, m in zip(pi_list, sigma_list, mu_list):
                    exponent = -((y - m) ** 2) / (2 * (s ** 2))
                    coeff = w / (s * math.sqrt(2 * math.pi))
                    pdf_val += coeff * math.exp(exponent)
                grid_pdf.append(pdf_val)
                
            pdf_df = pd.DataFrame({"Yield (t/ha)": grid_y, "Probability Density": grid_pdf})
            fig_pdf = px.area(
                pdf_df,
                x="Yield (t/ha)",
                y="Probability Density",
                title="Crop Yield Probability Distribution (GMM PDF Diagnostic)",
                template="plotly_white"
            )
            fig_pdf.update_traces(
                line=dict(color="#d97706", width=2),
                fillcolor="rgba(217,119,6,0.1)"
            )
            # Vertical line for prediction point estimate (dominant mode)
            fig_pdf.add_vline(
                x=prediction["predicted_yield"],
                line_dash="dash",
                line_color="#111827",
                annotation_text="Dominant Mode"
            )
            # Vertical lines for each mode
            for w, m in br.get("modes", []):
                fig_pdf.add_vline(
                    x=m,
                    line_dash="dot",
                    line_color="#f59e0b",
                    annotation_text=f"Mode ({w:.0%})"
                )
            fig_pdf.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                font=dict(family="Inter, -apple-system, sans-serif")
            )
            st.plotly_chart(fig_pdf, use_container_width=True)

st.markdown("---")

# ── Charts + Insights ────────────────────────────────────────────────────────
left, right = st.columns([1.6, 1.0])

with left:
    st.subheader("Yield Trend")
    if context["yield_history"].empty:
        st.info("No historical yield data for this region.")
    else:
        hdf = context["yield_history"].sort_values("year")
        fig = px.line(hdf, x="year", y="yield", markers=True, template="plotly_white")
        fig.update_traces(line=dict(color="#16a34a", width=2.5), marker=dict(color="#16a34a", size=8))
        if prediction:
            fig.add_trace(go.Scatter(x=[prediction["year"]], y=[prediction["predicted_yield"]], mode="markers", marker=dict(size=14, color="#d97706", symbol="diamond"), name="Forecast"))
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), xaxis_title="Year", yaxis_title="Yield (t/ha)", font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Vegetation Index (NDVI)")
    if active_ndvi:
        ndf = pd.DataFrame({"step": list(range(1, len(active_ndvi)+1)), "ndvi": active_ndvi})
        fn = px.area(ndf, x="step", y="ndvi", template="plotly_white")
        fn.update_traces(line=dict(color="#16a34a", width=2), fillcolor="rgba(22,163,74,0.08)")
        fn.add_hline(y=0.3, line_dash="dot", line_color="#d97706", annotation_text="Stress threshold")
        fn.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0), xaxis_title="Time Step", yaxis_title="NDVI", font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"))
        st.plotly_chart(fn, use_container_width=True)
    else:
        st.info("No NDVI series available for this region/year.")

with right:
    if prediction:
        # Attribution chart — use friendly labels (Flaw 8 fix) and theme colors (Flaw 9 fix)
        if view_mode != "🌾 Farmer":  # Farmers see simplified advice only, not technical attribution
            st.subheader("What Drove This Forecast")
            raw_attr = prediction["attribution"]
            friendly_attr = {_friendly(k): v for k, v in raw_attr.items()}
            adf = pd.DataFrame({"Factor": list(friendly_attr.keys()), "Impact": list(friendly_attr.values())}).sort_values("Impact")
            fa = px.bar(adf, x="Impact", y="Factor", orientation="h", template="plotly_white", color="Impact", color_continuous_scale=[THEME["primary_light"], THEME["primary"], THEME["primary_dark"]])
            fa.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False, font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"))
            st.plotly_chart(fa, use_container_width=True)

        st.subheader("Recommendations")
        for adv in prediction.get("recommendations", []):
            cc = "critical" if any(k in adv.lower() for k in ["emergency","🚨"]) else "warning" if any(k in adv.lower() for k in ["warning","⚠️","volatility"]) else ""
            st.markdown(f'<div class="advice-item {cc}">{adv}</div>', unsafe_allow_html=True)
    else:
        st.subheader("Getting Started")
        st.markdown('<div class="info-card">This dashboard does not fabricate predictions.<br><br><strong>To see a forecast:</strong><br>1. Select a region with processed data<br>2. Choose a year covered by the feature store<br>3. Press <strong>Run Forecast</strong></div>', unsafe_allow_html=True)

# ── Map ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Regional Overview")
if YIELD_HISTORY.empty:
    st.info("No historical yield records available for the map.")
else:
    ms = YIELD_HISTORY.groupby("site_id", as_index=False)["yield"].mean().rename(columns={"yield": "map_yield"})
    hist_averages = YIELD_HISTORY.groupby("site_id")["yield"].mean().to_dict()
    if prediction:
        ms.loc[ms["site_id"] == region, "map_yield"] = prediction["predicted_yield"]
    rl = {a["name"]: a for a in CONFIG.get("study_areas", [])}
    rows = []
    for _, r in ms.iterrows():
        a = rl.get(r["site_id"])
        if not a: continue
        rows.append({"site_id": r["site_id"], "yv": float(r["map_yield"]), "lat": a.get("lat"), "lon": a.get("lon")})
    if rows:
        mdf = pd.DataFrame(rows)
        ctr = mdf.loc[mdf["site_id"] == region].iloc[0]
        
        # Track region change to reset center/zoom, otherwise preserve client-side zoom/center state
        if "prev_map_region" not in st.session_state or st.session_state["prev_map_region"] != region:
            st.session_state["prev_map_region"] = region
            st.session_state["map_center"] = [ctr["lat"], ctr["lon"]]
            st.session_state["map_zoom"] = 5
            
        fm = folium.Map(
            location=st.session_state["map_center"],
            zoom_start=st.session_state["map_zoom"],
            tiles="CartoDB positron"
        )
        for _, r in mdf.iterrows():
            sel = r["site_id"] == region
            
            # Map marker color uses centralized THEME tokens (Flaw 9 fix)
            hist_avg = hist_averages.get(r["site_id"])
            if hist_avg:
                deviation = (hist_avg - r["yv"]) / hist_avg
                if deviation > 0.5:
                    fill_color = THEME["danger"]
                elif deviation > 0.2:
                    fill_color = THEME["warning"]
                else:
                    fill_color = THEME["primary"]
            else:
                fill_color = THEME["primary"]
                
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=18 if sel else 13,
                color="#1f2937" if sel else "#9ca3af",
                weight=3 if sel else 1,
                fill=True,
                fill_color=fill_color,
                fill_opacity=0.9,
                tooltip=f"{r['site_id']}: {r['yv']:.2f} t/ha"
            ).add_to(fm)
            
        map_out = st_folium(
            fm,
            use_container_width=True,
            height=400,
            key="regional_overview_map",
            returned_objects=["zoom", "center"]
        )
        
        # Save center and zoom state back into session state if panned/zoomed
        if map_out:
            if map_out.get("center"):
                st.session_state["map_center"] = [map_out["center"]["lat"], map_out["center"]["lng"]]
            if map_out.get("zoom"):
                st.session_state["map_zoom"] = map_out["zoom"]
    else:
        st.info("Could not populate the map from current configuration.")

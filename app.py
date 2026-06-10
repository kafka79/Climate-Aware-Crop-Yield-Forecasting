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
/* ── Offline indicator banner ── */
#offline-banner {
  display:none; position:fixed; top:0; left:0; right:0; z-index:9999;
  background:#92400e; color:#fffbeb; text-align:center;
  padding:0.5rem 1rem; font-size:0.88rem; font-weight:600;
}
/* ── Offline overlay to block interaction ── */
#offline-overlay {
  display:none; position:fixed; top:0; left:0; right:0; bottom:0; z-index:9998;
  background:rgba(255,255,255,0.5); backdrop-filter:blur(2px);
  cursor:not-allowed;
}
</style>

<!-- ── Connectivity Monitor ────────────────────────────────────────────────
     Streamlit is a server-side framework: every widget interaction sends a
     WebSocket message to the Python backend.  A Service Worker that caches
     the HTML shell is pointless — the app freezes on the first interaction
     without a live connection.

     Instead of faking interactive offline capability, we prevent broken actions:
     1. Display a top warning banner.
     2. Dim the interface with a blur overlay to block mouse and touch clicks.
     3. Explicitly disable HTML buttons and select options.
-->
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#16a34a">

<!-- Offline overlay & banner elements -->
<div id="offline-overlay"></div>
<div id="offline-banner">
  ⚠ Connection lost — Streamlit requires a live server connection to function.
  Forecasts and controls are paused. You can open the <a href="/app/static/index.html" target="_blank" style="color: #fef3c7; text-decoration: underline; font-weight: 700;">Offline Yield Forecast Tool</a> to run predictions locally via ONNX in your browser.
</div>

<script>
// Register Service Worker for offline PWA capabilities
if ('serviceWorker' in navigator) {
  window.addEventListener('load', function() {
    navigator.serviceWorker.register('/app/static/sw.js', { scope: '/app/static/' })
      .then(function(reg) {
        console.log('ServiceWorker registration successful with scope: ', reg.scope);
      }, function(err) {
        console.log('ServiceWorker registration failed: ', err);
      });
  });
}

function checkServerConnection() {
  var banner = document.getElementById('offline-banner');
  var overlay = document.getElementById('offline-overlay');
  
  // Check client-side navigator.onLine first
  if (!navigator.onLine) {
    showOffline(banner, overlay);
    return;
  }
  
  // Make a light HEAD request to verify the server is responsive
  fetch(window.location.href, { method: 'HEAD', cache: 'no-store' })
    .then(function(response) {
      if (response.ok) {
        if (banner) banner.style.display = 'none';
        if (overlay) overlay.style.display = 'none';
        toggleInputs(false);
      } else {
        showOffline(banner, overlay);
      }
    })
    .catch(function() {
      showOffline(banner, overlay);
    });
}

function showOffline(banner, overlay) {
  if (banner) banner.style.display = 'block';
  if (overlay) overlay.style.display = 'block';
  toggleInputs(true);
}

function toggleInputs(disabled) {
  // Use HTML5 standard inert attribute on the main app container to robustly block all interactions
  // (clicks, pointer events, keyboard focus, assistive tech) without mutating individual widgets
  // that Streamlit continuously redraws.
  var container = document.querySelector('[data-testid="stAppViewContainer"]');
  if (container) {
    if (disabled) {
      container.setAttribute('inert', '');
    } else {
      container.removeAttribute('inert');
    }
  }
}

// Check on online/offline events for instantaneous response
window.addEventListener('online', checkServerConnection);
window.addEventListener('offline', checkServerConnection);

// Check periodically for server responsiveness (every 15 seconds to save battery/data)
setInterval(checkServerConnection, 15000);
// Run on startup
setTimeout(checkServerConnection, 500);
</script>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 Crop Intelligence")
    region = st.selectbox("Region", REGIONS)
    year = st.selectbox("Year", YEARS if YEARS else [2023])

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
        st.subheader("What Drove This Forecast")
        adf = pd.DataFrame({"Modality": list(prediction["attribution"].keys()), "Score": list(prediction["attribution"].values())}).sort_values("Score")
        fa = px.bar(adf, x="Score", y="Modality", orientation="h", template="plotly_white", color="Score", color_continuous_scale=["#d1fae5","#16a34a","#14532d"])
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
        fm = folium.Map(location=[ctr["lat"], ctr["lon"]], zoom_start=5, tiles="CartoDB positron")
        for _, r in mdf.iterrows():
            sel = r["site_id"] == region
            
            # Map marker color according to deviation from historical average (design tokens: green, amber, red)
            hist_avg = hist_averages.get(r["site_id"])
            if hist_avg:
                deviation = (hist_avg - r["yv"]) / hist_avg
                if deviation > 0.5:
                    fill_color = "#ef4444"  # Red token (High Risk)
                elif deviation > 0.2:
                    fill_color = "#f59e0b"  # Amber token (Medium Risk)
                else:
                    fill_color = "#22c55e"  # Green token (Low Risk)
            else:
                fill_color = "#22c55e"
                
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
        st_folium(fm, use_container_width=True, height=400)
    else:
        st.info("Could not populate the map from current configuration.")

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
</style>

<!-- ── PWA: manifest + iOS meta tags ─────────────────────────────────────────
     [Marco · Apple]: "If a farmer is in a field with 2G connectivity, having
     this cached as a lightweight app would be the 10/10 design victory."
     Streamlit does not support a custom <head>; we inject via markdown.
-->
<link rel="manifest" href="/app/static/manifest.json">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="CropForecast">
<meta name="theme-color" content="#16a34a">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">

<!-- Offline banner (shown by SW when network is unavailable) -->
<div id="offline-banner">
  📡 You are offline — showing your last cached forecast. Results may be outdated.
</div>

<script>
// ── Register Service Worker ────────────────────────────────────────────────
if ('serviceWorker' in navigator) {
  window.addEventListener('load', function () {
    navigator.serviceWorker.register('/app/static/sw.js', { scope: '/' })
      .then(function (reg) {
        console.log('[PWA] Service worker registered. Scope:', reg.scope);
      })
      .catch(function (err) {
        console.warn('[PWA] Service worker registration failed:', err);
      });
  });
}

// ── Offline / Online indicator ─────────────────────────────────────────────
function updateOnlineStatus() {
  var banner = document.getElementById('offline-banner');
  if (banner) banner.style.display = navigator.onLine ? 'none' : 'block';
}
window.addEventListener('online',  updateOnlineStatus);
window.addEventListener('offline', updateOnlineStatus);
updateOnlineStatus();
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
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), xaxis_title="Year", yaxis_title="Yield (t/ha)", font=dict(family="Inter"))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Vegetation Index (NDVI)")
    if active_ndvi:
        ndf = pd.DataFrame({"step": list(range(1, len(active_ndvi)+1)), "ndvi": active_ndvi})
        fn = px.area(ndf, x="step", y="ndvi", template="plotly_white")
        fn.update_traces(line=dict(color="#16a34a", width=2), fillcolor="rgba(22,163,74,0.08)")
        fn.add_hline(y=0.3, line_dash="dot", line_color="#d97706", annotation_text="Stress threshold")
        fn.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0), xaxis_title="Time Step", yaxis_title="NDVI", font=dict(family="Inter"))
        st.plotly_chart(fn, use_container_width=True)
    else:
        st.info("No NDVI series available for this region/year.")

with right:
    if prediction:
        st.subheader("What Drove This Forecast")
        adf = pd.DataFrame({"Modality": list(prediction["attribution"].keys()), "Score": list(prediction["attribution"].values())}).sort_values("Score")
        fa = px.bar(adf, x="Score", y="Modality", orientation="h", template="plotly_white", color="Score", color_continuous_scale=["#d1fae5","#16a34a","#14532d"])
        fa.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False, font=dict(family="Inter"))
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
        lo, hi = mdf["yv"].min(), mdf["yv"].max()
        sc = max(hi - lo, 0.1)
        for _, r in mdf.iterrows():
            ratio = (r["yv"] - lo) / sc
            g, rd = int(100 + 120*ratio), int(180*(1-ratio))
            sel = r["site_id"] == region
            folium.CircleMarker(location=[r["lat"], r["lon"]], radius=18 if sel else 13, color="#1f2937" if sel else "#9ca3af", weight=3 if sel else 1, fill=True, fill_color=f"#{rd:02x}{g:02x}3a", fill_opacity=0.9, tooltip=f"{r['site_id']}: {r['yv']:.2f} t/ha").add_to(fm)
        st_folium(fm, use_container_width=True, height=400)
    else:
        st.info("Could not populate the map from current configuration.")

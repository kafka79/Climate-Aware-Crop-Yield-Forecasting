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

st.set_page_config(
    page_title="Climate-Aware Yield Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIG = load_runtime_config()
REGIONS = list_configured_regions(CONFIG)
YEARS = list_available_years(CONFIG)
YIELD_HISTORY = load_yield_history(CONFIG)

if "live_results" not in st.session_state:
    st.session_state["live_results"] = {}

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f4f7f2 0%, #ffffff 55%, #eef4eb 100%);
    }
    [data-testid="stSidebar"] {
        background: #f7faf5;
    }
    .status-card {
        padding: 0.9rem 1rem;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(30, 41, 59, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Crop Intelligence")
    st.caption("This dashboard only renders live forecasts for region/year pairs backed by real processed artifacts.")
    region = st.selectbox("Region", REGIONS)
    year = st.selectbox("Year", YEARS if YEARS else [2023])

context = build_region_context(region, int(year), CONFIG)
result_key = f"{region}:{year}"

with st.sidebar:
    run_live = st.button(
        "Run live inference",
        use_container_width=True,
        type="primary",
        disabled=not context["live_ready"],
    )
    st.caption(context["status"])
    if context["feature_years"]:
        years_text = ", ".join(str(item) for item in context["feature_years"])
        st.caption(f"Feature store years: {years_text}")
    else:
        st.caption("Feature store years: none")

if run_live:
    try:
        st.session_state["live_results"][result_key] = run_inference(region=region, year=int(year))
    except InferenceUnavailableError as exc:
        st.session_state["live_results"].pop(result_key, None)
        st.error(str(exc))

prediction = st.session_state["live_results"].get(result_key)
active_ndvi = prediction["ndvi_series"] if prediction else context["ndvi_series"]

st.title(f"Yield Forecast Workspace: {region} ({year})")
st.caption(
    "The dashboard shows historical context by default. Forecast metrics only appear after a real checkpoint "
    "runs against a matching processed feature store."
)

if prediction:
    st.success("Live forecast rendered from the stored checkpoint and processed Zarr feature store.")
elif context["live_ready"]:
    st.info("The selected region/year is ready. Run live inference to render a forecast.")
else:
    st.warning(context["status"])

summary_cols = st.columns(4)
summary_cols[0].metric(
    "Historical average",
    f"{context['historical_average']:.2f} t/ha" if context["historical_average"] is not None else "n/a",
)
summary_cols[1].metric(
    f"Observed yield ({year})",
    f"{context['observed_yield']:.2f} t/ha" if context["observed_yield"] is not None else "n/a",
)
summary_cols[2].metric(
    "Feature store",
    "Yes" if context["feature_store_ready"] else "No",
)
summary_cols[3].metric(
    "Live inference",
    "Ready" if context["live_ready"] else "Unavailable",
)

if prediction:
    forecast_cols = st.columns(4)
    forecast_cols[0].metric("Forecast", f"{prediction['predicted_yield']:.2f} t/ha")
    forecast_cols[1].metric(
        "95% interval",
        f"{prediction['lower_bound']:.2f} to {prediction['upper_bound']:.2f}",
    )
    forecast_cols[2].metric("Risk", prediction["risk"])
    forecast_cols[3].metric("Soil input", prediction["soil_source"])

left, right = st.columns([1.7, 1.0])

with left:
    st.subheader("Historical Yield Trend")
    if context["yield_history"].empty:
        st.info("No historical yield file is available for this region yet.")
    else:
        history_df = context["yield_history"].sort_values("year")
        fig_history = px.line(
            history_df,
            x="year",
            y="yield",
            markers=True,
            template="plotly_white",
        )
        fig_history.update_traces(line_color="#2f855a", marker_color="#2f855a")
        if prediction:
            fig_history.add_trace(
                go.Scatter(
                    x=[prediction["year"]],
                    y=[prediction["predicted_yield"]],
                    mode="markers",
                    marker=dict(size=14, color="#d97706", symbol="diamond"),
                    name="Live forecast",
                )
            )
        fig_history.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Year",
            yaxis_title="Yield (t/ha)",
        )
        st.plotly_chart(fig_history, use_container_width=True)

    st.subheader("NDVI Time Series")
    if active_ndvi:
        ndvi_df = pd.DataFrame(
            {
                "step": list(range(1, len(active_ndvi) + 1)),
                "ndvi": active_ndvi,
            }
        )
        fig_ndvi = px.line(
            ndvi_df,
            x="step",
            y="ndvi",
            markers=True,
            template="plotly_white",
        )
        fig_ndvi.update_traces(line_color="#4c956c", marker_color="#4c956c")
        fig_ndvi.add_hline(y=0.3, line_dash="dot", line_color="#b45309")
        fig_ndvi.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Time step",
            yaxis_title="NDVI",
        )
        st.plotly_chart(fig_ndvi, use_container_width=True)
        st.caption("NDVI is computed directly from B08 and B04 in the stored Sentinel-2 feature store.")
    else:
        st.info("No NDVI series is available for the selected region/year.")

with right:
    st.subheader("Data Status")
    st.markdown(
        f"""
        <div class="status-card">
        <strong>Region:</strong> {region}<br>
        <strong>Year:</strong> {year}<br>
        <strong>Status:</strong> {context["status"]}<br>
        <strong>Processed years:</strong> {", ".join(str(item) for item in context["feature_years"]) or "none"}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if prediction:
        st.subheader("Model Attribution by Modality")
        attribution_df = pd.DataFrame(
            {
                "Modality": list(prediction["attribution"].keys()),
                "Score": list(prediction["attribution"].values()),
            }
        ).sort_values("Score")
        fig_attr = px.bar(
            attribution_df,
            x="Score",
            y="Modality",
            orientation="h",
            template="plotly_white",
            color="Score",
            color_continuous_scale=["#c8e6c9", "#6aa84f", "#2f6f4f"],
        )
        fig_attr.update_layout(
            height=260,
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_attr, use_container_width=True)
        top_modality = max(prediction["attribution"], key=prediction["attribution"].get)
        st.caption(f"Highest attribution for this run: {top_modality}.")

        st.subheader("Agronomic Recommendations")
        for advice in prediction.get("recommendations", []):
            st.write(advice)
    else:
        st.subheader("Why No Forecast Yet")
        st.write(
            "This screen does not fabricate predictions. If live inference is unavailable, the dashboard stops at historical context and data readiness."
        )

    st.subheader("Operational Notes")
    notes = [
        "Crop selection has been removed from the dashboard because the committed artifacts are not crop-specific.",
        "Map values come from historical yield records, except the selected region when a live forecast has been computed.",
        "CLI predict mode now fails clearly when artifacts are missing or the checkpoint output is implausible.",
    ]
    for note in notes:
        st.write(f"- {note}")

st.subheader("Regional Yield Map")
if YIELD_HISTORY.empty:
    st.info("Historical yield records are not available, so the map cannot be populated.")
else:
    map_source = (
        YIELD_HISTORY.groupby("site_id", as_index=False)["yield"].mean().rename(
            columns={"yield": "map_yield"}
        )
    )
    if prediction:
        map_source.loc[map_source["site_id"] == region, "map_yield"] = prediction["predicted_yield"]

    region_lookup = {area["name"]: area for area in CONFIG.get("study_areas", [])}
    map_rows = []
    for _, row in map_source.iterrows():
        area = region_lookup.get(row["site_id"])
        if not area:
            continue
        map_rows.append(
            {
                "site_id": row["site_id"],
                "yield_value": float(row["map_yield"]),
                "lat": area.get("lat"),
                "lon": area.get("lon"),
            }
        )

    if map_rows:
        map_df = pd.DataFrame(map_rows)
        center = map_df.loc[map_df["site_id"] == region].iloc[0]
        fmap = folium.Map(
            location=[center["lat"], center["lon"]],
            zoom_start=5,
            tiles="CartoDB positron",
        )
        low = map_df["yield_value"].min()
        high = map_df["yield_value"].max()
        scale = max(high - low, 0.1)

        for _, row in map_df.iterrows():
            ratio = (row["yield_value"] - low) / scale
            red = int(210 * (1 - ratio))
            green = int(135 + (90 * ratio))
            color = f"#{red:02x}{green:02x}55"
            selected = row["site_id"] == region
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=16 if selected else 12,
                color="#1f2937",
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                tooltip=f"{row['site_id']}: {row['yield_value']:.2f} t/ha",
                popup=(
                    f"<b>{row['site_id']}</b><br>"
                    f"Yield value: {row['yield_value']:.2f} t/ha<br>"
                    f"{'Selected region' if selected else 'Historical average'}"
                ),
            ).add_to(fmap)

        st_folium(fmap, width="100%", height=420)
    else:
        st.info("The map could not be populated from the current study area configuration.")

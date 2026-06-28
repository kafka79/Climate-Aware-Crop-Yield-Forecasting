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

# ── Apple-grade Design System & Offline Workspace ─────────────────────────────
# White canvas, Inter font, high-contrast for outdoor screens, zero noise.
st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

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

/* ── Inline Offline Workspace Styles ── */
#offline-workspace {
  display: none;
  background: #ffffff;
  font-family: 'Inter', sans-serif;
  color: #111827;
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}
#offline-workspace h1 {
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin-bottom: 0.5rem;
}
#offline-workspace .offline-banner {
  background: #92400e;
  color: #fffbeb;
  padding: 0.75rem 1rem;
  border-radius: 8px;
  font-size: 0.88rem;
  font-weight: 600;
  margin-bottom: 2rem;
}
#offline-workspace .offline-grid {
  display: grid;
  grid-template-columns: 1.2fr 1.8fr;
  gap: 2.5rem;
}
@media (max-width: 900px) {
  #offline-workspace .offline-grid {
    grid-template-columns: 1fr;
  }
}
#offline-workspace .card {
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}
#offline-workspace .card h2 {
  font-size: 1.2rem;
  font-weight: 600;
  margin-bottom: 1.25rem;
  border-left: 4px solid #16a34a;
  padding-left: 0.5rem;
}
#offline-workspace .form-group {
  margin-bottom: 1rem;
}
#offline-workspace label {
  display: block;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  color: #6b7280;
  margin-bottom: 0.25rem;
}
#offline-workspace select, #offline-workspace input {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #fff;
  font-family: inherit;
  font-size: 0.95rem;
}
#offline-workspace .row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
  gap: 1rem;
  margin-bottom: 1rem;
}
#offline-workspace button {
  width: 100%;
  padding: 1rem;
  background: #16a34a;
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
}
#offline-workspace button:hover {
  background: #14532d;
}
#offline-workspace button:disabled {
  background: #e5e7eb;
  color: #9ca3af;
  cursor: not-allowed;
}
#offline-workspace .metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
}
#offline-workspace .metric-box {
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 1rem;
  text-align: center;
}
#offline-workspace .metric-val {
  font-size: 1.5rem;
  font-weight: 700;
}
#offline-workspace .metric-label {
  font-size: 0.7rem;
  color: #6b7280;
  text-transform: uppercase;
  font-weight: 600;
  margin-top: 0.25rem;
}
#offline-workspace .alert-box {
  background: #fef2f2;
  border-left: 4px solid #ef4444;
  padding: 1rem;
  border-radius: 0 8px 8px 0;
  color: #991b1b;
  font-size: 0.9rem;
  margin-bottom: 1.5rem;
  display: none;
}
#offline-workspace .alert-box.warning {
  background: #fffbeb;
  border-left-color: #f59e0b;
  color: #92400e;
}
#offline-workspace .chart-container {
  height: 300px;
  position: relative;
  width: 100%;
}
#offline-workspace .advice-container {
  margin-top: 1.5rem;
}
#offline-workspace .advice-container h3 {
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 0.75rem;
}
#offline-workspace .advice-item {
  background: #f0fdf4;
  border-left: 3px solid #22c55e;
  border-radius: 0 6px 6px 0;
  padding: 0.6rem 0.8rem;
  margin-bottom: 0.5rem;
  font-size: 0.88rem;
}
#offline-workspace .advice-item.warning {
  background: #fffbeb;
  border-left-color: #f59e0b;
}
#offline-workspace .advice-item.critical {
  background: #fef2f2;
  border-left-color: #ef4444;
}
</style>

<!-- Inlined Offline Workspace DOM Structure -->
<div id="offline-workspace">
  <div class="offline-banner">
    ⚠️ Connection Lost — You are operating in Offline Workspace mode. 
    Predictions are computed client-side in WebAssembly using your browser's local resources.
  </div>
  
  <div style="margin-bottom: 2rem;">
    <h1>Local Workspace Prediction</h1>
    <p style="color: #6b7280;">Adjust canopy, weather, and soil values to run simulations locally.</p>
  </div>
  
  <div class="offline-grid">
    <div>
      <div class="card">
        <h2>Modal Parameters</h2>
        <form id="offline-form">
          <div class="form-group">
            <label for="off-region">Region Profile</label>
            <select id="off-region">
              <option value="__manual__">✏️ Manual Input</option>
            </select>
          </div>
          
          <div class="form-group">
            <label for="off-preset">Canopy Condition Preset</label>
            <select id="off-preset">
              <option value="dense" selected>Dense Green / High Biomass</option>
              <option value="normal">Normal Green (Average)</option>
              <option value="pale">Pale Green (Mild Stress)</option>
              <option value="yellow">Yellow/Dry (Severe Drought)</option>
              <option value="soil">Bare Soil</option>
              <option value="custom">Custom</option>
            </select>
          </div>
          
          <div class="row">
            <div class="form-group">
              <label for="off-b02">B02 (Blue)</label>
              <input type="number" id="off-b02" value="0.03" step="0.01" min="0" max="1">
            </div>
            <div class="form-group">
              <label for="off-b03">B03 (Green)</label>
              <input type="number" id="off-b03" value="0.08" step="0.01" min="0" max="1">
            </div>
            <div class="form-group">
              <label for="off-b04">B04 (Red)</label>
              <input type="number" id="off-b04" value="0.02" step="0.01" min="0" max="1">
            </div>
            <div class="form-group">
              <label for="off-b08">B08 (NIR)</label>
              <input type="number" id="off-b08" value="0.75" step="0.01" min="0" max="1">
            </div>
            <div class="form-group">
              <label for="off-scl">SCL (Veg)</label>
              <input type="number" id="off-scl" value="0.9" step="0.1" min="0" max="1">
            </div>
          </div>
          
          <label>Weather (ERA5)</label>
          <div class="row">
            <div class="form-group">
              <input type="number" id="off-tmax" value="32.5" step="0.1">
              <span style="font-size: 0.7rem; color: #6b7280;">Max Temp (°C)</span>
            </div>
            <div class="form-group">
              <input type="number" id="off-tmin" value="23.2" step="0.1">
              <span style="font-size: 0.7rem; color: #6b7280;">Min Temp (°C)</span>
            </div>
            <div class="form-group">
              <input type="number" id="off-precip" value="12.4" step="0.1">
              <span style="font-size: 0.7rem; color: #6b7280;">Precip (mm)</span>
            </div>
          </div>
          
          <label>Soil (ISRIC)</label>
          <div class="row">
            <div class="form-group">
              <input type="number" id="off-ph" value="6.2" step="0.1" min="0" max="14">
              <span style="font-size: 0.7rem; color: #6b7280;">pH</span>
            </div>
            <div class="form-group">
              <input type="number" id="off-soc" value="12.5" step="0.1">
              <span style="font-size: 0.7rem; color: #6b7280;">SOC (g/kg)</span>
            </div>
            <div class="form-group">
              <input type="number" id="off-nit" value="1.4" step="0.1">
              <span style="font-size: 0.7rem; color: #6b7280;">N (g/kg)</span>
            </div>
          </div>
          
          <button type="submit" id="off-run-btn" disabled>Loading ONNX Local model...</button>
          <div id="off-model-status" style="font-size: 0.8rem; text-align: center; margin-top: 0.5rem; color: #6b7280;">
            Initializing WebAssembly runtime...
          </div>
        </form>
      </div>
    </div>
    
    <div>
      <div class="card">
        <h2>Local Predictions</h2>
        
        <div class="alert-box" id="off-alert-box"></div>
        
        <div class="metrics">
          <div class="metric-box">
            <div class="metric-val" id="off-yield-val">—</div>
            <div class="metric-label">Predicted Mean</div>
          </div>
          <div class="metric-box">
            <div class="metric-val" id="off-std-val">—</div>
            <div class="metric-label">Std Deviation</div>
          </div>
          <div class="metric-box">
            <div class="metric-val" id="off-dist-val">Unimodal</div>
            <div class="metric-label">Mode Type</div>
          </div>
        </div>
        
        <h2>GMM Distribution PDF</h2>
        <div class="chart-container">
          <canvas id="off-chart"></canvas>
        </div>
        
        <div class="advice-container" id="off-advice-box">
          <!-- Actionable recommendations will be added here -->
        </div>
      </div>
    </div>
  </div>
</div>

<script>
let offSession = null;
let offChart = null;
let offRegionalData = null;

// Initialize local ONNX runtime
async function initOffOnnx() {
  const statusDiv = document.getElementById("off-model-status");
  const runBtn = document.getElementById("off-run-btn");
  try {
    statusDiv.innerText = "Initializing WebAssembly local session...";
    try {
      offSession = await ort.InferenceSession.create('/app/static/model.onnx');
    } catch (err) {
      offSession = await ort.InferenceSession.create('static/model.onnx');
    }
    statusDiv.innerText = "Local Wasm model loaded successfully.";
    runBtn.removeAttribute("disabled");
    runBtn.innerText = "Compute Local Forecast";
  } catch (err) {
    console.error("ONNX Load error (offline):", err);
    statusDiv.innerHTML = "<span style='color:#dc2626'>⚠️ Local model not loaded. Check static/model.onnx exists.</span>";
  }
  
  await loadOffRegionalFeatures();
}

async function loadOffRegionalFeatures() {
  const regionSelect = document.getElementById("off-region");
  const manualOption = '<option value="__manual__" selected>✏️ Manual Input</option>';
  try {
    let response;
    try {
      response = await fetch('/app/static/regional_features.json');
    } catch (err) {
      response = await fetch('static/regional_features.json');
    }
    if (response.ok) {
      const data = await response.json();
      offRegionalData = data;
      const keys = Object.keys(data);
      if (keys.length > 0) {
        regionSelect.innerHTML = manualOption;
        for (const name of keys) {
          const opt = document.createElement("option");
          opt.value = name;
          opt.innerText = `${name} (Year ${data[name].year})`;
          regionSelect.appendChild(opt);
        }
      }
    }
  } catch (err) {
    console.warn("Failed to load regional profiles for offline container:", err);
  }
  
  regionSelect.addEventListener("change", (e) => {
    const val = e.target.value;
    if (val === "__manual__") return;
    if (offRegionalData && offRegionalData[val]) {
      const rd = offRegionalData[val];
      document.getElementById("off-b02").value = rd.satellite.b02;
      document.getElementById("off-b03").value = rd.satellite.b03;
      document.getElementById("off-b04").value = rd.satellite.b04;
      document.getElementById("off-b08").value = rd.satellite.b08;
      document.getElementById("off-scl").value = rd.satellite.scl;
      
      document.getElementById("off-tmax").value = rd.weather.tmax;
      document.getElementById("off-tmin").value = rd.weather.tmin;
      document.getElementById("off-precip").value = rd.weather.precip;
      
      document.getElementById("off-ph").value = rd.soil.ph;
      document.getElementById("off-soc").value = rd.soil.soc;
      document.getElementById("off-nit").value = rd.soil.nitrogen;
      
      document.getElementById("off-preset").value = "custom";
    }
  });
}

// Preset handler
const OFF_PRESETS = {
  dense:  { b02: 0.03, b03: 0.08, b04: 0.02, b08: 0.75, scl: 0.9 },
  normal: { b02: 0.04, b03: 0.09, b04: 0.04, b08: 0.55, scl: 0.7 },
  pale:   { b02: 0.05, b03: 0.11, b04: 0.08, b08: 0.42, scl: 0.5 },
  yellow: { b02: 0.06, b03: 0.14, b04: 0.15, b08: 0.28, scl: 0.3 },
  soil:   { b02: 0.08, b03: 0.12, b04: 0.16, b08: 0.18, scl: 0.1 }
};
document.getElementById("off-preset").addEventListener("change", (e) => {
  const p = OFF_PRESETS[e.target.value];
  if (p) {
    document.getElementById("off-b02").value = p.b02;
    document.getElementById("off-b03").value = p.b03;
    document.getElementById("off-b04").value = p.b04;
    document.getElementById("off-b08").value = p.b08;
    document.getElementById("off-scl").value = p.scl;
  }
});

// GMM PDF evaluator
function gmmPdf(y, pi, sigma, mu) {
  let pdf = 0;
  const numComponents = pi.length;
  for (let k = 0; k < numComponents; k++) {
    const w = pi[k];
    const s = sigma[k];
    const m = mu[k];
    const exponent = -Math.pow(y - m, 2) / (2 * Math.pow(s, 2));
    const coeff = w / (s * Math.sqrt(2 * Math.PI));
    pdf += coeff * Math.exp(exponent);
  }
  return pdf;
}

function getWeightedMean(pi, mu) {
  let sum = 0;
  for (let i = 0; i < pi.length; i++) {
    sum += pi[i] * mu[i];
  }
  return sum;
}

function getWeightedStd(pi, sigma, mu, mean) {
  let secondMoment = 0;
  for (let i = 0; i < pi.length; i++) {
    secondMoment += pi[i] * (Math.pow(sigma[i], 2) + Math.pow(mu[i], 2));
  }
  const variance = Math.max(secondMoment - Math.pow(mean, 2), 1e-6);
  return Math.sqrt(variance);
}

function generateOfflineAdvice(predicted, lower, upper, risk, topFactor, attr) {
  let advice = [];
  if (topFactor === "Weather") {
    advice.push("🌦️ <strong>Weather Dominance:</strong> High sensitivity to weather variations. If experiencing excessive seasonal precipitation, clear peripheral drainage trenches to prevent crop root rot. Under dry conditions, initiate deficit irrigation cycles and apply organic mulching to retard evaporation.");
  } else if (topFactor === "Satellite") {
    advice.push("🛰️ <strong>Biomass Signal:</strong> Yield is driven by crop vigor (NDVI). To maintain this progress, complete a split-nitrogen top-dressing before panicle initiation, and monitor canopy density closely to apply pest management protocols at the first sign of infestation.");
  } else if (topFactor === "Soil") {
    advice.push("🌱 <strong>Soil Constraints:</strong> Soil composition limits regional yield ceiling. To bypass root absorption constraints, apply a customized foliar spray of micro-nutrients (specifically Zinc and Boron) alongside a targeted mid-season NPK top-dress.");
  }
  
  if (risk.includes("HIGH")) {
    advice.push("🚨 <strong>Emergency Action:</strong> Yield is significantly below trend. Conduct a soil-moisture profile audit and check leaf tissue for nitrogen deficiency. Consider micro-irrigation or nitrogen foliar application if stress is confirmed.");
  } else if (risk.includes("LOW")) {
    advice.push("📈 <strong>Surplus Preparation:</strong> Expected yield is above average. Coordinate storage facility capacity, source drying equipment early to prevent post-harvest mold, and engage local distribution networks to lock in optimal pricing.");
  }
  
  let rangePct = (upper - lower) / Math.max(Math.abs(predicted), 0.01);
  if (rangePct > 0.4) {
    advice.push("⚠️ <strong>Risk Hedging (Data Volatility):</strong> High forecast variance indicates conflicting satellite and weather indicators. Postpone intensive fertilizer applications to avoid wasting inputs. Symmetrically prepare channels—clear drainage path (for potential wet spikes) and check pump readiness (for dry drops).");
  }
  return advice;
}

document.getElementById("off-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!offSession) return;
  
  const satVal = [
    parseFloat(document.getElementById("off-b02").value),
    parseFloat(document.getElementById("off-b03").value),
    parseFloat(document.getElementById("off-b04").value),
    parseFloat(document.getElementById("off-b08").value),
    parseFloat(document.getElementById("off-scl").value)
  ];
  const weatherVal = [
    parseFloat(document.getElementById("off-tmax").value),
    parseFloat(document.getElementById("off-tmin").value),
    parseFloat(document.getElementById("off-precip").value)
  ];
  const soilVal = [
    parseFloat(document.getElementById("off-ph").value),
    parseFloat(document.getElementById("off-soc").value),
    parseFloat(document.getElementById("off-nit").value)
  ];
  
  let T = 12;
  try {
    if (offSession.inputMetadata && offSession.inputMetadata.satellite && offSession.inputMetadata.satellite.dims) {
      const dims = offSession.inputMetadata.satellite.dims;
      if (dims && dims[1] && typeof dims[1] === 'number' && dims[1] > 0) {
        T = dims[1];
      }
    }
  } catch (err) {}

  const satFlattened = new Float32Array(T * 5);
  const weatherFlattened = new Float32Array(T * 3);
  for (let t = 0; t < T; t++) {
    satFlattened.set(satVal, t * 5);
    weatherFlattened.set(weatherVal, t * 3);
  }
  const soilFlattened = new Float32Array(soilVal);
  
  const satTensor = new ort.Tensor('float32', satFlattened, [1, T, 5]);
  const weatherTensor = new ort.Tensor('float32', weatherFlattened, [1, T, 3]);
  const soilTensor = new ort.Tensor('float32', soilFlattened, [1, 3]);
  
  try {
    const feeds = { satellite: satTensor, weather: weatherTensor, soil: soilTensor };
    const results = await offSession.run(feeds);
    
    let piData = Array.from(results.pi.data);
    let sigmaData = Array.from(results.sigma.data);
    let muData = Array.from(results.mu.data);
    
    const numMixtures = piData.length;
    const mean = getWeightedMean(piData, muData);
    const std = getWeightedStd(piData, sigmaData, muData, mean);
    
    const weightThreshold = 0.20;
    const separationThreshold = 1.5;
    let significant = [];
    for (let k = 0; k < numMixtures; k++) {
      if (piData[k] >= weightThreshold) {
        significant.push({ w: piData[k], m: muData[k], s: sigmaData[k] });
      }
    }
    significant.sort((a, b) => b.w - a.w);
    
    let isBimodal = false;
    let finalYield = mean;
    let warningBox = document.getElementById("off-alert-box");
    let distType = document.getElementById("off-dist-val");
    let risk = "Low Risk";
    if (finalYield < 2.5) risk = "High Risk";
    else if (finalYield < 4.0) risk = "Medium Risk";
    
    warningBox.style.display = "none";
    distType.innerText = "Unimodal";
    distType.style.color = "#111827";
    document.getElementById("off-yield-val").style.color = "#111827";
    
    if (significant.length >= 2) {
      const m1 = significant[0];
      const m2 = significant[1];
      const pooledSigma = Math.sqrt((m1.w * Math.pow(m1.s, 2) + m2.w * Math.pow(m2.s, 2)) / (m1.w + m2.w)) + 1e-8;
      const separation = Math.abs(m1.m - m2.m) / pooledSigma;
      
      if (separation >= separationThreshold) {
        isBimodal = true;
        finalYield = m1.m;
        const valleyDepth = (1.0 - Math.abs(m1.w - m2.w) / (m1.w + m2.w)).toFixed(2);
        distType.innerText = "Bimodal Risk";
        distType.style.color = "#ef4444";
        
        warningBox.className = "alert-box warning";
        warningBox.style.display = "block";
        warningBox.innerHTML = `
          <strong>⚠️ Two Distinct Scenarios Detected (valley depth=${valleyDepth})</strong><br>
          The model sees two plausible but very different outcomes: 
          <strong>${m1.m.toFixed(2)} t/ha</strong> (${(m1.w*100).toFixed(0)}% probability) vs. 
          <strong>${m2.m.toFixed(2)} t/ha</strong> (${(m2.w*100).toFixed(0)}% probability). 
          The displayed forecast uses the dominant scenario.
        `;
      }
    }
    
    document.getElementById("off-yield-val").innerText = `${finalYield.toFixed(2)} t/ha`;
    document.getElementById("off-std-val").innerText = `${std.toFixed(2)}`;
    
    // Determine top factor for advice
    let weatherSum = weatherVal.reduce((a,b)=>a+b, 0);
    let soilSum = soilVal.reduce((a,b)=>a+b, 0);
    let topFactor = "Satellite";
    let attr = { "Satellite": 0.4, "Weather": 0.3, "Soil": 0.3 };
    if (weatherSum > 40) {
      topFactor = "Weather";
      attr = { "Satellite": 0.2, "Weather": 0.6, "Soil": 0.2 };
    } else if (soilSum < 10) {
      topFactor = "Soil";
      attr = { "Satellite": 0.2, "Weather": 0.2, "Soil": 0.6 };
    }
    
    const adviceList = generateOfflineAdvice(finalYield, finalYield - std, finalYield + std, risk, topFactor, attr);
    const adviceBox = document.getElementById("off-advice-box");
    adviceBox.innerHTML = "<h3>Actionable Agronomic Advice</h3>";
    for (const adv of adviceList) {
      const div = document.createElement("div");
      div.className = "advice-item";
      if (adv.includes("🚨") || adv.includes("Emergency")) div.className = "advice-item critical";
      else if (adv.includes("⚠️") || adv.includes("Volatility")) div.className = "advice-item warning";
      div.innerHTML = adv;
      adviceBox.appendChild(div);
    }
    
    plotOffGmmPdf(piData, sigmaData, muData);
    
  } catch (err) {
    console.error("Offline execution failed:", err);
    alert("Offline prediction failed: " + err.message);
  }
});

function plotOffGmmPdf(pi, sigma, mu) {
  const yields = [];
  const densities = [];
  for (let y = 0.0; y <= 12.0; y += 0.1) {
    yields.push(parseFloat(y.toFixed(1)));
    densities.push(gmmPdf(y, pi, sigma, mu));
  }
  const ctx = document.getElementById('off-chart').getContext('2d');
  if (offChart) offChart.destroy();
  offChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: yields,
      datasets: [{
        label: 'Density',
        data: densities,
        borderColor: '#16a34a',
        backgroundColor: 'rgba(22, 163, 74, 0.06)',
        fill: true,
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false } },
        y: { beginAtZero: true, grid: { color: 'rgba(229, 231, 235, 0.3)' } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

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
  // Query Streamlit's native health endpoint
  fetch(window.location.origin + '/_stcore/health', { method: 'GET', cache: 'no-store' })
    .then(function(response) {
      if (response.ok) {
        var connectionWarning = document.querySelector('[data-testid="stConnectionStatus"]');
        var isDisconnected = connectionWarning && (
          connectionWarning.textContent.includes("Connecting") || 
          connectionWarning.textContent.includes("Offline")
        );
        
        if (isDisconnected) {
          showOffline();
        } else {
          hideOffline();
        }
      } else {
        showOffline();
      }
    })
    .catch(function() {
      showOffline();
    });
}

function showOffline() {
  var stContainer = document.querySelector('[data-testid="stAppViewContainer"]');
  if (stContainer) {
    stContainer.style.display = 'none';
  }
  
  var offWorkspace = document.getElementById('offline-workspace');
  if (offWorkspace) {
    offWorkspace.style.display = 'block';
    if (!offSession) {
      initOffOnnx();
    }
  }
}

function hideOffline() {
  var stContainer = document.querySelector('[data-testid="stAppViewContainer"]');
  if (stContainer) {
    stContainer.style.display = 'block';
  }
  
  var offWorkspace = document.getElementById('offline-workspace');
  if (offWorkspace) {
    offWorkspace.style.display = 'none';
  }
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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator, model_validator
import torch
import os
import math
from typing import List, Dict, Any
from src.models.mdn import mdn_expected_value, mdn_prune_components
from src.models.transformer import initialize_model, load_model_weights
from src.utils.config import load_config
from src.explainability.integrated_gradients import explain_prediction

app = FastAPI(title="Crop Yield Prediction API", version="1.0.0")

# Load configuration and model at startup
CONFIG_PATH = "configs/model_config.yaml"
MODEL_PATH = "models/checkpoints/best_model.pth"

config = load_config(CONFIG_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Expected dimensions from model config (used for validation)
_EXPECTED_SAT_CHANNELS = config.get("transformer", {}).get("input_dim", 5)
_EXPECTED_WEATHER_FEATURES = config.get("transformer", {}).get("temporal_dim", 3)
_EXPECTED_SOIL_DIM = config.get("transformer", {}).get("soil_dim", 3)

# Global model variable
model = None

@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = initialize_model(config)
        load_model_weights(model, MODEL_PATH, device)
        model.to(device)
        model.eval()
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Prediction endpoints will fail.")

class PredictionRequest(BaseModel):
    """Validated prediction input with domain-aware constraints.

    Enforces the same soil/weather/satellite validity rules as the
    preprocessing pipeline (src.schema.validators) so that out-of-distribution
    inputs cannot bypass validation and cause model hallucination.
    """
    sat: List[List[float]]     # (T, C) — satellite spectral features
    weather: List[List[float]] # (T, F_w) — weather temporal features
    soil: List[float]          # (F_s) — soil static features [ph, soc, nitrogen]

    @field_validator("soil")
    @classmethod
    def validate_soil(cls, v: List[float]) -> List[float]:
        if len(v) != _EXPECTED_SOIL_DIM:
            raise ValueError(
                f"Soil vector must have exactly {_EXPECTED_SOIL_DIM} features "
                f"(ph, soc, nitrogen), got {len(v)}"
            )
        # Validate individual soil features per SoilSchema contract
        ph = v[0]
        if not (0.0 <= ph <= 14.0):
            raise ValueError(f"Soil pH must be between 0.0 and 14.0, got {ph}")
        for i, (name, val) in enumerate(zip(["soc", "nitrogen"], v[1:])):
            if val < 0.0:
                raise ValueError(f"Soil {name} must be non-negative, got {val}")
        # Reject NaN/Inf values
        for i, val in enumerate(v):
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Soil feature at index {i} is NaN or Inf")
        return v

    @field_validator("sat")
    @classmethod
    def validate_sat(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) == 0:
            raise ValueError("Satellite input must have at least 1 timestep")
        for t, row in enumerate(v):
            if len(row) < _EXPECTED_SAT_CHANNELS:
                raise ValueError(
                    f"Satellite timestep {t} has {len(row)} channels, "
                    f"need at least {_EXPECTED_SAT_CHANNELS}"
                )
            for i, val in enumerate(row):
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(
                        f"Satellite value at timestep {t}, channel {i} is NaN or Inf"
                    )
        return v

    @field_validator("weather")
    @classmethod
    def validate_weather(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) == 0:
            raise ValueError("Weather input must have at least 1 timestep")
        all_zero = True
        for t, row in enumerate(v):
            if len(row) < _EXPECTED_WEATHER_FEATURES:
                raise ValueError(
                    f"Weather timestep {t} has {len(row)} features, "
                    f"need at least {_EXPECTED_WEATHER_FEATURES}"
                )
            for i, val in enumerate(row):
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(
                        f"Weather value at timestep {t}, feature {i} is NaN or Inf"
                    )
                if val != 0.0:
                    all_zero = False
        if all_zero:
            raise ValueError(
                "All weather values are zero — this indicates missing data "
                "and would cause the model to produce unreliable predictions"
            )
        return v

    @model_validator(mode="after")
    def validate_temporal_alignment(self) -> "PredictionRequest":
        """Ensure satellite and weather sequences have the same length."""
        if len(self.sat) != len(self.weather):
            raise ValueError(
                f"Satellite and weather sequences must have the same number of "
                f"timesteps: sat has {len(self.sat)}, weather has {len(self.weather)}"
            )
        return self

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

class PredictionResponse(BaseModel):
    yield_prediction: float
    explanation: Dict[str, Any] = None

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    from src.utils.telemetry import TelemetryTracker, log_business_metric
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    with TelemetryTracker("api_predict") as tracker:
        tracker.set_attribute("input_sat_shape", [len(request.sat), len(request.sat[0]) if request.sat else 0])
        tracker.set_attribute("input_weather_shape", [len(request.weather), len(request.weather[0]) if request.weather else 0])
        tracker.set_attribute("input_soil_shape", len(request.soil))
        
        try:
            # Convert input to tensors
            sat = torch.tensor(request.sat, dtype=torch.float32).unsqueeze(0).to(device)
            weather = torch.tensor(request.weather, dtype=torch.float32).unsqueeze(0).to(device)
            soil = torch.tensor(request.soil, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                output = model(sat, weather, soil)
                if isinstance(output, tuple):
                    # Prune negligible mixture components before computing estimate
                    pi, sigma, mu = output
                    pi, sigma, mu = mdn_prune_components(pi, sigma, mu)
                    prediction = mdn_expected_value(pi, sigma, mu).item()
                else:
                    prediction = output.item()
            
            tracker.set_attribute("yield_prediction", float(prediction))
            log_business_metric("api_crop_yield_prediction", float(prediction), "t/ha", {})
            
            # Optional: Generate explanation
            sample = {"sat": sat.squeeze(0), "weather": weather.squeeze(0), "soil": soil.squeeze(0)}
            explanation_summary, _ = explain_prediction(model, sample)
            
            return {
                "yield_prediction": float(prediction),
                "explanation": explanation_summary
            }
        except Exception as e:
            tracker.record_exception(e)
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

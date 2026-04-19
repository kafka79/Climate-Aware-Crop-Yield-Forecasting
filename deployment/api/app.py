from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import os
from typing import List, Dict, Any
from src.models.transformer import initialize_model
from src.utils.config import load_config
from src.explainability.integrated_gradients import explain_prediction

app = FastAPI(title="Crop Yield Prediction API", version="1.0.0")

# Load configuration and model at startup
CONFIG_PATH = "configs/model_config.yaml"
MODEL_PATH = "models/checkpoints/best_model.pth"

config = load_config(CONFIG_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model variable
model = None

@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = initialize_model(config)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Prediction endpoints will fail.")

class PredictionRequest(BaseModel):
    sat: List[List[float]] # (T, C)
    weather: List[List[float]] # (T, F_w)
    soil: List[float] # (F_s)

class PredictionResponse(BaseModel):
    yield_prediction: float
    explanation: Dict[str, Any] = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Yield Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to tensors
        sat = torch.tensor(request.sat, dtype=torch.float32).unsqueeze(0).to(device)
        weather = torch.tensor(request.weather, dtype=torch.float32).unsqueeze(0).to(device)
        soil = torch.tensor(request.soil, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(sat, weather, soil)
            if isinstance(output, tuple):
                # If MDN, pick the mean of the mixture (simplified)
                pi, sigma, mu = output
                prediction = torch.sum(pi * mu, dim=1).item()
            else:
                prediction = output.item()
        
        # Optional: Generate explanation
        sample = {"sat": sat.squeeze(0), "weather": weather.squeeze(0), "soil": soil.squeeze(0)}
        explanation_summary, _ = explain_prediction(model, sample)
        
        return {
            "yield_prediction": float(prediction),
            "explanation": explanation_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import math
import torch
import torch.nn as nn
from loguru import logger
from typing import Dict, Any, Tuple
from src.models.mdn import MixtureDensityNetwork


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encodings.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1)]


class MultiModalTransformer(nn.Module):
    """
    Multi-Modal Transformer for Crop Yield Prediction.
    Fuses Satellite (Spectral), Weather (Temporal), and Soil (Static) data.
    """
    def __init__(self, config: Dict[str, Any]):
        super(MultiModalTransformer, self).__init__()
        self.config = config["transformer"]
        # Input perturbation/jittering regularization to add noise to soil inputs during training.
        self.use_jitter = self.config.get("use_jitter", False)
        self.jitter_noise_scale = self.config.get("jitter_noise_scale", 0.1)

        soil_dim = self.config.get("soil_dim", 3)
        hidden_dim = self.config["hidden_dim"]

        # Satellite encoder: 1D temporal convolution extracts local temporal
        # patterns across spectral channels (NOT spatial super-resolution).
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.config["input_dim"], hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        )
        # Linear projection (residual path) for the satellite encoder
        self.sat_encoder = nn.Linear(self.config["input_dim"], hidden_dim)
        self.weather_encoder = nn.LSTM(self.config["temporal_dim"],
                                      hidden_dim,
                                      batch_first=True)
        self.soil_encoder = nn.Linear(soil_dim, hidden_dim)

        # Transformer Layers (applied to temporal features ONLY)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                  nhead=self.config["num_heads"],
                                                  dropout=self.config["dropout"])
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=self.config["num_layers"])

        # Cross-Modal Attention: satellite queries attend to weather and soil context
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                num_heads=self.config["num_heads"])

        # Static per-feature sensitivity for input perturbation noise scaling.
        sensitivity_vals = self.config.get("soil_sensitivity", [1.0] * soil_dim)
        self.register_buffer("soil_sensitivity", torch.tensor(sensitivity_vals, dtype=torch.float32))

        # Learnable gating parameters initialized to 1.0 for backward compatibility
        self.gate_cross = nn.Parameter(torch.ones(1))
        self.gate_weather = nn.Parameter(torch.ones(1))
        self.gate_soil = nn.Parameter(torch.ones(1))

        # MDN Output Head instead of simple Linear
        self.mdn_head = MixtureDensityNetwork(
            input_dim=hidden_dim,
            num_mixtures=config["mdn"]["num_mixtures"],
            output_dim=config["mdn"]["output_dim"]
        )

    def forward(self, sat, weather, soil):
        """
        sat:     (B, T, C)   - Spectral Features
        weather: (B, T, F_w) - Temporal Features
        soil:    (B, F_s)    - Static Features
        """
        # 1. Encode satellite: temporal conv + linear residual path
        sat_t = sat.transpose(1, 2)                          # (B, C, T)
        sat_conv = self.temporal_conv(sat_t).transpose(1, 2)  # (B, T, D)
        sat_res = self.sat_encoder(sat)                       # (B, T, D)
        sat_enc = sat_conv + sat_res                          # (B, T, D) residual sum

        # 2. Encode weather
        weather_enc, _ = self.weather_encoder(weather)        # (B, T, D)

        # 3. Input perturbation/jittering regularization
        if self.training and self.use_jitter:
            sensitivity = self.soil_sensitivity.abs() + 1e-8  # (F_s,)
            noise = torch.randn_like(soil) * self.jitter_noise_scale * sensitivity
            soil = soil + noise

        # 4. Encode soil
        soil_enc = self.soil_encoder(soil).unsqueeze(1)       # (B, 1, D)

        # 5. Cross-modal attention: satellite queries attend to weather/soil context
        # nn.MultiheadAttention expects (Seq, Batch, Dim)
        sat_q = sat_enc.permute(1, 0, 2)                      # (T, B, D)
        weather_kv = weather_enc.permute(1, 0, 2)             # (T, B, D)
        soil_kv = soil_enc.permute(1, 0, 2)                   # (1, B, D)
        
        # Combine weather and soil into a single context sequence for cross-attention
        context_kv = torch.cat([weather_kv, soil_kv], dim=0)  # (T+1, B, D)
        
        cross_out, _ = self.cross_attn(sat_q, context_kv, context_kv)  # (T, B, D)
        cross_out = cross_out.permute(1, 0, 2)                # (B, T, D)

        # 6. Gated multimodal fusion (preserving temporal alignment)
        # Broadcast soil_enc (B, 1, D) to (B, T, D) during addition
        fused = self.gate_cross * cross_out + self.gate_weather * weather_enc + self.gate_soil * soil_enc  # (B, T, D)
        
        # 7. Transformer self-attention refinement
        fused = fused.permute(1, 0, 2)                        # (T, B, D)
        out = self.transformer_encoder(fused)
        out = out.permute(1, 0, 2)                            # (B, T, D)

        # 8. Global Average Pooling over time
        out = torch.mean(out, dim=1)                          # (B, D)

        # 9. MDN Output
        return self.mdn_head(out)


def initialize_model(config: Dict[str, Any]):
    """
    Factory function to initialize the model from configuration.
    """
    logger.info("Initializing MultiModalTransformer...")
    return MultiModalTransformer(config)


def load_model_weights(model: nn.Module, model_path: str, device: torch.device) -> None:
    """
    Loads model weights, mapping legacy keys (e.g. super_res -> temporal_conv)
    and handling missing layers with strict=False to ensure backward compatibility.
    """
    logger.info(f"Loading weights from {model_path} onto {device}...")
    state_dict = torch.load(model_path, map_location=device)
    
    # Map legacy super_res keys to temporal_conv
    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith("super_res."):
            new_key = key.replace("super_res.", "temporal_conv.")
            state_dict[new_key] = state_dict.pop(key)
            logger.info(f"Mapped legacy weight key: {key} -> {new_key}")
            
    # Load state dict with strict=False
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys in state_dict (initialized randomly): {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys in state_dict (ignored): {unexpected}")


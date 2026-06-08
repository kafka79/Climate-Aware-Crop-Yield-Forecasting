import torch
import torch.nn as nn
from loguru import logger
from typing import Dict, Any, Tuple
from src.models.mdn import MixtureDensityNetwork


class MultiModalTransformer(nn.Module):
    """
    Multi-Modal Transformer for Crop Yield Prediction.
    Fuses Satellite (Spectral), Weather (Temporal), and Soil (Static) data
    using cross-modal attention before self-attention refinement.

    Architecture (post-fix):
    ────────────────────────
    1. Satellite encoder  → (B, T, D)  via 1D conv + linear residual
    2. Weather encoder    → (B, T, D)  via LSTM
    3. Cross-modal attn   → (B, T, D)  satellite queries attend to weather
    4. Temporal self-attn  → (B, 2T, D) refine temporal features ONLY
    5. Temporal pooling   → (B, D)     pool time dimension
    6. Concat soil        → (B, 2D)    static features join AFTER pooling
    7. Fusion MLP         → (B, D)     project fused repr to hidden dim
    8. MDN output head    → GMM params

    Previous version concatenated soil (B, 1, D) into the temporal sequence
    (B, 2T+1, D) and then averaged, diluting soil by a factor of 2T+1.
    """
    def __init__(self, config: Dict[str, Any]):
        super(MultiModalTransformer, self).__init__()
        self.config = config["transformer"]
        self.use_privacy = config.get("use_privacy", False)
        self.epsilon = config.get("privacy_epsilon", 0.1)

        soil_dim = self.config.get("soil_dim", 4)
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

        # Cross-Modal Attention: satellite queries attend to weather keys/values
        # to learn which weather conditions are most relevant for each spectral timestep.
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                num_heads=self.config["num_heads"])

        # Fusion MLP: projects concatenated [temporal_pool ‖ soil] → hidden_dim
        # so the MDN head sees a properly weighted combination.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config["dropout"]),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Static per-feature sensitivity for differential-privacy noise scaling.
        # This prevents the model from driving the sensitivity to zero via backpropagation
        # to bypass the privacy noise. The sensitivity is set based on the expected dynamic 
        # range of each feature (e.g. soil_sensitivity config: [1.0, 5.0, 20.0]).
        sensitivity_vals = self.config.get("soil_sensitivity", [1.0] * soil_dim)
        self.register_buffer("soil_sensitivity", torch.tensor(sensitivity_vals, dtype=torch.float32))

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

        # 3. Differential privacy noise — SCALED per feature dimension.
        # Previous version applied uniform randn * epsilon, which destroyed
        # low-magnitude features (pH ~6.5) while barely affecting high-magnitude
        # ones (potassium ~200). Now noise is proportional to learned sensitivity.
        if self.training and self.use_privacy:
            sensitivity = self.soil_sensitivity.abs() + 1e-8  # (F_s,)
            noise = torch.randn_like(soil) * self.epsilon * sensitivity
            soil = soil + noise

        # 4. Encode soil (static — NOT concatenated into the temporal sequence)
        soil_enc = self.soil_encoder(soil)                    # (B, D)

        # 5. Cross-modal attention: satellite queries attend to BOTH weather and soil context
        # nn.MultiheadAttention expects (Seq, Batch, Dim)
        sat_q = sat_enc.permute(1, 0, 2)                      # (T, B, D)
        weather_kv = weather_enc.permute(1, 0, 2)             # (T, B, D)
        soil_kv = soil_enc.unsqueeze(0)                       # (1, B, D)
        
        # Combine weather and soil into a single context sequence for cross-attention
        context_kv = torch.cat([weather_kv, soil_kv], dim=0)  # (T+1, B, D)
        
        cross_out, _ = self.cross_attn(sat_q, context_kv, context_kv)  # (T, B, D)
        cross_out = cross_out.permute(1, 0, 2)                # (B, T, D)

        # 6. Concatenate TEMPORAL features only, then refine via self-attention
        temporal_fused = torch.cat([cross_out, weather_enc], dim=1)  # (B, 2T, D)
        temporal_fused = temporal_fused.permute(1, 0, 2)      # (2T, B, D)
        temporal_out = self.transformer_encoder(temporal_fused)
        temporal_out = temporal_out.permute(1, 0, 2)          # (B, 2T, D)

        # 7. Pool temporal dimension → (B, D)
        temporal_pooled = torch.mean(temporal_out, dim=1)     # (B, D)

        # 8. Concatenate pooled temporal with STATIC soil (no dilution)
        fused = torch.cat([temporal_pooled, soil_enc], dim=1) # (B, 2D)

        # 9. Fusion MLP projects back to hidden_dim
        fused = self.fusion_mlp(fused)                        # (B, D)

        # 10. MDN Output
        return self.mdn_head(fused)


def initialize_model(config: Dict[str, Any]):
    """
    Factory function to initialize the model from configuration.
    """
    logger.info("Initializing MultiModalTransformer...")
    return MultiModalTransformer(config)


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
    """
    def __init__(self, config: Dict[str, Any]):
        super(MultiModalTransformer, self).__init__()
        self.config = config["transformer"]
        self.use_privacy = config.get("use_privacy", False)
        self.epsilon = config.get("privacy_epsilon", 0.1)
        
        # Satellite encoder: 1D temporal convolution extracts local temporal
        # patterns across spectral channels (NOT spatial super-resolution).
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.config["input_dim"], self.config["hidden_dim"], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.config["hidden_dim"], self.config["hidden_dim"], kernel_size=1)
        )
        # Linear projection (residual path) for the satellite encoder
        self.sat_encoder = nn.Linear(self.config["input_dim"], self.config["hidden_dim"])
        self.weather_encoder = nn.LSTM(self.config["temporal_dim"], 
                                      self.config["hidden_dim"], 
                                      batch_first=True)
        self.soil_encoder = nn.Linear(self.config.get("soil_dim", 4), self.config["hidden_dim"])
        
        # Transformer Layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config["hidden_dim"], 
                                                  nhead=self.config["num_heads"], 
                                                  dropout=self.config["dropout"])
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=self.config["num_layers"])
        
        # Cross-Modal Attention: satellite queries attend to weather keys/values
        # to learn which weather conditions are most relevant for each spectral timestep.
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.config["hidden_dim"], 
                                                num_heads=self.config["num_heads"])
        
        # MDN Output Head instead of simple Linear
        self.mdn_head = MixtureDensityNetwork(
            input_dim=self.config["hidden_dim"], 
            num_mixtures=config["mdn"]["num_mixtures"],
            output_dim=config["mdn"]["output_dim"]
        )

    def forward(self, sat, weather, soil):
        """
        sat: (B, T, C) - Spectral Features
        weather: (B, T, F_w) - Temporal Features
        soil: (B, F_s) - Static Features
        """
        # 1. Encode satellite: temporal conv + linear residual path
        sat_t = sat.transpose(1, 2)                         # (B, C, T)
        sat_conv = self.temporal_conv(sat_t).transpose(1, 2) # (B, T, D)
        sat_res = self.sat_encoder(sat)                      # (B, T, D)
        sat_enc = sat_conv + sat_res                         # (B, T, D) residual sum
        
        # 2. Encode weather and soil
        weather_enc, _ = self.weather_encoder(weather)       # (B, T, D)
        
        if self.training and self.use_privacy:
            noise = torch.randn_like(soil) * self.epsilon
            soil = soil + noise
            
        soil_enc = self.soil_encoder(soil).unsqueeze(1)      # (B, 1, D)
        
        # 3. Cross-modal attention: satellite queries attend to weather context
        # nn.MultiheadAttention expects (Seq, Batch, Dim)
        sat_q = sat_enc.permute(1, 0, 2)                     # (T, B, D)
        weather_kv = weather_enc.permute(1, 0, 2)            # (T, B, D)
        cross_out, _ = self.cross_attn(sat_q, weather_kv, weather_kv)  # (T, B, D)
        cross_out = cross_out.permute(1, 0, 2)               # (B, T, D)
        
        # 4. Concatenate cross-attended satellite, weather, and soil
        fused = torch.cat([cross_out, weather_enc, soil_enc], dim=1)  # (B, 2T+1, D)
        
        # 5. Transformer self-attention refinement
        fused = fused.permute(1, 0, 2)                        # (2T+1, B, D)
        out = self.transformer_encoder(fused)
        out = out.permute(1, 0, 2)                            # (B, 2T+1, D)
        
        # 6. Global Average Pooling over time/modalities
        out = torch.mean(out, dim=1)                          # (B, D)
        
        # 7. MDN Output
        return self.mdn_head(out)

def initialize_model(config: Dict[str, Any]):
    """
    Factory function to initialize the model from configuration.
    """
    logger.info("Initializing MultiModalTransformer...")
    return MultiModalTransformer(config)

import torch
import torch.nn as nn
from loguru import logger
from typing import Dict, Any

class MultiModalTransformer(nn.Module):
    """
    State-of-the-art Multi-Modal Transformer for Crop Yield Prediction.
    Fuses Satellite (Spectral), Weather (Temporal), and Soil (Static) data.
    """
    def __init__(self, config: Dict[str, Any]):
        super(MultiModalTransformer, self).__init__()
        self.config = config["transformer"]
        
        # Encoders
        self.sat_encoder = nn.Linear(self.config["input_dim"], self.config["hidden_dim"])
        self.weather_encoder = nn.LSTM(self.config["temporal_dim"], 
                                      self.config["hidden_dim"], 
                                      batch_first=True)
        self.soil_encoder = nn.Linear(4, self.config["hidden_dim"]) # Static 4 features
        
        # Transformer Layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config["hidden_dim"], 
                                                  nhead=self.config["num_heads"], 
                                                  dropout=self.config["dropout"])
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=self.config["num_layers"])
        
        # Cross-Modal Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.config["hidden_dim"], 
                                                num_heads=self.config["num_heads"])
        
        # Output Head
        self.fc = nn.Linear(self.config["hidden_dim"], 1) # Single output (Yield)

    def forward(self, sat, weather, soil):
        """
        sat: (B, T, C) - Spectral Features
        weather: (B, T, F_w) - Temporal Features
        soil: (B, F_s) - Static Features
        """
        # 1. Encode modalities
        sat_enc = self.sat_encoder(sat) # (B, T, D)
        weather_enc, _ = self.weather_encoder(weather) # (B, T, D)
        soil_enc = self.soil_encoder(soil).unsqueeze(1) # (B, 1, D)
        
        # 2. Fuse modalities (Simplified concatenation for now)
        fused = torch.cat([sat_enc, weather_enc, soil_enc], dim=1) # (B, 2T+1, D)
        
        # 3. Transformer Processing
        # Transformer expects (Seq_Len, Batch, Dim)
        fused = fused.permute(1, 0, 2)
        out = self.transformer_encoder(fused)
        out = out.permute(1, 0, 2)
        
        # 4. Global Average Pooling over time/modalities
        out = torch.mean(out, dim=1)
        
        # 5. Output
        return self.fc(out)

def initialize_model(config: Dict[str, Any]):
    """
    Factory function to initialize the model from configuration.
    """
    logger.info("Initializing MultiModalTransformer...")
    return MultiModalTransformer(config)

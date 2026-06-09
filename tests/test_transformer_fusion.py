import torch
import pandas as pd
import numpy as np
from src.models.transformer import initialize_model
from src.schema.validators import align_and_validate_soil

def test_multimodal_transformer_temporal_alignment():
    # Setup model config matchingconfigs/model_config.yaml
    config = {
        "transformer": {
            "input_dim": 5,
            "temporal_dim": 3,
            "soil_dim": 3,
            "hidden_dim": 128,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.3,
            "soil_sensitivity": [1.0, 5.0, 20.0]
        },
        "mdn": {
            "num_mixtures": 5,
            "hidden_dim": 128,
            "output_dim": 1
        }
    }
    
    model = initialize_model(config)
    model.eval()
    
    # Batch size = 2, Sequence length = 12
    B, T = 2, 12
    sat = torch.randn(B, T, 5)
    weather = torch.randn(B, T, 3)
    soil = torch.randn(B, 3)
    
    with torch.no_grad():
        pi, sigma, mu = model(sat, weather, soil)
        
    # Check that output shapes conform to the GMM mixtures
    assert pi.shape == (B, 5)
    assert sigma.shape == (B, 5, 1)
    assert mu.shape == (B, 5, 1)

def test_align_and_validate_soil_empty():
    # Create empty soil dataframe matching schema
    empty_df = pd.DataFrame(columns=["ph", "soc", "nitrogen"])
    
    # Validate should return a padded zero vector instead of crashing with IndexError
    soil_values = align_and_validate_soil(empty_df, soil_dim=3)
    assert isinstance(soil_values, np.ndarray)
    assert np.all(soil_values == 0.0)
    assert len(soil_values) == 3

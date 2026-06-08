import pandera as pa
from pandera.typing import Series
import numpy as np

class SoilSchema(pa.DataFrameModel):
    """
    Strict data contract for Soil features.
    Ensures upstream APIs have not drifted and data is numerically valid.
    """
    ph: Series[float] = pa.Field(ge=0.0, le=14.0, description="Soil pH value must be between 0 and 14")
    soc: Series[float] = pa.Field(ge=0.0, description="Soil Organic Carbon must be non-negative")
    nitrogen: Series[float] = pa.Field(ge=0.0, description="Nitrogen content must be non-negative")

    class Config:
        strict = False  # Allows additional columns to be present, but expected ones MUST match
        coerce = True   # Tries to coerce types to float if they are passed as integers

def align_and_validate_soil(soil_df, soil_dim: int) -> np.ndarray:
    """
    Validates soil dataframe using Pandera, and returns the strictly ordered numpy array.
    """
    # Fail loudly if contract is violated
    validated_df = SoilSchema.validate(soil_df)
    
    expected_cols = ["ph", "soc", "nitrogen"]
    soil_values = validated_df[expected_cols].iloc[0].to_numpy(dtype=np.float32)
    
    if len(soil_values) >= soil_dim:
        return soil_values[:soil_dim]

    padded = np.zeros(soil_dim, dtype=np.float32)
    padded[: len(soil_values)] = soil_values
    return padded

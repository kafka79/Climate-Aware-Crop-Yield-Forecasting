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
    if len(expected_cols) != soil_dim:
        raise ValueError(
            f"Dimension mismatch: validation schema defines {len(expected_cols)} soil features "
            f"({expected_cols}), but the model requires soil_dim={soil_dim}."
        )

    # Handle empty DataFrame (e.g. only headers present) robustly without raising IndexError on iloc[0]
    if validated_df.empty:
        return np.zeros(soil_dim, dtype=np.float32)
        
    soil_values = validated_df[expected_cols].iloc[0].to_numpy(dtype=np.float32)
    return soil_values

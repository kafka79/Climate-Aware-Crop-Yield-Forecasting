import pytest
import pandas as pd
import numpy as np
from src.temporal.sequence_builder import SequenceBuilder

def test_sequence_builder():
    builder = SequenceBuilder(window_size=3, step_size=1)
    
    # Mock data with 6 rows. Columns: f1, f2, yield
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6],
        "yield": [10, 20, 30, 40, 50, 60]
    })
    
    # Can form 3 sequences:
    # 1: window(1,2,3), label=40
    # 2: window(2,3,4), label=50
    # 3: window(3,4,5), label=60
    
    X, y = builder.create_sequences(df, target="yield")
    
    assert len(X) == 3
    assert len(y) == 3
    assert y[0] == 40
    assert y[2] == 60
    
    assert X.shape == (3, 3, 1) # 3 sequences, window=3, 1 feature

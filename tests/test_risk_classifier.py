import pytest
import numpy as np
from src.risk.risk_classifier import YieldRiskClassifier

@pytest.fixture
def classifier():
    return YieldRiskClassifier(thresholds={"low": 0.2, "high": 0.5})

def test_low_risk(classifier):
    """Yield at 95% of average → Low Risk."""
    risk = classifier.classify_risk(predicted_yield=3.8, average_yield=4.0)
    assert risk == "Low Risk"

def test_medium_risk(classifier):
    """Yield at 70% of average → Medium Risk."""
    risk = classifier.classify_risk(predicted_yield=2.8, average_yield=4.0)
    assert risk == "Medium Risk"

def test_high_risk(classifier):
    """Yield at 40% of average → High Risk."""
    risk = classifier.classify_risk(predicted_yield=1.5, average_yield=4.0)
    assert risk == "High Risk"

def test_uncertainty_elevates_risk(classifier):
    """Low Risk with high std should elevate to Medium Risk."""
    # mean=3.9 → Low Risk baseline; std=0.9 > 15% of mean → elevated
    risk = classifier.calibrate_with_uncertainty(mean=3.9, std=0.9, average_yield=4.0)
    assert "Medium Risk" in risk or "Elevated" in risk

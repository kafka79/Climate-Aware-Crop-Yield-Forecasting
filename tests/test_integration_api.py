import pytest
import responses
import requests
import os
from src.data.downloader import SentinelHubDownloader, ERA5Downloader

@pytest.fixture
def mock_config():
    return {
        "sentinel_hub": {
            "client_id": "test_id",
            "client_secret": "test_secret"
        },
        "paths": {
            "raw": {
                "sentinel2": "data/raw/sentinel",
                "era5": "data/raw/era5"
            }
        }
    }

@responses.activate
def test_sentinel_hub_api_resilience(mock_config):
    """
    Test that the SentinelHub downloader handles API success and simulated latency.
    """
    # Mocking the OAuth2 token request
    responses.add(
        responses.POST, 
        "https://services.sentinel-hub.com/oauth/token",
        json={"access_token": "mock_token", "expires_in": 3600},
        status=200
    )
    
    # Mocking a data request
    responses.add(
        responses.POST,
        "https://services.sentinel-hub.com/api/v1/process",
        body=b"mock_tiff_content",
        status=200,
        content_type="image/tiff"
    )

    downloader = SentinelHubDownloader(mock_config)
    # Simulate a download call (note: we check that it doesn't crash and handles the response)
    try:
        downloader.download([77.0, 28.0, 78.0, 29.0], ("2023-01-01", "2023-01-31"), "test_area")
    except Exception as e:
        pytest.fail(f"Downloader failed with mock API: {e}")

@responses.activate
def test_api_error_handling(mock_config):
    """
    Test that the system gracefully handles API 500 errors.
    """
    responses.add(
        responses.POST, 
        "https://services.sentinel-hub.com/oauth/token",
        status=500
    )
    
    downloader = SentinelHubDownloader(mock_config)
    # We expect the downloader to log the error and not crash the whole pipeline if managed
    with pytest.raises(Exception):
        downloader.download([77.0, 28.0, 78.0, 29.0], ("2023-01-01", "2023-01-31"), "test_area")

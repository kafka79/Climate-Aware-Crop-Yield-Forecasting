import os
import yaml
from loguru import logger
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.debug(f"Loaded config from {config_path}")
    return config

def load_secrets(secrets_path: str = "configs/secrets.yaml") -> Dict[str, Any]:
    """
    Load secrets from secrets.yaml (which is gitignored).
    Falls back gracefully to environment variables if the file doesn't exist,
    making it compatible with cloud deployment (Docker, k8s, etc).
    """
    if os.path.exists(secrets_path):
        with open(secrets_path, "r") as f:
            secrets = yaml.safe_load(f)
        logger.debug("Loaded secrets from secrets.yaml")
        return secrets
    
    # Fallback: read from environment variables for production/cloud deployments
    logger.warning(
        f"'{secrets_path}' not found. Falling back to environment variables. "
        "Copy configs/secrets.yaml.template to configs/secrets.yaml to run locally."
    )
    return {
        "sentinel_hub": {
            "client_id": os.getenv("SENTINELHUB_CLIENT_ID", ""),
            "client_secret": os.getenv("SENTINELHUB_CLIENT_SECRET", ""),
            "instance_id": os.getenv("SENTINELHUB_INSTANCE_ID", ""),
        },
        "upag": {
            "api_key": os.getenv("UPAG_API_KEY", ""),
            "base_url": os.getenv("UPAG_BASE_URL", "https://upag.gov.in/api/v1"),
        },
        "isric": {
            "api_key": os.getenv("ISRIC_API_KEY", ""),
            "base_url": os.getenv("ISRIC_BASE_URL", "https://rest.isric.org/soilgrids/v2.0"),
        },
        "era5": {
            "cds_uid": os.getenv("CDS_UID", ""),
            "cds_api_key": os.getenv("CDS_API_KEY", ""),
        },
    }

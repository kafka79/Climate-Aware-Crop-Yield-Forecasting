import yaml
import os
from loguru import logger

def load_config(config_path: str):
    """
    Load project configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at: {config_path}")
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
            logger.debug(f"Loaded config from {config_path}")
            return config
        except yaml.YAMLError as exc:
            logger.error(f"Error parsing YAML file: {exc}")
            raise exc

def save_config(config: dict, config_path: str):
    """
    Save current configuration to a YAML file.
    """
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved config to {config_path}")

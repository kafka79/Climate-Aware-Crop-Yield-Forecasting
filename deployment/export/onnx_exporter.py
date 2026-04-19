import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from loguru import logger
from src.models.transformer import initialize_model
from src.utils.config import load_config

def export_to_onnx(
    config_path: str = "configs/model_config.yaml",
    model_path: str = "models/checkpoints/best_model.pth",
    output_path: str = "deployment/mobile/onnx_model/model.onnx",
):
    """
    Export the trained PyTorch MultiModalTransformer to ONNX format.
    Enables inference on mobile (CoreML via AppleConvert) or edge (TensorRT).
    """
    logger.info("Loading model configuration...")
    config = load_config(config_path)

    logger.info("Initializing model architecture...")
    model = initialize_model(config)

    if os.path.exists(model_path):
        logger.info(f"Loading trained weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        logger.warning(f"No trained weights found at {model_path}. Exporting untrained model.")

    model.eval()

    # Build dummy inputs matching the model's expected shapes
    T = 6   # Time steps matching sample inputs
    C = config["transformer"]["input_dim"]
    Fw = config["transformer"]["temporal_dim"]
    Fs = config["transformer"].get("soil_dim", 4)

    dummy_sat     = torch.randn(1, T, C)
    dummy_weather = torch.randn(1, T, Fw)
    dummy_soil    = torch.randn(1, Fs)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"Exporting to ONNX at {output_path}...")
    torch.onnx.export(
        model,
        args=(dummy_sat, dummy_weather, dummy_soil),
        f=output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["satellite", "weather", "soil"],
        output_names=["yield_prediction"],
        dynamic_axes={
            "satellite": {0: "batch_size"},
            "weather":   {0: "batch_size"},
            "soil":      {0: "batch_size"},
            "yield_prediction": {0: "batch_size"},
        },
    )
    logger.success(f"ONNX model exported successfully → {output_path}")
    return output_path


if __name__ == "__main__":
    export_to_onnx()

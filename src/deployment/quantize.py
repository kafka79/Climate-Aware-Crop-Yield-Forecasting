import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from loguru import logger
import os

def quantize_for_mobile(model_path: str, output_path: str):
    """
    Converts a PyTorch checkpoint to a quantized ONNX model for mobile deployment.
    """
    logger.info(f"Quantizing model: {model_path} -> {output_path}...")
    
    # 1. Export to ONNX (FP32)
    onnx_fp32 = output_path.replace(".onnx", "_fp32.onnx")
    dummy_sat = torch.randn(1, 12, 10) # B, T, C
    dummy_weather = torch.randn(1, 12, 5) # B, T, F
    dummy_soil = torch.randn(1, 4) # B, S
    
    # We need a wrapper if the model takes multiple inputs
    # Assuming model forward is forward(sat, weather, soil)
    torch.onnx.export(
        torch.load(model_path), 
        (dummy_sat, dummy_weather, dummy_soil),
        onnx_fp32,
        input_names=["satellite", "weather", "soil"],
        output_names=["yield_prediction"],
        dynamic_axes={'satellite': {0: 'batch_size'}, 'weather': {0: 'batch_size'}, 'soil': {0: 'batch_size'}},
        opset_version=14
    )
    
    # 2. Dynamic Quantization to INT8
    quantize_dynamic(
        onnx_fp32,
        output_path,
        weight_type=QuantType.QInt8
    )
    
    # Cleanup intermediate
    if os.path.exists(onnx_fp32):
        os.remove(onnx_fp32)
        
    logger.success(f"Quantized ONNX model saved at: {output_path}")
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Final Model Size: {size_mb:.2f} MB")

if __name__ == "__main__":
    # Example usage (placeholders)
    # quantize_for_mobile("models/checkpoints/best_model.pth", "deployment/export/model_quantized.onnx")
    pass

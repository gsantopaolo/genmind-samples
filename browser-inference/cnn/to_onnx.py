"""
Export trained PyTorch CNN model to ONNX format

This script exports the trained EfficientNet-B0 model to ONNX format for:
- Cross-platform deployment (web, mobile, edge devices)
- Faster inference with ONNX Runtime
- Compatibility with non-Python environments

Usage:
    python3 to_onnx.py
    
Output:
    - model.onnx: Exported ONNX model
    - Verification of export success
"""

import logging
import os
from pathlib import Path

os.environ['TORCH_HOME'] = str(Path(__file__).parent / 'models')

import torch
import onnx
import onnxruntime as ort
import numpy as np

from cnn_train import create_efficientnet_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def export_to_onnx(
    checkpoint_path: str = "checkpoints/best_model.pt",
    output_path: str = "models_onnx/model.onnx",
    num_classes: int = 3,
    img_size: int = 224,
    opset_version: int = 18
):
    """
    Export trained PyTorch model to ONNX format.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        output_path: Output path for ONNX model
        num_classes: Number of output classes
        img_size: Input image size
        opset_version: ONNX opset version (18 is PyTorch 2.x standard)
    """
    logger.info("🚀 Starting ONNX export...")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trained model
    device = torch.device("cpu")  # Export on CPU for compatibility
    model = create_efficientnet_model(num_classes=num_classes, freeze_features=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    logger.info(f"✅ Loaded model from {checkpoint_path}")
    
    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)
    
    # Export model to ONNX with FIXED batch size for WebGL compatibility
    # WebGL execution provider doesn't support dynamic batch dimensions
    # Using fixed batch_size=1 for browser inference
    logger.info(f"📦 Exporting to ONNX (opset_version={opset_version})...")
    logger.info("Using FIXED batch size (batch=1) for WebGL/browser compatibility")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        dynamo=False,
        input_names=['input'],
        output_names=['output']
        # NO dynamic_axes - fixed batch size for WebGL
    )
    logger.info(f"✅ ONNX model saved to {output_path}")
    
    # Verify ONNX model
    logger.info("🔍 Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("✅ ONNX model is valid")
    
    # Get model info
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"📊 Model size: {file_size_mb:.2f} MB")
    
    # Test inference with ONNX Runtime
    logger.info("🧪 Testing ONNX Runtime inference...")
    ort_session = ort.InferenceSession(output_path)
    
    # Run PyTorch inference
    with torch.no_grad():
        pytorch_output = model(dummy_input).numpy()
    
    # Run ONNX inference
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # Compare outputs
    max_diff = np.abs(pytorch_output - ort_output).max()
    logger.info(f"📈 Max difference between PyTorch and ONNX: {max_diff:.6f}")
    
    if max_diff < 1e-5:
        logger.info("✅ ONNX export successful! Outputs match PyTorch.")
    else:
        logger.warning(f"⚠️  Outputs differ by {max_diff:.6f} (may be acceptable)")
    
    # Print model info
    logger.info("\n" + "="*70)
    logger.info("📋 ONNX Model Information:")
    logger.info("="*70)
    logger.info(f"Input name: {ort_session.get_inputs()[0].name}")
    logger.info(f"Input shape: {ort_session.get_inputs()[0].shape}")
    logger.info(f"Input type: {ort_session.get_inputs()[0].type}")
    logger.info(f"Output name: {ort_session.get_outputs()[0].name}")
    logger.info(f"Output shape: {ort_session.get_outputs()[0].shape}")
    logger.info(f"Output type: {ort_session.get_outputs()[0].type}")
    logger.info("="*70)
    
    logger.info("\n🎉 Export complete!")
    logger.info(f"💾 ONNX model: {output_path}")
    logger.info(f"📏 Model size: {file_size_mb:.2f} MB")
    logger.info("\nUsage example:")
    logger.info("  import onnxruntime as ort")
    logger.info(f"  session = ort.InferenceSession('{output_path}')")
    logger.info("  outputs = session.run(None, {{'input': image_array}})")


def main():
    """Export model to ONNX"""
    checkpoint_path = "checkpoints/best_model.pt"
    output_path = "models_onnx/model.onnx"
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        logger.error("Please train the model first using: python3 cnn_train.py")
        return
    
    # Export to ONNX
    export_to_onnx(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        num_classes=3,
        img_size=224,
        opset_version=18  # PyTorch 2.x standard version
    )


if __name__ == "__main__":
    main()

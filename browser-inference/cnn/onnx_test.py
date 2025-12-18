"""
Test Script for ONNX Model using ONNX Runtime

This script loads the exported ONNX model and evaluates it on test data.
It provides the same metrics as cnn_test.py for comparison with PyTorch.

Metrics include:
- Overall accuracy
- Per-class accuracy
- Confusion matrix
- Sample predictions with confidence scores
- Inference time comparison
"""

"""
ONNX Runtime Testing - NO PyTorch dependency
This script uses only ONNX Runtime for inference, no PyTorch required.
"""

import logging
import os
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from datasets import load_dataset, DownloadMode
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress HuggingFace HTTP metadata check logging
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("datasets.builder").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def preprocess_image(image: Image.Image, img_size: int = 224) -> np.ndarray:
    """
    Preprocess image for ONNX model (no PyTorch dependency).
    
    Args:
        image: PIL Image
        img_size: Target size
        
    Returns:
        Preprocessed numpy array [1, 3, H, W]
    """
    # Resize image
    image = image.resize((img_size, img_size), Image.BILINEAR)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # Apply normalization: (img - mean) / std
    img_array = (img_array - mean) / std
    
    # Convert from HWC to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    return img_array


def load_test_data(datasets_dir: str = "datasets", img_size: int = 224):
    """Load test dataset from HuggingFace (offline mode - no downloads)"""
    datasets_path = Path(datasets_dir)
    datasets_path.mkdir(exist_ok=True)
    
    cache_exists = any(datasets_path.glob("**/dataset_info.json"))
    
    if cache_exists:
        logger.info("📦 Using cached dataset (100% offline - no HTTP requests)")
        # Force complete offline mode
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        
        # Suppress HuggingFace HTTP logging
        logging.getLogger("datasets.builder").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        
        dataset = load_dataset(
            "beans",
            cache_dir=datasets_dir,
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
        )
        
        # Clean up env vars
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("HF_DATASETS_OFFLINE", None)
    else:
        logger.info("⬇️  Downloading dataset...")
        dataset = load_dataset("beans", cache_dir=datasets_dir)
    
    test_dataset = dataset["test"]
    class_names = ["angular_leaf_spot", "bean_rust", "healthy"]
    
    logger.info(f"✅ Test data loaded: {len(test_dataset)} samples")
    logger.info(f"   Classes: {', '.join(class_names)}")
    
    # Convert dataset to list of (image, label) tuples
    processed_data = []
    for item in test_dataset:
        image = item['image'].convert('RGB')
        label = item['labels']
        processed_data.append((image, label))
    
    return processed_data, class_names


def compute_confusion_matrix(predictions: np.ndarray, targets: np.ndarray, num_classes: int):
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted class indices [N]
        targets: True class indices [N]
        num_classes: Number of classes
        
    Returns:
        Confusion matrix as numpy array [num_classes, num_classes]
    """
    # Initialize confusion matrix: rows=true labels, cols=predicted labels
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # Count predictions: conf_matrix[true_label][predicted_label] += 1
    for pred, target in zip(predictions, targets):
        conf_matrix[target, pred] += 1
    
    return conf_matrix


def print_confusion_matrix(conf_matrix: np.ndarray, class_names: list[str]):
    """Pretty print confusion matrix"""
    logger.info("\n📊 Confusion Matrix:")
    logger.info("-" * 70)
    
    header = "True\\Pred".ljust(20) + " ".join(f"{name[:15]:>15}" for name in class_names)
    logger.info(header)
    logger.info("-" * 70)
    
    for i, name in enumerate(class_names):
        row = f"{name[:18]:18}" + " ".join(f"{conf_matrix[i, j]:>15}" for j in range(len(class_names)))
        logger.info(row)
    logger.info("-" * 70)


def compute_per_class_metrics(conf_matrix: np.ndarray, class_names: list[str]):
    """
    Compute precision, recall, and F1 score per class.
    """
    logger.info("\n📈 Per-Class Metrics:")
    logger.info("-" * 70)
    logger.info(f"{'Class':<20} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    logger.info("-" * 70)
    
    for i, name in enumerate(class_names):
        # True Positives: correctly predicted as this class
        tp = conf_matrix[i, i]
        # False Positives: predicted as this class but was actually another class
        fp = conf_matrix[:, i].sum() - tp
        # False Negatives: actually this class but predicted as another class
        fn = conf_matrix[i, :].sum() - tp
        
        # Precision: of all predictions for this class, how many were correct?
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        # Recall: of all actual instances of this class, how many did we find?
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        # F1 Score: harmonic mean of precision and recall
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info(f"{name:<20} {precision:>12.3f} {recall:>12.3f} {f1:>12.3f}")
    logger.info("-" * 70)


def softmax(x: np.ndarray) -> np.ndarray:
    """Apply softmax to convert logits to probabilities"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def test_onnx_model(
    ort_session: ort.InferenceSession,
    test_data: list,
    class_names: list[str],
    img_size: int = 224,
    batch_size: int = 32,
    show_samples: int = 5
):
    """
    Run inference on test data using ONNX Runtime (NO PyTorch).
    
    Args:
        ort_session: ONNX Runtime inference session
        test_data: List of (PIL Image, label) tuples
        class_names: List of class names
        img_size: Input image size
        batch_size: Batch size for inference
        show_samples: Number of sample predictions to display
        
    Returns:
        Tuple of (accuracy, confusion_matrix)
    """
    logger.info("🧪 Running ONNX Runtime inference on test data...")
    
    # Store all predictions and ground truth
    all_preds = []
    all_targets = []
    all_probs = []
    
    # Track inference time
    inference_times = []
    
    # Get input name from ONNX model
    input_name = ort_session.get_inputs()[0].name
    
    # Process in batches
    num_samples = len(test_data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        # Get batch data
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_data = test_data[start_idx:end_idx]
        
        # Preprocess batch (pure numpy, no PyTorch)
        batch_images = []
        batch_labels = []
        for image, label in batch_data:
            img_array = preprocess_image(image, img_size)
            batch_images.append(img_array)
            batch_labels.append(label)
        
        # Stack into batch
        images_np = np.stack(batch_images, axis=0)
        labels_np = np.array(batch_labels)
        
        # Measure inference time
        start_time = time.perf_counter()
        
        # Run ONNX inference
        ort_inputs = {input_name: images_np}
        logits = ort_session.run(None, ort_inputs)[0]
        
        end_time = time.perf_counter()
        inference_times.append(end_time - start_time)
        
        # Convert logits to probabilities and predictions
        probs = softmax(logits)
        preds = np.argmax(logits, axis=1)
        
        # Collect results
        all_preds.extend(preds.tolist())
        all_targets.extend(labels_np.tolist())
        all_probs.extend(probs.tolist())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Calculate overall accuracy
    accuracy = (all_preds == all_targets).mean()
    
    # Calculate average inference time
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
    total_time = sum(inference_times)
    
    # Calculate per-image time for fair comparison
    per_image_time = avg_inference_time / batch_size
    
    logger.info(f"\n✅ Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"⏱️  Batch size: {batch_size}")
    logger.info(f"⏱️  Average inference time: {avg_inference_time:.2f} ms/batch ({per_image_time:.2f} ms/image)")
    logger.info(f"⏱️  Total inference time: {total_time:.3f} seconds")
    
    # Compute and display confusion matrix
    conf_matrix = compute_confusion_matrix(all_preds, all_targets, len(class_names))
    print_confusion_matrix(conf_matrix, class_names)
    
    # Compute per-class metrics
    compute_per_class_metrics(conf_matrix, class_names)
    
    # Show sample predictions
    logger.info(f"\n🔍 Sample Predictions (showing first {show_samples}):")
    logger.info("-" * 90)
    
    for i in range(min(show_samples, len(all_preds))):
        true_class = class_names[all_targets[i]]
        pred_class = class_names[all_preds[i]]
        confidence = all_probs[i][all_preds[i]]
        
        status = "✅" if all_preds[i] == all_targets[i] else "❌"
        logger.info(
            f"{status} True: {true_class:<20} | "
            f"Predicted: {pred_class:<20} | "
            f"Confidence: {confidence:.4f}"
        )
    logger.info("-" * 90)
    
    return accuracy, conf_matrix


def main(
    onnx_path: str = "models_onnx/model.onnx",
    datasets_dir: str = "datasets",
    img_size: int = 224,
    batch_size: int = 1  # FIXED batch size for WebGL compatibility
):
    """Main testing function - 100% ONNX Runtime, no PyTorch"""
    logger.info("🚀 Starting ONNX model testing (NO PyTorch dependency)...")
    
    # Check if ONNX model exists
    if not Path(onnx_path).exists():
        logger.error(f"❌ ONNX model not found: {onnx_path}")
        logger.error("Please export the model first using: python3 to_onnx.py")
        return
    
    # Load test data (returns raw PIL images, no PyTorch)
    test_data, class_names = load_test_data(
        datasets_dir=datasets_dir,
        img_size=img_size
    )
    
    # Load ONNX model with GPU acceleration
    logger.info(f"📦 Loading ONNX model from {onnx_path}")
    
    # Configure execution providers with GPU support
    # Try CoreML (Mac GPU/Neural Engine) first, fallback to CPU
    providers = []
    provider_options = []
    
    # CoreML for Mac M1/M2/M3 (GPU + Neural Engine + CPU)
    if ort.get_available_providers().__contains__('CoreMLExecutionProvider'):
        providers.append('CoreMLExecutionProvider')
        provider_options.append({
            'MLComputeUnits': 'ALL',  # Use GPU, Neural Engine, or CPU (auto-select best)
            'ModelFormat': 'MLProgram',  # Modern format (iOS 15+, macOS 12+)
            'EnableOnSubgraphs': 0
        })
        logger.info("🚀 CoreML Execution Provider available (GPU/Neural Engine/CPU)")
    
    # CPU fallback (always available)
    providers.append('CPUExecutionProvider')
    provider_options.append({})
    
    # Create session with GPU support
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=providers,
        provider_options=provider_options
    )
    
    # Log which provider is actually being used
    active_providers = ort_session.get_providers()
    logger.info(f"✅ ONNX model loaded successfully")
    logger.info(f"   Using: {active_providers[0]}")
    
    # Display model info
    logger.info(f"   Input: {ort_session.get_inputs()[0].name} {ort_session.get_inputs()[0].shape}")
    logger.info(f"   Output: {ort_session.get_outputs()[0].name} {ort_session.get_outputs()[0].shape}")
    
    # Run tests (pure ONNX Runtime inference)
    accuracy, conf_matrix = test_onnx_model(
        ort_session=ort_session,
        test_data=test_data,
        class_names=class_names,
        img_size=img_size,
        batch_size=batch_size,
        show_samples=10
    )
    
    logger.info(f"\n🎉 Testing complete! Final accuracy: {accuracy:.4f}")
    logger.info("✅ This test used ONLY ONNX Runtime - NO PyTorch required!")


if __name__ == "__main__":
    main()

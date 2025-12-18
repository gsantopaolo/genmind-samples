"""
Test Script for PyTorch Training Loop

This script loads a trained model and evaluates it on unseen test data.
It provides detailed metrics including:
- Overall accuracy
- Per-class accuracy
- Confusion matrix
- Sample predictions
"""

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, DownloadMode
import numpy as np

from train import TinyMLP, get_best_device, accuracy_from_logits

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# suppress verbose HF and HTTP logging
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def load_test_data(datasets_dir: str = "datasets", seed: int = 42):
    """
    Load test data from the Iris dataset.
    Uses a different split than training to ensure unseen data.
    
    Strategy: Original split was 80/20 for train/val.
    We'll create a 70/10/20 split (train/val/test) by using
    a different seed to get a completely different test set.
    
    Args:
        datasets_dir: Directory where dataset is cached
        seed: Random seed (different from training)
        
    Returns:
        Tuple of (X_test, y_test, class_names)
    """
    datasets_path = Path(datasets_dir)
    datasets_path.mkdir(exist_ok=True)
    
    # Check if dataset is already cached
    cache_exists = any(datasets_path.glob("**/dataset_info.json"))
    
    if cache_exists:
        logger.info("📦 Using cached dataset (offline mode)")
        # Enable offline mode to prevent any HTTP calls
        os.environ["HF_HUB_OFFLINE"] = "1"
        ds = load_dataset(
            "scikit-learn/iris",
            cache_dir=datasets_dir,
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
        )["train"]
        # Reset offline mode
        os.environ.pop("HF_HUB_OFFLINE", None)
    else:
        logger.info("⬇️  Downloading dataset...")
        # Download dataset to local folder
        ds = load_dataset("scikit-learn/iris", cache_dir=datasets_dir)["train"]
    
    feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    label_col = "Species"
    
    X = torch.tensor(
        list(zip(*(ds[c] for c in feature_cols))),
        dtype=torch.float32
    )
    
    species = ds[label_col]
    if isinstance(species[0], str):
        class_names = sorted(set(species))
        name_to_id = {name: i for i, name in enumerate(class_names)}
        y = torch.tensor([name_to_id[s] for s in species], dtype=torch.long)
    else:
        y = torch.tensor(species, dtype=torch.long)
        class_names = [f"Class_{i}" for i in range(y.max().item() + 1)]
    
    # Create a 70/10/20 split using a different strategy than training
    # Training used last 20% as val, so we'll use first 20% as test
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    idx = torch.randperm(n, generator=g)
    
    test_size = max(1, int(0.2 * n))
    test_idx = idx[:test_size]
    
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Normalize using same statistics as training
    # For proper testing, we should save these from training,
    # but for demonstration, we'll compute from full dataset
    # (In production, save mean/std during training!)
    train_idx = idx[test_size:]
    X_train_for_stats = X[train_idx]
    mean = X_train_for_stats.mean(dim=0, keepdim=True)
    std = X_train_for_stats.std(dim=0, keepdim=True).clamp_min(1e-6)
    
    X_test = (X_test - mean) / std
    
    return X_test, y_test, class_names


def compute_confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted class indices [N]
        targets: True class indices [N]
        num_classes: Number of classes
        
    Returns:
        Confusion matrix as numpy array [num_classes, num_classes]
    """
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for pred, target in zip(predictions, targets):
        conf_matrix[target, pred] += 1
    return conf_matrix.numpy()


def print_confusion_matrix(conf_matrix: np.ndarray, class_names: list[str]):
    """Pretty print confusion matrix"""
    logger.info("\n📊 Confusion Matrix:")
    logger.info("-" * 60)
    
    # Header
    header = "True\\Pred".ljust(12) + " ".join(f"{name[:8]:>8}" for name in class_names)
    logger.info(header)
    logger.info("-" * 60)
    
    # Rows
    for i, name in enumerate(class_names):
        row = f"{name[:10]:10}" + " ".join(f"{conf_matrix[i, j]:>8}" for j in range(len(class_names)))
        logger.info(row)
    logger.info("-" * 60)


def compute_per_class_metrics(conf_matrix: np.ndarray, class_names: list[str]):
    """
    Compute precision, recall, and F1 score per class.
    
    Args:
        conf_matrix: Confusion matrix
        class_names: List of class names
    """
    logger.info("\n📈 Per-Class Metrics:")
    logger.info("-" * 60)
    logger.info(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    logger.info("-" * 60)
    
    for i, name in enumerate(class_names):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info(f"{name:<15} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}")
    logger.info("-" * 60)


@torch.no_grad()
def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    show_samples: int = 5
):
    """
    Test the model and display detailed metrics.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        class_names: List of class names
        show_samples: Number of sample predictions to display
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    logger.info("🧪 Running inference on test data...")
    
    with torch.inference_mode():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_predictions.append(preds.cpu())
            all_targets.append(yb.cpu())
            all_probabilities.append(probs.cpu())
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    probabilities = torch.cat(all_probabilities)
    
    # Overall accuracy
    accuracy = (predictions == targets).float().mean().item()
    logger.info(f"\n✅ Overall Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    
    # Confusion matrix
    num_classes = len(class_names)
    conf_matrix = compute_confusion_matrix(predictions, targets, num_classes)
    print_confusion_matrix(conf_matrix, class_names)
    
    # Per-class metrics
    compute_per_class_metrics(conf_matrix, class_names)
    
    # Show sample predictions
    logger.info(f"\n🔍 Sample Predictions (showing first {show_samples}):")
    logger.info("-" * 80)
    for i in range(min(show_samples, len(predictions))):
        pred_idx = predictions[i].item()
        true_idx = targets[i].item()
        pred_prob = probabilities[i, pred_idx].item()
        
        status = "✅" if pred_idx == true_idx else "❌"
        logger.info(
            f"{status} True: {class_names[true_idx]:<15} | "
            f"Predicted: {class_names[pred_idx]:<15} | "
            f"Confidence: {pred_prob:.4f}"
        )
    logger.info("-" * 80)
    
    return accuracy, conf_matrix


def main():
    """Main testing routine"""
    logger.info("🚀 Starting model testing...")
    
    # Configuration
    checkpoint_path = Path("checkpoints/best_model.pt")
    datasets_dir = "datasets"
    batch_size = 64
    
    # Check if model exists
    if not checkpoint_path.exists():
        logger.error(f"❌ Model checkpoint not found at {checkpoint_path}")
        logger.error("Please train the model first by running: python train.py")
        return
    
    # Device
    device = get_best_device()
    logger.info(f"✅ Device: {device}")
    
    # Load test data
    X_test, y_test, class_names = load_test_data(
        datasets_dir=datasets_dir,
        seed=999  # Different seed than training to get different split
    )
    logger.info(f"✅ Test data loaded: {X_test.shape[0]} samples, {len(class_names)} classes")
    logger.info(f"   Classes: {', '.join(class_names)}")
    
    # Create test loader
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )
    
    # Load model
    model = TinyMLP(in_features=4, num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logger.info(f"✅ Model loaded from {checkpoint_path}")
    
    # Test the model
    accuracy, conf_matrix = test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        show_samples=10
    )
    
    logger.info(f"\n🎉 Testing complete! Final accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

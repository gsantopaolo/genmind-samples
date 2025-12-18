"""
Test Script for CNN Image Classification

This script loads a trained CNN model and evaluates it on test data.
It provides detailed metrics including:
- Overall accuracy
- Per-class accuracy
- Confusion matrix
- Sample predictions with confidence scores
"""

import logging
import os
from pathlib import Path

os.environ['TORCH_HOME'] = str(Path(__file__).parent / 'models')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DownloadMode
import numpy as np
from torchvision import transforms
from PIL import Image

from cnn_train import create_efficientnet_model, get_best_device, accuracy_from_logits, get_transforms, BeansDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def load_test_data(datasets_dir: str = "datasets", img_size: int = 224):
    """
    Load test data from the Beans dataset.
    
    Args:
        datasets_dir: Directory where dataset is cached
        img_size: Target image size for resizing
        
    Returns:
        Tuple of (test_dataset, class_names)
    """
    datasets_path = Path(datasets_dir)
    datasets_path.mkdir(parents=True, exist_ok=True)
    
    cache_exists = any(datasets_path.glob("**/dataset_info.json"))
    
    if cache_exists:
        logger.info("📦 Using cached dataset (offline mode)")
        os.environ["HF_HUB_OFFLINE"] = "1"
        dataset = load_dataset(
            "beans",
            cache_dir=datasets_dir,
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
        )
        os.environ.pop("HF_HUB_OFFLINE", None)
    else:
        logger.info("⬇️  Downloading Beans dataset...")
        dataset = load_dataset("beans", cache_dir=datasets_dir)
    
    test_dataset = dataset["test"]
    class_names = ["angular_leaf_spot", "bean_rust", "healthy"]
    
    logger.info(f"✅ Test data loaded: {len(test_dataset)} samples")
    logger.info(f"   Classes: {', '.join(class_names)}")
    
    return test_dataset, class_names


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
    # Initialize confusion matrix: rows=true labels, cols=predicted labels
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    # Count predictions: conf_matrix[true_label][predicted_label] += 1
    for pred, target in zip(predictions, targets):
        conf_matrix[target, pred] += 1
    
    return conf_matrix.numpy()


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
    
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    probabilities = torch.cat(all_probabilities)
    
    accuracy = (predictions == targets).float().mean().item()
    logger.info(f"\n✅ Overall Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    
    num_classes = len(class_names)
    conf_matrix = compute_confusion_matrix(predictions, targets, num_classes)
    print_confusion_matrix(conf_matrix, class_names)
    
    compute_per_class_metrics(conf_matrix, class_names)
    
    logger.info(f"\n🔍 Sample Predictions (showing first {show_samples}):")
    logger.info("-" * 90)
    for i in range(min(show_samples, len(predictions))):
        pred_idx = predictions[i].item()
        true_idx = targets[i].item()
        pred_prob = probabilities[i, pred_idx].item()
        
        status = "✅" if pred_idx == true_idx else "❌"
        logger.info(
            f"{status} True: {class_names[true_idx]:<20} | "
            f"Predicted: {class_names[pred_idx]:<20} | "
            f"Confidence: {pred_prob:.4f}"
        )
    logger.info("-" * 90)
    
    return accuracy, conf_matrix


def main():
    """Main testing routine"""
    logger.info("🚀 Starting CNN model testing...")
    
    checkpoint_path = Path("checkpoints/best_model.pt")
    datasets_dir = "datasets"
    batch_size = 32
    img_size = 224
    
    if not checkpoint_path.exists():
        logger.error(f"❌ Model checkpoint not found at {checkpoint_path}")
        logger.error("Please train the model first by running: python cnn_train.py")
        return
    
    device = get_best_device()
    logger.info(f"✅ Device: {device}")
    
    test_hf_dataset, class_names = load_test_data(
        datasets_dir=datasets_dir,
        img_size=img_size
    )
    
    test_transforms = get_transforms(img_size, is_training=False)
    test_dataset = BeansDataset(test_hf_dataset, transform=test_transforms)
    
    pin_mem = device.type == "cuda"
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if device.type == "mps" else 2,
        pin_memory=pin_mem
    )
    
    model = create_efficientnet_model(num_classes=len(class_names), freeze_features=False)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logger.info(f"✅ Model loaded from {checkpoint_path}")
    
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

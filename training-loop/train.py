"""
PyTorch Training Loop: A Production-Ready Template

This script demonstrates a clean, reusable training loop with:
- Device-agnostic execution (CUDA/MPS/CPU)
- Train/validation split with proper eval mode
- Loss tracking and visualization
- Optional mixed precision training (AMP)
- Best model checkpointing

Dataset: HuggingFace scikit-learn/iris (150 samples, 4 features, 3 classes)
"""

import logging
import os
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, DownloadMode
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress verbose HuggingFace and HTTP logging
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    use_amp: bool = False  # Mixed precision training
    checkpoint_dir: str = "checkpoints"
    datasets_dir: str = "datasets"


def get_best_device() -> torch.device:
    """
    Automatically select the best available device.
    Priority: CUDA -> MPS -> CPU
    """
    # CUDA: NVIDIA GPUs (Linux/Windows)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS: Apple Silicon GPUs (M1/M2/M3)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    # Fallback to CPU
    return torch.device("cpu")


def load_iris_tensors(datasets_dir: str = "datasets", seed: int = 42):
    """
    Load Iris dataset from HuggingFace and prepare train/val tensors.
    
    Args:
        datasets_dir: Directory to cache the dataset
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
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
    # Convert string labels to integer indices
    if isinstance(species[0], str):
        names = sorted(set(species))
        name_to_id = {name: i for i, name in enumerate(names)}
        y = torch.tensor([name_to_id[s] for s in species], dtype=torch.long)
    else:
        y = torch.tensor(species, dtype=torch.long)
    
    # Reproducible train/val split
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    idx = torch.randperm(n, generator=g)
    
    # 80/20 train/val split
    val_size = max(1, int(0.2 * n))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Normalize features using training statistics
    # Important: fit on train, apply to both train and val
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    
    return X_train, y_train, X_val, y_val


class TinyMLP(nn.Module):
    """
    Simple MLP for multi-class classification.
    Uses Dropout to make train()/eval() behavior visible.
    """
    def __init__(self, in_features: int = 4, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),  # Disabled during eval()
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),  # Raw logits (no softmax)
        )
    
    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Compute accuracy from raw logits"""
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler = None
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: The neural network
        loader: Training data loader
        optimizer: Optimizer instance
        loss_fn: Loss function
        device: Device to run on
        use_amp: Whether to use automatic mixed precision
        scaler: GradScaler for AMP (required if use_amp=True)
        
    Returns:
        Average training loss for the epoch
    """
    model.train()  # Enable Dropout, BatchNorm training mode
    total_loss = 0.0
    n_batches = 0
    
    for xb, yb in loader:
        # Move batch to device (GPU/CPU)
        xb = xb.to(device)
        yb = yb.to(device)
        
        if use_amp:
            # Mixed precision: FP16 forward/backward for speed
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = loss_fn(logits, yb)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard FP32 training
            logits = model(xb)
            loss = loss_fn(logits, yb)
            
            optimizer.zero_grad(set_to_none=True)  # Clear gradients from previous step
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(1, n_batches)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """
    Evaluate the model on validation/test data.
    
    Args:
        model: The neural network
        loader: Validation data loader
        loss_fn: Loss function
        device: Device to run on
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()  # Disable Dropout, BatchNorm uses running stats
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    
    # No gradient tracking for validation (faster, less memory)
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            logits = model(xb)
            loss = loss_fn(logits, yb)
            acc = accuracy_from_logits(logits, yb)
            
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1
    
    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


def plot_loss_curves(
    epoch_count: list[int],
    train_loss_values: list[float],
    val_loss_values: list[float],
    save_path: str = "training-loop/loss_curve.png"
):
    """
    Plot and save training and validation loss curves.
    
    Args:
        epoch_count: List of epoch numbers
        train_loss_values: Training loss per epoch
        val_loss_values: Validation loss per epoch
        save_path: Where to save the plot
    """
    # Create loss curve plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_count, train_loss_values, label='Train Loss', linewidth=2)
    plt.plot(epoch_count, val_loss_values, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"📊 Loss curve saved to {save_path}")


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    config: TrainingConfig
) -> tuple[list[float], list[float], list[int]]:
    """
    Execute the training loop.
    
    Returns:
        Tuple of (train_loss_values, val_loss_values, epoch_count)
    """
    # Initialize GradScaler for mixed precision (CUDA only)
    scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    
    # Track metrics for plotting
    train_loss_values = []
    val_loss_values = []
    epoch_count = []
    
    best_val_loss = float("inf")
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    logger.info(f"\n🚀 Starting training for {config.epochs} epochs...")
    logger.info("-" * 60)
    
    # Main training loop
    for epoch in range(1, config.epochs + 1):
        # Train for one epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            use_amp=config.use_amp, scaler=scaler
        )
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        
        # Track metrics
        train_loss_values.append(train_loss)
        val_loss_values.append(val_loss)
        epoch_count.append(epoch)
        
        # Save checkpoint if validation improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
        
        # Log progress
        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.3f}"
            )
    
    logger.info("-" * 60)
    logger.info(f"✅ Training complete! Best val loss: {best_val_loss:.4f}")
    logger.info(f"💾 Best model saved to {checkpoint_dir / 'best_model.pt'}")
    
    return train_loss_values, val_loss_values, epoch_count


def main():
    """Setup and execute training"""
    config = TrainingConfig()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    device = get_best_device()
    logger.info(f"✅ Device: {device}")
    
    # Load data
    X_train, y_train, X_val, y_val = load_iris_tensors(
        datasets_dir=config.datasets_dir,
        seed=42
    )
    logger.info(f"✅ Dataset loaded: {X_train.shape[0]} train, {X_val.shape[0]} val")
    
    # Create data loaders for mini-batch training
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config.batch_size,
        shuffle=True  # Shuffle training data each epoch
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=64,
        shuffle=False  # No need to shuffle validation
    )
    
    # Initialize model, loss, optimizer
    model = TinyMLP(in_features=4, num_classes=3).to(device)  # Move model to device
    loss_fn = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay  # L2 regularization
    )
    
    # Train
    train_loss_values, val_loss_values, epoch_count = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config=config
    )
    
    # Plot results
    plot_loss_curves(epoch_count, train_loss_values, val_loss_values)


if __name__ == "__main__":
    main()

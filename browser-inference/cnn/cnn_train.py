"""
PyTorch CNN Training for Image Classification - Beans Dataset

This script demonstrates a CNN training pipeline with:
- Device-agnostic execution (CUDA/MPS/CPU)
- Image data loading from HuggingFace
- Train/validation split with proper eval mode
- Loss tracking and visualization
- Best model checkpointing
- Data augmentation

Dataset: HuggingFace beans dataset (1,034 train, 133 val images)
Classes: 3 (angular_leaf_spot, bean_rust, healthy)
"""

import logging
import os
from pathlib import Path
from dataclasses import dataclass

os.environ['TORCH_HOME'] = str(Path(__file__).parent / 'models')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DownloadMode
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, models
from PIL import Image

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


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    use_amp: bool = False
    checkpoint_dir: str = "checkpoints"
    datasets_dir: str = "datasets"
    img_size: int = 224
    unfreeze_after_epoch: int = 10


def get_best_device() -> torch.device:
    """
    Automatically select the best available device.
    Priority: CUDA -> MPS -> CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_transforms(img_size: int = 224, is_training: bool = True):
    """
    Get image transforms compatible with ImageNet pretrained models.
    Uses stronger augmentation for better generalization.
    
    Args:
        img_size: Target image size
        is_training: If True, apply data augmentation
        
    Returns:
        torchvision.transforms.Compose object
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def load_beans_dataset(datasets_dir: str = "datasets", img_size: int = 224):
    """
    Load Beans dataset from HuggingFace.
    
    Args:
        datasets_dir: Directory to cache the dataset
        img_size: Target image size for resizing
        
    Returns:
        Tuple of (train_dataset, val_dataset, class_names)
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
    
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    
    class_names = ["angular_leaf_spot", "bean_rust", "healthy"]
    
    logger.info(f"✅ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
    logger.info(f"   Classes: {', '.join(class_names)}")
    
    return train_dataset, val_dataset, class_names


class BeansDataset(torch.utils.data.Dataset):
    """Custom Dataset wrapper for Beans with transforms"""
    
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["labels"]
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_efficientnet_model(num_classes: int = 3, freeze_features: bool = True):
    """
    Create EfficientNet-B0 with pretrained ImageNet weights.
    Transfer learning approach: freeze feature extractor, train only classifier.
    
    Args:
        num_classes: Number of output classes
        freeze_features: If True, freeze feature extractor layers
        
    Returns:
        EfficientNet model ready for training
    """
    # Load pretrained EfficientNet-B0 weights from ImageNet
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    
    # Freeze backbone layers to retain pretrained features
    # Only the classifier head will be trained initially
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False  # Don't update these weights during training
    
    # Replace the classifier head with a new one for our 3 classes
    # Original classifier outputs 1000 classes (ImageNet), we need 3
    in_features = model.classifier[1].in_features  # Get input size (1280 for EfficientNet-B0)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),  # Dropout for regularization
        nn.Linear(in_features, num_classes)  # Final classification layer
    )
    
    return model


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
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = loss_fn(logits, yb)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
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
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    
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


def smooth_curve(values: list[float], weight: float = 0.6) -> np.ndarray:
    """
    Apply exponential moving average smoothing.
    
    Args:
        values: List of values to smooth
        weight: Smoothing factor (0-1), higher = more smoothing
        
    Returns:
        Smoothed numpy array
    """
    smoothed = []
    last = values[0]
    for value in values:
        smoothed_val = last * weight + value * (1 - weight)
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def plot_loss_curves(
    epoch_count: list[int],
    train_loss_values: list[float],
    val_loss_values: list[float],
    save_path: str = "loss_curve.png"
):
    """
    Plot and save training and validation loss curves with smoothing.
    """
    train_smooth = smooth_curve(train_loss_values, weight=0.6)
    val_smooth = smooth_curve(val_loss_values, weight=0.6)
    
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('default')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(epoch_count, train_smooth, label='Train Loss (smoothed)', 
            linewidth=2.5, color='#2E86AB', alpha=0.9)
    ax.plot(epoch_count, val_smooth, label='Val Loss (smoothed)', 
            linewidth=2.5, color='#A23B72', alpha=0.9)
    
    ax.scatter(epoch_count, train_loss_values, s=30, color='#2E86AB', 
               alpha=0.3, label='Train Loss (raw)')
    ax.scatter(epoch_count, val_loss_values, s=30, color='#A23B72', 
               alpha=0.3, label='Val Loss (raw)')
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('Training and Validation Loss Curves', fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"📊 Loss curve saved to {save_path}")


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    config: TrainingConfig,
    scheduler = None
) -> tuple[list[float], list[float], list[int]]:
    """
    Execute the training loop with early stopping.
    
    Returns:
        Tuple of (train_loss_values, val_loss_values, epoch_count)
    """
    scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    
    train_loss_values = []
    val_loss_values = []
    epoch_count = []
    
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience = 8
    patience_counter = 0
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n🚀 Starting training for {config.epochs} epochs...")
    logger.info("Phase 1: Training classifier only (frozen backbone)")
    logger.info("-" * 60)
    
    for epoch in range(1, config.epochs + 1):
        # Two-phase training: start fine-tuning entire network after initial epochs
        if hasattr(config, 'unfreeze_after_epoch') and epoch == config.unfreeze_after_epoch:
            logger.info(f"\n🔓 Epoch {epoch}: Unfreezing backbone for fine-tuning")
            logger.info("Phase 2: Fine-tuning entire network")
            # Unfreeze all layers for fine-tuning
            for param in model.features.parameters():
                param.requires_grad = True  # Now we update all weights
            # Reduce learning rate for fine-tuning to avoid destroying pretrained features
            optimizer.param_groups[0]['lr'] = config.learning_rate * 0.1
            logger.info(f"Reduced learning rate to {optimizer.param_groups[0]['lr']:.6f}")
            logger.info("-" * 60)
        
        # Train for one epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            use_amp=config.use_amp, scaler=scaler
        )
        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        
        # Track metrics for plotting
        train_loss_values.append(train_loss)
        val_loss_values.append(val_loss)
        epoch_count.append(epoch)
        
        # Save model if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0  # Reset early stopping counter
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
        else:
            patience_counter += 1  # Increment early stopping counter
        
        # Reduce learning rate if validation loss plateaus
        if scheduler is not None:
            scheduler.step(val_loss)
        
        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%) | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
        
        if patience_counter >= patience:
            logger.info(f"\n⚠️  Early stopping triggered after {epoch} epochs (patience={patience})")
            break
    
    logger.info("-" * 60)
    logger.info(f"✅ Training complete! Best val accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    logger.info(f"✅ Best val loss: {best_val_loss:.4f}")
    logger.info(f"💾 Best model saved to {checkpoint_dir / 'best_model.pt'}")
    
    return train_loss_values, val_loss_values, epoch_count


def main():
    """Setup and execute training"""
    config = TrainingConfig()
    
    torch.manual_seed(42)
    device = get_best_device()
    logger.info(f"✅ Device: {device}")
    
    train_hf_dataset, val_hf_dataset, class_names = load_beans_dataset(
        datasets_dir=config.datasets_dir,
        img_size=config.img_size
    )
    
    train_transforms = get_transforms(config.img_size, is_training=True)
    val_transforms = get_transforms(config.img_size, is_training=False)
    
    train_dataset = BeansDataset(train_hf_dataset, transform=train_transforms)
    val_dataset = BeansDataset(val_hf_dataset, transform=val_transforms)
    
    pin_mem = device.type == "cuda"
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0 if device.type == "mps" else 2,
        pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0 if device.type == "mps" else 2,
        pin_memory=pin_mem
    )
    
    model = create_efficientnet_model(num_classes=len(class_names), freeze_features=True)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Model initialized: {total_params:,} parameters")
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    train_loss_values, val_loss_values, epoch_count = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config=config,
        scheduler=scheduler
    )
    
    plot_loss_curves(epoch_count, train_loss_values, val_loss_values)


if __name__ == "__main__":
    main()

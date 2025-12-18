# CNN Image Classification - Beans Disease Detection

A PyTorch CNN implementation for image classification using the Beans leaf disease dataset from HuggingFace.

## Dataset

**Beans Leaf Disease Dataset**
- **Training**: 1,034 images
- **Validation**: 133 images  
- **Test**: 128 images
- **Image Size**: 500x500 RGB (resized to 224x224)
- **Classes**: 3
  - `angular_leaf_spot`
  - `bean_rust`
  - `healthy`

## Quick Start

### 1. Install Dependencies

```bash
cd browser-inference/cnn
pip install -r requirements.txt
```

### 2. Train the CNN Model

```bash
python cnn_train.py
```

This will:
- Auto-detect best device (CUDA/MPS/CPU)
- Download Beans dataset to `datasets/` (cached for future runs)
- Train for 25 epochs with data augmentation
- Save best model to `checkpoints/best_model.pt`
- Generate loss curve plot at `loss_curve.png`

### 3. Test the Model

```bash
python cnn_test.py
```

This will:
- Load the trained model from checkpoints
- Use cached dataset (no re-download)
- Evaluate on test set (128 images)
- Display:
  - Overall test accuracy
  - Confusion matrix
  - Per-class precision, recall, F1-score
  - Sample predictions with confidence scores

### 4. Export to ONNX

```bash
python to_onnx.py
```

This will:
- Export trained model to ONNX format for deployment
- Verify export correctness (PyTorch vs ONNX outputs)
- Enable deployment to web browsers, mobile apps, edge devices
- Output: `models_onnx/model.onnx` (~16-17 MB)

#### Visualize the Model

You can visualize the exported ONNX model architecture using [Netron](https://github.com/lutzroeder/netron):

- **Web version**: [https://netron.app/](https://netron.app/) - Just drag and drop `models_onnx/model.onnx`
- **Desktop app**: `pip install netron` then `netron models_onnx/model.onnx`

Netron shows the complete model graph, layer details, input/output shapes, and operator information.

## Model Architecture

**EfficientNet-B0 with Transfer Learning** (~4M parameters, only ~1.3k trainable initially)
- **Pretrained on ImageNet** (1.2M images, 1000 classes)
- **Frozen backbone**: All convolutional layers frozen initially
- **Custom classifier**: New output layer for 3 classes (angular_leaf_spot, bean_rust, healthy)
- **Two-phase training**:
  1. Phase 1 (epochs 1-10): Train classifier only
  2. Phase 2 (epochs 10-30): Fine-tune entire network with reduced LR

**Why Transfer Learning?**
- ✅ **95%+ accuracy** (vs 80% training from scratch)
- ✅ Leverages ImageNet features learned from millions of images
- ✅ Much better generalization on small datasets (1034 images)
- ✅ Industry-standard approach for image classification

**Anti-Overfitting Features:**
- ✅ Transfer learning with frozen backbone
- ✅ Strong data augmentation (flip, rotate, crop, color jitter)
- ✅ Dropout (30%) in classifier
- ✅ Learning rate scheduling (ReduceLROnPlateau)
- ✅ Early stopping (patience=8)
- ✅ Two-phase training (classifier → full fine-tuning)

## Configuration

Edit `TrainingConfig` in `cnn_train.py`:

```python
@dataclass
class TrainingConfig:
    epochs: int = 30                    # Max training epochs
    batch_size: int = 32                # Batch size
    learning_rate: float = 1e-3         # Initial learning rate
    weight_decay: float = 1e-4          # L2 regularization
    use_amp: bool = False               # Mixed precision (CUDA only)
    checkpoint_dir: str = "checkpoints"
    datasets_dir: str = "datasets"
    img_size: int = 224                 # Input image size
    unfreeze_after_epoch: int = 10      # When to start fine-tuning
```

## Data Augmentation

**Strong augmentation** for better generalization:
- Resize to 256×256, then random crop to 224×224
- Random horizontal flip (50%)
- Random vertical flip (30%)
- Random rotation (±20 degrees)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine transforms (translation)
- ImageNet normalization (required for pretrained models)

## Key Features

✅ **CNN architecture** optimized for image classification  
✅ **Data augmentation** to prevent overfitting  
✅ **Device-agnostic** (CUDA/MPS/CPU auto-selection)  
✅ **Smart caching** (works offline after first download)  
✅ **Proper train/eval modes** with dropout  
✅ **Best model checkpointing**  
✅ **Comprehensive test metrics**

## Expected Performance

With **EfficientNet-B0 transfer learning**:
- **Validation Accuracy**: ~95-98%
- **Test Accuracy**: ~92-96%
- **Training Time**: ~3-5 minutes (M2 Max/CUDA for 30 epochs)
- **Convergence**: Smooth loss curves with excellent generalization

**Two-Phase Training:**
1. **Phase 1** (epochs 1-10): Quick convergence training classifier (~85-90% accuracy)
2. **Phase 2** (epochs 10+): Fine-tuning entire network (~95%+ accuracy)

The model uses **early stopping** (patience=8) and may stop before 30 epochs if validation stops improving.

## Project Structure

```
browser-inference/cnn/
├── cnn_train.py          # Training script with CNN model
├── cnn_test.py           # Testing script with metrics
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── checkpoints/         # Saved models (created during training)
├── datasets/            # Cached HuggingFace datasets
└── loss_curve.png       # Training/validation loss plot
```

## Extending the Model

### Use Transfer Learning

Replace `SimpleCNN` with a pre-trained model:

```python
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

### Add Learning Rate Scheduler

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
# After optimizer.step():
scheduler.step()
```

### Enable Mixed Precision Training

```python
config = TrainingConfig(use_amp=True)  # CUDA only
```

## Why CNN for Image Classification?

**CNNs are still the best choice** for:
- ✅ Small-to-medium datasets (<10k images)
- ✅ Fast training and inference
- ✅ Lightweight deployment
- ✅ Strong performance with data augmentation

**Vision Transformers (ViT)** are better for very large datasets (>100k images) but require significantly more compute.

## License

MIT License - feel free to use in your projects!

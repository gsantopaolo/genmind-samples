# CNN Image Classification - Beans Disease Detection

A production-ready image classifier using **EfficientNet-B0 transfer learning** on the Beans leaf disease dataset. Achieves **97.66% test accuracy** with PyTorch and ONNX export for deployment.

## 🎯 Highlights

- ✅ **97.66% test accuracy** using transfer learning
- ✅ **EfficientNet-B0** pretrained on ImageNet
- ✅ **ONNX export** for cross-platform deployment
- ✅ **Performance comparison** (PyTorch vs ONNX Runtime)
- ✅ **Production-ready** with comprehensive metrics

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

### 2. Train the Model

```bash
python cnn_train.py
```

This will:
- Auto-detect best device (CUDA/MPS/CPU)
- Download Beans dataset to `datasets/` (cached for future runs)
- **Two-phase training**: Classifier only (epochs 1-10), then full fine-tuning (10-30)
- Save best model to `checkpoints/best_model.pt`
- Generate loss curve plot at `loss_curve.png`
- **Expected**: ~3-5 minutes on M2 Max/CUDA, achieves ~95-98% validation accuracy

### 3. Test the Model (PyTorch)

```bash
python cnn_test.py
```

This will:
- Load the trained PyTorch model from checkpoints
- Evaluate on test set (128 images)
- **Display metrics**:
  - Overall test accuracy (~97.66%)
  - Inference time per batch
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

#### Test ONNX Model

```bash
python onnx_test.py
```

This will:
- Run inference using ONNX Runtime
- Compare performance with PyTorch model
- Display identical metrics (accuracy, confusion matrix, per-class metrics)
- Show inference time comparison

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

## 📊 Actual Performance Results

**Tested on:** MacBook Pro M2 Max (32GB RAM)

### Achieved Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **97.66%** (125/128 correct) |
| **Validation Accuracy** | 95-98% |
| **Training Time** | ~3-5 min |
| **Inference Time (PyTorch/MPS)** | 83 ms/batch |
| **Inference Time (ONNX/CPU)** | 168 ms/batch |
| **Model Size (ONNX)** | 15.29 MB |
| **Parameters** | ~4M total, ~1.3k trainable initially |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|
| angular_leaf_spot | 100.0% | 93.0% | 96.4% |
| bean_rust | 93.5% | 100.0% | 96.6% |
| healthy | 100.0% | 100.0% | 100.0% |

### Confusion Matrix (Test Set)

```
True\Pred        angular_leaf_spot    bean_rust    healthy
----------------------------------------------------------------
angular_leaf_spot         40              3            0
bean_rust                  0             43            0
healthy                    0              0           42
```

**Only 3 errors out of 128 test images!**

### Training Process

**Two-Phase Training:**
1. **Phase 1** (epochs 1-10): Train classifier only (~90% accuracy)
2. **Phase 2** (epochs 10-30): Fine-tune entire network (~97%+ accuracy)
3. **Early stopping** (patience=8) prevents overfitting

### Performance Comparison (M2 Max 32GB) - Real Measured Results

**Test Setup**: 128 test images, EfficientNet-B0, MacBook Pro M2 Max 32GB

| Implementation | Device | Batch | Time/Image | Throughput | Accuracy |
|---------------|--------|-------|------------|------------|----------|
| **PyTorch MPS** | M2 Max GPU | 32 | **2.63 ms** ⭐ | 380 img/sec | 97.66% |
| **ONNX CoreML** | GPU/ANE/CPU | 1 | **4.77 ms** | 210 img/sec | 97.66% |
| **Web WebGL** | Browser GPU | 1 | **26.76 ms** | 37 img/sec | 97.66% |

**Key Findings:**
- ✅ **All three achieve identical 97.66% accuracy** - lossless model conversion
- ⚡ **PyTorch MPS fastest** at 2.63ms/image with batch processing
- 🥈 **ONNX CoreML only 1.8x slower** (4.77ms) - excellent for production
- 🌐 **Web WebGL 10x slower** (26.76ms) but runs entirely client-side
- 🎯 **Same 3 errors** out of 128 test images across all platforms
- 💾 **ONNX Runtime**: No PyTorch dependency (deployment 10x lighter)

**When to use:**
- **PyTorch + MPS**: Training, research, maximum speed batch inference
- **ONNX Runtime + CoreML**: Production Mac/iOS apps, edge deployment
- **Web + WebGL**: Interactive browser apps, client-side ML, zero server cost

See `../PERFORMANCE_COMPARISON.md` for detailed analysis.

## 📁 Project Structure

```
browser-inference/cnn/
├── cnn_train.py          # Training script (EfficientNet-B0)
├── cnn_test.py           # PyTorch testing with performance metrics
├── onnx_test.py          # ONNX Runtime testing with performance metrics
├── to_onnx.py            # Export model to ONNX format
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
├── checkpoints/          # Saved PyTorch models (*.pt)
│   └── best_model.pt     # Best model checkpoint
├── datasets/             # Cached HuggingFace datasets
├── models/               # Pretrained ImageNet weights
│   └── hub/checkpoints/  # EfficientNet-B0 weights
├── models_onnx/          # Exported ONNX models
│   └── model.onnx        # Production-ready ONNX model
└── loss_curve.png        # Training/validation loss plot
```

## 🚀 Deployment Options

### 1. PyTorch Deployment

Use `cnn_test.py` as a template:
```python
from cnn_train import create_efficientnet_model

model = create_efficientnet_model(num_classes=3)
model.load_state_dict(torch.load('checkpoints/best_model.pt'))
model.eval()
```

### 2. ONNX Runtime Deployment (Recommended)

Faster inference, no PyTorch dependency:
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('models_onnx/model.onnx')
outputs = session.run(None, {'input': image_array})
```

### 3. Web Deployment

Use ONNX.js for browser inference:
```javascript
import * as ort from 'onnxruntime-web';

const session = await ort.InferenceSession.create('model.onnx');
const results = await session.run(feeds);
```

### 4. Mobile Deployment

- **iOS**: Convert ONNX → CoreML
- **Android**: Use ONNX Runtime Mobile

## Why CNN for Image Classification?

**CNNs are still the best choice** for:
- ✅ Small-to-medium datasets (<10k images)
- ✅ Fast training and inference
- ✅ Lightweight deployment
- ✅ Strong performance with data augmentation

**Vision Transformers (ViT)** are better for very large datasets (>100k images) but require significantly more compute.

## License

MIT License - feel free to use in your projects!

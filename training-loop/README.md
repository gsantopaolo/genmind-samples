# PyTorch Training Loop Template

A production-ready PyTorch training loop with device-agnostic execution, proper train/eval modes, loss tracking, and comprehensive testing.

**📖 Full Blog Post:** [PyTorch Training Loop: A Production-Ready Template for AI Engineers](https://genmind.ch/posts/PyTorch-Training-Loop-A-Production-Ready-Template-for-AI-Engineers/)

## Quick Start

### 1. Clone repo and install Dependencies

```bash
git clone https://github.com/gsantopaolo/genmind-samples.git
cd genmind-samples/training-loop
conda create -n "genmind-samples" python=3.11.7  
conda activate genmind-samples
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

This will:
- Auto-detect best device (CUDA/MPS/CPU)
- Download Iris dataset to `datasets/` folder (cached for future runs)
- Train for 50 epochs
- Save best model to `checkpoints/best_model.pt`
- Generate loss curve plot at `training-loop/loss_curve.png`

**Note:** The dataset is cached locally in `datasets/`. Subsequent runs will use the cached version in **offline mode** (no network calls at all).

### 3. Test the Model

```bash
python test.py
```

This will:
- Load the trained model from `checkpoints/best_model.pt`
- Use cached dataset from `datasets/` (no re-download)
- Evaluate on unseen test data (different split than training)
- Display:
  - Overall test accuracy
  - Confusion matrix
  - Per-class precision, recall, F1-score
  - Sample predictions with confidence scores

## Files

- **`train.py`** - Main training script with clean, reusable functions
- **`test.py`** - Comprehensive testing script with detailed metrics
- **`requirements.txt`** - Python dependencies

## Key Features

### Training (`train.py`)
- ✅ Device-agnostic (CUDA/MPS/CPU auto-selection)
- ✅ Proper `train()`/`eval()` modes
- ✅ `torch.inference_mode()` for efficient validation
- ✅ Loss tracking and visualization
- ✅ Automatic checkpointing (saves best model)
- ✅ Optional mixed precision training (AMP)
- ✅ Logging with timestamps
- ✅ Smart dataset caching (works offline after first run)

### Testing (`test.py`)
- ✅ Comprehensive evaluation metrics
- ✅ Confusion matrix visualization
- ✅ Per-class precision/recall/F1
- ✅ Sample predictions with confidence
- ✅ Unseen test data (different from training)

## Configuration

Edit `TrainingConfig` in `train.py`:

```python
@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    use_amp: bool = False  # Toggle mixed precision
    checkpoint_dir: str = "checkpoints"
    datasets_dir: str = "datasets"
```

## Example Output

### Training
```
2025-12-18 15:25:30 - INFO - ✅ Device: cuda
2025-12-18 15:25:30 - INFO - ✅ Dataset loaded: 120 train, 30 val
2025-12-18 15:25:30 - INFO - 🚀 Starting training for 50 epochs...
2025-12-18 15:25:31 - INFO - Epoch 001 | Train Loss: 1.0854 | Val Loss: 1.0234 | Val Acc: 0.367
2025-12-18 15:25:32 - INFO - Epoch 010 | Train Loss: 0.3421 | Val Loss: 0.2891 | Val Acc: 0.933
...
2025-12-18 15:25:45 - INFO - ✅ Training complete! Best val loss: 0.0567
2025-12-18 15:25:45 - INFO - 💾 Best model saved to checkpoints/best_model.pt
2025-12-18 15:25:45 - INFO - 📊 Loss curve saved to training-loop/loss_curve.png
```

### Testing
```
2025-12-18 15:26:00 - INFO - ✅ Device: cuda
2025-12-18 15:26:00 - INFO - ✅ Test data loaded: 30 samples, 3 classes
2025-12-18 15:26:00 - INFO - 🧪 Running inference on test data...
2025-12-18 15:26:00 - INFO - ✅ Overall Test Accuracy: 0.9667 (96.67%)

📊 Confusion Matrix:
------------------------------------------------------------
True\Pred    Iris-setosa Iris-versicolor Iris-virginica
------------------------------------------------------------
Iris-setosa           10        0        0
Iris-versicolor        0        9        1
Iris-virginica         0        0       10
------------------------------------------------------------

📈 Per-Class Metrics:
------------------------------------------------------------
Class           Precision     Recall  F1-Score
------------------------------------------------------------
Iris-setosa         1.000      1.000     1.000
Iris-versicolor     1.000      0.900     0.947
Iris-virginica      0.909      1.000     0.952
------------------------------------------------------------
```

## Extending the Template

### Add Gradient Clipping
```python
# In train_one_epoch(), after loss.backward():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Add Learning Rate Scheduler
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
# After optimizer.step() in training loop:
scheduler.step()
```

### Use Your Own Dataset

Replace `load_iris_tensors()` in `train.py`:

```python
def load_your_data():
    # Your data loading logic
    X_train, y_train = ...
    X_val, y_val = ...
    return X_train, y_train, X_val, y_val
```

## License

MIT License - feel free to use in your projects!

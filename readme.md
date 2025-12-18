# genmind-samples

Code samples for Gian Paolo's blog at [genmind.ch](https://genmind.ch)

## Blog Post Code

This repository contains runnable code examples that accompany blog posts on genmind.ch. Each folder corresponds to a specific post.

### 📁 [`browser-inference/`](./browser-inference) ⭐ NEW

**Complete ML Deployment Pipeline: From Training to Browser**

Train a CNN model and deploy it across **three platforms** with GPU acceleration:

| Platform | Device | Speed | Throughput | Accuracy |
|----------|--------|-------|------------|----------|
| **PyTorch MPS** | M2 Max GPU | **2.63 ms/image** ⭐ | 380 img/sec | 97.66% |
| **ONNX CoreML** | GPU/ANE/CPU | **4.77 ms/image** | 210 img/sec | 97.66% |
| **Web WebGL** | Browser GPU | **26.76 ms/image** | 37 img/sec | 97.66% |

**What's Included:**
- ✅ **PyTorch Training** on Apple M2 Max (MPS backend)
- ✅ **ONNX Export** with fixed batch size for WebGL compatibility
- ✅ **ONNX Runtime + CoreML** for Mac/iOS deployment
- ✅ **React TypeScript Web App** with GPU-accelerated inference
- ✅ **Comprehensive benchmarks** across all three platforms
- ✅ **Performance analysis** and deployment recommendations

**Quick Start:**
```bash
cd browser-inference/cnn
python3 cnn_train.py        # Train model
python3 to_onnx.py          # Export to ONNX
python3 cnn_test.py         # Benchmark PyTorch
python3 onnx_test.py        # Benchmark ONNX

cd ../web-app
npm install && npm run dev  # Run web app
```

See `browser-inference/PERFORMANCE_COMPARISON.md` for detailed analysis.

### 📁 [`training-loop/`](./training-loop)

**Blog Post:** [PyTorch Training Loop: A Production-Ready Template for AI Engineers](https://genmind.ch/posts/PyTorch-Training-Loop-A-Production-Ready-Template-for-AI-Engineers/)

A complete, production-ready PyTorch training loop template with:
- ✅ Device-agnostic execution (CUDA/MPS/CPU)
- ✅ Smart dataset caching (works offline)
- ✅ Professional logging with timestamps
- ✅ Automatic checkpointing
- ✅ Loss tracking and visualization
- ✅ Comprehensive testing utilities

---

## Contributing

Found an issue or want to suggest improvements? Open an issue or PR!

## License

MIT License - see individual folders for specific licensing information.

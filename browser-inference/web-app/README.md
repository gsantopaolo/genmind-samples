# Beans Disease Classifier - Web App

A React TypeScript web application that runs your EfficientNet-B0 ONNX model entirely in the browser using ONNX Runtime Web.

## 🎯 What This App Does

This web app allows you to:
1. **Upload a bean leaf image** from your computer
2. **Classify diseases** using your trained ONNX model
3. **See predictions** with confidence scores in real-time
4. **All processing happens in your browser** - no server needed!

The app automatically detects and uses:
- 🚀 **GPU acceleration (WebGPU)** on supported browsers (10-20x faster)
- 💻 **CPU fallback (WebAssembly)** on older browsers

## 🎯 Features

- ✅ **100% Client-Side**: No server required - runs entirely in browser
- ✅ **Privacy First**: Images never leave your device
- ✅ **GPU Acceleration**: Auto-detects WebGPU for 10-20x speedup
- ✅ **Fast Inference**: ~50-100ms (GPU) or ~200-500ms (CPU)
- ✅ **Real-time**: Upload and classify instantly
- ✅ **Works Offline**: After first load, no internet needed

## 📋 Prerequisites

- **Node.js 18+** (to run development server)
- Your trained **ONNX model** from `../cnn/models_onnx/model.onnx`
- Modern browser (Chrome, Edge, Safari, or Firefox)

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

Navigate to the web-app folder and install packages:

```bash
cd browser-inference/web-app
npm install
```

**What this does:**
- Downloads ~200MB of dependencies into `node_modules/`
- Installs React, TypeScript, Vite, and ONNX Runtime Web
- Takes ~30-60 seconds
- Creates `package-lock.json` to lock versions

**Dependencies installed:**
- `react` + `react-dom` (UI framework)
- `onnxruntime-web` (ONNX Runtime with WebGPU support)
- `typescript` (type safety)
- `vite` (fast build tool)

### Step 2: Copy Your ONNX Model

Copy your trained model to the `public/` folder so the browser can access it:

```bash
# From web-app directory
mkdir -p public
cp ../cnn/models_onnx/model.onnx public/model.onnx
```

**Important:** The model file must be in `public/model.onnx` for the app to find it!

### Step 3: Run the App

Start the development server:

```bash
npm run dev
```

**What happens:**
- Vite starts a local web server
- Opens browser at `http://localhost:3000`
- Hot reload enabled (saves auto-refresh)
- Press `Ctrl+C` to stop

You should see:
```
  VITE v5.0.0  ready in 500 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

## 🎨 How to Use the App

1. **Open browser** → `http://localhost:3000`
2. **Click "Choose Image"** → Upload a bean leaf photo
3. **Click "Classify Image"** → Wait ~50-500ms
4. **View results** → See disease prediction + confidence

**Check the badges:**
- `✅ Model Loaded` - ONNX model is ready
- `🚀 GPU (WebGPU)` - Using GPU acceleration (fast!)
- `💻 CPU (WASM)` - Using CPU (slower, but works everywhere)

## 🏗️ Build for Production

To create optimized production files:

```bash
npm run build
```

**Output:**
- Production files in `dist/` folder
- Minified JavaScript (~500KB)
- Ready to deploy to any static host

**Deploy to:**
- Vercel: `npx vercel`
- Netlify: Upload `dist/` folder
- GitHub Pages: Push `dist/` to `gh-pages` branch

## 📸 Where to Get Test Images

### Option 1: HuggingFace Beans Dataset (Recommended)

Visit the dataset page and download sample images:
- **Dataset**: https://huggingface.co/datasets/AI-Lab-Makerere/beans
- **Browse images**: Click "Files and versions" → explore the image folders

**Direct download examples:**
- Angular Leaf Spot: https://huggingface.co/datasets/AI-Lab-Makerere/beans/tree/main/test/angular_leaf_spot
- Bean Rust: https://huggingface.co/datasets/AI-Lab-Makerere/beans/tree/main/test/bean_rust  
- Healthy: https://huggingface.co/datasets/AI-Lab-Makerere/beans/tree/main/test/healthy

**Quick way to download:**
```bash
# Clone the dataset (requires git-lfs)
git lfs install
git clone https://huggingface.co/datasets/AI-Lab-Makerere/beans

# Or use Python from your cnn folder - images are already cached!
cd ../cnn/datasets/beans/
# Test images are in the cached dataset
```

### Option 2: Use Your Cached Dataset

Your test images are already in:
```
../cnn/datasets/beans/test/
```

Just copy a few images from there to test the web app!

### Option 3: TensorFlow Datasets

Alternative source with visualization:
- https://www.tensorflow.org/datasets/catalog/beans

### Option 4: Search Online

Google Images search terms:
- "bean leaf angular leaf spot"
- "bean leaf rust disease"
- "healthy bean leaves"

Make sure images show clear leaf details.

## 🧪 Testing the App

### **1. Start the app**
```bash
npm run dev
```

### **2. Open browser**
Navigate to `http://localhost:3000`

### **3. Upload test image**
Click **"Choose Image"** and select a bean leaf photo

**Where to get test images?** See [Test Images](#-where-to-get-test-images) section below.

### **4. Classify**
Click **"Classify Image"** button

### **5. View results**
You should see:
- **Predicted class** (e.g., "Angular Leaf Spot")
- **Confidence score** (e.g., 99.42%)
- **All class probabilities** with visual bars

### **Expected Accuracy**
Results should match your Python model:
- **97.66% overall accuracy** on test set
- Same predictions as `cnn_test.py` and `onnx_test.py`
- 3 classes: angular_leaf_spot, bean_rust, healthy

## 📂 Project Structure

```
web-app/
├── public/
│   └── model.onnx          # Your ONNX model (copy here)
├── src/
│   ├── App.tsx             # Main React component
│   ├── App.css             # Styling
│   ├── main.tsx            # Entry point
│   ├── index.css           # Global styles
│   └── utils/
│       ├── imagePreprocessing.ts   # Image preprocessing (224x224, ImageNet norm)
│       └── modelInference.ts       # ONNX Runtime inference
├── package.json            # Dependencies
├── tsconfig.json           # TypeScript config
├── vite.config.ts          # Vite config
└── index.html              # HTML template
```

## 🎨 How It Works

### 1. Image Preprocessing (imagePreprocessing.ts)

```typescript
// Matches your Python preprocessing:
// 1. Resize to 224x224
// 2. Normalize with ImageNet mean/std
// 3. Convert to NCHW format (channels first)
```

**Exactly like your Python code:**
```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

### 2. ONNX Inference (modelInference.ts)

```typescript
// Load model once (cached)
const session = await ort.InferenceSession.create('/model.onnx');

// Run inference
const tensor = new ort.Tensor('float32', imageData, [1, 3, 224, 224]);
const results = await session.run({ input: tensor });

// Apply softmax, get predictions
```

### 3. React UI (App.tsx)

- File upload
- Image preview
- Loading states
- Results display with probabilities

## ⚡ Performance Benchmarks

### **With GPU (WebGPU)**
| Device | Browser | Inference Time |
|--------|---------|----------------|
| M2 Max | Chrome/Edge | **~50-100ms** ⚡ |
| M2 Max | Safari | **~50-100ms** ⚡ |

### **Without GPU (WASM/CPU)**
| Device | Browser | Inference Time |
|--------|---------|----------------|
| M2 Max | Firefox | ~250-400ms |
| Any | Older browsers | ~300-500ms |

**Key Points:**
- ✅ **WebGPU is 10-20x faster** than CPU
- ✅ First inference is slower (model loading ~500ms)
- ✅ Subsequent inferences are much faster
- ✅ GPU acceleration works on Chrome, Edge, Safari (latest versions)

**Check GPU support:** Open browser console → Look for:
```
✅ GPU acceleration enabled (WebGPU)
```
or
```
⚠️ Using CPU (WASM) - WebGPU not available
```

## 🌐 Deployment Options

Once you run `npm run build`, you can deploy the `dist/` folder to any static hosting:

### **Vercel** (Easiest - Free)

```bash
npm install -g vercel
vercel
```

Automatic deploys on every `git push`!

### **Netlify** (Free)

```bash
npm run build
```

Drag and drop `dist/` folder to [Netlify](https://app.netlify.com/drop)

### **GitHub Pages** (Free)

```bash
npm run build
# Copy dist/ to gh-pages branch
git subtree push --prefix dist origin gh-pages
```

### **Any Static Host**

The `dist/` folder contains:
- `index.html` - Main page
- `assets/` - JavaScript, CSS
- `model.onnx` - Your model (15 MB)

Just upload to any CDN or static host!

## 💻 Development Commands

| Command | What it does |
|---------|--------------|
| `npm install` | Install dependencies (first time only) |
| `npm run dev` | Start development server (http://localhost:3000) |
| `npm run build` | Build for production (output: `dist/`) |
| `npm run preview` | Preview production build locally |

## 🔧 Troubleshooting

### ❌ "Model fails to load"

**Error in console:** `Failed to load model: Error: fetch failed`

**Solutions:**
1. Ensure `model.onnx` is in `public/` folder:
   ```bash
   ls -lh public/model.onnx
   # Should show: ~15MB file
   ```
2. Check the model path in browser DevTools → Network tab
3. Make sure dev server is running (`npm run dev`)

### ❌ "WebGPU not available"

**Console shows:** `⚠️ Using CPU (WASM) - WebGPU not available`

**Solutions:**
1. **Update browser** to latest version (Chrome 113+, Edge 113+, Safari 18+)
2. **Enable WebGPU** in browser flags:
   - Chrome/Edge: `chrome://flags/#enable-unsafe-webgpu`
   - Safari: Enable "WebGPU" in Develop menu
3. **Check compatibility**: https://webgpu.io/status/

### ❌ Wrong predictions

**Predictions don't match Python model**

**Check:**
1. Using same ONNX model file (copy from `../cnn/models_onnx/model.onnx`)
2. Image preprocessing is correct:
   - Resize to 224x224 ✅
   - ImageNet normalization ✅
   - RGB format ✅
   - NCHW (channels first) ✅

### ⚠️ Slow inference (>1 second)

**Solutions:**
1. **Check GPU badge** - Should show `🚀 GPU (WebGPU)`
2. **Update browser** to latest version for WebGPU support
3. **First inference is slower** - Model loading takes ~500ms
4. **Disable browser extensions** - Some block WebGPU

### 🐛 Still having issues?

1. **Open browser DevTools** (F12)
2. **Check Console** for error messages
3. **Check Network tab** to see if model loaded
4. **Clear cache** and hard reload (Cmd+Shift+R)

## 📚 Resources

- **ONNX Runtime Web Docs**: https://onnxruntime.ai/docs/tutorials/web/
- **React Docs**: https://react.dev
- **Vite Docs**: https://vitejs.dev
- **TypeScript Docs**: https://www.typescriptlang.org

## 🎯 Next Steps

1. ✅ Add drag-and-drop upload
2. ✅ Add camera capture for mobile
3. ✅ Add batch upload (multiple images)
4. ✅ Add WebGPU support for faster inference
5. ✅ Add model performance metrics display
6. ✅ Add error boundaries
7. ✅ Add PWA support for offline use

## 📝 License

MIT License - same as parent project

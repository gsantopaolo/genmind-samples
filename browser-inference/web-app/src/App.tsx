import { useState, useRef, useEffect } from 'react';
import * as ort from 'onnxruntime-web';
import { preprocessImage, createImagePreview } from './utils/imagePreprocessing';
import { loadModel, runInference, formatClassName, getClassEmoji, PredictionResult } from './utils/modelInference';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [executionProvider, setExecutionProvider] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [inferenceTime, setInferenceTime] = useState<number>(0);
  const [totalInferences, setTotalInferences] = useState<number>(0);
  const [avgInferenceTime, setAvgInferenceTime] = useState<number>(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const inferenceTimes = useRef<number[]>([]);

  // Load model on page load
  useEffect(() => {
    const loadModelOnMount = async () => {
      try {
        console.log('⏳ Loading model on page load...');
        const loadStart = performance.now();
        
        const session = await loadModel('/model.onnx');
        sessionRef.current = session;
        setModelLoaded(true);
        
        const loadEnd = performance.now();
        const loadTime = (loadEnd - loadStart).toFixed(2);
        
        setExecutionProvider('🚀 GPU (WebGL)');
        console.log(`✅ Model loaded in ${loadTime}ms`);
        console.log('✅ Ready for inference!');
      } catch (err) {
        setError(`Failed to load model: ${err}`);
        console.error(err);
      }
    };
    
    loadModelOnMount();
  }, []);

  // Model should already be loaded
  const ensureModelLoaded = () => {
    if (!sessionRef.current) {
      throw new Error('Model not loaded yet. Please wait...');
    }
    return sessionRef.current;
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file');
      return;
    }

    setSelectedFile(file);
    setPrediction(null);
    setError('');

    // Create preview
    try {
      const url = await createImagePreview(file);
      setPreviewUrl(url);
    } catch (err) {
      setError('Failed to create image preview');
    }
  };

  const handleClassify = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError('');

    try {
      // Model is already loaded on page load
      const session = ensureModelLoaded();

      // Preprocess image
      const preprocessed = await preprocessImage(selectedFile);

      // Run inference (timing happens inside runInference)
      const result = await runInference(session, preprocessed);
      setPrediction(result);
      
      // Update performance metrics
      setInferenceTime(result.inferenceTimeMs);
      inferenceTimes.current.push(result.inferenceTimeMs);
      setTotalInferences(prev => prev + 1);
      
      // Calculate average
      const avg = inferenceTimes.current.reduce((a, b) => a + b, 0) / inferenceTimes.current.length;
      setAvgInferenceTime(avg);
    } catch (err) {
      setError(`Classification failed: ${err}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl('');
    setPrediction(null);
    setError('');
    setInferenceTime(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🌱 Beans Disease Classifier</h1>
        <p className="subtitle">
          Upload a bean leaf image to detect diseases using ONNX Runtime Web
        </p>
        <div className="info-badges">
          <div className="info-badge">
            {modelLoaded ? '✅ Model Loaded' : '⚠️ Model not loaded yet'}
          </div>
          {executionProvider && (
            <div className="info-badge">
              {executionProvider}
            </div>
          )}
        </div>
      </header>

      <main className="main">
        <div className="upload-section">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="file-input"
            id="file-upload"
          />
          <label htmlFor="file-upload" className="upload-button">
            📁 Choose Image
          </label>
          
          {selectedFile && (
            <div className="file-info">
              Selected: {selectedFile.name}
            </div>
          )}
        </div>

        {previewUrl && (
          <div className="preview-section">
            <img src={previewUrl} alt="Preview" className="preview-image" />
            
            <div className="button-group">
              <button
                onClick={handleClassify}
                disabled={loading}
                className="classify-button"
              >
                {loading ? '🔄 Classifying...' : '🔍 Classify Image'}
              </button>
              <button onClick={handleReset} className="reset-button">
                🔄 Reset
              </button>
            </div>
          </div>
        )}

        {error && (
          <div className="error-message">
            ❌ {error}
          </div>
        )}

        {prediction && (
          <div className="results-section">
            <h2>Prediction Results</h2>
            
            <div className="prediction-card">
              <div className="prediction-main">
                <span className="class-emoji">{getClassEmoji(prediction.class)}</span>
                <div>
                  <h3>{formatClassName(prediction.class)}</h3>
                  <p className="confidence">
                    Confidence: {(prediction.confidence * 100).toFixed(2)}%
                  </p>
                </div>
              </div>
            </div>

            <div className="performance-metrics">
              <h3>⚡ Performance Metrics (WebGL GPU)</h3>
              <div className="metrics-grid">
                <div className="metric-item">
                  <span className="metric-label">Device:</span>
                  <span className="metric-value">WebGL (M2 Max GPU)</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Batch Size:</span>
                  <span className="metric-value">1</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Current Time:</span>
                  <span className="metric-value">{inferenceTime.toFixed(2)} ms/image</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Average Time:</span>
                  <span className="metric-value">{avgInferenceTime.toFixed(2)} ms/image</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Total Inferences:</span>
                  <span className="metric-value">{totalInferences}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Throughput:</span>
                  <span className="metric-value">{(1000 / avgInferenceTime).toFixed(1)} img/sec</span>
                </div>
              </div>
              <div className="comparison-note">
                <strong>📊 Comparison on M2 Max:</strong> PyTorch MPS: 2.63ms/image (⭐ fastest) | ONNX CoreML: 4.77ms/image | Web WebGL: ~{avgInferenceTime.toFixed(1)}ms/image (this!)
              </div>
            </div>

            <div className="probabilities">
              <h3>All Class Probabilities:</h3>
              {Object.entries(prediction.probabilities).map(([className, prob]) => (
                <div key={className} className="probability-bar-container">
                  <div className="probability-label">
                    {getClassEmoji(className)} {formatClassName(className)}
                  </div>
                  <div className="probability-bar-wrapper">
                    <div
                      className="probability-bar"
                      style={{ width: `${prob * 100}%` }}
                    />
                  </div>
                  <div className="probability-value">
                    {(prob * 100).toFixed(2)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="instructions">
          <h3>ℹ️ Instructions:</h3>
          <ol>
            <li>Place your <code>model.onnx</code> file in the <code>public/</code> folder</li>
            <li>Upload a bean leaf image (from test dataset or your own)</li>
            <li>Click "Classify Image" to get predictions</li>
          </ol>
          <p className="note">
            <strong>Note:</strong> Model runs entirely in your browser - no data sent to servers!
          </p>
        </div>
      </main>
    </div>
  );
}

export default App;

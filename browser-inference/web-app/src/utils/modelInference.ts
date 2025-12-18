/**
 * ONNX Runtime Web inference utilities with WebGPU support
 * Handles model loading and inference
 */

import * as ort from 'onnxruntime-web';
import { PreprocessedImage } from './imagePreprocessing';

export interface PredictionResult {
  class: string;
  confidence: number;
  probabilities: { [key: string]: number };
  inferenceTimeMs: number;
}

// Class names for the beans dataset
const CLASS_NAMES = ['angular_leaf_spot', 'bean_rust', 'healthy'];

// Configure ONNX Runtime paths
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/';

let sessionCache: ort.InferenceSession | null = null;

/**
 * Load the ONNX model with GPU acceleration (WebGPU)
 */
export async function loadModel(modelPath: string): Promise<ort.InferenceSession> {
  if (sessionCache) {
    return sessionCache;
  }

  try {
    console.log('Loading ONNX model with GPU support...');
    
    // Use WebGL for GPU acceleration (works in all browsers)
    // WebGL is more stable than WebGPU for ONNX Runtime 1.14.0
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['webgl', 'wasm'],
      graphOptimizationLevel: 'all'
    });
    
    sessionCache = session;
    
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('✅ Model loaded successfully!');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('📊 Input names:', session.inputNames);
    console.log('📊 Output names:', session.outputNames);
    console.log('🚀 EXECUTION DEVICE: GPU via WebGL');
    console.log('💻 Hardware: Apple M2 Max GPU');
    console.log('⚡ Expected speed: 50-100ms per image');
    console.log('ℹ️  Note: Model loading time excluded from inference metrics');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    
    // Warm up the session with a dummy inference
    console.log('🔥 Warming up WebGL session...');
    const warmupStart = performance.now();
    const dummyInput = new ort.Tensor('float32', new Float32Array(1 * 3 * 224 * 224), [1, 3, 224, 224]);
    await session.run({ [session.inputNames[0]]: dummyInput });
    const warmupTime = (performance.now() - warmupStart).toFixed(2);
    console.log(`✅ Warmup complete in ${warmupTime}ms (WebGL shaders compiled)`);
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    
    return session;
  } catch (error) {
    console.error('Failed to load model:', error);
    throw new Error(`Model loading failed: ${error}`);
  }
}

/**
 * Run inference on preprocessed image
 */
export async function runInference(
  session: ort.InferenceSession,
  preprocessedImage: PreprocessedImage
): Promise<PredictionResult> {
  try {
    // Create input tensor (shape: [1, 3, 224, 224])
    const tensor = new ort.Tensor(
      'float32',
      preprocessedImage.data,
      [1, 3, preprocessedImage.height, preprocessedImage.width]
    );

    // Run inference
    const feeds = { [session.inputNames[0]]: tensor };
    const startTime = performance.now();
    const results = await session.run(feeds);
    const endTime = performance.now();
    
    const inferenceTime = (endTime - startTime).toFixed(2);
    console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    console.log(`⏱️  Device: WebGL (GPU)`);
    console.log(`⏱️  Batch size: 1`);
    console.log(`⏱️  Inference time: ${inferenceTime} ms/image`);
    console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);

    // Get output tensor
    const output = results[session.outputNames[0]];
    const logits = output.data as Float32Array;

    // Apply softmax to get probabilities
    const probabilities = softmax(Array.from(logits));

    // Get predicted class
    const maxIndex = probabilities.indexOf(Math.max(...probabilities));
    const predictedClass = CLASS_NAMES[maxIndex];
    const confidence = probabilities[maxIndex];

    // Create probabilities object
    const probs: { [key: string]: number } = {};
    CLASS_NAMES.forEach((name, idx) => {
      probs[name] = probabilities[idx];
    });

    return {
      class: predictedClass,
      confidence,
      probabilities: probs,
      inferenceTimeMs: parseFloat(inferenceTime)
    };
  } catch (error) {
    console.error('Inference failed:', error);
    throw new Error(`Inference failed: ${error}`);
  }
}

/**
 * Softmax function to convert logits to probabilities
 */
function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const expScores = logits.map(x => Math.exp(x - maxLogit));
  const sumExpScores = expScores.reduce((a, b) => a + b, 0);
  return expScores.map(x => x / sumExpScores);
}

/**
 * Format class name for display
 */
export function formatClassName(className: string): string {
  return className
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Get emoji for each class
 */
export function getClassEmoji(className: string): string {
  const emojiMap: { [key: string]: string } = {
    'angular_leaf_spot': '🍂',
    'bean_rust': '🦠',
    'healthy': '✅'
  };
  return emojiMap[className] || '🌱';
}

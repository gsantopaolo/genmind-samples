/**
 * Image preprocessing utilities for ONNX model
 * Matches the Python preprocessing: resize to 224x224, ImageNet normalization
 */

export interface PreprocessedImage {
  data: Float32Array;
  width: number;
  height: number;
}

/**
 * Preprocess image for ONNX model inference
 * - Resize to 224x224
 * - Convert to RGB
 * - Normalize with ImageNet mean/std
 * - Convert to NCHW format (batch, channels, height, width)
 */
export async function preprocessImage(imageFile: File): Promise<PreprocessedImage> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const reader = new FileReader();

    reader.onload = (e) => {
      img.onload = () => {
        try {
          const processed = processImageElement(img);
          resolve(processed);
        } catch (error) {
          reject(error);
        }
      };
      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = e.target?.result as string;
    };

    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsDataURL(imageFile);
  });
}

/**
 * Process an HTML Image element
 */
function processImageElement(img: HTMLImageElement): PreprocessedImage {
  const targetSize = 224;
  
  // Create canvas for resizing
  const canvas = document.createElement('canvas');
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext('2d')!;
  
  // Draw and resize image
  ctx.drawImage(img, 0, 0, targetSize, targetSize);
  
  // Get image data
  const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
  const { data } = imageData; // RGBA format
  
  // ImageNet normalization constants
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  
  // Convert to NCHW format (1, 3, 224, 224) with normalization
  const float32Data = new Float32Array(1 * 3 * targetSize * targetSize);
  
  for (let i = 0; i < targetSize * targetSize; i++) {
    const r = data[i * 4 + 0] / 255.0;     // Red
    const g = data[i * 4 + 1] / 255.0;     // Green  
    const b = data[i * 4 + 2] / 255.0;     // Blue
    
    // Apply ImageNet normalization: (pixel - mean) / std
    float32Data[i] = (r - mean[0]) / std[0];                           // R channel
    float32Data[targetSize * targetSize + i] = (g - mean[1]) / std[1]; // G channel
    float32Data[2 * targetSize * targetSize + i] = (b - mean[2]) / std[2]; // B channel
  }
  
  return {
    data: float32Data,
    width: targetSize,
    height: targetSize
  };
}

/**
 * Create a preview URL for the uploaded image
 */
export function createImagePreview(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target?.result as string);
    reader.onerror = () => reject(new Error('Failed to create preview'));
    reader.readAsDataURL(file);
  });
}

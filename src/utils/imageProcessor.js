import * as faceapi from 'face-api.js';
import { removeBackground } from '@imgly/background-removal';

let modelsLoaded = false;

export const loadModels = async () => {
  if (modelsLoaded) return true;
  
  try {
    const MODEL_URL = '/models';
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL)
    ]);
    modelsLoaded = true;
    return true;
  } catch (error) {
    console.error("Error loading models:", error);
    return false;
  }
};

export const processImage = async (file, config) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        const img = new Image();
        img.onload = async () => {
          try {
            const resultBlob = await cropAndProcess(img, file, config);
            const url = URL.createObjectURL(resultBlob);
            resolve({
              blob: resultBlob,
              url,
              fileName: generateOutputName(file.name, config.format)
            });
          } catch (err) {
            reject(err);
          }
        };
        img.onerror = () => reject(new Error("Failed to load image"));
        img.src = e.target.result;
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
};

const aiCache = new Map();

// Helper to downscale large images specifically for face detection speedup
const getDetectionImage = (img, maxDim = 800) => {
  const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
  if (scale === 1) return { detectionCanvas: img, scale: 1 };
  
  const canvas = document.createElement('canvas');
  canvas.width = img.width * scale;
  canvas.height = img.height * scale;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  return { detectionCanvas: canvas, scale };
};

const cropAndProcess = async (originalImg, originalFile, config) => {
  await loadModels();

  // Create a robust unique key for caching heavy AI operations
  const fileKey = `${originalFile.name}_${originalFile.size}_${originalFile.lastModified}`;
  if (!aiCache.has(fileKey)) {
    aiCache.set(fileKey, {});
  }
  const cacheEntry = aiCache.get(fileKey);

  let imgToCrop = originalImg;

  // 1. Background Removal
  if (config.bgColor !== 'keep' || config.sizePreset === 'cutoutOnly') {
    if (cacheEntry.bgRemovedImg && cacheEntry.cutoutQuality === config.cutoutQuality) {
      imgToCrop = cacheEntry.bgRemovedImg;
    } else {
      try {
        const bgRemovedBlob = await removeBackground(originalFile, {
          model: config.cutoutQuality || 'medium'
        });
        imgToCrop = await blobToImage(bgRemovedBlob);
        cacheEntry.bgRemovedImg = imgToCrop;
        cacheEntry.cutoutQuality = config.cutoutQuality || 'medium';
      } catch (e) {
        console.warn("Background removal failed:", e);
      }
    }
  }

  // If "仅抠图" mode is selected, skip face detection and crop
  if (config.sizePreset === 'cutoutOnly') {
    const canvas = document.createElement('canvas');
    canvas.width = originalImg.width;
    canvas.height = originalImg.height;
    const ctx = canvas.getContext('2d');
    
    // Apply Background Color
    if (config.bgColor !== 'keep' && config.bgColor !== 'transparent') {
      ctx.fillStyle = config.bgColor;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    
    // Apply Beauty Filter
    if (config.beautyFilter) {
      ctx.filter = 'brightness(1.05) contrast(1.05) saturate(1.1) blur(0.5px)';
    }
    
    ctx.drawImage(imgToCrop, 0, 0);
    
    // Apply Watermark
    if (config.watermark) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.font = `${Math.max(12, canvas.width * 0.05)}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.translate(canvas.width / 2, canvas.height / 2);
      ctx.rotate(-Math.PI / 4);
      for (let i = -2; i <= 2; i++) {
        ctx.fillText(config.watermark, 0, i * (canvas.height * 0.3));
      }
      ctx.resetTransform();
    }
    
    const format = (config.bgColor === 'transparent' || config.bgColor === 'keep') && config.format !== 'jpg' ? 'png' : config.format;
    return canvasToBlob(canvas, format, config.quality);
  }

  // 2. Face Detection
  let detection = cacheEntry.detection;
  
  if (!detection) {
    // For much faster face detection on high-res images, we downscale first
    const { detectionCanvas } = getDetectionImage(imgToCrop, 800);
    
    const rawDetection = await faceapi.detectSingleFace(
      detectionCanvas, 
      new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 })
    ).withFaceLandmarks();
    
    if (!rawDetection) {
      throw new Error("No face detected in the image. Please try another photo with a clearer face.");
    }

    // Scale the results back up to the original high-res image dimensions
    detection = faceapi.resizeResults(rawDetection, { width: imgToCrop.width, height: imgToCrop.height });
    cacheEntry.detection = detection;
  }

  const box = detection.detection.box;
  const landmarks = detection.landmarks;
  
  // 3. Leveling (Rotation Alignment) and Precise Centering
  const leftEye = landmarks.getLeftEye();
  const rightEye = landmarks.getRightEye();
  
  const getCenter = (pts) => {
    return pts.reduce((acc, pt) => ({ x: acc.x + pt.x, y: acc.y + pt.y }), { x: 0, y: 0 });
  };
  
  const leCenter = getCenter(leftEye);
  leCenter.x /= leftEye.length; leCenter.y /= leftEye.length;
  
  const reCenter = getCenter(rightEye);
  reCenter.x /= rightEye.length; reCenter.y /= rightEye.length;
  
  const dy = reCenter.y - leCenter.y;
  const dx = reCenter.x - leCenter.x;
  const angle = Math.atan2(dy, dx); // Face tilt angle in radians

  // True face horizontal center (midpoint between eyes is much more stable than bounding box)
  const faceCenterX = (leCenter.x + reCenter.x) / 2;

  // 4. Cropping Calculations with Fine-Tuning Parameters
  const targetWidth = parseInt(config.width);
  const targetHeight = parseInt(config.height);
  const targetRatio = targetWidth / targetHeight;
  
  const faceRatio = (config.faceRatio || 45) / 100;
  const headroom = (config.headroom || 60) / 100;
  
  const faceHeight = box.height;
  const cropHeight = faceHeight / faceRatio;
  const cropWidth = cropHeight * targetRatio;
  
  const faceTopY = box.y;
  const startY = faceTopY - (faceHeight * headroom);
  
  const cropCenterX = faceCenterX;
  const cropCenterY = startY + (cropHeight / 2);

  // 5. Render to Canvas with Transformations
  const canvas = document.createElement('canvas');
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const ctx = canvas.getContext('2d');
  
  // Apply Background Color
  if (config.bgColor !== 'keep' && config.bgColor !== 'transparent') {
    ctx.fillStyle = config.bgColor;
    ctx.fillRect(0, 0, targetWidth, targetHeight);
  } else {
    ctx.clearRect(0, 0, targetWidth, targetHeight);
  }
  
  ctx.save();
  
  // Apply Beauty Filter
  if (config.beautyFilter) {
    ctx.filter = 'brightness(1.05) contrast(1.05) saturate(1.1) blur(0.5px)';
  }
  
  // Transform canvas to achieve leveling and cropping
  ctx.translate(targetWidth / 2, targetHeight / 2); // Move origin to center of target canvas
  ctx.rotate(-angle); // Counter-rotate the canvas to level the face
  
  const scale = targetWidth / cropWidth;
  ctx.scale(scale, scale); // Scale the image to fit the crop area
  
  ctx.translate(-cropCenterX, -cropCenterY); // Move the specific crop center of the image to the origin
  
  // Draw the image
  ctx.drawImage(imgToCrop, 0, 0);
  
  ctx.restore();

  // Apply Watermark
  if (config.watermark) {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.font = `${Math.max(12, targetWidth * 0.05)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    // Draw diagonally across the image multiple times
    ctx.translate(targetWidth / 2, targetHeight / 2);
    ctx.rotate(-Math.PI / 4);
    for (let i = -2; i <= 2; i++) {
      ctx.fillText(config.watermark, 0, i * (targetHeight * 0.3));
    }
    ctx.resetTransform();
  }

  // 5. Print Layout (4x6 tiling)
  if (config.printLayout) {
    return generatePrintLayout(canvas, targetWidth, targetHeight, config.format, config.quality);
  }

  // 6. Export Single Blob
  return canvasToBlob(canvas, config.format, config.quality);
};

const generatePrintLayout = async (croppedCanvas, singleWidth, singleHeight, format, quality) => {
  // Standard 4x6 inch at 300 DPI is 1200x1800 or 1800x1200
  // We'll use 1800x1200 (Landscape)
  const layoutCanvas = document.createElement('canvas');
  layoutCanvas.width = 1800;
  layoutCanvas.height = 1200;
  const lctx = layoutCanvas.getContext('2d');
  
  lctx.fillStyle = '#ffffff';
  lctx.fillRect(0, 0, layoutCanvas.width, layoutCanvas.height);
  
  // Padding and spacing
  const marginX = 100;
  const marginY = 100;
  const spacingX = 50;
  const spacingY = 50;
  
  // Calculate how many can fit
  const cols = Math.floor((layoutCanvas.width - marginX * 2 + spacingX) / (singleWidth + spacingX));
  const rows = Math.floor((layoutCanvas.height - marginY * 2 + spacingY) / (singleHeight + spacingY));
  
  // Center the grid
  const gridWidth = cols * singleWidth + (cols - 1) * spacingX;
  const gridHeight = rows * singleHeight + (rows - 1) * spacingY;
  const startX = (layoutCanvas.width - gridWidth) / 2;
  const startY = (layoutCanvas.height - gridHeight) / 2;
  
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = startX + c * (singleWidth + spacingX);
      const y = startY + r * (singleHeight + spacingY);
      
      // Draw a subtle border for cutting
      lctx.strokeStyle = '#cccccc';
      lctx.lineWidth = 2;
      lctx.strokeRect(x - 2, y - 2, singleWidth + 4, singleHeight + 4);
      
      lctx.drawImage(croppedCanvas, x, y);
    }
  }
  
  return canvasToBlob(layoutCanvas, format, quality);
};

const blobToImage = (blob) => {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = reject;
    img.src = url;
  });
};

const canvasToBlob = (canvas, format, quality) => {
  const mimeType = format === 'png' ? 'image/png' : 'image/jpeg';
  const q = format === 'png' ? undefined : quality / 100;
  return new Promise((resolve) => {
    canvas.toBlob(resolve, mimeType, q);
  });
};

const generateOutputName = (originalName, format) => {
  const lastDot = originalName.lastIndexOf('.');
  const baseName = lastDot === -1 ? originalName : originalName.substring(0, lastDot);
  return `${baseName}_idphoto.${format}`;
};

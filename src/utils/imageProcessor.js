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
    console.error('Error loading models:', error);
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
        img.onerror = () => reject(new Error('Failed to load image'));
        img.src = e.target.result;
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
};

const aiCache = new Map();

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

const clamp = (value, min = 0, max = 255) => Math.max(min, Math.min(max, value));

const smoothstep = (edge0, edge1, value) => {
  const x = clamp((value - edge0) / (edge1 - edge0), 0, 1);
  return x * x * (3 - 2 * x);
};

const colorDistance = (a, b) => {
  const dr = a[0] - b[0];
  const dg = a[1] - b[1];
  const db = a[2] - b[2];
  return Math.sqrt(dr * dr + dg * dg + db * db);
};

const luminance = (r, g, b) => r * 0.299 + g * 0.587 + b * 0.114;

// Edge-aware alpha and spill cleanup. Pixels with solid alpha are protected.
const refineAlpha = (img, strength, originalImg = img) => {
  const cleanStrength = Math.max(0, Math.min(20, Number(strength) || 0));
  if (cleanStrength <= 0) return img;

  const width = img.width;
  const height = img.height;
  const strength01 = cleanStrength / 20;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);

  const originalCanvas = document.createElement('canvas');
  originalCanvas.width = width;
  originalCanvas.height = height;
  const originalCtx = originalCanvas.getContext('2d', { willReadFrequently: true });
  originalCtx.drawImage(originalImg, 0, 0, width, height);

  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  const src = new Uint8ClampedArray(data);
  const originalData = originalCtx.getImageData(0, 0, width, height).data;

  const protectAlpha = 245;
  const killAlpha = Math.min(86, 4 + cleanStrength * 4.1);
  const bgAlpha = Math.max(8, killAlpha * 0.38);
  const searchRadius = Math.round(3 + cleanStrength * 0.35);

  const pixelOffset = (x, y) => ((y * width + x) << 2);
  const alphaAt = (x, y) => src[pixelOffset(x, y) + 3];

  const getEdgeStats = (x, y) => {
    let minAlpha = 255;
    let maxAlpha = 0;

    for (let yy = Math.max(0, y - 1); yy <= Math.min(height - 1, y + 1); yy++) {
      for (let xx = Math.max(0, x - 1); xx <= Math.min(width - 1, x + 1); xx++) {
        const alpha = alphaAt(xx, yy);
        minAlpha = Math.min(minAlpha, alpha);
        maxAlpha = Math.max(maxAlpha, alpha);
      }
    }

    return {
      minAlpha,
      maxAlpha,
      contrast: maxAlpha - minAlpha
    };
  };

  const sampleColorByAlpha = (x, y, radius, preferred, fallback) => {
    const tests = fallback ? [preferred, fallback] : [preferred];

    for (const testAlpha of tests) {
      for (let r = 1; r <= radius; r++) {
        let totalWeight = 0;
        let red = 0;
        let green = 0;
        let blue = 0;

        for (let yy = Math.max(0, y - r); yy <= Math.min(height - 1, y + r); yy++) {
          for (let xx = Math.max(0, x - r); xx <= Math.min(width - 1, x + r); xx++) {
            if (Math.abs(xx - x) !== r && Math.abs(yy - y) !== r) continue;

            const targetOffset = pixelOffset(xx, yy);
            if (!testAlpha(src[targetOffset + 3])) continue;

            const sampleWeight = 1 / (1 + Math.abs(xx - x) + Math.abs(yy - y));
            red += originalData[targetOffset] * sampleWeight;
            green += originalData[targetOffset + 1] * sampleWeight;
            blue += originalData[targetOffset + 2] * sampleWeight;
            totalWeight += sampleWeight;
          }
        }

        if (totalWeight > 0) {
          return [red / totalWeight, green / totalWeight, blue / totalWeight];
        }
      }
    }

    return null;
  };

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const offset = pixelOffset(x, y);
      const alpha = src[offset + 3];
      if (alpha === 0) continue;

      const edgeStats = getEdgeStats(x, y);
      const isEdge =
        edgeStats.contrast >= 12 &&
        edgeStats.minAlpha < protectAlpha &&
        alpha < protectAlpha;

      let nextAlpha = alpha;

      if (isEdge || alpha <= killAlpha) {
        if (alpha <= killAlpha) {
          nextAlpha = 0;
        } else if (alpha < protectAlpha) {
          const edgeFade = 1 - smoothstep(killAlpha, protectAlpha, alpha);
          const gradientWeight = 0.6 + 0.4 * clamp(edgeStats.contrast / 180, 0, 1);
          const alphaLoss =
            strength01 *
            (18 + cleanStrength * 1.9) *
            Math.pow(edgeFade, 1.25) *
            gradientWeight;

          nextAlpha = Math.max(0, Math.round(alpha - alphaLoss));
          if (nextAlpha < killAlpha * (1.05 + strength01 * 0.45)) {
            nextAlpha = 0;
          }
        }

        data[offset + 3] = nextAlpha;
      }

      if (!isEdge || nextAlpha === 0) continue;

      const solidColor = sampleColorByAlpha(
        x,
        y,
        searchRadius,
        sampleAlpha => sampleAlpha >= protectAlpha,
        sampleAlpha => sampleAlpha >= 210
      );
      const backgroundColor = sampleColorByAlpha(
        x,
        y,
        searchRadius + 2,
        sampleAlpha => sampleAlpha <= bgAlpha,
        sampleAlpha => sampleAlpha <= Math.min(28, killAlpha)
      );

      if (!solidColor || !backgroundColor) continue;

      const originalColor = [
        originalData[offset],
        originalData[offset + 1],
        originalData[offset + 2]
      ];
      const alpha01 = clamp(nextAlpha / 255, 0.08, 1);
      const edgeWeight = 1 - smoothstep(0.72, 0.99, alpha01);
      const contrastWeight = smoothstep(24, 110, colorDistance(backgroundColor, solidColor));
      const spillLikelihood = clamp(
        (colorDistance(originalColor, solidColor) - colorDistance(originalColor, backgroundColor) + 55) / 140,
        0,
        1
      );
      const whiteSpillBoost =
        smoothstep(176, 238, luminance(...backgroundColor)) *
        smoothstep(8, 80, luminance(...originalColor) - luminance(...solidColor));
      const cleanupAmount = clamp(
        strength01 *
          edgeWeight *
          contrastWeight *
          (0.35 + 0.65 * spillLikelihood) *
          (1 + 0.65 * whiteSpillBoost),
        0,
        0.92
      );

      if (cleanupAmount <= 0.015) continue;

      const matteFactor = Math.pow(1 - alpha01, 0.65) * cleanupAmount * 1.4;

      for (let channel = 0; channel < 3; channel++) {
        const originalValue = originalColor[channel];
        const spillVector = backgroundColor[channel] - solidColor[channel];
        let cleanedValue = originalValue - spillVector * matteFactor;

        const solidPull = cleanupAmount * 0.18 * (1 - alpha01);
        cleanedValue = cleanedValue * (1 - solidPull) + solidColor[channel] * solidPull;

        const lowerBound = Math.max(0, Math.min(originalValue, solidColor[channel]) - 18);
        const upperBound = Math.min(255, Math.max(originalValue, solidColor[channel]) + 18);
        data[offset + channel] = clamp(Math.round(cleanedValue), lowerBound, upperBound);
      }

      const cleanedLuma = luminance(data[offset], data[offset + 1], data[offset + 2]);
      const solidLuma = luminance(...solidColor);
      const bgLuma = luminance(...backgroundColor);
      const lumaCeiling = solidLuma + 22 + 65 * alpha01;

      if (bgLuma > solidLuma + 30 && cleanedLuma > lumaCeiling) {
        const pull = clamp(((cleanedLuma - lumaCeiling) / 90) * 0.45 * strength01, 0, 0.45);

        for (let channel = 0; channel < 3; channel++) {
          data[offset + channel] = clamp(
            Math.round(data[offset + channel] * (1 - pull) + solidColor[channel] * pull)
          );
        }
      }
    }
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas;
};

const cropAndProcess = async (originalImg, originalFile, config) => {
  await loadModels();

  const fileKey = `${originalFile.name}_${originalFile.size}_${originalFile.lastModified}`;
  if (!aiCache.has(fileKey)) {
    aiCache.set(fileKey, {});
  }
  const cacheEntry = aiCache.get(fileKey);

  let imgToCrop = originalImg;

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
        console.warn('Background removal failed:', e);
      }
    }

    if (cacheEntry.bgRemovedImg) {
      imgToCrop = refineAlpha(cacheEntry.bgRemovedImg, config.edgeShift || 0, originalImg);
    }
  }

  if (config.sizePreset === 'cutoutOnly') {
    const canvas = document.createElement('canvas');
    canvas.width = originalImg.width;
    canvas.height = originalImg.height;
    const ctx = canvas.getContext('2d');

    if (config.bgColor !== 'keep' && config.bgColor !== 'transparent') {
      ctx.fillStyle = config.bgColor;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    if (config.beautyFilter) {
      ctx.filter = 'brightness(1.05) contrast(1.05) saturate(1.1) blur(0.5px)';
    }

    ctx.drawImage(imgToCrop, 0, 0);

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

    const exportFormat = config.bgColor === 'transparent' ? 'png' : config.format;
    return canvasToBlob(canvas, exportFormat, config.quality);
  }

  let detection = cacheEntry.detection;

  if (!detection) {
    const { detectionCanvas } = getDetectionImage(imgToCrop, 800);

    const rawDetection = await faceapi.detectSingleFace(
      detectionCanvas,
      new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 })
    ).withFaceLandmarks();

    if (!rawDetection) {
      throw new Error('No face detected in the image. Please try another photo with a clearer face.');
    }

    detection = faceapi.resizeResults(rawDetection, { width: imgToCrop.width, height: imgToCrop.height });
    cacheEntry.detection = detection;
  }

  const box = detection.detection.box;
  const landmarks = detection.landmarks;

  const leftEye = landmarks.getLeftEye();
  const rightEye = landmarks.getRightEye();

  const getCenter = (pts) => {
    return pts.reduce((acc, pt) => ({ x: acc.x + pt.x, y: acc.y + pt.y }), { x: 0, y: 0 });
  };

  const leCenter = getCenter(leftEye);
  leCenter.x /= leftEye.length;
  leCenter.y /= leftEye.length;

  const reCenter = getCenter(rightEye);
  reCenter.x /= rightEye.length;
  reCenter.y /= rightEye.length;

  const dy = reCenter.y - leCenter.y;
  const dx = reCenter.x - leCenter.x;
  const angle = Math.atan2(dy, dx);
  const faceCenterX = (leCenter.x + reCenter.x) / 2;

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

  const canvas = document.createElement('canvas');
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const ctx = canvas.getContext('2d');

  if (config.bgColor !== 'keep' && config.bgColor !== 'transparent') {
    ctx.fillStyle = config.bgColor;
    ctx.fillRect(0, 0, targetWidth, targetHeight);
  } else {
    ctx.clearRect(0, 0, targetWidth, targetHeight);
  }

  ctx.save();

  if (config.beautyFilter) {
    ctx.filter = 'brightness(1.05) contrast(1.05) saturate(1.1) blur(0.5px)';
  }

  ctx.translate(targetWidth / 2, targetHeight / 2);
  ctx.rotate(-angle);

  const scale = targetWidth / cropWidth;
  ctx.scale(scale, scale);

  ctx.translate(-cropCenterX, -cropCenterY);
  ctx.drawImage(imgToCrop, 0, 0);
  ctx.restore();

  if (config.watermark) {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.font = `${Math.max(12, targetWidth * 0.05)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    ctx.translate(targetWidth / 2, targetHeight / 2);
    ctx.rotate(-Math.PI / 4);
    for (let i = -2; i <= 2; i++) {
      ctx.fillText(config.watermark, 0, i * (targetHeight * 0.3));
    }
    ctx.resetTransform();
  }

  const exportFormat = config.bgColor === 'transparent' ? 'png' : config.format;

  if (config.printLayout) {
    return generatePrintLayout(canvas, targetWidth, targetHeight, exportFormat, config.quality);
  }

  return canvasToBlob(canvas, exportFormat, config.quality);
};

const generatePrintLayout = async (croppedCanvas, singleWidth, singleHeight, format, quality) => {
  const layoutCanvas = document.createElement('canvas');
  layoutCanvas.width = 1800;
  layoutCanvas.height = 1200;
  const lctx = layoutCanvas.getContext('2d');

  lctx.fillStyle = '#ffffff';
  lctx.fillRect(0, 0, layoutCanvas.width, layoutCanvas.height);

  const marginX = 100;
  const marginY = 100;
  const spacingX = 50;
  const spacingY = 50;

  const cols = Math.floor((layoutCanvas.width - marginX * 2 + spacingX) / (singleWidth + spacingX));
  const rows = Math.floor((layoutCanvas.height - marginY * 2 + spacingY) / (singleHeight + spacingY));

  const gridWidth = cols * singleWidth + (cols - 1) * spacingX;
  const gridHeight = rows * singleHeight + (rows - 1) * spacingY;
  const startX = (layoutCanvas.width - gridWidth) / 2;
  const startY = (layoutCanvas.height - gridHeight) / 2;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = startX + c * (singleWidth + spacingX);
      const y = startY + r * (singleHeight + spacingY);

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

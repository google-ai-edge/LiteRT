/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {CompiledModel, Tensor} from '@litertjs/core';

/**
 * Options for depth estimation.
 */
export interface DepthEstimatorOptions {
  sourceImage: HTMLImageElement;
  model: CompiledModel;
  colormap?: 'grayscale'|'spectral_r';
  progressCallback: (progress: {message: string; value: number}) => void;
}

/**
 * Runs depth estimation on an image using a LiteRT model.
 */
export async function runDepthEstimation({
  sourceImage,
  model,
  colormap = 'grayscale',
  progressCallback,
}: DepthEstimatorOptions): Promise<HTMLCanvasElement> {
  // Get Model Dimensions
  const inputDetails = model.getInputDetails()[0];
  const outputDetails = model.getOutputDetails()[0];

  const [, channels, inputHeight, inputWidth] = inputDetails.shape;
  const [, outputHeight, outputWidth] = outputDetails.shape;

  // Prepare Source Image Data
  progressCallback({message: 'Preparing image data...', value: 0});
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = inputWidth;
  srcCanvas.height = inputHeight;
  const srcCtx = srcCanvas.getContext('2d')!;
  srcCtx.drawImage(sourceImage, 0, 0, inputWidth, inputHeight);
  const srcImageData =
      srcCtx.getImageData(0, 0, inputWidth, inputHeight);

  const float32Data = new Float32Array(inputWidth * inputHeight * channels);

  // The model expects the input to be in the format [1, 3, H, W].
  // We need to convert the image data from [H, W, 4] (RGBA) to [3, H, W] (RGB).
  for (let y = 0; y < inputHeight; y++) {
    for (let x = 0; x < inputWidth; x++) {
      const i = (y * inputWidth + x) * 4;
      const r = srcImageData.data[i];
      const g = srcImageData.data[i + 1];
      const b = srcImageData.data[i + 2];

      // Store in CHW format
      float32Data[y * inputWidth + x] = r / 255.0;
      float32Data[inputWidth * inputHeight + y * inputWidth + x] = g / 255.0;
      float32Data[2 * inputWidth * inputHeight + y * inputWidth + x] = b / 255.0;
    }
  }


  progressCallback({message: 'Running inference...', value: 0.5});
  const inputTensor =
      new Tensor(float32Data, [1, channels, inputHeight, inputWidth]);

  const [outputTensor] = model.run([inputTensor]);
  inputTensor.delete();

  const outputData = outputTensor.toTypedArray() as Float32Array;
  outputTensor.delete();

  // Post-process the output to create a grayscale depth map
  progressCallback({message: 'Creating depth map...', value: 0.8});

  // Create a temporary canvas to hold the raw output data
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = outputWidth;
  tempCanvas.height = outputHeight;
  const tempCtx = tempCanvas.getContext('2d')!;
  const tempImageData = tempCtx.createImageData(outputWidth, outputHeight);

  // Find the min and max values in the output data to normalize it.
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < outputData.length; i++) {
    if (outputData[i] < min) {
      min = outputData[i];
    }
    if (outputData[i] > max) {
      max = outputData[i];
    }
  }

  const range = max - min;
  for (let i = 0; i < outputData.length; i++) {
    const j = i * 4;
    const value = (outputData[i] - min) / range;
    const [r, g, b] = getColor(value, colormap);
    tempImageData.data[j] = r;      // R
    tempImageData.data[j + 1] = g;  // G
    tempImageData.data[j + 2] = b;  // B
    tempImageData.data[j + 3] = 255;    // Alpha
  }
  tempCtx.putImageData(tempImageData, 0, 0);

  // Create the final canvas with the same dimensions as the source image
  const outCanvas = document.createElement('canvas');
  outCanvas.width = sourceImage.width;
  outCanvas.height = sourceImage.height;
  const outCtx = outCanvas.getContext('2d')!;

  // Draw the temporary canvas onto the final canvas, scaling it to fit
  outCtx.drawImage(tempCanvas, 0, 0, sourceImage.width, sourceImage.height);

  progressCallback({message: 'Done.', value: 1});
  return outCanvas;
}

function getColor(value: number, colormap: 'grayscale'|'spectral_r'):
    [number, number, number] {
  if (colormap === 'grayscale') {
    const v = value * 255;
    return [v, v, v];
  }

  // Spectral-like colormap
  return getSpectralColor(value);
}

function hueToRgb(p: number, q: number, t: number): number {
  if (t < 0) t += 1;
  if (t > 1) t -= 1;
  if (t < 1 / 6) return p + (q - p) * 6 * t;
  if (t < 1 / 2) return q;
  if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
  return p;
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  let r: number;
  let g: number;
  let b: number;

  if (s === 0) {
    r = g = b = l;  // achromatic
  } else {
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hueToRgb(p, q, h + 1 / 3);
    g = hueToRgb(p, q, h);
    b = hueToRgb(p, q, h - 1 / 3);
  }

  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

// Generates a color from a spectral-like colormap
function getSpectralColor(value: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, value));
  // Map value to a hue range. This is a simplified spectral-like mapping.
  // Blue/Violet (0.7) -> Red (0.0)
  const hue = 0.7 * Math.pow(1.0 - v, 1.5);
  return hslToRgb(hue, 0.7, 0.5);
}

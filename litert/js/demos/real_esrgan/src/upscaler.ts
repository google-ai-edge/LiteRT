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
 * Options for upscaling an image.
 */
export interface UpscaleOptions {
  sourceImage: HTMLImageElement;
  model: CompiledModel;
  overlapPercent: number;
  normalizationRange: [number, number];
  progressCallback: (progress: {message: string; value: number}) => void;
}

/**
 * Upscales an image using a LiteRT model with a tiling strategy.
 */
export async function upscaleImageWithTiling({
  sourceImage,
  model,
  overlapPercent,
  normalizationRange,
  progressCallback,
}: UpscaleOptions): Promise<HTMLCanvasElement> {
  // Get Model Dimensions
  const inputDetails = model.getInputDetails()[0];
  const outputDetails = model.getOutputDetails()[0];

  const [, inputHeight, inputWidth] = inputDetails.shape;
  const [, outputHeight, outputWidth] = outputDetails.shape;
  const scale = outputHeight / inputHeight;

  if (outputWidth / inputWidth !== scale) {
    throw new Error(
        'Model scale factor is not consistent between height and width.');
  }

  // Prepare Source Image Data
  progressCallback({message: 'Preparing image data...', value: 0});
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = sourceImage.width;
  srcCanvas.height = sourceImage.height;
  const srcCtx = srcCanvas.getContext('2d')!;
  srcCtx.drawImage(sourceImage, 0, 0);
  const srcImageData =
      srcCtx.getImageData(0, 0, sourceImage.width, sourceImage.height);

  const float32Data =
      new Float32Array(sourceImage.width * sourceImage.height * 3);
  const [min, max] = normalizationRange;
  const scaleFactor = (max - min) / 255.0;

  for (let i = 0; i < srcImageData.data.length; i += 4) {
    const j = (i / 4) * 3;
    float32Data[j] = srcImageData.data[i] * scaleFactor + min;          // R
    float32Data[j + 1] = srcImageData.data[i + 1] * scaleFactor + min;  // G
    float32Data[j + 2] = srcImageData.data[i + 2] * scaleFactor + min;  // B
  }

  // Calculate Tiling Strategy
  const overlapX = Math.floor(inputWidth * (overlapPercent / 100));
  const overlapY = Math.floor(inputHeight * (overlapPercent / 100));
  const stepSizeX = inputWidth - overlapX;
  const stepSizeY = inputHeight - overlapY;

  const numTilesX = sourceImage.width <= inputWidth ?
      1 :
      Math.ceil((sourceImage.width - inputWidth) / stepSizeX) + 1;
  const numTilesY = sourceImage.height <= inputHeight ?
      1 :
      Math.ceil((sourceImage.height - inputHeight) / stepSizeY) + 1;
  const totalTiles = numTilesX * numTilesY;

  // Prepare Output Canvas
  const outCanvas = document.createElement('canvas');
  const outWidth = sourceImage.width * scale;
  const outHeight = sourceImage.height * scale;
  outCanvas.width = outWidth;
  outCanvas.height = outHeight;
  const outCtx = outCanvas.getContext('2d')!;
  const outImageData = outCtx.createImageData(outWidth, outHeight);

  // Run Inference on Each Tile
  for (let tileY = 0; tileY < numTilesY; tileY++) {
    for (let tileX = 0; tileX < numTilesX; tileX++) {
      const tileIndex = tileY * numTilesX + tileX;
      progressCallback({
        message: `Upscaling tile ${tileIndex + 1} of ${totalTiles}`,
        value: (tileIndex + 1) / totalTiles,
      });

      // Determine the source coordinates for the tile
      let startX = tileX * stepSizeX;
      let startY = tileY * stepSizeY;

      // Clamp coordinates to image boundaries
      if (startX + inputWidth > sourceImage.width) {
        startX = sourceImage.width - inputWidth;
      }
      if (startY + inputHeight > sourceImage.height) {
        startY = sourceImage.height - inputHeight;
      }

      // Extract tile data from the full source float32 array
      const tileData = new Float32Array(inputWidth * inputHeight * 3);
      for (let y = 0; y < inputHeight; y++) {
        for (let x = 0; x < inputWidth; x++) {
          const srcIdx = ((startY + y) * sourceImage.width + (startX + x)) * 3;
          const destIdx = (y * inputWidth + x) * 3;
          tileData[destIdx] = float32Data[srcIdx];
          tileData[destIdx + 1] = float32Data[srcIdx + 1];
          tileData[destIdx + 2] = float32Data[srcIdx + 2];
        }
      }

      // Create the tensor on the CPU ('wasm' accelerator)
      const cpuInputTensor =
          new Tensor(tileData, [1, inputHeight, inputWidth, 3]);

      // Move the tensor to the GPU ('webgpu' accelerator)
      const gpuInputTensor = await cpuInputTensor.moveTo('webgpu');

      // Run the model with the GPU tensor
      const [outputTensor] = model.run([gpuInputTensor]);

      // The GPU input tensor can now be deleted
      gpuInputTensor.delete();

      const outputCpu = await outputTensor.moveTo('wasm');
      const outputData = outputCpu.toTypedArray() as Float32Array;
      outputCpu.delete();

      // Stitch the resulting tile into the output image data
      const destStartX = Math.round(startX * scale);
      const destStartY = Math.round(startY * scale);

      for (let y = 0; y < outputHeight; y++) {
        for (let x = 0; x < outputWidth; x++) {
          const outX = destStartX + x;
          const outY = destStartY + y;
          if (outX >= outWidth || outY >= outHeight) continue;

          const srcIdx = (y * outputWidth + x) * 3;
          const destIdx = (outY * outWidth + outX) * 4;

          // Rescale the data to [0, 255] and copy it to the output array.
          outImageData.data[destIdx] = (outputData[srcIdx] - min) / scaleFactor;
          outImageData.data[destIdx + 1] =
              (outputData[srcIdx + 1] - min) / scaleFactor;
          outImageData.data[destIdx + 2] =
              (outputData[srcIdx + 2] - min) / scaleFactor;
          outImageData.data[destIdx + 3] = 255;  // Alpha
        }
      }
    }
  }

  progressCallback({message: 'Finalizing image...', value: 1});
  outCtx.putImageData(outImageData, 0, 0);
  return outCanvas;
}

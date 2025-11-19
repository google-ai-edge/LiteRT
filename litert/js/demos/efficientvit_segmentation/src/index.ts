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

import '@tensorflow/tfjs-backend-webgpu';

import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
import {runWithTfjsTensors} from '@litertjs/tfjs-interop';
import {loadAndCompile, loadLiteRt, setWebGpuDevice} from '@litertjs/core';
import {WebGPUBackend} from '@tensorflow/tfjs-backend-webgpu';
import * as tf from '@tensorflow/tfjs-core';
import {webcam} from '@tensorflow/tfjs-data';
// Placeholder for internal dependency on trusted resource url

async function main() {
  await tf.setBackend('webgpu');

  await loadLiteRt('/litert_wasm_internal.js');
  const backend = tf.backend() as WebGPUBackend;
  setWebGpuDevice(backend.device);

  const video = document.getElementById('video') as HTMLVideoElement;
  const cam = await webcam(video, {
    facingMode: 'user',
  });

  const {colors} =
      (await (await fetch('./static/ade20k_class_colors.json')).json()) as {
    classes: string[];
    colors: Array<[number, number, number]>;
  };
  const colorsTensor = tf.tensor(colors, undefined, 'int32');

  const compileOptions = {accelerator: 'webgpu'} as const;
  const model = await loadAndCompile(
      'https://storage.googleapis.com/tfweb/litertjs_demo_models/efficientvit/efficientvit_seg_l2_ade20k_r512x512.tflite',
      compileOptions);

  const loadingMessage =
      document.getElementById('loading-message') as HTMLParagraphElement;
  loadingMessage.innerText = 'Loaded';

  const canvas = document.getElementById('canvas') as HTMLCanvasElement;

  let timesRun = 0;
  const intervalTime = 1000;
  setInterval(() => {
    console.log(`Approximate fps: ${(timesRun / intervalTime) * 1000}`);
    timesRun = 0;
  }, intervalTime);

  async function animate() {
    timesRun++;
    const image = await cam.capture();
    tf.tidy(() => {
      // https://github.com/mit-han-lab/efficientvit/blob/master/applications/efficientvit_seg/demo_efficientvit_seg_model.py
      const imageData = image.div(255)
                            .resizeBilinear([512, 512])
                            .sub([0.485, 0.456, 0.406])
                            .div([0.229, 0.224, 0.225])
                            .expandDims(0);

      const output = runWithTfjsTensors(model, imageData);
      const probabilities = output[0];
      const segmentationClasses = probabilities.argMax(3);
      const segmentation = colorsTensor.gather(segmentationClasses).squeeze();
      const pixels = segmentation.resizeNearestNeighbor([
        image.shape[0],
        image.shape[1],
      ]);

      tf.browser.draw(pixels as tf.Tensor3D, canvas);
    });
    image.dispose();
    requestAnimationFrame(animate);
  }

  return animate();
}

main();

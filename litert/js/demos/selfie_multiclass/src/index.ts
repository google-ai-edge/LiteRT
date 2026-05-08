/**
 * @license
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {CompiledModel, getGlobalLiteRt, loadAndCompile, loadLiteRt, Tensor} from '@litertjs/core';
import '@tensorflow/tfjs-backend-webgpu';
import * as tf from '@tensorflow/tfjs-core';
// Placeholder for internal dependency on trusted resource url

const MODEL_URL =
    'https://raw.githubusercontent.com/google-ai-edge/litert-samples/main/compiled_model_api/image_segmentation/c%2B%2B_segmentation/build_from_source/models/selfie_multiclass_256x256.tflite';
const video = document.getElementById("webcam") as HTMLVideoElement;
const canvasElement = document.getElementById("canvas") as HTMLCanvasElement;
const canvasCtx = canvasElement.getContext("2d")!;
const webcamButton = document.getElementById("webcamButton") as HTMLButtonElement;
const statusMsg = document.getElementById("status") as HTMLParagraphElement;
const backendSelect = document.getElementById("backendSelect") as HTMLSelectElement;

let liteRtModel: CompiledModel | null = null;
let webcamRunning = false;
let isCompiling = false;
let isPredicting = false;
let lastVideoTime = -1;
let imageData: ImageData|null = null;

// Format: [R, G, B, Alpha]
const categoryColors = [
  [0, 0, 0, 0],         // 0: Background (Transparent)
  [255, 99, 71, 150],   // 1: Category 1 (e.g., Hair - Tomato Red)
  [46, 139, 87, 150],   // 2: Category 2 (e.g., Body Skin - Sea Green)
  [65, 105, 225, 150],  // 3: Category 3 (e.g., Face Skin - Royal Blue)
  [255, 215, 0, 150],   // 4: Category 4 (e.g., Clothes - Gold)
  [218, 112, 214, 150], // 5: Category 5 (e.g., Accessories - Orchid)
  [0, 255, 255, 150]    // 6: Fallback (Cyan)
];

// 1. Initialize LiteRT
async function init() {
  try {
    statusMsg.innerText = "Initializing TFJS...";
    await tf.ready();
    statusMsg.innerText = "Loading LiteRT Wasm...";
    await loadLiteRt('./wasm/');

    // Workaround Emscripten WebIDL enum binding bug: named properties are undefined on the constructor.
    // Monkey-patch them using the populated values map so comparisons in core package work.
    const liteRtWasm = (getGlobalLiteRt() as any).liteRtWasm;
    if (liteRtWasm && liteRtWasm.LiteRtTensorBufferType && liteRtWasm.LiteRtTensorBufferType.values) {
      const Enum = liteRtWasm.LiteRtTensorBufferType;
      Enum.UNKNOWN = Enum.values[0];
      Enum.HOST_MEMORY = Enum.values[1];
      Enum.WEB_GPU_BUFFER = Enum.values[20];
      Enum.WEB_GPU_BUFFER_FP16 = Enum.values[21];
      Enum.WEB_GPU_BUFFER_PACKED = Enum.values[26];
    }

    const initialBackend = backendSelect.value as 'wasm' | 'webgpu' | 'webnn';
    await switchBackend(initialBackend);
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    statusMsg.innerText = `Error initializing: ${message}`;
    console.error(e);
  }
}

init();

async function switchBackend(backend: 'wasm' | 'webgpu' | 'webnn') {
  if (isCompiling) return;
  isCompiling = true;
  backendSelect.disabled = true;
  // Wait for active prediction to finish
  if (isPredicting) {
    statusMsg.innerText = "Waiting for active inference to finish...";
    while (isPredicting) {
      await new Promise(resolve => setTimeout(resolve, 10));
    }
  }
  statusMsg.innerText = `Loading and Compiling TFLite Model for ${backend.toUpperCase()}...`;
  const wasWebcamRunning = webcamRunning;
  try {
    if (liteRtModel) {
      liteRtModel.delete();
      liteRtModel = null;
    }
    liteRtModel = await loadAndCompile(MODEL_URL, {accelerator: backend});
    statusMsg.innerText = `Multiclass Model Ready on ${backend.toUpperCase()}!`;
    if (wasWebcamRunning) {
      void predictWebcam();
    }
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    statusMsg.innerText = `Error switching to ${backend}: ${message}`;
    console.error(e);
  } finally {
    isCompiling = false;
    backendSelect.disabled = false;
  }
}

backendSelect.addEventListener("change", async () => {
  const selectedBackend = backendSelect.value as 'wasm' | 'webgpu' | 'webnn';
  await switchBackend(selectedBackend);
});

// 2. Webcam Logic
async function enableCam() {
  if (!liteRtModel || isCompiling) return;
  if (webcamRunning) {
    webcamRunning = false;
    webcamButton.innerText = "ENABLE WEBCAM";
    const stream = video.srcObject;
    if (stream && stream instanceof MediaStream) {
      const tracks = stream.getTracks();
      tracks.forEach((track: MediaStreamTrack) => track.stop());
    }
    video.srcObject = null;
  } else {
    webcamRunning = true;
    webcamButton.innerText = 'DISABLE WEBCAM';
    const constraints = { video: { width: 640, height: 480 } };
    video.srcObject = await navigator.mediaDevices.getUserMedia(constraints);
    video.addEventListener("loadeddata", predictWebcam);
  }
}

webcamButton.addEventListener('click', enableCam);
// 3. Real-time Inference Loop
async function predictWebcam() {
  if (!webcamRunning || !liteRtModel || isCompiling) return;
  if (video.currentTime !== lastVideoTime && !isPredicting) {
    lastVideoTime = video.currentTime;
    isPredicting = true;
    try {
      // PRE-PROCESSING (TFJS for math operations)
      const imgTensor = tf.browser.fromPixels(video);
      const resized = tf.image.resizeBilinear(imgTensor, [256, 256]);
      const normalized = resized.div(127.5).sub(1.0).expandDims(0);
      const inputArray = await normalized.data();
      tf.dispose([imgTensor, resized, normalized]);
      // INFERENCE (Native LiteRT)
      const inputLiteRtTensor =
          new Tensor(inputArray as Float32Array, [1, 256, 256, 3]);
      if (liteRtModel && !isCompiling) {
        const outputs = await liteRtModel.run([inputLiteRtTensor]);
        const outputTensor = outputs[0];
        const outputData = await outputTensor.data() as Float32Array;
        inputLiteRtTensor.delete();
        outputTensor.delete();
        // POST-PROCESSING
        drawSegmentation(outputData);
      } else {
        inputLiteRtTensor.delete();
      }
    } catch (error) {
      console.error("LiteRT Inference Error:", error);
    } finally {
      isPredicting = false;
    }
  }
  if (webcamRunning && !isCompiling) {
    window.requestAnimationFrame(predictWebcam);
  }
}

// 4. Draw the multiclass overlay using Argmax
function drawSegmentation(outputData: Float32Array) {
  const width = 256;
  const height = 256;
  // Sync canvas dimensions with the model's output mask
  if (canvasElement.width !== width || imageData === null) {
    canvasElement.width = width;
    canvasElement.height = height;
    imageData = canvasCtx.createImageData(width, height);
  }
  const data = imageData.data;
  // Output data size is 256x256x6
  for (let i = 0; i < width * height; i++) {
    const pixelOffset = i * 6;
    let maxProb = -Infinity;
    let maxClass = 0;
    // Find highest probability class for this pixel
    for (let c = 0; c < 6; c++) {
      if (outputData[pixelOffset + c] > maxProb) {
        maxProb = outputData[pixelOffset + c];
        maxClass = c;
      }
    }
    const n = i * 4;
    const color = categoryColors[maxClass];
    data[n] = color[0];     // R
    data[n + 1] = color[1]; // G
    data[n + 2] = color[2]; // B
    data[n + 3] = color[3]; // Alpha
  }
  canvasCtx.putImageData(imageData!, 0, 0);
}

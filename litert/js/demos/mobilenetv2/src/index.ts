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

import '@tensorflow/tfjs-backend-webgpu'; // side-effect to register the backend

import {runWithTfjsTensors} from '@litertjs/tfjs-interop';
import {CompileOptions, loadAndCompile, loadLiteRt, setWebGpuDevice} from '@litertjs/core';
import {type WebGPUBackend} from '@tensorflow/tfjs-backend-webgpu';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
// Placeholder for internal dependency on object url from safe source and trusted resource url

async function main(useWebGpu = true) {
  await tf.ready();
  let compileOptions: CompileOptions;
  // Load LiteRt's Wasm module.
  await loadLiteRt('./wasm/');
  if (useWebGpu) {
    await tf.setBackend('webgpu');
    const backend = tf.backend() as WebGPUBackend;
    setWebGpuDevice(backend.device);
    compileOptions = {accelerator: 'webgpu'} as CompileOptions;
  } else {
    compileOptions = {
      accelerator: 'wasm',
    } as CompileOptions;
  }

  const imagenetLabels =
      await (await fetch('/static/imagenet_labels.txt')).text();
  const classes = imagenetLabels.split('\n');

  const model = await loadAndCompile(
      'https://storage.googleapis.com/tfweb/litertjs_demo_models/mobilenetv2/torchvision_mobilenet_v2.tflite',
      compileOptions);

  const image = document.getElementById('infer-image') as HTMLImageElement;
  const uploadButton =
      document.getElementById('image-input') as HTMLInputElement;
  uploadButton.onchange = () => {
    const file = (uploadButton.files as FileList)[0];
    if (file) {
      image.src = URL.createObjectURL(file).toString();
    }
  };

  const results = document.createElement('div');

  const loadingMessage =
      document.getElementById('loading-message') as HTMLParagraphElement;
  loadingMessage.innerText = 'Loaded';

  const runTimesInput =
      document.getElementById('run-times') as HTMLInputElement;
  let runTimes = Number(runTimesInput.value);
  runTimesInput.onchange = () => {
    runTimes = Number(runTimesInput.value);
  };

  const runButton =
      document.getElementById('run-inference') as HTMLButtonElement;
  runButton.onclick = async () => {
    results.innerText = '';
    await runInference(runTimes);
  };
  runButton.disabled = false;

  document.body.appendChild(results);

  async function runInference(times = 1) {
    let totalTime = 0;
    let indices: ArrayLike<number>;
    let values: ArrayLike<number>;
    const imageData = tf.tidy(() => {
      // Get RGB data values from the image element and convert it to range
      // [0, 1).
      const imageTensor = tf.browser.fromPixels(image, 3).div(255.0);

      // The source model is PyTorch mobilenet_v2:
      // https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html#mobilenet-v2
      // The pre-processing is referred from
      // MobileNet_V2_Weights.IMAGENET1K_V2.transforms:
      // https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py#L204
      const imageData = imageTensor.resizeBilinear([224, 224])
                            .sub([0.485, 0.456, 0.406])
                            .div([0.229, 0.224, 0.225])
                            .reshape([1, 224, 224, 3])
                            .transpose([0, 3, 1, 2]);
      return imageData;
    });

    for (let i = 0; i < times; ++i) {
      let start: number;
      const top5 = tf.tidy(() => {
        const inputs = {
          'serving_default_args_0:0': imageData,
        };

        start = performance.now();

        const outputs = runWithTfjsTensors(model, inputs);
        const output = outputs['StatefulPartitionedCall:0'];
        return tf.topk(output, 5);
      });

      values = await top5.values.data();
      indices = await top5.indices.data();

      top5.values.dispose();
      top5.indices.dispose();

      const stop = performance.now();
      totalTime += stop - start!;
    }
    const averageTime = Math.round(totalTime / times * 1000) / 1000;
    function append(text: string) {
      const element = document.createElement('div');
      element.appendChild(document.createTextNode(text));
      results.appendChild(element);
    }
    imageData.dispose();
    for (let i = 0; i < 5; ++i) {
      append(`${classes[indices![i]]}: ${values![i]}`);
    }
    append(`Average inference and readback time: ${averageTime}ms`);

    const tfMem = tf.memory();
    append(`TFJS Tensors allocated: ${tfMem.numTensors} <-- Should be 0`);
  }
}

main().catch(console.error);

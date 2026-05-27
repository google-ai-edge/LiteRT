// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Wrapper for LiteRT WASM module.
 * @suppress {missingProperties, globalThis}
 */

let nextRunnerId = 1;

/**
 * Helper to get the JS object behind a pointer ID.
 * @param {number|!Object} ptrId
 * @param {!Object} module
 * @return {!Object|null}
 */
const getBrowserGpuBuffer = function(ptrId, module) {
  if (!ptrId) return null;
  if (typeof ptrId === 'object' && ptrId !== null) return /** @type {!Object} */ (ptrId);

  const webGpu = module.WebGPU || globalThis.WebGPU || (typeof window !== 'undefined' ? window.WebGPU : null);
  if (webGpu && webGpu.Internals && webGpu.Internals.jsObjects) {
    const unsignedPtr = ptrId >>> 0;
    const jsBuffer = webGpu.Internals.jsObjects[unsignedPtr];
    return jsBuffer || null;
  }
  return null;
};

class LiteRTRunner {
  constructor(underlyingRunner, module) {
    this.runner = underlyingRunner;
    this.module = module;
    this.id = nextRunnerId++;
  }

  getId() {
    return this.id;
  }

  async run() {
    return this.runner.run();
  }

  getInput(nameOrIndex) {
    if (typeof nameOrIndex === 'number') {
      if (this.runner.getInputByIndex) {
        return this.runner.getInputByIndex(nameOrIndex);
      }
      throw new Error("This runner does not support access by index.");
    }
    return this.runner.getInput(nameOrIndex);
  }

  getOutput(nameOrIndex) {
    if (typeof nameOrIndex === 'number') {
      if (this.runner.getOutputByIndex) {
        return this.runner.getOutputByIndex(nameOrIndex);
      }
      throw new Error("This runner does not support access by index.");
    }
    return this.runner.getOutput(nameOrIndex);
  }

  setInput(name, tensor) {
    return this.runner.setInput(name, tensor);
  }

  setInputBinary(name, array) {
    return this.runner.setInputBinary(name, array);
  }

  setInputData(name, data) {
    if (Array.isArray(data)) {
      const tensor = this.getInput(name);
      const type = tensor.getType();
      const typeVal = type.value !== undefined ? type.value : type;
      let typedArray;
      switch (typeVal) {
        case 14: // FP32
          typedArray = new Float32Array(data);
          break;
        case 6: // I32
          typedArray = new Int32Array(data);
          break;
        case 4: // I8
          typedArray = new Int8Array(data);
          break;
        case 9: // U8
          typedArray = new Uint8Array(data);
          break;
        case 1: // BOOL
          typedArray = new Uint8Array(data);
          break;
        default:
          throw new Error(`Unsupported tensor type for setInputData: ${typeVal}`);
      }
      return this.runner.setInputBinary(name, new Uint8Array(typedArray.buffer));
    }
    return this.runner.setInputBinary(name, data);
  }

  delete() {
    this.runner.delete();
  }
}
class LiteRtMultiSignatureRunner {
  constructor(underlyingRunner, module) {
    this.runner = underlyingRunner;
    this.module = module;
    this.id = nextRunnerId++;
  }

  getId() {
    return this.id;
  }

  async run(signatureName) {
    return this.runner.runSig(signatureName);
  }

  getInput(signatureName, name) {
    return this.runner.getInputSig(signatureName, name);
  }

  getOutput(signatureName, name) {
    return this.runner.getOutputSig(signatureName, name);
  }

  setInput(signatureName, name, tensor) {
    return this.runner.setInputSig(signatureName, name, tensor);
  }

  setInputBinary(signatureName, name, array) {
    return this.runner.setInputBinarySig(signatureName, name, array);
  }

  setInputData(signatureName, name, data) {
    if (Array.isArray(data)) {
      const tensor = this.getInput(signatureName, name);
      const type = tensor.getType();
      const typeVal = type.value !== undefined ? type.value : type;
      let typedArray;
      switch (typeVal) {
        case 14: // FP32
          typedArray = new Float32Array(data);
          break;
        case 6: // I32
          typedArray = new Int32Array(data);
          break;
        case 4: // I8
          typedArray = new Int8Array(data);
          break;
        case 9: // U8
          typedArray = new Uint8Array(data);
          break;
        case 1: // BOOL
          typedArray = new Uint8Array(data);
          break;
        default:
          throw new Error(`Unsupported tensor type for setInputData: ${typeVal}`);
      }
      return this.runner.setInputBinarySig(signatureName, name, new Uint8Array(typedArray.buffer));
    }
    return this.runner.setInputBinarySig(signatureName, name, data);
  }

  getOutputWebGpuBuffer(signatureName, name) {
    return this.runner.getOutputWebGpuBuffer(signatureName, name);
  }

  getInputWebGpuBuffer(signatureName, name) {
    return this.runner.getInputWebGpuBuffer(signatureName, name);
  }

  delete() {
    this.runner.delete();
  }
}

/**
 * Wraps the generated LiteRT WASM module and adds idiomatic JS methods.
 * @param {!Object} module The generated Emscripten module instance.
 * @return {!Object} The augmented module instance.
 */
export function wrapLiteRTModule(module) {
  const originalCreateTensor = module.createTensor;
  let unnamedTensorIdx = 0;
  
  module.createTensor = function(options) {
    let type = options.type;
    if (typeof type === 'string') {
      type = type.toUpperCase();
      if (module.TensorType && module.TensorType[type]) {
        type = module.TensorType[type];
      } else {
        console.warn(`Unknown tensor type: ${type}, available:`, module.TensorType);
      }
    }
    
    if (originalCreateTensor) {
      const t = originalCreateTensor({ ...options, type: type });
      if (options.name) {
        t.setName(options.name);
      } else {
        t.setName(`js_unnamed_tensor_${unnamedTensorIdx++}`);
      }
      if (options.storage) {
        t.storage = options.storage;
      }
      t._aliasedBuffers = new Map();
      return t;
    }
    

    const shape = options.shape || [1];
    const t = module.createPlaceholderTensor(type.value !== undefined ? type.value : type, shape, options.name || null);
    if (options.storage) {
      t.storage = options.storage;
    }
    t._aliasedBuffers = new Map();
    return t;
  };

  // 2. setData method on Tensor prototype
  if (module.Tensor && module.Tensor.prototype) {
    module.Tensor.prototype.setData = async function(data) {
      let mutableData = await this.getMutableData();
      if (!mutableData) {
        this.allocateBuffer();
        mutableData = await this.getMutableData();
      }
      
      if (mutableData) {
        mutableData.set(data);
      } else {
        throw new Error("Cannot set data on this tensor (maybe not mutable or wrong type).");
      }
    };

    const originalConcatenation = module.Tensor.prototype.concatenation;
    module.Tensor.prototype.concatenation = function(others, axis) {
      const vector = new module.TensorVector();
      for (const t of others) {
        vector.push_back(t);
      }
      const result = originalConcatenation.call(this, vector, axis);
      vector.delete();
      return result;
    };

    const originalGetWgpuBuf = module.Tensor.prototype.getWebGpuBuffer;
    if (originalGetWgpuBuf) {
      module.Tensor.prototype.getWebGpuBuffer = function() {
        const ptrId = originalGetWgpuBuf.call(this);
        return getBrowserGpuBuffer(ptrId, module);
      };
    }
  }

  let eagerMode = false;
  module.setEagerMode = function(enable) { eagerMode = enable; };

  if (module.Tensor && module.Tensor.prototype) {
    module.Tensor.prototype.evaluate = function() {
      return module.runEager([this]);
    };

    const ops = ["abs", "relu", "relu6", "elu", "hardSwish", "logSoftmax",
                 "softmax", "logistic", "gelu", "neg", "sqrt", "cos", "sin",
                 "exp", "log", "ceil", "floor", "sign", "round", "logicalNot",
                 "add", "mul", "sub", "div", "pow", "minimum", "maximum",
                 "less", "greater", "lessEqual", "greaterEqual", "equal",
                 "notEqual", "logicalAnd", "logicalOr", "floorDiv", "floorMod"];

    ops.forEach(opName => {
      const originalOp = module.Tensor.prototype[opName];
      if (originalOp) {
        const wrapper = function(...args) {
          this[opName] = originalOp;
          try {
            const result = originalOp.apply(this, args);
            if (eagerMode) {
              const success = module.runEager([result]);
              if (!success) {
                throw new Error(`Eager execution failed for operation '${opName}'. This usually happens if the graph contains placeholder inputs without data.`);
              }
            }
            return result;
          } finally {
            delete this[opName];
          }
        };
        Object.assign(wrapper, originalOp);
        module.Tensor.prototype[opName] = wrapper;
      }
    });
  }

  // 3. Wrapper functions for runners to handle enums and unify interface
  module.createGraphRunner = async function(inputs, outputs, accelerators) {
    let acc = accelerators;
    if (acc && typeof acc === 'object' && acc.value !== undefined) {
      acc = acc.value;
    }
    if (acc === undefined) {
      acc = module.HwAccelerators.GPU.value;
    }
    const rawRunner = await module.createStaticLambdaRunner(inputs, outputs, acc);
    return new LiteRTRunner(rawRunner, module);
  };

  module.writeToBuffer = function(ptr, data) {
    if (!module.HEAPU8) {
      throw new Error("Could not find HEAPU8 in WASM module.");
    }
    const buffer = module.HEAPU8.buffer;

    if (data instanceof Float32Array) {
      const heap = new Float32Array(buffer);
      heap.set(data, ptr / 4);
    } else if (data instanceof Int32Array) {
      const heap = new Int32Array(buffer);
      heap.set(data, ptr / 4);
    } else if (data instanceof Int8Array || data instanceof Uint8Array) {
      const heap = new Uint8Array(buffer);
      heap.set(data, ptr);
    } else {
      throw new Error("Unsupported typed array for writeToBuffer");
    }
  };

  module.createModelRunner = function(buffer, accelerators) {
    let acc = 3; // Default to both
    if (accelerators !== undefined) {
      acc = typeof accelerators === 'object' ? accelerators.value : accelerators;
    }
    const rawRunner = module.createDynamicRunnerFromBuffer(buffer, acc);
    return new LiteRTRunner(rawRunner, module);
  };

  module.createMultiSignatureRunner = function(signatures, accelerators) {
    let acc = 3; // Default to both
    if (accelerators !== undefined) {
      acc = typeof accelerators === 'object' ? accelerators.value : accelerators;
    }
    const rawRunner = module.createMultiSignatureRunnerInternal(signatures, acc);
    return new LiteRtMultiSignatureRunner(rawRunner, module);
  };

  module.jit = function(func, options = {}) {
    let compiledRunner = null;
    let inputNames = [];
    let outputNames = [];
    let isArrayOutput = false;

    const jittedFn = async function(...args) {
      if (!compiledRunner) {
        const inputBindings = {};
        inputNames = args.map((_, index) => `jit_input_${index}`);

        const traceArgs = args.map((arg, index) => {
          if (!arg || !arg.getName) {
            throw new Error(`jit() arguments must be LiteRT Tensors. Argument at index ${index} is not a Tensor.`);
          }
          const name = inputNames[index];
          const placeholder = module.createTensor({
            name: name,
            type: arg.getType(),
            shape: arg.getShape()
          });
          inputBindings[name] = placeholder;
          return placeholder;
        });

        const wasEager = eagerMode;
        eagerMode = false;

        let tracedOutput;
        try {
          tracedOutput = func(...traceArgs);
        } finally {
          eagerMode = wasEager;
        }

        // Normalize output list to handle both single Tensors and Arrays beautifully!
        let outputsList = [];
        if (Array.isArray(tracedOutput)) {
          outputsList = tracedOutput;
          isArrayOutput = true;
        } else {
          outputsList = [tracedOutput];
          isArrayOutput = false;
        }

        outputsList.forEach((out, idx) => {
          if (!out || !out.getName) {
            throw new Error(`The jitted function returned an invalid element at index ${idx}. Every element must be a valid LiteRT Tensor.`);
          }
        });

        const outputBindings = {};
        outputNames = outputsList.map((out, idx) => {
          const name = `jit_output_${idx}`;
          out.setName(name);
          outputBindings[name] = out;
          return name;
        });

        const rawRunner = await module.createStaticLambdaRunner(
          inputBindings,
          outputBindings,
          options.accelerators !== undefined ? (typeof options.accelerators === 'object' ? options.accelerators.value : options.accelerators) : (module.HwAccelerators.GPU ? module.HwAccelerators.GPU.value : 2)
        );
        compiledRunner = new LiteRTRunner(rawRunner, module);

        // Multi-Runner Registry and Dynamic Buffer Aliasing!
        for (let i = 0; i < args.length; i++) {
          if (args[i] && args[i].storage === 'webgpu') {
            const internalInput = compiledRunner.getInput(inputNames[i]);
            const pointerId = internalInput.getWebGpuBuffer();
            
            // Register the mapping inside the placeholder's Map
            args[i]._aliasedBuffers.set(compiledRunner.getId(), pointerId);
            
            // Override getWebGpuBuffer to fetch contextually based on target compiled runner
            args[i].getWebGpuBuffer = function(targetRunner) {
              const runnerKey = targetRunner ? (targetRunner.getId ? targetRunner.getId() : targetRunner) : Array.from(this._aliasedBuffers.keys())[0];
              const ptr = this._aliasedBuffers.get(runnerKey);
              return getBrowserGpuBuffer(ptr, module);
            };
          }
        }
      }

      const runnerId = compiledRunner.getId();
      for (let i = 0; i < args.length; i++) {
        // Self-healing bypass check: skip only if natively aliased to this specific runner!
        if (args[i] && args[i]._aliasedBuffers && args[i]._aliasedBuffers.has(runnerId)) continue;
        compiledRunner.setInput(inputNames[i], args[i]);
      }

      await compiledRunner.run();

      // Return single output or conformed Array of outputs symmetrically!
      const outputs = outputNames.map(name => compiledRunner.getOutput(name));
      return isArrayOutput ? outputs : outputs[0];
    };

    jittedFn.getRunner = () => compiledRunner;
    jittedFn.getInputName = (index) => `jit_input_${index}`;
    jittedFn.compile = async function(...sampleArgs) {
      if (compiledRunner) return;
      await jittedFn(...sampleArgs);
    };

    return jittedFn;
  };

  module.jitMulti = function(sigConfigs, options = {}) {
    let compiledRunner = null;
    const sigInputNames = {};
    const sigOutputNames = {};
    const jittedObject = {};

    async function compileAll(sampleInputs = null) {
      const signaturesBindings = {};
      const wasEager = eagerMode;
      eagerMode = false;

      try {
        for (const [sigName, config] of Object.entries(sigConfigs)) {
          let func, inputs;
          if (typeof config === 'function') {
            func = config;
            if (!sampleInputs || !sampleInputs[sigName]) {
              throw new Error(`Cannot compile signature '${sigName}'. No sample inputs supplied.`);
            }
            inputs = sampleInputs[sigName].map(arg => ({
              type: arg.getType(),
              shape: arg.getShape()
            }));
          } else {
            func = config.func;
            inputs = config.inputs;
            if (!func || !inputs) {
              throw new Error(`Invalid configuration for signature '${sigName}'. Must contain 'func' and 'inputs'.`);
            }
          }

          sigInputNames[sigName] = inputs.map((_, idx) => `sig_${sigName}_in_${idx}`);
          const traceArgs = inputs.map((inputDesc, idx) => {
            return module.createTensor({
              name: sigInputNames[sigName][idx],
              type: inputDesc.type,
              shape: inputDesc.shape
            });
          });

          let tracedOutputs = func(...traceArgs);
          if (!Array.isArray(tracedOutputs)) {
            tracedOutputs = [tracedOutputs];
          }

          sigOutputNames[sigName] = tracedOutputs.map((out, idx) => {
            if (!out || !out.getName) {
              throw new Error(`Signature '${sigName}' must return LiteRT Tensor(s).`);
            }
            const outName = `sig_${sigName}_out_${idx}`;
            out.setName(outName);
            return outName;
          });

          signaturesBindings[sigName] = {
            outputs: tracedOutputs
          };
        }
      } finally {
        eagerMode = wasEager;
      }

      let acc = options.accelerators;
      if (acc && typeof acc === 'object' && acc.value !== undefined) {
        acc = acc.value;
      }
      if (acc === undefined) {
        acc = module.HwAccelerators.GPU ? module.HwAccelerators.GPU.value : 2;
      }

      const rawRunner = module.createMultiSignatureRunnerInternal(signaturesBindings, acc);
      compiledRunner = new LiteRtMultiSignatureRunner(rawRunner, module);
    }

    let canCompileImmediately = true;
    for (const config of Object.values(sigConfigs)) {
      if (typeof config === 'function' || !config.inputs) {
        canCompileImmediately = false;
        break;
      }
    }

    let compilationPromise = null;
    if (canCompileImmediately) {
      compilationPromise = compileAll();
    }

    for (const sigName of Object.keys(sigConfigs)) {
      jittedObject[sigName] = async function(...args) {
        if (compilationPromise) {
          await compilationPromise;
        }
        if (!compiledRunner) {
          throw new Error("jitMulti: Runner is not compiled yet. Call .compile(sampleInputs) first.");
        }

        const inputNames = sigInputNames[sigName];
        const runnerId = compiledRunner.getId();
        for (let i = 0; i < args.length; i++) {
          if (args[i] && args[i]._aliasedBuffers && args[i]._aliasedBuffers.has(runnerId)) continue;
          compiledRunner.setInput(sigName, inputNames[i], args[i]);
        }

        await compiledRunner.run(sigName);

        const outputNames = sigOutputNames[sigName];
        const results = outputNames.map(name => compiledRunner.getOutput(sigName, name));
        return results.length === 1 ? results[0] : results;
      };
    }

    jittedObject.compile = async function(sampleInputs) {
      if (compiledRunner) {
        throw new Error("jitMulti is already compiled.");
      }
      compilationPromise = compileAll(sampleInputs);
      await compilationPromise;

      // Multi-Runner Registry and Dynamic Buffer Aliasing for jitMulti!
      if (sampleInputs && typeof sampleInputs === 'object' && sampleInputs !== null) {
        for (const [sigName, sigArgs] of Object.entries(sampleInputs)) {
          if (Array.isArray(sigArgs)) {
            const inputNames = sigInputNames[sigName];
            for (let i = 0; i < sigArgs.length; i++) {
              if (sigArgs[i] && sigArgs[i].storage === 'webgpu') {
                const internalInput = compiledRunner.getInput(sigName, inputNames[i]);
                const pointerId = internalInput.getWebGpuBuffer();
                
                // Register the mapping inside the placeholder's Map
                sigArgs[i]._aliasedBuffers.set(compiledRunner.getId(), pointerId);
                
                // Override getWebGpuBuffer contextually
                sigArgs[i].getWebGpuBuffer = function(targetRunner) {
                  const runnerKey = targetRunner ? (targetRunner.getId ? targetRunner.getId() : targetRunner) : Array.from(this._aliasedBuffers.keys())[0];
                  const ptr = this._aliasedBuffers.get(runnerKey);
                  return getBrowserGpuBuffer(ptr, module);
                };
              }
            }
          }
        }
      }
    };

    return jittedObject;
  };

  return module;
}

/**
 * Loads the LiteRT WASM module and returns the augmented instance.
 * Assumes createTensorModule is available globally (e.g., loaded via script tag).
 * @param {Object=} options Emscripten module options.
 * @return {!Promise<!Object>} The augmented module instance.
 */
export async function createLiteRT(options = {}) {
  let loader = options.createTensorModule || globalThis.createTensorModule || (typeof createTensorModule !== 'undefined' ? createTensorModule : null);
  if (loader && loader.default) {
    loader = loader.default;
  }
  if (typeof loader !== 'function') {
    throw new Error("createTensorModule is not defined. Make sure to load tensor_wasm_internal.js first.");
  }
  const module = await loader(options);
  
  // Register the preinitialized WebGPU device pointer ID inside C++ to share device contexts seamlessly!
  if (module.setWebGpuDeviceId && options && options.preinitializedWebGPUDevice) {
    if (module.emscripten_webgpu_get_device) {
      const deviceId = module.emscripten_webgpu_get_device();
      module.setWebGpuDeviceId(deviceId);
    }
  }

  return wrapLiteRTModule(module);
}

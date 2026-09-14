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

class LiteRTRunner {
  constructor(underlyingRunner, module) {
    this.runner = underlyingRunner;
    this.module = module;
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
      return t;
    }
    

    const shape = options.shape || [1];
    const t = module.createPlaceholderTensor(type.value !== undefined ? type.value : type, shape, options.name || null);
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

  return module;
}

/**
 * Loads the LiteRT WASM module and returns the augmented instance.
 * Assumes createTensorModule is available globally (e.g., loaded via script tag).
 * @param {!Object} options Emscripten module options.
 * @return {!Promise<!Object>} The augmented module instance.
 */
export async function createLiteRT(options) {
  if (typeof createTensorModule !== 'function') {
    throw new Error("createTensorModule is not defined. Make sure to load tensor_wasm_internal.js first.");
  }
  const module = await createTensorModule(options);
  return wrapLiteRTModule(module);
}

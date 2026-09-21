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

import {CompiledModel} from './compiled_model';
import {DType} from './datatypes';
import {getGlobalLiteRt} from './global_litert';
import {Model} from './model';
import {CompileOptions, fillCompileOptions} from './model_types';
import {Dimensions, Tensor} from './tensor';
import {
  Deletable,
  LiteRtCompiledModel,
  LiteRtSignatureGraphSpec,
} from './wasm_binding_types';

// tslint:disable:no-any

/**
 * Specification of an input tensor shape and data type for precompilation.
 */
export interface PlaceholderSpec {
  shape: Dimensions;
  dataType?: DType;
}

/**
 * An authored model input, which can be a concrete Tensor or a PlaceholderSpec.
 */
export type InputTensorOrSpec = Tensor | PlaceholderSpec;

/**
 * A user-authored tensor function to be traced and JIT-compiled into a LiteRT model graph.
 */
export type TensorFunction = (...args: any[]) => any;

/**
 * Explicit specification for a named signature in a multi-signature compilation.
 */
export interface SignatureSpec {
  name?: string;
  inputs: any[] | Record<string, any> | any;
}

interface SignatureMetadata {
  compiledModel: CompiledModel;
  signatureName: string;
  cacheKey: string;
  outputFormat: 'single' | 'array' | 'dict';
  outputDictKeys?: string[];
}

interface NormalizedInput {
  shape: Dimensions;
  dtype: DType;
}

interface ParsedInputs {
  normalizedInputs: NormalizedInput[];
  inputTensors?: Tensor[];
  cacheKey: string;
  isSingleInput: boolean;
  isArrayInput: boolean;
  isDictInput: boolean;
  dictKeys: string[];
}

function isPlaceholderSpec(value: any): value is PlaceholderSpec {
  return (
    typeof value === 'object' &&
    value !== null &&
    !(value instanceof Tensor) &&
    'shape' in value &&
    Array.isArray(value.shape) &&
    (value.shape.length === 0 || typeof value.shape[0] === 'number')
  );
}

function normalizeCompileArgs(
  args: any[],
): Array<{name?: string; rawInputs: any[]}> {
  if (args.length === 0) {
    throw new Error('compile() requires at least one input specification.');
  }

  if (args.length === 1) {
    const arg = args[0];

    // Case 1: Explicit configuration object with `signatures` field
    if (
      typeof arg === 'object' &&
      arg !== null &&
      !(arg instanceof Tensor) &&
      !isPlaceholderSpec(arg) &&
      'signatures' in arg
    ) {
      const sigs = (arg as {signatures: any}).signatures;
      if (Array.isArray(sigs)) {
        return sigs.map((item, idx) => {
          if (Array.isArray(item)) {
            return {name: `signature_${idx}`, rawInputs: item};
          } else if (
            typeof item === 'object' &&
            item !== null &&
            'inputs' in item
          ) {
            const rawInputs = Array.isArray(item.inputs)
              ? item.inputs
              : [item.inputs];
            return {name: item.name ?? `signature_${idx}`, rawInputs};
          } else {
            return {name: `signature_${idx}`, rawInputs: [item]};
          }
        });
      } else if (typeof sigs === 'object' && sigs !== null) {
        return Object.keys(sigs).map((name) => {
          const val = sigs[name];
          const rawInputs = Array.isArray(val) ? val : [val];
          return {name, rawInputs};
        });
      }
    }

    // Case 2: Array of signatures: e.g. [[specA], [specB]] or [{name, inputs}, ...]
    if (Array.isArray(arg)) {
      const isMultiSig =
        arg.length > 0 &&
        arg.every(
          (item) =>
            Array.isArray(item) ||
            (typeof item === 'object' &&
              item !== null &&
              !(item instanceof Tensor) &&
              !isPlaceholderSpec(item) &&
              'inputs' in item),
        );
      if (isMultiSig) {
        return arg.map((item, idx) => {
          if (Array.isArray(item)) {
            return {name: `signature_${idx}`, rawInputs: item};
          } else {
            const rawInputs = Array.isArray(item.inputs)
              ? item.inputs
              : [item.inputs];
            return {name: item.name ?? `signature_${idx}`, rawInputs};
          }
        });
      }
    }

    // Case 3: Record of named signatures: { sig1: [spec1], sig2: [spec2] }
    if (
      typeof arg === 'object' &&
      arg !== null &&
      !(arg instanceof Tensor) &&
      !isPlaceholderSpec(arg) &&
      !Array.isArray(arg)
    ) {
      const keys = Object.keys(arg);
      const isDictInput = keys.every(
        (k) => arg[k] instanceof Tensor || isPlaceholderSpec(arg[k]),
      );
      if (!isDictInput && keys.length > 0) {
        return keys.map((name) => {
          const val = arg[name];
          const rawInputs = Array.isArray(val) ? val : [val];
          return {name, rawInputs};
        });
      }
    }
  }

  // Fallback: Single signature with `args` as raw inputs
  return [{name: 'serving_default', rawInputs: args}];
}

/**
 * Represents a JIT-compiled model authored directly in JavaScript / TypeScript.
 *
 * Supports single and multi-signature ahead-of-time compilation. When multiple signatures
 * are specified in a single compile() call, all signatures are compiled into a single `CompiledModel`
 * instance sharing constant weight buffers. Multiple compile() calls are supported, with each
 * call generating a new `CompiledModel`.
 */
export class AuthoredModel implements Deletable {
  private readonly compiledModels: CompiledModel[] = [];
  private compilationPromise?: Promise<CompiledModel>;
  private readonly cacheKeyToSignature = new Map<string, SignatureMetadata>();
  private deletedInternal = false;

  constructor(
    readonly fn: TensorFunction,
    readonly options: CompileOptions = {},
  ) {}

  get compiledModel(): CompiledModel | undefined {
    return this.compiledModels.length > 0
      ? this.compiledModels[this.compiledModels.length - 1]
      : undefined;
  }

  get deleted(): boolean {
    return this.deletedInternal;
  }

  private ensureNotDeleted() {
    if (this.deletedInternal) {
      throw new Error('AuthoredModel is deleted and cannot be used.');
    }
  }

  private parseInputs(args: any[], requireTensors: boolean): ParsedInputs {
    let rawInputs: any[] = [];
    let isSingleInput = false;
    let isArrayInput = false;
    let isDictInput = false;
    let dictKeys: string[] = [];

    if (args.length === 1) {
      const first = args[0];
      if (first instanceof Tensor || isPlaceholderSpec(first)) {
        rawInputs = [first];
        isSingleInput = true;
      } else if (Array.isArray(first)) {
        rawInputs = first;
        isArrayInput = true;
      } else if (typeof first === 'object' && first !== null) {
        dictKeys = Object.keys(first).sort();
        rawInputs = dictKeys.map((k) => first[k]);
        isDictInput = true;
      }
    } else {
      rawInputs = args;
    }

    const inputTensors: Tensor[] = [];
    const normalizedInputs: NormalizedInput[] = [];

    for (let i = 0; i < rawInputs.length; i++) {
      const item = rawInputs[i];
      if (item instanceof Tensor) {
        inputTensors.push(item);
        normalizedInputs.push({
          shape: item.type.layout.dimensions,
          dtype: item.type.dtype,
        });
      } else if (!requireTensors && isPlaceholderSpec(item)) {
        const dtype = item.dataType ?? 'float32';
        normalizedInputs.push({
          shape: item.shape,
          dtype,
        });
      } else {
        const expected = requireTensors
          ? 'Tensor'
          : 'Tensor or PlaceholderSpec ({ shape: Dimensions, dataType?: DType })';
        throw new Error(`Input at index ${i} is not a valid ${expected}.`);
      }
    }

    const cacheKey = isDictInput
      ? dictKeys
          .map(
            (k, i) =>
              `${k}:${normalizedInputs[i].dtype}[${normalizedInputs[i].shape.join(',')}]`,
          )
          .join(';')
      : normalizedInputs
          .map((n) => `${n.dtype}[${n.shape.join(',')}]`)
          .join(';');

    return {
      normalizedInputs,
      inputTensors: requireTensors ? inputTensors : undefined,
      cacheKey,
      isSingleInput,
      isArrayInput,
      isDictInput,
      dictKeys,
    };
  }

  private async compileSignaturesInternal(
    specs: Array<{name?: string; rawInputs: any[]}>,
  ): Promise<CompiledModel> {
    const globalLiteRt = getGlobalLiteRt();
    const liteRtWasm = globalLiteRt.liteRtWasm;
    const environment =
      this.options.environment ?? globalLiteRt.getDefaultEnvironment();

    const parsedSignatures: Array<{
      name: string;
      parsed: ParsedInputs;
    }> = [];

    // Verify uniqueness of input specs within this compilation batch
    const batchSeenKeys = new Set<string>();
    for (let i = 0; i < specs.length; i++) {
      const spec = specs[i];
      const parsed = this.parseInputs(spec.rawInputs, /* requireTensors= */ false);
      if (batchSeenKeys.has(parsed.cacheKey)) {
        throw new Error(
          `Duplicate signature input specification for '${parsed.cacheKey}'. Each signature in a compile call must have distinct input shapes or dtypes.`,
        );
      }
      batchSeenKeys.add(parsed.cacheKey);
      const sigName =
        spec.name ?? (specs.length === 1 ? 'serving_default' : `signature_${i}`);
      parsedSignatures.push({name: sigName, parsed});
    }

    const wasmSignatureSpecs: LiteRtSignatureGraphSpec[] = [];
    const allPlaceholders: Tensor[] = [];
    const allOutputs: Tensor[] = [];

    let modelDataPtr = 0;
    let modelSize = 0;

    const signatureOutputsMeta: Array<{
      outputFormat: 'single' | 'array' | 'dict';
      outputDictKeys?: string[];
    }> = [];

    try {
      for (const {name, parsed} of parsedSignatures) {
        // Create symbolic placeholder tensors for this signature
        const placeholders: Tensor[] = parsed.normalizedInputs.map(
          (norm, idx) => {
            const placeholderName = parsed.isDictInput
              ? parsed.dictKeys[idx]
              : `input_${idx}`;
            return Tensor.createPlaceholder({
              shape: norm.shape,
              dataType: norm.dtype,
              environment,
              name: placeholderName,
            });
          },
        );
        allPlaceholders.push(...placeholders);

        // Call the user's function with placeholders to trace the graph
        let symbolicOutputs: any;
        if (parsed.isSingleInput) {
          symbolicOutputs = this.fn(placeholders[0]);
        } else if (parsed.isArrayInput) {
          symbolicOutputs = this.fn(placeholders);
        } else if (parsed.isDictInput) {
          const dictArg: Record<string, Tensor> = {};
          parsed.dictKeys.forEach((k, idx) => {
            dictArg[k] = placeholders[idx];
          });
          symbolicOutputs = this.fn(dictArg);
        } else {
          symbolicOutputs = this.fn(...placeholders);
        }

        // Unpack symbolic output tensors
        let outputTensors: Tensor[] = [];
        let outputFormat: 'single' | 'array' | 'dict' = 'single';
        let outputDictKeys: string[] | undefined;

        if (symbolicOutputs instanceof Tensor) {
          outputTensors = [symbolicOutputs];
          outputFormat = 'single';
        } else if (Array.isArray(symbolicOutputs)) {
          outputTensors = symbolicOutputs;
          outputFormat = 'array';
        } else if (
          typeof symbolicOutputs === 'object' &&
          symbolicOutputs !== null
        ) {
          outputDictKeys = Object.keys(symbolicOutputs);
          outputTensors = outputDictKeys.map((k) => symbolicOutputs[k]);
          outputFormat = 'dict';
        } else {
          throw new Error(
            'AuthoredModel function must return a Tensor, Tensor[], or Record<string, Tensor>.',
          );
        }

        for (let i = 0; i < outputTensors.length; i++) {
          if (!(outputTensors[i] instanceof Tensor)) {
            throw new Error(
              `Output at index ${i} for signature '${name}' is not a Tensor.`,
            );
          }
        }
        allOutputs.push(...outputTensors);

        wasmSignatureSpecs.push({
          name,
          inputHandles: placeholders.map((p) => p.liteRtTensorHandle),
          outputHandles: outputTensors.map((o) => o.liteRtTensorHandle),
        });

        signatureOutputsMeta.push({
          outputFormat,
          outputDictKeys,
        });
      }

      // Generate a FlatBuffer via ModelFactory (multi-signature if multiple specs provided)
      const modelData =
        liteRtWasm.createModelDataFromTensorGraph(wasmSignatureSpecs);
      modelDataPtr = modelData.modelDataPtr;
      modelSize = modelData.modelSize;
    } finally {
      // Clean up all symbolic tensors from all tracing passes
      for (const p of allPlaceholders) {
        p.delete();
      }
      for (const o of allOutputs) {
        o.delete();
      }
    }

    // Load and compile the model
    const wasmModel = liteRtWasm.loadModel(
      environment.liteRtEnvironment,
      modelDataPtr,
      modelSize,
    );

    const filledCompileOptions = fillCompileOptions(
      this.options,
      environment,
      liteRtWasm.getThreadCount(),
    );

    let wasmCompiledModel: LiteRtCompiledModel;
    try {
      wasmCompiledModel = await liteRtWasm.compileModel(
        environment.liteRtEnvironment,
        wasmModel,
        filledCompileOptions,
      );
    } catch (e) {
      liteRtWasm._free(modelDataPtr);
      wasmModel.delete();
      throw e;
    }

    const loadedModel = new Model(wasmModel, () => {
      liteRtWasm._free(modelDataPtr);
    });

    const compiledModel = new CompiledModel(
      loadedModel,
      wasmCompiledModel,
      filledCompileOptions,
      () => {},
    );

    this.compiledModels.push(compiledModel);

    // Register each signature with its corresponding compiled model
    for (let i = 0; i < parsedSignatures.length; i++) {
      const {name, parsed} = parsedSignatures[i];
      const meta = signatureOutputsMeta[i];
      this.cacheKeyToSignature.set(parsed.cacheKey, {
        compiledModel,
        signatureName: name,
        cacheKey: parsed.cacheKey,
        outputFormat: meta.outputFormat,
        outputDictKeys: meta.outputDictKeys,
      });
    }

    return compiledModel;
  }

  /**
   * Compiles the model graph ahead of time into a CompiledModel.
   *
   * Can be invoked with:
   * 1. A single signature (e.g. `compile({ shape: [1, 256, 256, 3] })` or `compile(specA, specB)`)
   * 2. Multiple signatures as an array of input tuples (e.g. `compile([ [{shape: [1, 10]}], [{shape: [4, 10]}] ])`)
   * 3. Multiple named signatures as a dictionary (e.g. `compile({ lowRes: [{shape: [1, 10]}], highRes: [{shape: [4, 10]}] })`)
   *
   * When multiple signatures are provided in a single compile() call, they share the same constant weight buffers.
   * Multiple compile() calls are permitted; each call generates a new CompiledModel.
   */
  async compile(...args: any[]): Promise<void> {
    this.ensureNotDeleted();

    const specs = normalizeCompileArgs(args);
    if (specs.length === 1 && specs[0].name === 'signature_0') {
      specs[0].name = 'serving_default';
    }

    this.compilationPromise = this.compileSignaturesInternal(specs);
    await this.compilationPromise;
  }

  /**
   * Returns true if a compiled signature exists in the model for the given input specs or tensors.
   */
  hasCompiledSignature(...args: any[]): boolean {
    this.ensureNotDeleted();
    if (this.compiledModels.length === 0) {
      return false;
    }
    const {cacheKey} = this.parseInputs(args, /* requireTensors= */ false);
    return this.cacheKeyToSignature.has(cacheKey);
  }

  /**
   * Executes the authored tensor model with concrete inputs.
   *
   * If the input shape/type has not been compiled yet, it is compiled silently under the hood
   * with a notice logged to the console.
   */
  async run(...args: any[]): Promise<any> {
    this.ensureNotDeleted();
    const parsed = this.parseInputs(args, /* requireTensors= */ true);

    // Wait for any currently ongoing compilation to finish
    if (this.compilationPromise) {
      await this.compilationPromise;
    }

    this.ensureNotDeleted();

    let signature = this.cacheKeyToSignature.get(parsed.cacheKey);
    if (!signature) {
      // Silently compile a new CompiledModel for this unseen tensor shape/type
      console.log(
        `[LiteRT] JIT compiling model for input shape/type '${parsed.cacheKey}' under the hood...`,
      );
      this.compilationPromise = this.compileSignaturesInternal([
        {name: 'serving_default', rawInputs: args},
      ]);
      await this.compilationPromise;
      signature = this.cacheKeyToSignature.get(parsed.cacheKey)!;
    }

    // Run the compiled model with concrete input tensors for this signature
    const rawOutputs = await signature.compiledModel.run(
      signature.signatureName,
      parsed.inputTensors!,
    );

    // Format return value matching the function's return structure
    switch (signature.outputFormat) {
      case 'single':
        return rawOutputs[0];
      case 'array':
        return rawOutputs;
      case 'dict': {
        const result: Record<string, Tensor> = {};
        signature.outputDictKeys!.forEach((k, idx) => {
          result[k] = rawOutputs[idx];
        });
        return result;
      }
      default: {
        const exhaustiveCheck: never = signature.outputFormat;
        throw new Error(`Unhandled output format: ${exhaustiveCheck}`);
      }
    }
  }

  delete() {
    if (this.deletedInternal) {
      return;
    }
    this.deletedInternal = true;
    for (const model of this.compiledModels) {
      model.delete();
    }
    this.compiledModels.length = 0;
    this.cacheKeyToSignature.clear();
  }
}

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

// Placeholder for internal dependency on trusted resource url type

import {runScript} from './run_script_helper';
import {WasmModule} from './wasm_module';

type UrlString = string;

/**
 * Internal type of constructors used for initializing wasm modules.
 */
export type WasmConstructor<LibType> =
    (new (
         module: WasmModule, canvas?: HTMLCanvasElement|OffscreenCanvas|null) =>
         LibType);

/**
 * Simple interface for allowing users to set the directory where internal
 * wasm-loading and asset-loading code looks (e.g. for .wasm and .data file
 * locations).
 */
export declare interface FileLocator {
  locateFile: (filename: string) => string;
  mainScriptUrlOrBlob?: string;
}

/**
 * Global function interface to initialize Wasm blob and load runtime assets for
 *     a specialized Wasm library. Standard implementation is
 *     `createWasmLib<LibType>`.
 * @param constructorFcn The name of the class to instantiate via "new".
 * @param wasmLoaderScript Url for the wasm-runner script; produced by the build
 *     process.
 * @param assetLoaderScript Url for the asset-loading script; produced by the
 *     build process.
 * @param fileLocator A function to override the file locations for assets
 *     loaded by the Wasm library.
 * @return promise A promise which will resolve when initialization has
 *     completed successfully.
 */
export interface CreateWasmLibApi {
  <LibType>(
      constructorFcn: WasmConstructor<LibType>,
      wasmLoaderScript?: UrlString|null, assetLoaderScript?: UrlString|null,
      glCanvas?: HTMLCanvasElement|OffscreenCanvas|null,
      fileLocator?: FileLocator): Promise<LibType>;
}

// Global declarations, for tapping into Window for Wasm blob running
declare global {
  interface Window {
    // Created by us using wasm-runner script
    Module?: WasmModule|FileLocator;
    // Created by wasm-runner script
    ModuleFactory?: (fileLocator: FileLocator) => Promise<WasmModule>;
  }
}

/** {@override CreateWasmLibApi} */
export const createWasmLib: CreateWasmLibApi = async<LibType>(
    constructorFcn: WasmConstructor<LibType>, wasmLoaderScript?: UrlString|null,
    assetLoaderScript?: UrlString|null,
    glCanvas?: HTMLCanvasElement|OffscreenCanvas|null,
    fileLocator?: FileLocator): Promise<LibType> => {
  // Run wasm-loader script here
  if (wasmLoaderScript) {
    await runScript(wasmLoaderScript);
  }

  if (!self.ModuleFactory) {
    throw new Error('ModuleFactory not set.');
  }

  // Run asset-loader script here; must be run after wasm-loader script if we
  // are re-wrapping the existing MODULARIZE export.
  if (assetLoaderScript) {
    await runScript(assetLoaderScript);
    if (!self.ModuleFactory) {
      throw new Error('ModuleFactory not set.');
    }
  }

  // Until asset scripts work nicely with MODULARIZE, when we are given both
  // self.Module and a fileLocator, we manually merge them into self.Module and
  // use that. TODO(b/277980571): Remove this when asset scripts are fixed.
  if (self.Module && fileLocator) {
    const moduleFileLocator = self.Module as FileLocator;
    moduleFileLocator.locateFile = fileLocator.locateFile;
    if (fileLocator.mainScriptUrlOrBlob) {
      moduleFileLocator.mainScriptUrlOrBlob = fileLocator.mainScriptUrlOrBlob;
    }
  }
  // TODO(mrschmidt): Ensure that fileLocator is passed in by all users
  // and make it required
  const module =
      await self.ModuleFactory(self.Module as FileLocator || fileLocator);
  // Don't reuse factory or module seed
  self.ModuleFactory = self.Module = undefined;
  return new constructorFcn(module, glCanvas);
};

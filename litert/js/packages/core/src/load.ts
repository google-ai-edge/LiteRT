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

import {
  createWasmLib,
  type FileLocator,
  type WasmModuleFactory,
} from '@litertjs/wasm-utils';

import {LiteRt} from './litert_web';
import {appendPathSegment, pathToString, UrlPath} from './url_path_utils';
import {supportsFeature, throwIfFeatureNotSupported} from './wasm_feature_detect';

const WASM_JS_FILE_NAME = 'litert_wasm_internal.js';
const WASM_JS_COMPAT_FILE_NAME = 'litert_wasm_compat_internal.js';
const WASM_JS_THREADED_FILE_NAME = 'litert_wasm_threaded_internal.js';
const WASM_JS_JSPI_FILE_NAME = 'litert_wasm_jspi_internal.js';
const WASM_MJS_FILE_NAME = 'litert_wasm_internal.mjs';
const WASM_MJS_COMPAT_FILE_NAME = 'litert_wasm_compat_internal.mjs';
const WASM_MJS_THREADED_FILE_NAME = 'litert_wasm_threaded_internal.mjs';
const WASM_MJS_JSPI_FILE_NAME = 'litert_wasm_jspi_internal.mjs';

export type WasmModuleSource = UrlPath|WasmModuleFactory;
export type WasmLoaderType = 'script'|'module';

/**
 * Options for loading LiteRT's Wasm module.
 *
 * @property threads Whether to load the threaded version of the Wasm module.
 *     Defaults to false. Unused when specifying a .js file directly instead of
 *     a directory containing the Wasm files.
 * @property jspi Whether to load the JSPI version of the Wasm module. Defaults
 *     to false. Unused when specifying a .js file directly instead of a
 *     directory containing the Wasm files.
 * @property wasmLoaderType Whether to load Emscripten's generated glue as a
 *     classic script or as an ES module. Defaults to 'script'. When set to
 *     'module' and a directory path is provided, LiteRT.js selects the
 *     corresponding `.mjs` file.
 * @property fileLocator File locator overrides passed to Emscripten's module
 *     factory.
 **/
export interface LoadOptions {
  threads?: boolean;
  jspi?: boolean;
  wasmLoaderType?: WasmLoaderType;
  fileLocator?: FileLocator;
}

/**
 * Load the LiteRt library with WASM from the given URL. Does not set the
 * global LiteRT instance.
 */
export async function load(
    source: WasmModuleSource, options?: LoadOptions): Promise<LiteRt> {
  if (typeof source === 'function') {
    await validateLoadOptions(
        options, /* isFullFilePath= */ true,
        'the provided Wasm module factory');
    return createWasmLib(LiteRt, {
      fileLocator: options?.fileLocator,
      moduleFactory: source,
    });
  }

  const pathString = pathToString(source);
  const isModule = options?.wasmLoaderType === 'module' ||
      pathString.endsWith('.mjs');
  const isFullFilePath = pathString.endsWith('.wasm') ||
      pathString.endsWith('.js') || pathString.endsWith('.mjs');

  const relaxedSimd =
      await validateLoadOptions(options, isFullFilePath, pathString);
  let fileName = isModule ? WASM_MJS_COMPAT_FILE_NAME : WASM_JS_COMPAT_FILE_NAME;
  if (relaxedSimd) {
    if (options?.threads) {
      fileName =
          isModule ? WASM_MJS_THREADED_FILE_NAME : WASM_JS_THREADED_FILE_NAME;
    } else if (options?.jspi) {
      fileName = isModule ? WASM_MJS_JSPI_FILE_NAME : WASM_JS_JSPI_FILE_NAME;
    } else {
      fileName = isModule ? WASM_MJS_FILE_NAME : WASM_JS_FILE_NAME;
    }
  }

  let jsFilePath = source;
  if (pathString.endsWith('.wasm')) {
    throw new Error(
        'Please load the `.js` file corresponding to the `.wasm` file, or ' +
        'load the directory containing it.');
  } else if (!pathString.endsWith('.js') && !pathString.endsWith('.mjs')) {
    jsFilePath = appendPathSegment(source, fileName);
  }

  if (isModule) {
    const wasmModuleFactory = await importWasmModuleFactory(jsFilePath);
    return createWasmLib(LiteRt, {
      fileLocator: options?.fileLocator,
      moduleFactory: wasmModuleFactory,
    });
  }

  return createWasmLib(LiteRt, jsFilePath, null, null, options?.fileLocator);
}

async function validateLoadOptions(
    options: LoadOptions|undefined, isFullFilePath: boolean,
    pathString: string): Promise<boolean> {
  const relaxedSimd = await supportsFeature('relaxedSimd');
  if (options?.threads) {
    if (options?.jspi) {
      throw new Error(
          'The `threads` and `jspi` options are mutually exclusive.');
    }
    if (isFullFilePath) {
      console.warn(
          `The \`threads\` option was specified, but the wasm path ${
              pathString} is a full ` +
          `file path. Whether threads are available or not will depend on the ` +
          `loaded file. To allow LiteRT.js to load the threaded wasm file, ` +
          `use a directory path instead of a full file path.`);
    }
    if (!relaxedSimd) {
      throw new Error(
          'Threads are only supported with relaxed SIMD, and the current ' +
          'browser does not support relaxed SIMD.');
    }
    await throwIfFeatureNotSupported('threads');
  }

  if (options?.jspi) {
    if (isFullFilePath) {
      console.warn(
          `The \`jspi\` option was specified, but the wasm path ${
              pathString} is a full ` +
          `file path. Whether JSPI is available or not will depend on the ` +
          `loaded file. To allow LiteRT.js to load the JSPI wasm file, ` +
          `use a directory path instead of a full file path.`);
    }
    await throwIfFeatureNotSupported('jspi');
  }
  return relaxedSimd;
}

async function importWasmModuleFactory(
    jsFilePath: UrlPath): Promise<WasmModuleFactory> {
  const moduleUrl = resolveModuleImportUrl(jsFilePath);
  const wasmModule = await import(/* @vite-ignore */ moduleUrl);
  if (typeof wasmModule.default !== 'function') {
    throw new Error(
        `LiteRT Wasm ES module ${moduleUrl} must have a ` +
        'default export module factory.');
  }
  return wasmModule.default as WasmModuleFactory;
}

function resolveModuleImportUrl(jsFilePath: UrlPath): string {
  const pathString = pathToString(jsFilePath);
  const baseUrl = getResourceBaseUrl();
  if (!baseUrl) return pathString;

  try {
    return new URL(pathString, baseUrl).href;
  } catch {
    return pathString;
  }
}

function getResourceBaseUrl(): string|undefined {
  if (typeof document !== 'undefined') {
    return document.baseURI;
  }
  if (typeof location !== 'undefined') {
    return location.href;
  }
  return undefined;
}

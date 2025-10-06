// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_VENDORS_C_LITERT_COMPILER_PLUGIN_H_
#define ODML_LITERT_LITERT_VENDORS_C_LITERT_COMPILER_PLUGIN_H_

#include <stddef.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LiteRtCompilerPlugin);

// Artifact produced from compiling a selected partition of ops.
LITERT_DEFINE_HANDLE(LiteRtCompiledResult);

//
// Plugin
//

LITERT_CAPI_EXPORT LiteRtStatus
LiteRtGetCompilerPluginVersion(LiteRtApiVersion* api_version);

// Name associated with the manufacturer this plugin relates to (e.g,
// GoogleTensor, Qualcomm).
LITERT_CAPI_EXPORT const char* LiteRtGetCompilerPluginSocManufacturer();

// Initialize a compiler plugin with options provided by ther caller. The caller
// retains ownership of `env` and `options` and guarantees pointers are valid
// while the plugin is alive. These are read-only (TODO: update api for const
// correctness). It is OK for these to be null, in which case the plugin should
// use default values.
LITERT_CAPI_EXPORT LiteRtStatus
LiteRtCreateCompilerPlugin(LiteRtCompilerPlugin* compiler_plugin,
                           LiteRtEnvironmentOptions env, LiteRtOptions options);

LITERT_CAPI_EXPORT void LiteRtDestroyCompilerPlugin(
    LiteRtCompilerPlugin compiler_plugin);

// Return the HW supported by this plugin (e.g., GPU, NPU)
LITERT_CAPI_EXPORT LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators* supported_hardware);

// Number of SoC models supported by this plugin.
LITERT_CAPI_EXPORT LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex* num_supported_soc_models);

// Gets the name of the SoC model at the given index. The memory
// associated with the returned name is owned by the plugin.
LITERT_CAPI_EXPORT LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name);

// Select desired ops for compilation. This will only be called once
// per subgraph, plugins should select all supportable ops.
LITERT_CAPI_EXPORT LiteRtStatus LiteRtCompilerPluginPartition(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtSubgraph subgraph, LiteRtOpList selected_ops);

// Prepare result to pass to the runtime for given model containing partitioned
// subgraphs. Optionally, handles a SoC model (parameter `soc_model` can be NULL
// to specify a default SoC model).
LITERT_CAPI_EXPORT LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result);
//
// Compiled Partition
//

LITERT_CAPI_EXPORT void LiteRtDestroyCompiledResult(
    LiteRtCompiledResult result);

// Get the buffer for the compiled byte code for the given index.
LITERT_CAPI_EXPORT LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size);

// The number of individual byte code modules.
LITERT_CAPI_EXPORT LiteRtStatus LiteRtCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_byte_code);

// Get per-op info related to a particular compiled partition as well as the
// index of the respective byte code buffer.
LITERT_CAPI_EXPORT LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size,
    LiteRtParamIndex* byte_code_idx);

// Get the number of calls that will be made to the HAL for this graph.
// This should equal the number of partitions given for compilation which
// is equal to the number of custom ops in the final model.
LITERT_CAPI_EXPORT LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls);

// Allow the compiler plugin to registers all the graph transformations
// required. This function populates the provided arrays with pattern functions
// and their corresponding names. Registered patterns will be applied to the
// graph before partition and compilation.
//
// Experimental: Unstable ABI, function signature is subject to change.
LiteRtStatus LiteRtCompilerPluginRegisterAllTransformations(
    LiteRtCompilerPlugin compiler_plugin, LiteRtPatternFn** pattern_fns,
    const char*** transformation_names, LiteRtParamIndex* num_patterns);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_VENDORS_C_LITERT_COMPILER_PLUGIN_H_

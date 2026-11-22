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

#ifndef ODML_LITERT_LITERT_VENDORS_C_LITERT_COMPILER_PLUGIN_API_H_
#define ODML_LITERT_LITERT_VENDORS_C_LITERT_COMPILER_PLUGIN_API_H_

#include <assert.h>
#include <stddef.h>

#include "litert/c/internal/litert_abi_header.h"
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/vendors/c/litert_compiler_plugin.h"  // IWYU pragma: export

// Wrapper for dynamically loaded LiteRtCompilerPlugin library. See
// "litert_compiler_plugin.h".

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Current ABI version for LiteRT Compiler Plugin interface tables.
#define LITERT_COMPILER_PLUGIN_ABI_VERSION_MAJOR 1
#define LITERT_COMPILER_PLUGIN_ABI_VERSION_MINOR 1
#define LITERT_COMPILER_PLUGIN_ABI_VERSION_PATCH 0

//
// Api Interface
//

typedef LiteRtStatus (*LiteRtGetCompilerPluginVersionT)(LiteRtApiVersion*);

typedef const char* (*LiteRtGetCompilerPluginSocManufacturerT)();

typedef LiteRtStatus (*LiteRtCreateCompilerPluginT)(
    const LiteRtCompilerContext*, LiteRtCompilerPlugin*,
    LiteRtEnvironmentOptions, LiteRtOptions);

typedef void (*LiteRtDestroyCompilerPluginT)(LiteRtCompilerPlugin);

typedef LiteRtStatus (*LiteRtGetCompilerPluginSupportedHardwareT)(
    LiteRtCompilerPlugin, LiteRtHwAccelerators*);

typedef LiteRtStatus (*LiteRtGetNumCompilerPluginSupportedSocModelsT)(
    LiteRtCompilerPlugin, LiteRtParamIndex*);

typedef LiteRtStatus (*LiteRtGetCompilerPluginSupportedSocModelT)(
    LiteRtCompilerPlugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name);

typedef LiteRtStatus (*LiteRtGetCompilerPluginSDKVersionT)(LiteRtCompilerPlugin,
                                                           const char**);

typedef LiteRtStatus (*LiteRtCompilerPluginPartitionT)(
    LiteRtCompilerPlugin, const char* soc_model, LiteRtSubgraph subgraph,
    LiteRtOpList selected_ops);

typedef LiteRtStatus (*LiteRtCompilerPluginCompileT)(
    LiteRtCompilerPlugin, const char* soc_model, LiteRtModel partitions,
    LiteRtCompiledResult* compiled_result);

typedef void (*LiteRtDestroyCompiledResultT)(LiteRtCompiledResult);

typedef LiteRtStatus (*LiteRtGetCompiledResultByteCodeT)(
    LiteRtCompiledResult, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size);

typedef LiteRtStatus (*LiteRtGetCompiledResultHandleT)(
    LiteRtCompiledResult, LiteRtParamIndex call_idx,
    LiteRtJitExecutable* handle);

typedef LiteRtStatus (*LiteRtCompiledResultNumByteCodeModulesT)(
    LiteRtCompiledResult, LiteRtParamIndex* num_byte_code);

typedef LiteRtStatus (*LiteRtGetCompiledResultCallInfoT)(
    LiteRtCompiledResult, LiteRtParamIndex call_idx, const void** call_info,
    size_t* call_info_size, LiteRtParamIndex* byte_code_idx);

typedef LiteRtStatus (*LiteRtGetNumCompiledResultCallsT)(
    LiteRtCompiledResult, LiteRtParamIndex* num_calls);

typedef LiteRtStatus (*LiteRtCompilerPluginRegisterAllTransformationsT)(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtTransformation** transformations, LiteRtParamIndex* num_patterns);

typedef LiteRtStatus (*LiteRtCompilerPluginCheckCompilerCompatibilityT)(
    LiteRtApiVersion api_version, LiteRtCompilerPlugin compiler_plugin,
    LiteRtEnvironmentOptions env, LiteRtOptions options,
    const char* soc_model_name);

/// Selects ops in `model`'s subgraph at `subgraph_index` for compilation.
///
/// Same contract as `LiteRtCompilerPluginPartitionT`, except that the plugin
/// receives the owning `model` and `subgraph_index` so it can follow op
/// subgraph references (via `LiteRtCompilerContext::get_num_op_subgraph_refs`,
/// `get_op_subgraph_ref`, and `get_model_subgraph`) to inspect control-flow
/// bodies, region computations, and composite decompositions before selecting.
///
/// The plugin must treat `model` as read-only, must select only ops belonging
/// to `subgraph_index`, and must tolerate being called more than once on the
/// same subgraph (for example, after an unselected composite is inlined).
typedef LiteRtStatus (*LiteRtCompilerPluginPartitionV2T)(
    LiteRtCompilerPlugin, const char* soc_model, LiteRtModel model,
    LiteRtParamIndex subgraph_index, LiteRtOpList selected_ops);

/// Compiles `partitions`, whose subgraphs are laid out as
/// `[entry subgraphs..., helper subgraphs...]` with `num_entry_subgraphs`
/// leading entries.
///
/// Helpers (indices `[num_entry_subgraphs, num_subgraphs)`) are private
/// subgraphs referenced transitively by ops in the entries (region bodies,
/// control-flow branches, composite decompositions). The plugin must emit
/// exactly one call per entry subgraph (call `i` for entry `i`) and none for
/// helpers.
typedef LiteRtStatus (*LiteRtCompilerPluginCompileV2T)(
    LiteRtCompilerPlugin, const char* soc_model, LiteRtModel partitions,
    LiteRtParamIndex num_entry_subgraphs,
    LiteRtCompiledResult* compiled_result);

/// Identifiers for the versioned compiler plugin interface tables queried via
/// LiteRtCompilerPluginQueryInterface.
///
/// @note This enum is shared across the runtime and compiler plugins, so
///     enumerator values must remain ABI stable.
typedef enum {
  kLiteRtCompilerPluginInterfaceBasic = 0,
} LiteRtCompilerPluginInterfaceId;

typedef struct LiteRtCompilerPluginInterface_V1 {
  LiteRtAbiHeader abi_header;

  LiteRtGetCompilerPluginVersionT get_compiler_plugin_version;
  LiteRtGetCompilerPluginSocManufacturerT get_compiler_plugin_soc_manufacturer;
  LiteRtCreateCompilerPluginT create_compiler_plugin;
  LiteRtDestroyCompilerPluginT destroy_compiler_plugin;

  LiteRtGetCompilerPluginSupportedHardwareT
      get_compiler_plugin_supported_hardware;
  LiteRtGetNumCompilerPluginSupportedSocModelsT
      get_num_compiler_plugin_supported_models;
  LiteRtGetCompilerPluginSupportedSocModelT
      get_compiler_plugin_supported_soc_model;

  LiteRtCompilerPluginPartitionT compiler_plugin_partition;
  LiteRtCompilerPluginCompileT compiler_plugin_compile;

  LiteRtDestroyCompiledResultT destroy_compiled_result;
  LiteRtGetCompiledResultByteCodeT get_compiled_result_byte_code;
  LiteRtCompiledResultNumByteCodeModulesT get_compiled_result_num_byte_code;
  LiteRtGetCompiledResultCallInfoT get_compiled_result_call_info;
  LiteRtGetNumCompiledResultCallsT get_num_compiled_result_calls;
  LiteRtCompilerPluginRegisterAllTransformationsT register_all_transformations;

  LiteRtGetCompilerPluginSDKVersionT get_compiler_plugin_sdk_version;
  LiteRtGetCompiledResultHandleT get_compiled_result_handle;
  LiteRtCompilerPluginCheckCompilerCompatibilityT check_compiler_compatibility;

  // Added in version 1.1.
  LiteRtCompilerPluginPartitionV2T compiler_plugin_partition_v2;
  LiteRtCompilerPluginCompileV2T compiler_plugin_compile_v2;
} LiteRtCompilerPluginInterface_V1;

static_assert(
    offsetof(LiteRtCompilerPluginInterface_V1, abi_header) == 0,
    "LiteRtCompilerPluginInterface_V1 abi_header must be at offset 0");
#if defined(__cplusplus) && defined(__SIZEOF_POINTER__) && \
    __SIZEOF_POINTER__ == 8
static_assert(offsetof(LiteRtCompilerPluginInterface_V1,
                       compiler_plugin_partition_v2) == 152,
              "LiteRtCompilerPluginInterface_V1 "
              "compiler_plugin_partition_v2 offset mismatch");
static_assert(offsetof(LiteRtCompilerPluginInterface_V1,
                       compiler_plugin_compile_v2) == 160,
              "LiteRtCompilerPluginInterface_V1 "
              "compiler_plugin_compile_v2 offset mismatch");
static_assert(sizeof(LiteRtCompilerPluginInterface_V1) == 168,
              "LiteRtCompilerPluginInterface_V1 size mismatch");
#endif  // __cplusplus

typedef LiteRtStatus (*LiteRtCompilerPluginQueryInterfaceT)(
    LiteRtCompilerPluginInterfaceId interface_id,
    LiteRtApiVersion litert_runtime_version, LiteRtInterface* out_interface);

LITERT_CAPI_EXPORT LiteRtStatus LiteRtCompilerPluginQueryInterface(
    LiteRtCompilerPluginInterfaceId interface_id,
    LiteRtApiVersion litert_runtime_version, LiteRtInterface* out_interface);

#ifdef __cplusplus
}
#endif  // __cplusplus

#ifdef __cplusplus
#include "absl/strings/string_view.h"  // from @com_google_absl

static constexpr absl::string_view kLiteRtCompilerPluginQueryInterface =
    "LiteRtCompilerPluginQueryInterface";

#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_VENDORS_C_LITERT_COMPILER_PLUGIN_API_H_

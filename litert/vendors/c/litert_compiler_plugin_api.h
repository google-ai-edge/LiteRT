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
#include <stdint.h>

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

/// Optional features a compiler plugin can report through
/// `get_compiler_plugin_capabilities`.
///
/// @note Values are ABI stable bit flags.
typedef enum {
  kLiteRtCompilerPluginCapabilityNone = 0,
  /// `compiler_plugin_compile` accepts a partitions model whose subgraphs
  /// [num_entry_subgraphs, num_subgraphs) are private helper subgraphs (region
  /// bodies, control-flow branches, composite decompositions) referenced by
  /// ops in the entry subgraphs, and emits exactly one call per entry
  /// subgraph. The entry count is available through
  /// `LiteRtCompilerContext::get_num_entry_subgraphs`. Without this capability
  /// every subgraph of the partitions model is an entry subgraph.
  kLiteRtCompilerPluginCapabilityRegionSubgraphs = 1 << 0,
} LiteRtCompilerPluginCapability;

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

/// Reports a bitwise OR of `LiteRtCompilerPluginCapability` values.
typedef LiteRtStatus (*LiteRtGetCompilerPluginCapabilitiesT)(
    LiteRtCompilerPlugin compiler_plugin, uint64_t* capabilities);

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
  LiteRtGetCompilerPluginCapabilitiesT get_compiler_plugin_capabilities;
} LiteRtCompilerPluginInterface_V1;

static_assert(
    offsetof(LiteRtCompilerPluginInterface_V1, abi_header) == 0,
    "LiteRtCompilerPluginInterface_V1 abi_header must be at offset 0");
#if defined(__cplusplus) && defined(__SIZEOF_POINTER__) && \
    __SIZEOF_POINTER__ == 8
static_assert(offsetof(LiteRtCompilerPluginInterface_V1,
                       get_compiler_plugin_capabilities) == 152,
              "LiteRtCompilerPluginInterface_V1 "
              "get_compiler_plugin_capabilities offset mismatch");
static_assert(sizeof(LiteRtCompilerPluginInterface_V1) == 160,
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

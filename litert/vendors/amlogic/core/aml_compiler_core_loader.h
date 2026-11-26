// Copyright 2026 Amlogic, Inc.
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
 * @file    aml_compiler_core_loader.h
 * @module  amlogic_compiler_plugin
 * @brief   Run-time function table for the dlopen'ed aml_compiler_core SDK.
 */

#ifndef AML_COMPILER_CORE_LOADER_H_
#define AML_COMPILER_CORE_LOADER_H_

#include "aml_npu/compiler_core/compiler_core_context.h" // from @aml_npu_nn_runtime

namespace aml_compiler_core {

/** @brief Library file name dlopen'ed by Funcs(). */
inline constexpr char kSoName[] = "libaml_compiler_core.so";

/**
 * @brief Entry points of libaml_compiler_core.so, resolved with dlsym.
 *
 * SDK headers stay compile-time only (types and enums); no SDK symbol is
 * linked, so processes that never compile never map the SDK .so or its
 * libcompiler_adla.so dependency (a DT_NEEDED of the core).
 */
struct FuncTable {
  AmlContext (*AmlContextCreate)(void);
  void (*AmlContextDestroy)(AmlContext ctx);
  AmlCompilerStatus (*AmlGraphCreate)(AmlContext ctx, AmlGraph* graph);
  AmlCompilerStatus (*AmlGraphSetTarget)(AmlGraph graph, const char* target);
  AmlCompilerStatus (*AmlGraphSetName)(AmlGraph graph, const char* name);
  AmlCompilerStatus (*AmlGraphAddInput)(AmlGraph graph, AmlTensor tensor);
  AmlCompilerStatus (*AmlGraphAddOutput)(AmlGraph graph, AmlTensor tensor);
  AmlCompilerStatus (*AmlTensorCreate)(AmlGraph graph, AmlTensor* tensor);
  AmlCompilerStatus (*AmlTensorSetName)(AmlTensor tensor, const char* name);
  AmlCompilerStatus (*AmlTensorSetDtype)(AmlTensor tensor, int dtype);
  AmlCompilerStatus (*AmlTensorSetIndex)(AmlTensor tensor, int tensor_index);
  AmlCompilerStatus (*AmlTensorSetDims)(AmlTensor tensor, const int* dims,
                                        uint32_t dims_count);
  AmlCompilerStatus (*AmlTensorSetQuant)(AmlTensor tensor, const int* zp,
                                         uint32_t zp_count,
                                         const float* scale,
                                         uint32_t scale_count);
  AmlCompilerStatus (*AmlTensorSetConstData)(AmlTensor tensor,
                                             const void* data,
                                             size_t data_size);
  AmlCompilerStatus (*AmlNodeCreate)(AmlGraph graph, int op_type,
                                     int node_index, AmlNode* node);
  AmlCompilerStatus (*AmlNodeAddInput)(AmlNode node, AmlTensor tensor);
  AmlCompilerStatus (*AmlNodeAddOutput)(AmlNode node, AmlTensor tensor);
  AmlCompilerStatus (*AmlNodeSetParamI)(AmlNode node, const char* key,
                                        int value);
  AmlCompilerStatus (*AmlNodeSetParamF)(AmlNode node, const char* key,
                                        float value);
  void (*AmlCompilerCoreOptionsInitDefaults)(AmlCompilerCoreOptions* options);
  void (*AmlCompilerExternalOptionsInitDefaults)(
      AmlCompilerExternalOptions* external_options);
  AmlCompilerStatus (*AmlContextCompile)(
      AmlContext ctx, const char* model_name, const char* model_path,
      const AmlCompilerCoreOptions* options,
      AmlCompilerCoreCompiledArtifact* compiled_artifact);
  void (*AmlCompilerCoreDestroyCompiledArtifact)(
      AmlCompilerCoreCompiledArtifact compiled_artifact);
  AmlCompilerStatus (*AmlCompilerCoreGetCompiledArtifactAdlaBinary)(
      AmlCompilerCoreCompiledArtifact compiled_artifact,
      const void** adla_bin, size_t* adla_bin_size);
  AmlCompilerStatus (*AmlCompilerCoreGetCompiledArtifactAdlaPath)(
      AmlCompilerCoreCompiledArtifact compiled_artifact,
      const char** adla_path);
  AmlCompilerStatus (*AmlCompilerCoreGetCompiledArtifactFingerprint)(
      AmlCompilerCoreCompiledArtifact compiled_artifact,
      const char** fingerprint);
  AmlCompilerStatus (*AmlCompilerCoreGetCompiledArtifactCacheHit)(
      AmlCompilerCoreCompiledArtifact compiled_artifact, int* cache_hit);
  // Optional: LiteRtGetCompilerPluginSDKVersion falls back when missing.
  const AmlCompilerCoreVersion* (*AmlCompilerCoreGetVersion)(void);
};

/**
 * @brief SDK entry points, or nullptr when libaml_compiler_core.so is
 *        unavailable.
 *
 * With allow_dlopen=false the lookup succeeds only when the library is
 * already mapped (RTLD_NOLOAD), so a version query never forces the SDK
 * into the process. The library is never dlclosed: libcompiler_adla.so uses
 * OpenMP worker threads and is not safe to unmap (compiler_core_api.h). A
 * failed real load is sticky so later callers fail fast without re-logging.
 *
 * @param allow_dlopen Whether a not-yet-mapped library may be dlopen'ed.
 */
const FuncTable* Funcs(bool allow_dlopen);

}  // namespace aml_compiler_core

#endif  // AML_COMPILER_CORE_LOADER_H_

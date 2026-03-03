// Copyright 2025 Google LLC.
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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_MEDIATEK_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_MEDIATEK_OPTIONS_H_

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct LrtMediatekOptions LrtMediatekOptions;

// Creates a mediatek options object.
// The caller is responsible for freeing the returned options using
// `LrtDestroyMediatekOptions`.
LiteRtStatus LrtCreateMediatekOptions(LrtMediatekOptions** options);

// Creates a mediatek options object from a TOML payload.
LiteRtStatus LrtCreateMediatekOptionsFromToml(const char* toml_payload,
                                              LrtMediatekOptions** options);

// Destroys a mediatek options object.
void LrtDestroyMediatekOptions(LrtMediatekOptions* options);

// Serializes mediatek options and returns the components needed to create
// opaque options. The caller is responsible for passing these to
// `LiteRtCreateOpaqueOptions` and freeing the returned payload using
// `payload_deleter`.
LiteRtStatus LrtGetOpaqueMediatekOptionsData(const LrtMediatekOptions* options,
                                             const char** identifier,
                                             void** payload,
                                             void (**payload_deleter)(void*));

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

// sdk_version_type ----------------------------------------------------------
typedef enum LiteRtMediatekOptionsNeronSDKVersionType {
  kLiteRtMediatekOptionsNeronSDKVersionTypeVersion7 = 0,
  kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8 = 1,
  kLiteRtMediatekOptionsNeronSDKVersionTypeVersion9 = 2,
} LiteRtMediatekOptionsNeronSDKVersion;

LiteRtStatus LrtSetMediatekOptionsNeronSDKVersionType(
    LrtMediatekOptions* options,
    enum LiteRtMediatekOptionsNeronSDKVersionType sdk_version_type);

LiteRtStatus LrtGetMediatekOptionsNeronSDKVersionType(
    const LrtMediatekOptions* options,
    enum LiteRtMediatekOptionsNeronSDKVersionType* sdk_version_type);

// gemma_compiler_optimizations ----------------------------------------------
LiteRtStatus LrtSetMediatekOptionsGemmaCompilerOptimizations(
    LrtMediatekOptions* options, bool gemma_compiler_optimizations);

LiteRtStatus LrtGetMediatekOptionsGemmaCompilerOptimizations(
    const LrtMediatekOptions* options, bool* gemma_compiler_optimizations);

// neuron_adapter_peformance_mode --------------------------------------------

// Configures MTK devices to optimize for performance or power efficiency.
// See NeuronAdapterPreferenceCode in mtk_sdk. By default, it
// will use  Sustained Speed answer.

typedef enum LiteRtMediatekNeuronAdapterPerformanceMode {
  /* Prefer executing in a way that minimizes battery drain. */
  kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferLowPower = 0,
  /* Prefer executing as fast as possible. (more power consumption)*/
  kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferFastSingleAnswer = 1,
  /* Prefer maximizing the throughput of successive frames */
  kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferSustainedSpeed = 2,
  /* Prefer executing with turbo boost. (most power consumption) */
  kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferTurboBoost = 3,
} LiteRtMediatekNeuronAdapterPerformanceMode;

LiteRtStatus LrtSetMediatekOptionsPerformanceMode(
    LrtMediatekOptions* options,
    LiteRtMediatekNeuronAdapterPerformanceMode performance_mode);

LiteRtStatus LrtGetMediatekOptionsPerformanceMode(
    const LrtMediatekOptions* options,
    LiteRtMediatekNeuronAdapterPerformanceMode* performance_mode);

// l1_cache_optimizations ----------------------------------------------------
LiteRtStatus LrtSetMediatekOptionsL1CacheOptimizations(
    LrtMediatekOptions* options, bool l1_cache_optimizations);

LiteRtStatus LrtGetMediatekOptionsL1CacheOptimizations(
    const LrtMediatekOptions* options, bool* l1_cache_optimizations);

// neuron_optimization_hints -------------------------------------------------

// Configures MTK devices with optimization hints..
// By default, it will use NEURON_OPTIMIZATION_NORMAL.

typedef enum LiteRtMediatekNeuronAdapterOptimizationHint {
  /* Normal optimization. Default Value */
  kLiteRtMediatekNeuronAdapterOptimizationHintNormal = 0,
  /* Reduce latency by utilizing as many APU cores as possible.*/
  kLiteRtMediatekNeuronAdapterOptimizationHintLowLatency = 1 << 0,
  /* Reducing DRAM access as more as possible.*/
  kLiteRtMediatekNeuronAdapterOptimizationHintDeepFusion = 1 << 1,
  /*Reduce latency by using as many APU cores as possible in batch-dimension.
   * (For models with batch > 1)*/
  kLiteRtMediatekNeuronAdapterOptimizationHintBatchProcessing = 1 << 2,
} LiteRtMediatekNeuronAdapterOptimizationHint;

LiteRtStatus LrtSetMediatekOptionsOptimizationHint(
    LrtMediatekOptions* options,
    LiteRtMediatekNeuronAdapterOptimizationHint optimization_hint);

LiteRtStatus LrtGetMediatekOptionsOptimizationHint(
    const LrtMediatekOptions* options,
    LiteRtMediatekNeuronAdapterOptimizationHint* optimization_hint);

// disable_dla_dir_removal ---------------------------------------------------
LiteRtStatus LrtSetMediatekOptionsDisableDlaDirRemoval(
    LrtMediatekOptions* options, bool disable_dla_dir_removal);

LiteRtStatus LrtGetMediatekOptionsDisableDlaDirRemoval(
    const LrtMediatekOptions* options, bool* disable_dla_dir_removal);

// mediatek_dla_dir ----------------------------------------------------------

LiteRtStatus LrtSetMediatekOptionsMediatekDlaDir(LrtMediatekOptions* options,
                                                 const char* mediatek_dla_dir);

LiteRtStatus LrtGetMediatekOptionsMediatekDlaDir(
    const LrtMediatekOptions* options, const char** mediatek_dla_dir);

// AoT compilation options --------------------------------------------
LiteRtStatus LrtSetMediatekOptionsAotCompilationOptions(
    LrtMediatekOptions* options, const char* aot_compilation_options);

LiteRtStatus LrtGetMediatekOptionsAotCompilationOptions(
    const LrtMediatekOptions* options, const char** aot_compilation_options);

#ifdef __cplusplus

}  // extern "C"

#endif  // __cplusplus
#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_MEDIATEK_OPTIONS_H_

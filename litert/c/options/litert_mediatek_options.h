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
#include "litert/c/litert_opaque_options.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Create a mediatek options object that is type erased. The actual option
// data can be accessed from the payload.
LiteRtStatus LiteRtMediatekOptionsCreate(LiteRtOpaqueOptions* options);
LITERT_DEFINE_HANDLE(LiteRtMediatekOptions);

// The a string identifier that discriminates mediatek options within
// type erased options.
const char* LiteRtMediatekOptionsGetIdentifier();

// Attempt to retrieve mediatek options from the opaque options. Fails
// unless the opaque options are of another type.
LiteRtStatus LiteRtMediatekOptionsGet(LiteRtOpaqueOptions options,
                                      LiteRtMediatekOptions* options_data);

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

// sdk_version_type ----------------------------------------------------------
typedef enum LiteRtMediatekOptionsNeronSDKVersionType {
  kLiteRtMediatekOptionsNeronSDKVersionTypeVersion7 = 0,
  kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8 = 1,
  kLiteRtMediatekOptionsNeronSDKVersionTypeVersion9 = 2,
} LiteRtMediatekOptionsNeronSDKVersion;

LiteRtStatus LiteRtMediatekOptionsSetNeronSDKVersionType(
    LiteRtMediatekOptions options,
    enum LiteRtMediatekOptionsNeronSDKVersionType sdk_version_type);

LiteRtStatus LiteRtMediatekOptionsGetNeronSDKVersionType(
    LiteRtMediatekOptions options,
    enum LiteRtMediatekOptionsNeronSDKVersionType* sdk_version_type);

// gemma_compiler_optimizations ----------------------------------------------
LiteRtStatus LiteRtMediatekOptionsSetGemmaCompilerOptimizations(
    LiteRtMediatekOptions options, bool gemma_compiler_optimizations);

LiteRtStatus LiteRtMediatekOptionsGetGemmaCompilerOptimizations(
    LiteRtMediatekOptions options, bool* gemma_compiler_optimizations);

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

LiteRtStatus LiteRtMediatekOptionsSetPerformanceMode(
    LiteRtMediatekOptions options,
    LiteRtMediatekNeuronAdapterPerformanceMode performance_mode);

LiteRtStatus LiteRtMediatekOptionsGetPerformanceMode(
    LiteRtMediatekOptions options,
    LiteRtMediatekNeuronAdapterPerformanceMode* performance_mode);

// l1_cache_optimizations ----------------------------------------------------
LiteRtStatus LiteRtMediatekOptionsSetL1CacheOptimizations(
    LiteRtMediatekOptions options, bool l1_cache_optimizations);

LiteRtStatus LiteRtMediatekOptionsGetL1CacheOptimizations(
    LiteRtMediatekOptions options, bool* l1_cache_optimizations);

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

LiteRtStatus LiteRtMediatekOptionsSetOptimizationHint(
    LiteRtMediatekOptions options,
    LiteRtMediatekNeuronAdapterOptimizationHint optimization_hint);

LiteRtStatus LiteRtMediatekOptionsGetOptimizationHint(
    LiteRtMediatekOptions options,
    LiteRtMediatekNeuronAdapterOptimizationHint* optimization_hint);

// disable_dla_dir_removal ---------------------------------------------------
LiteRtStatus LiteRtMediatekOptionsSetDisableDlaDirRemoval(
    LiteRtMediatekOptions options, bool disable_dla_dir_removal);

LiteRtStatus LiteRtMediatekOptionsGetDisableDlaDirRemoval(
    LiteRtMediatekOptions options, bool* disable_dla_dir_removal);

// mediatek_dla_dir ----------------------------------------------------------

LiteRtStatus LiteRtMediatekOptionsSetMediatekDlaDir(
    LiteRtMediatekOptions options, const char* mediatek_dla_dir);

LiteRtStatus LiteRtMediatekOptionsGetMediatekDlaDir(
    LiteRtMediatekOptions options, const char** mediatek_dla_dir);

#ifdef __cplusplus

}  // extern "C"

#endif  // __cplusplus
#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_MEDIATEK_OPTIONS_H_

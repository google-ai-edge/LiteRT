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

#include "litert/vendors/mediatek/compiler/compile_model.h"

#include <cstdint>
#include <optional>
#include <string>

#include "neuron/api/NeuronAdapter.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_mediatek_options.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

constexpr int kDecodePartitionIndex = 0;
constexpr int kPrefillPartitionIndex = 1;

Expected<NeuronCompilationPtr> CompileModel(
    const NeuronAdapterApi& neuron_adapter_api, NeuronModel* model,
    std::optional<std::string> soc_model,
    ::litert::Expected<litert::mediatek::MediatekOptions>& mediatek_opts,
    const int subgraph_index, bool get_supported_mode) {
  // LITERT_USE_JIT is automatically defined based on the build target.
  // It is defined on devices with MediaTek hardwares.
#if LITERT_USE_JIT
  if (soc_model) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "JIT compilation for a specific SoC is not supported");
  }
#endif

  // Per MediaTek recommendation, Compilation_create,
  // Compilation_createWithOptions, and Compilation_setOptimizationString
  // should be used as follow:
  // - AOT Compilation: Compilation_createWithOptions only
  // - JIT Compilation: Compilation_create and Compilation_setOptimizationString
  // The code below takes care of those conditions.

  // NOLINTBEGIN
  auto compile_options =
#if LITERT_USE_JIT
      std::string(neuron_adapter_api.JitCompileOptions());
#else
      std::string(neuron_adapter_api.AotCompileOptions());
#endif
  // NOLINTEND

  if (mediatek_opts->GetEnableGemmaCompilerOptimizations()) {
    if (subgraph_index == kDecodePartitionIndex) {
      compile_options = " --option-bundle=gemma-decode-accuracy";
    }

    if (subgraph_index == kPrefillPartitionIndex) {
      compile_options = " --option-bundle=gemma-prefill-accuracy";
    }
  }

  // This is needed in order to support FP32 acativations since TFLite doesn't
  // contain support for FP16 activations currently.
  if (auto status = neuron_adapter_api.api().relax_fp32_to_fp16(model, true);
      status != NEURON_NO_ERROR) {
    LITERT_LOG(
        LITERT_INFO,
        "NeuronModel_relaxComputationFloat32toFloat16 failed with error %d",
        status);
    return Error(kLiteRtStatusErrorRuntimeFailure, "Failed to relaxFp32ToFp16");
  }

  auto compilation =
#if LITERT_USE_JIT
      neuron_adapter_api.CreateCompilation(model);
#else
      neuron_adapter_api.CreateCompilation(model, compile_options);
#endif
  if (!compilation) {
    return compilation.Error();
  }

  if (neuron_adapter_api.api().compilation_set_priority(
          compilation->get(), NEURON_PRIORITY_HIGH) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to set compilation priority");
  }

  LITERT_LOG(LITERT_INFO,
             "NeuronCompilation_setPreference being set with value: %d",
             mediatek_opts->GetPerformanceMode());

  if (neuron_adapter_api.api().compilation_set_preference(
          compilation->get(), mediatek_opts->GetPerformanceMode()) !=
      NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to set compilation preference");
  }

  if (mediatek_opts->GetEnableL1CacheOptimizations()) {
    uint32_t apu_mem_size = 0;
    if (neuron_adapter_api.api().get_l1_memory_size_kb(&apu_mem_size) ==
            NEURON_NO_ERROR &&
        apu_mem_size > 0) {
      if (neuron_adapter_api.api().compilation_set_l1_memory_size_kb(
              compilation->get(), apu_mem_size) != NEURON_NO_ERROR) {
        LITERT_LOG(LITERT_INFO,
                   "NeuronCompilation_setL1MemorySizeKb failed with error %d",
                   neuron_adapter_api.api().compilation_set_l1_memory_size_kb(
                       compilation->get(), apu_mem_size));
      }
    }
  }

  if (auto status = neuron_adapter_api.api().compilation_set_optimization_hint(
          compilation->get(), mediatek_opts->GetOptimizationHint());
      status != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_INFO,
               "NeuronCompilation_setOptimizationHint failed with error %d",
               status);
    LITERT_LOG(LITERT_INFO,
               "NeuronCompilation_setOptimizationHint failed attempting to set "
               "optimization hint enum value to %d",
               mediatek_opts->GetOptimizationHint());
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to set optimization hint");
  }

  if (auto status =
          neuron_adapter_api.api().compilation_set_optimization_string(
              compilation->get(), compile_options.c_str());
      status != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_INFO,
               "NeuronCompilation_setOptimizationString failed with error %d",
               status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to set optimization string");
  }

  // NeuronCompilation_getSupportedOperations must be called before
  // calling NeuronCompilation_finish
  if (get_supported_mode) {
    return compilation;
  }

  if (auto status =
          neuron_adapter_api.api().compilation_finish(compilation->get());
      status != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_INFO, "NeuronCompilation_finish failed with error %d",
               status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to finish compilation");
  }

  return compilation;
}

Expected<void> GetSupportedOperations(
    const NeuronAdapterApi& neuron_adapter_api, NeuronModel* model,
    std::optional<std::string> soc_model,
    ::litert::Expected<litert::mediatek::MediatekOptions>& mediatek_opts,
    const int subgraph_index, bool* support_flags, int num_ops) {
  auto compilation =
      CompileModel(neuron_adapter_api, model, soc_model, mediatek_opts,
                   subgraph_index, /*get_supported_mode*/ true);
  if (!compilation) {
    return compilation.Error();
  }
  if (auto status =
          neuron_adapter_api.api().compilation_get_supported_opertations(
              compilation->get(), num_ops, support_flags);
      status != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_INFO,
               "NeuronCompilation_getSupportedOperations failed with error %d",
               status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to get supported op of compilation");
  }
  return {};
}

}  // namespace litert::mediatek

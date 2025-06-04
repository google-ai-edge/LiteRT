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

#include <optional>
#include <string>

#include "neuron/api/NeuronAdapter.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_mediatek_options.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

Expected<NeuronCompilationPtr> CompileModel(
    const NeuronAdapterApi& neuron_adapter_api, NeuronModel* model,
    std::optional<std::string> soc_model,
    ::litert::Expected<litert::mediatek::MediatekOptions>& mediatek_opts) {
#if defined(__ANDROID__)
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
#if __ANDROID__
      std::string(neuron_adapter_api.JitCompileOptions());
#else
      std::string(neuron_adapter_api.AotCompileOptions());
#endif
  // NOLINTEND

  if (mediatek_opts->GetEnableGemmaCompilerOptimizations()) {
    compile_options = "--option-bundle=gemma";
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
#if __ANDROID__
      neuron_adapter_api.CreateCompilation(model);
#else
      neuron_adapter_api.CreateCompilation(model, compile_options);
#endif
  if (!compilation) {
    return compilation.Error();
  }

  if (neuron_adapter_api.api().compilation_set_priority(
          compilation.Value().get(), NEURON_PRIORITY_HIGH) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to set compilation priority");
  }

  LITERT_LOG(LITERT_INFO,
             "NeuronCompilation_setPreference being set with value: %d",
             mediatek_opts->GetPerformanceMode());

  if (neuron_adapter_api.api().compilation_set_preference(
          compilation.Value().get(), mediatek_opts->GetPerformanceMode()) !=
      NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to set compilation preference");
  }

  if (auto status =
          neuron_adapter_api.api().compilation_set_optimization_string(
              compilation.Value().get(), compile_options.c_str());
      status != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_INFO,
               "NeuronCompilation_setOptimizationString failed with error %d",
               status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to set optimization string");
  }

  if (auto status = neuron_adapter_api.api().compilation_finish(
          compilation.Value().get());
      status != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_INFO, "NeuronCompilation_finish failed with error %d",
               status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to finish compilation");
  }

  return compilation;
}

}  // namespace litert::mediatek

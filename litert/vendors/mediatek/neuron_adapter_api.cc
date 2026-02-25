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

#include "litert/vendors/mediatek/neuron_adapter_api.h"

#include <dlfcn.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "neuron/api/NeuronAdapter.h"
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_mediatek_options.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_mediatek_options.h"

#define LOAD_SYMB(S, H)                                                   \
  if (auto maybe_H = dlib_.LookupSymbol<void*>(#S); maybe_H.HasValue()) { \
    H = reinterpret_cast<decltype(&S)>(std::move(maybe_H.Value()));       \
  } else {                                                                \
    LITERT_LOG(LITERT_WARNING, "Failed to load symbol %s: %s", #S,        \
               dlib_.DlError());                                          \
    H = nullptr;                                                          \
  }

namespace litert {
namespace mediatek {

NeuronAdapterApi::NeuronAdapterApi() : api_(new Api) {}

litert::Expected<NeuronAdapterApi::Ptr> NeuronAdapterApi::Create(
    std::optional<std::string> shared_library_dir,
    ::litert::Expected<litert::mediatek::MediatekOptions>& options) {
  std::unique_ptr<NeuronAdapterApi> neuron_adapter_api(new NeuronAdapterApi);

  if (auto status = neuron_adapter_api->LoadSymbols(
          shared_library_dir, options->GetNeronSDKVersionType());
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to load NeuronAdapter shared library: %s",
               status.Error().Message().c_str());
    return status.Error();
  }

  // Read the user provided aot compilation options, if any, otherwise use the
  // default options.
  auto aot_compilation_options = options->GetAotCompilationOptions();
  if (!aot_compilation_options.empty()) {
    neuron_adapter_api->aot_compilation_options_ =
        std::string(aot_compilation_options);
  } else {
    neuron_adapter_api->aot_compilation_options_ =
        kDefaultAotCompilationOptions;
  }

  LITERT_RETURN_IF_ERROR(neuron_adapter_api->GetNeuronVersion());

  return neuron_adapter_api;
}

litert::Expected<void> NeuronAdapterApi::LoadSymbols(
    std::optional<std::string> shared_library_dir,
    LiteRtMediatekOptionsNeronSDKVersionType sdk_version) {
  constexpr auto kLibNeuronAdapterLib = "libneuron_adapter.so";
  std::vector<std::string> so_paths;

  // Add preinstalled libraries for system partition applications.
  so_paths.push_back("libneuronusdk_adapter.mtk.so");
  // Check if the device need to use higher version of usdk.
  auto magic_number = GetNeuroPilotMagicNumber();
  if (magic_number && magic_number.Value() >= kMinMagicNumberForNeuronService) {
    so_paths.push_back("libneuronusdk_adapter.9.mtk.so");
  }
  so_paths.push_back("libneuron_adapter_mgvi.so");
  // Some platforms have non-usdk non-mgvi build.
  so_paths.push_back(kLibNeuronAdapterLib);

  // Add the library from the provided shared lib directory if available.
  if (shared_library_dir.has_value()) {
    so_paths.push_back(
        absl::StrCat(*shared_library_dir, "/", kLibNeuronAdapterLib));
  }

#if !defined(__ANDROID__)
  // Add SDK specific paths for Linux.
  std::string sdk_path;
  switch (sdk_version) {
    case kLiteRtMediatekOptionsNeronSDKVersionTypeVersion7:
      sdk_path = "third_party/neuro_pilot/v7_latest/host/lib";
      break;
    case kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8:
      sdk_path = "third_party/neuro_pilot/v8_latest/host/lib";
      break;
    case kLiteRtMediatekOptionsNeronSDKVersionTypeVersion9:
      sdk_path = "third_party/neuro_pilot/v9_latest/host/lib";
      break;
    default:
      return litert::Error(kLiteRtStatusErrorInvalidArgument,
                           "Invalid sdk_version");
  }
  so_paths.push_back(absl::StrCat(sdk_path, "/", kLibNeuronAdapterLib));
#endif

  for (auto& so_path : so_paths) {
    auto maybe_dlib = SharedLibrary::Load(so_path, RtldFlags::Default());
    if (maybe_dlib.HasValue()) {
      LITERT_LOG(LITERT_INFO, "Loading MediaTek NeuronAdapter .so from: %s",
                 so_path.c_str());
      dlib_ = std::move(maybe_dlib.Value());
    }
  }

  if (!dlib_.Loaded()) {
    return litert::Error(kLiteRtStatusErrorDynamicLoading,
                         "Failed to load NeuronAdapter shared library");
  }

  LITERT_LOG(LITERT_INFO, "Loaded NeuronAdapter shared library.");

  // Binds all supported symbols from the shared library to the function
  // pointers.
  LOAD_SYMB(NeuronCompilation_create, api_->compilation_create);
  LOAD_SYMB(NeuronCompilation_createWithOptions,
            api_->compilation_create_with_options);
  LOAD_SYMB(NeuronCompilation_finish, api_->compilation_finish);
  LOAD_SYMB(NeuronCompilation_free, api_->compilation_free);
  LOAD_SYMB(NeuronCompilation_getInputPaddedDimensions,
            api_->compilation_get_input_padded_dimensions);
  LOAD_SYMB(NeuronCompilation_getInputPaddedSize,
            api_->compilation_get_input_padded_size);
  LOAD_SYMB(NeuronCompilation_getOutputPaddedDimensions,
            api_->compilation_get_output_padded_dimensions);
  LOAD_SYMB(NeuronCompilation_getOutputPaddedSize,
            api_->compilation_get_output_padded_size);
  LOAD_SYMB(NeuronCompilation_setOptimizationString,
            api_->compilation_set_optimization_string);
  LOAD_SYMB(NeuronCompilation_setPreference, api_->compilation_set_preference);
  LOAD_SYMB(NeuronCompilation_setPriority, api_->compilation_set_priority);
  LOAD_SYMB(NeuronExecution_compute, api_->execution_compute);
  LOAD_SYMB(NeuronExecution_create, api_->execution_create);
  LOAD_SYMB(NeuronExecution_free, api_->execution_free);
  LOAD_SYMB(NeuronCompilation_getCompiledNetworkSize,
            api_->compilation_get_compiled_network_size);
  LOAD_SYMB(NeuronCompilation_storeCompiledNetwork,
            api_->compilation_store_compiled_network);
  LOAD_SYMB(NeuronExecution_setBoostHint, api_->execution_set_boost_hint);
  LOAD_SYMB(NeuronExecution_setInputFromMemory,
            api_->execution_set_input_from_memory);
  LOAD_SYMB(NeuronExecution_setOutputFromMemory,
            api_->execution_set_output_from_memory);
  LOAD_SYMB(NeuronMemory_createFromAHardwareBuffer,
            api_->memory_create_from_ahwb);
  LOAD_SYMB(NeuronMemory_createFromFd, api_->memory_create_from_fd);
  LOAD_SYMB(NeuronMemory_free, api_->memory_free);
  LOAD_SYMB(NeuronModel_addOperand, api_->model_add_operand);
  LOAD_SYMB(NeuronModel_addOperation, api_->model_add_operation);
  LOAD_SYMB(NeuronModel_create, api_->model_create);
  LOAD_SYMB(NeuronModel_finish, api_->model_finish);
  LOAD_SYMB(NeuronModel_free, api_->model_free);
  LOAD_SYMB(NeuronModel_getExtensionOperandType,
            api_->model_get_extension_operand_type);
  LOAD_SYMB(NeuronModel_getExtensionOperationType,
            api_->model_get_extension_operation_type);
  LOAD_SYMB(NeuronModel_identifyInputsAndOutputs,
            api_->model_identify_inputs_and_outputs);
  LOAD_SYMB(NeuronModel_restoreFromCompiledNetwork,
            api_->model_restore_from_compiled_network);
  LOAD_SYMB(NeuronModel_setName, api_->model_set_name);
  LOAD_SYMB(NeuronModel_setOperandValue, api_->model_set_operand_value);
  LOAD_SYMB(NeuronModel_setOperandSymmPerChannelQuantParams,
            api_->model_set_symm_per_channel_quant_params);
  LOAD_SYMB(Neuron_getVersion, api_->get_version);
  LOAD_SYMB(NeuronModel_relaxComputationFloat32toFloat16,
            api_->relax_fp32_to_fp16);
  LOAD_SYMB(Neuron_getL1MemorySizeKb, api_->get_l1_memory_size_kb);
  LOAD_SYMB(NeuronCompilation_setL1MemorySizeKb,
            api_->compilation_set_l1_memory_size_kb);
  LOAD_SYMB(NeuronCompilation_setOptimizationHint,
            api_->compilation_set_optimization_hint);
  LOAD_SYMB(NeuronCompilation_getSupportedOperations,
            api_->compilation_get_supported_opertations);

  LITERT_LOG(LITERT_INFO, "NeuronAdapter symbols loaded");
  return {};
}

Expected<NeuronModelPtr> NeuronAdapterApi::CreateModel() const {
  NeuronModel* model;
  if (api().model_create(&model) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to create NeuroModel");
  }
  return NeuronModelPtr{model, api().model_free};
}

Expected<NeuronCompilationPtr> NeuronAdapterApi::CreateCompilation(
    NeuronModel* model) const {
  NeuronCompilation* compilation;
  if (api().compilation_create(model, &compilation) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to create NeuronCompilation");
  }
  return NeuronCompilationPtr{compilation, api().compilation_free};
}

Expected<NeuronCompilationPtr> NeuronAdapterApi::CreateCompilation(
    NeuronModel* model, const std::string& compile_options) const {
  NeuronCompilation* compilation;
  if (auto status = api().compilation_create_with_options(
          model, &compilation, compile_options.c_str());
      status != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_ERROR,
               "NeuronCompilation_createWithOptions failed with error %d",
               status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to create NeuronCompilation");
  }
  return NeuronCompilationPtr{compilation, api().compilation_free};
}

Expected<NeuronExecutionPtr> NeuronAdapterApi::CreateExecution(
    NeuronCompilation* compilation) const {
  NeuronExecution* execution;
  if (api().execution_create(compilation, &execution) != NEURON_NO_ERROR) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to create execution");
  }
  return NeuronExecutionPtr{execution, api().execution_free};
}

litert::Expected<void> NeuronAdapterApi::GetNeuronVersion() {
  if (api().get_version(&runtime_version_) != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_ERROR, "Fail to get neuron api version");
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Fail to get neuron api version");
  }
  LITERT_LOG(LITERT_INFO, "Neuron api version: %d.%d.%d",
             static_cast<int>(runtime_version_.major),
             static_cast<int>(runtime_version_.minor),
             static_cast<int>(runtime_version_.patch));
  return {};
}

bool NeuronAdapterApi::IsFeatureEnabled(NeuronFeatureType feature) const {
  if (feature < 0 || feature >= NEURON_FEATURE_COUNT) {
    return false;
  }

  auto is_version_greater = [this](const NeuronRuntimeVersion& min_ver) {
    if (runtime_version_.major > min_ver.major) return true;
    if (runtime_version_.major < min_ver.major) return false;
    if (runtime_version_.minor > min_ver.minor) return true;
    if (runtime_version_.minor < min_ver.minor) return false;
    if (runtime_version_.patch >= min_ver.patch) return true;
    return false;
  };

  return is_version_greater(kNeuronFeatureMinVersion[feature]);
}

litert::Expected<int32_t> NeuronAdapterApi::GetNeuroPilotMagicNumber() {
  std::string lib_path = "libneuron_sys_util.mtk.so";
  // Load library
  auto maybe_dlib = SharedLibrary::Load(lib_path, RtldFlags::Default());
  if (!maybe_dlib.HasValue()) {
    LITERT_LOG(LITERT_INFO, "libneuron_sys_util.mtk.so not found");
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "libneuron_sys_util.mtk.so not found");
  }
  SharedLibrary& lib = maybe_dlib.Value();

  if (!lib.Loaded()) {
    return litert::Error(kLiteRtStatusErrorDynamicLoading,
                         "Failed to load neuron_sys_util shared library");
  }

  // Load symbol
  using GetMagicNumberFunc = int (*)(int32_t*);
  auto maybe_func =
      lib.LookupSymbol<void*>("NeuronService_getNeuroPilotMagicNumber");
  if (!maybe_func.HasValue()) {
    lib.Close();
    return litert::Error(kLiteRtStatusErrorDynamicLoading,
                         "NeuronService_getNeuroPilotMagicNumber not found in "
                         "neuron_sys_util.mtk.so");
  }
  auto get_magic_number =
      reinterpret_cast<GetMagicNumberFunc>(maybe_func.Value());

  // Get magic number
  int32_t magic_number = 0;
  int ret = get_magic_number(&magic_number);
  lib.Close();
  if (ret != 0) {
    return litert::Error(
        kLiteRtStatusErrorDynamicLoading,
        "NeuronService_getNeuroPilotMagicNumber returns an error");
  }
  LITERT_LOG(LITERT_INFO, "Magic number: %d", magic_number);

  return magic_number;
}

}  // namespace mediatek
}  // namespace litert

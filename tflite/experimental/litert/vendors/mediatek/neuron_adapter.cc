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

#include "tflite/experimental/litert/vendors/mediatek/neuron_adapter.h"

#include <dlfcn.h>

#include <memory>
#include <optional>
#include <string>

#include "tflite/experimental/litert/c/litert_common.h"
#include "tflite/experimental/litert/c/litert_logging.h"
#include "tflite/experimental/litert/cc/litert_expected.h"
#include "tflite/experimental/litert/core/dynamic_loading.h"

#define LOAD_SYMB(S, H)                                            \
  H = reinterpret_cast<decltype(&S)>(::dlsym(dlib_handle_, #S));   \
  if (!H) {                                                        \
    LITERT_LOG(LITERT_WARNING, "Failed to load symbol %s: %s", #S, \
               ::dlerror());                                       \
  }

namespace litert {
namespace mediatek {

NeuronAdapter::NeuronAdapter() : api_(new Api) {}

NeuronAdapter::~NeuronAdapter() {
  if (dlib_handle_) {
    litert::internal::CloseLib(dlib_handle_);
  }
}

litert::Expected<NeuronAdapter::Ptr> NeuronAdapter::Create(
    std::optional<std::string> shared_library_dir) {
  std::unique_ptr<NeuronAdapter> neuron_adapter(new NeuronAdapter);
  if (auto status = neuron_adapter->LoadSymbols(shared_library_dir); !status) {
    return status.Error();
  }

  return neuron_adapter;
}

litert::Expected<void> NeuronAdapter::LoadSymbols(
    std::optional<std::string> shared_library_dir) {
  // The following preinstalled library is for system partition applications.
  if (litert::internal::OpenLib("libneuronusdk_adapter.mtk.so",
                                &dlib_handle_) != kLiteRtStatusOk) {
    // The next preinstalled library is in the vendor partition.
    if (litert::internal::OpenLib("libneuron_adapter_mgvi.so", &dlib_handle_) !=
        kLiteRtStatusOk) {
      // Finally, the app may want to provide their own version of the library.
      constexpr auto kLibNeuronAdapterLib = "libneuron_adapter.so";
      std::string library_path =
          shared_library_dir.has_value()
              ? *shared_library_dir + kLibNeuronAdapterLib
              : kLibNeuronAdapterLib;
      if (litert::internal::OpenLib(library_path, &dlib_handle_) !=
          kLiteRtStatusOk) {
        return litert::Unexpected(
            kLiteRtStatusErrorRuntimeFailure,
            "Failed to load NeuronAdapter shared library");
      }
    }
  }

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
  LOAD_SYMB(NeuronExecution_setBoostHint, api_->execution_set_boost_hint);
  LOAD_SYMB(NeuronExecution_setInputFromMemory,
            api_->execution_set_input_from_memory);
  LOAD_SYMB(NeuronExecution_setOutputFromMemory,
            api_->execution_set_output_from_memory);
  LOAD_SYMB(NeuronMemory_createFromAHardwareBuffer,
            api_->memory_create_from_ahwb);
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
  LOAD_SYMB(NeuronModel_setOperandValue, api_->model_set_operand_value);
  LOAD_SYMB(Neuron_getVersion, api_->get_version);

  LITERT_LOG(LITERT_INFO, "NeuronAdapter symbols loaded");
  return {};
}

}  // namespace mediatek
}  // namespace litert

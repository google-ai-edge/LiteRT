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

#include "litert/vendors/google_tensor/adapter_odc.h"

#include <dlfcn.h>
#include <sys/mman.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/google_tensor/adapter.h"
#include "litert/vendors/google_tensor/compiler_service_api.h"
#include "litert/vendors/google_tensor/compiler_service_options.pb.h"

#define Load(H, S)                                                      \
  H = reinterpret_cast<decltype(&S)>(::dlsym(dlib_handle_, #S));        \
  if (!H) {                                                             \
    std::string error_message =                                         \
        "Failed to load symbol " #S ": " + std::string(dlerror());      \
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, error_message); \
  }

namespace litert::google_tensor {

// EdgeTPU LiteRT shared library path.
constexpr const char* kLiteRtLibPath = "/vendor/lib64/libedgetpu_litert.so";

litert::Expected<Adapter::Ptr> Adapter::Create(
    std::optional<std::string> shared_library_dir) {
  // Always load the TfLite ODC library from the vendor partition.
  (void)shared_library_dir;
  AdapterOdc::Ptr adapter = std::make_unique<AdapterOdc>();
  LITERT_RETURN_IF_ERROR(adapter->LoadSymbols(shared_library_dir));
  return Adapter::Ptr(adapter.release());
}

AdapterOdc::AdapterOdc() : api_(std::make_unique<Api>()) {}

AdapterOdc::~AdapterOdc() {
  if (api_->context != nullptr) {
    api_->destroy(api_->context);
  }
  if (dlib_handle_ != nullptr) {
    dlclose(dlib_handle_);
  }
}

litert::Expected<void> AdapterOdc::LoadSymbols(
    std::optional<std::string> shared_library_dir) {
  // Always load the ODC API library from the vendor partition.
  (void)shared_library_dir;

  dlib_handle_ = ::dlopen(kLiteRtLibPath, RTLD_NOW | RTLD_LOCAL);
  if (!dlib_handle_) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to load Southbound shared library");
  }

  // Binds all supported symbols from the shared library to the function
  // pointers.
  Load(api_->create, CompilerServiceCreate);
  Load(api_->compile_subgraph_flatbuffer, CompilerServiceSubgraphFlatbuffer);
  Load(api_->free_compiled_code, CompilerServiceFreeCompiledCode);
  Load(api_->free_error_message, CompilerServiceFreeErrorMessage);
  Load(api_->destroy, CompilerServiceDestroy);

  LITERT_LOG(LITERT_INFO, "ODC symbols loaded");

  int status = api_->create(&api_->context);
  if (status != 0 || api_->context == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create CompilerServiceContext");
  }
  return {};
}

// Create CompilerServiceOptions from the opaque options.
Expected<CompilerServiceOptions> CreateCompilerServiceOptions(
    LiteRtOpaqueOptions options) {
  CompilerServiceOptions compiler_service_options;
  // TODO(chanchalraj): Parse options from the opaque options.
  return compiler_service_options;
}

Expected<void> AdapterOdc::Compile(const char* tfl_buffer_data,
                                   size_t tfl_buffer_size,
                                   const char* soc_model_data,
                                   size_t soc_model_size,
                                   LiteRtOpaqueOptions options,
                                   char** compiled_code_data,
                                   size_t* compiled_code_size) {
  // Parse soc model.
  CompilerServiceOptions::SocModel soc_model =
      CompilerServiceOptions::SOC_MODEL_UNSPECIFIED;
  if (soc_model_size > 0 && soc_model_data != nullptr) {
    if (strcmp(soc_model_data, "Tensor_G3") == 0) {
      soc_model = CompilerServiceOptions::SOC_MODEL_TENSOR_G3;
    } else if (strcmp(soc_model_data, "Tensor_G4") == 0) {
      soc_model = CompilerServiceOptions::SOC_MODEL_TENSOR_G4;
    } else if (strcmp(soc_model_data, "Tensor_G5") == 0) {
      soc_model = CompilerServiceOptions::SOC_MODEL_TENSOR_G5;
    } else {
      soc_model = CompilerServiceOptions::SOC_MODEL_UNSPECIFIED;
    }
  }

  LITERT_ASSIGN_OR_RETURN(auto compiler_service_options,
                          CreateCompilerServiceOptions(options));
  compiler_service_options.set_soc_model(soc_model);
  std::string serialized_options;
  if (!compiler_service_options.SerializeToString(&serialized_options)) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        "Failed to serialize compiler service options into proto");
  }

  char* error_message = nullptr;
  // Ensure memory allocated by the C API is freed.
  absl::Cleanup error_cleanup = [&] {
    if (error_message) {
      api_->free_error_message(api_->context, error_message);
    }
  };

  int compile_status = api_->compile_subgraph_flatbuffer(
      api_->context, tfl_buffer_data, tfl_buffer_size,
      serialized_options.data(), serialized_options.size(), compiled_code_data,
      compiled_code_size, &error_message);
  if (compile_status != 0) {
    std::string error_str = "Failed to compile model";
    if (error_message) {
      absl::StrAppend(&error_str, ": ", error_message);
    }
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, error_str);
  }
  return {};
}

void AdapterOdc::FreeCompiledCode(char* compiled_code_data) {
  api_->free_compiled_code(api_->context, compiled_code_data);
}

}  // namespace litert::google_tensor

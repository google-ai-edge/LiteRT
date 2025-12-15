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

#include <sys/mman.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/google_tensor/adapter.h"
#include "litert/vendors/google_tensor/edgetpu_compiler_api.h"
#include "litert/vendors/google_tensor/edgetpu_compiler_options.pb.h"

#define LOAD_SYMBOL(H, S) \
  LITERT_ASSIGN_OR_RETURN(H, dlib_.LookupSymbol<decltype(&S)>(#S));

namespace litert::google_tensor {

// EdgeTPU LiteRT shared library path.
constexpr const char* kLiteRtLibPath = "/vendor/lib64/libedgetpu_litert.so";

litert::Expected<Adapter::Ptr> Adapter::Create(
    std::optional<std::string> shared_library_dir) {
  // Always load the TfLite ODC library from the vendor partition.
  (void)shared_library_dir;
  AdapterOdc::Ptr adapter = std::make_unique<AdapterOdc>();
  LITERT_RETURN_IF_ERROR(adapter->LoadSymbols());
  return Adapter::Ptr(adapter.release());
}

AdapterOdc::AdapterOdc() : api_(std::make_unique<Api>()) {}

AdapterOdc::~AdapterOdc() {
  if (api_->context != nullptr) {
    api_->destroy(api_->context);
  }
}

litert::Expected<void> AdapterOdc::LoadSymbols() {
  // Load the shared library.
  LITERT_ASSIGN_OR_RETURN(
      dlib_, SharedLibrary::Load(kLiteRtLibPath, RtldFlags::Default()));

  // Binds all supported symbols from the shared library to the function
  // pointers.
  LOAD_SYMBOL(api_->create, EdgeTpuCompilerCreate);
  LOAD_SYMBOL(api_->compile_flatbuffer, EdgeTpuCompilerCompileFlatbuffer);
  LOAD_SYMBOL(api_->free_compiled_code, EdgeTpuCompilerFreeCompiledCode);
  LOAD_SYMBOL(api_->free_error_message, EdgeTpuCompilerFreeErrorMessage);
  LOAD_SYMBOL(api_->destroy, EdgeTpuCompilerDestroy);

  LITERT_LOG(LITERT_DEBUG, "ODC symbols loaded");

  int status = api_->create(&api_->context);
  if (status != 0 || api_->context == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create EdgeTpuCompilerContext");
  }
  return {};
}

// Create EdgeTpuCompilerOptions from the opaque options.
Expected<EdgeTpuCompilerOptions> CreateEdgeTpuCompilerOptions(
    const char* options, size_t options_size) {
  EdgeTpuCompilerOptions edgetpu_compiler_options;
  // TODO: b/467884692 - Parse options from the opaque options.
  return edgetpu_compiler_options;
}

Expected<void> AdapterOdc::Compile(
    const char* tfl_buffer_data, size_t tfl_buffer_size,
    const char* soc_model_data, size_t soc_model_size,
    const char* options, size_t options_size, char*** compiled_code_data,
    size_t** compiled_code_sizes, size_t* num_bytecodes) {
  LITERT_ASSIGN_OR_RETURN(auto edgetpu_compiler_options,
                          CreateEdgeTpuCompilerOptions(options, options_size));
  std::string serialized_options;
  if (!edgetpu_compiler_options.SerializeToString(&serialized_options)) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to serialize EdgeTpuCompilerOptions into proto");
  }

  char* error_message = nullptr;
  // Ensure memory allocated by the C API is freed.
  absl::Cleanup error_cleanup = [&] {
    if (error_message) {
      api_->free_error_message(api_->context, error_message);
    }
  };

  int compile_status = api_->compile_flatbuffer(
      api_->context, tfl_buffer_data, tfl_buffer_size,
      serialized_options.data(), serialized_options.size(), compiled_code_data,
      compiled_code_sizes, num_bytecodes, &error_message);
  if (compile_status != 0) {
    std::string error_str = "Failed to compile model";
    if (error_message) {
      absl::StrAppend(&error_str, ": ", error_message);
    }
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, error_str);
  }
  return {};
}

void AdapterOdc::FreeCompiledCode(char** compiled_code_data,
                                  size_t* compiled_code_sizes,
                                  size_t num_bytecodes) {
  api_->free_compiled_code(api_->context, compiled_code_data,
                           compiled_code_sizes, num_bytecodes);
}

}  // namespace litert::google_tensor

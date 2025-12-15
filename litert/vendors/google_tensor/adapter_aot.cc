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

#include "litert/vendors/google_tensor/adapter_aot.h"

#include <dlfcn.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/debugging/leak_check.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/google_tensor/adapter.h"

namespace litert {
namespace google_tensor {

AdapterAot::AdapterAot() : api_(std::make_unique<Api>()) {}

AdapterAot::~AdapterAot() {
  if (dlib_handle_) {
    dlclose(dlib_handle_);  // Use dlclose directly
  }
}

litert::Expected<Adapter::Ptr> Adapter::Create(
    std::optional<std::string> shared_library_dir) {
  AdapterAot::Ptr adapter = std::make_unique<AdapterAot>();
  auto status = adapter->LoadSymbols(shared_library_dir);
  if (!status.HasValue()) {
    LITERT_LOG(LITERT_ERROR, "Failed to create Adapter: %s",
               status.Error().Message().c_str());
    return status.Error();
  }
  return Adapter::Ptr(adapter.release());
}

litert::Expected<void> AdapterAot::LoadSymbols(
    std::optional<std::string> shared_library_dir) {
  constexpr auto kLibTensorTPUCompiler = "libcompiler_api_wrapper.so";

  const std::vector<std::string> so_paths = {
      shared_library_dir.has_value()
          ? absl::StrCat(*shared_library_dir, "/", kLibTensorTPUCompiler)
          : kLibTensorTPUCompiler};

  // Use dlopen directly
  for (const auto& path : so_paths) {
    dlib_handle_ = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (dlib_handle_) {
      void* init_func = dlsym(dlib_handle_, "GoogleTensorInitialize");
      if (init_func) {
        absl::LeakCheckDisabler disabler;
        (*reinterpret_cast<void (*)()>(init_func))();
      }
      break;  // Found the library
    }
  }

  if (!dlib_handle_) {
    const std::string error_message =
        "Failed to load Tensor TPU compiler library: " + std::string(dlerror());
    LITERT_LOG(LITERT_ERROR, "Failed to load Tensor TPU compiler library: %s",
               error_message.c_str());  // Include dlerror() for more info
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, error_message);
  }

  api_->compile = reinterpret_cast<::litert::google_tensor::Compile>(
      dlsym(dlib_handle_, "GoogleTensorCompileFlatbuffer"));
  if (!api_->compile) {
    const std::string error_message =
        "Failed to load Tensor TPU compiler API: " + std::string(dlerror());
    LITERT_LOG(LITERT_ERROR, "Failed to load Tensor TPU compiler API: %s",
               error_message.c_str());  // Include dlerror()
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, error_message);
  }
  api_->free_compiled_code = reinterpret_cast<CompilerFreeCompiledCode>(
      dlsym(dlib_handle_, "GoogleTensorCompilerFreeCompiledCode"));
  if (!api_->free_compiled_code) {
    const std::string error_message =
        "Failed to load Tensor TPU compiler API: " + std::string(dlerror());
    LITERT_LOG(LITERT_ERROR, "Failed to load Tensor TPU compiler API: %s",
               error_message.c_str());  // Include dlerror()
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, error_message);
  }
  api_->free_error_message = reinterpret_cast<CompilerFreeErrorMessage>(
      dlsym(dlib_handle_, "GoogleTensorCompilerFreeErrorMessage"));
  if (!api_->free_error_message) {
    const std::string error_message =
        "Failed to load Tensor TPU compiler API: " + std::string(dlerror());
    LITERT_LOG(LITERT_ERROR, "Failed to load Tensor TPU compiler API: %s",
               error_message.c_str());  // Include dlerror()
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, error_message);
  }

  LITERT_LOG(LITERT_INFO, "Tensor TPU compiler API symbols loaded");
  return {};
}

Expected<void> AdapterAot::Compile(
    const char* tfl_buffer_data, size_t tfl_buffer_size,
    const char* soc_model_data, size_t soc_model_size, const char* options_data,
    size_t options_size, char*** compiled_code_data,
    size_t** compiled_code_sizes, size_t* num_bytecodes) {
  char* error_message = nullptr;
  // Ensure memory allocated by the C API is freed.
  absl::Cleanup error_cleanup = [&] {
    if (error_message) {
      api_->free_error_message(error_message);
    }
  };
  bool compile_status = api_->compile(
      tfl_buffer_data, tfl_buffer_size, soc_model_data, soc_model_size,
      options_data, options_size, compiled_code_data, compiled_code_sizes,
      num_bytecodes, &error_message);
  if (!compile_status) {
    std::string error_str = "Failed to compile model";
    if (error_message) {
      absl::StrAppend(&error_str, ": ", error_message);
    }
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, error_str);
  }
  return {};
}

void AdapterAot::FreeCompiledCode(char** compiled_code_data,
                                  size_t* compiled_code_sizes,
                                  size_t num_bytecodes) {
  api_->free_compiled_code(compiled_code_data, compiled_code_sizes,
                           num_bytecodes);
}

}  // namespace google_tensor
}  // namespace litert

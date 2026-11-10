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
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
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
  constexpr auto kLibTensorTPUCompiler = "liblitert_plugin_compiler.so";

  const std::vector<std::string> so_paths = {
      shared_library_dir.has_value()
          ? absl::StrCat(*shared_library_dir, "/", kLibTensorTPUCompiler)
          : kLibTensorTPUCompiler};

  // Use dlopen directly
  for (const auto& path : so_paths) {
    dlib_handle_ = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (dlib_handle_) {
      break;  // Found the library
    }
  }

  if (!dlib_handle_) {
    const std::string error_message =
        absl::StrCat("Failed to load Tensor TPU compiler library: ", dlerror());
    LITERT_LOG(LITERT_ERROR, "Failed to load Tensor TPU compiler library: %s",
               error_message.c_str());  // Include dlerror() for more info
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, error_message);
  }

  api_->compile = reinterpret_cast<::litert::google_tensor::Compile>(
      dlsym(dlib_handle_, "GoogleTensorCompileFlatbuffer"));
  if (!api_->compile) {
    const std::string error_message =
        absl::StrCat("Failed to load Tensor TPU compiler API: ", dlerror());
    LITERT_LOG(LITERT_ERROR, "Failed to load Tensor TPU compiler API: %s",
               error_message.c_str());  // Include dlerror()
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, error_message);
  }
  api_->free_compiled_code = reinterpret_cast<CompilerFreeCompiledCode>(
      dlsym(dlib_handle_, "GoogleTensorCompilerFreeCompiledCode"));
  if (!api_->free_compiled_code) {
    const std::string error_message =
        absl::StrCat("Failed to load Tensor TPU compiler API: ", dlerror());
    LITERT_LOG(LITERT_ERROR, "Failed to load Tensor TPU compiler API: %s",
               error_message.c_str());  // Include dlerror()
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, error_message);
  }
  api_->free_error_message = reinterpret_cast<CompilerFreeErrorMessage>(
      dlsym(dlib_handle_, "GoogleTensorCompilerFreeErrorMessage"));
  if (!api_->free_error_message) {
    const std::string error_message =
        absl::StrCat("Failed to load Tensor TPU compiler API: ", dlerror());
    LITERT_LOG(LITERT_ERROR, "Failed to load Tensor TPU compiler API: %s",
               error_message.c_str());  // Include dlerror()
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, error_message);
  }

  api_->get_unsupported_ops = reinterpret_cast<CompilerGetUnsupportedOps>(
      dlsym(dlib_handle_, "GoogleTensorGetUnsupportedOps"));
  if (!api_->get_unsupported_ops) {
    const std::string error_message =
        absl::StrCat("Failed to load Tensor TPU compiler API: ", dlerror());
    LITERT_LOG(LITERT_ERROR, "Failed to load Tensor TPU compiler API: %s",
               error_message.c_str());  // Include dlerror()
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, error_message);
  }

  api_->free_unsupported_ops = reinterpret_cast<CompilerFreeUnsupportedOps>(
      dlsym(dlib_handle_, "GoogleTensorFreeUnsupportedOps"));
  if (!api_->free_unsupported_ops) {
    const std::string error_message =
        absl::StrCat("Failed to load Tensor TPU compiler API: ", dlerror());
    LITERT_LOG(LITERT_ERROR, "Failed to load Tensor TPU compiler API: %s",
               error_message.c_str());  // Include dlerror()
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, error_message);
  }

  api_->validate_composite_ops = reinterpret_cast<CompilerValidateCompositeOps>(
      dlsym(dlib_handle_, "GoogleTensorValidateCompositeOps"));
  api_->free_composite_op_validation_results =
      reinterpret_cast<CompilerFreeCompositeOpValidationResults>(
          dlsym(dlib_handle_, "GoogleTensorFreeCompositeOpValidationResults"));

  LITERT_LOG(LITERT_INFO, "Tensor TPU compiler API symbols loaded");
  return {};
}

Expected<void> AdapterAot::Compile(
    const char* tfl_buffer_data, size_t tfl_buffer_size,
    const char* options_data, size_t options_size, char*** compiled_code_data,
    size_t** compiled_code_sizes, size_t* num_bytecodes) {
  char* error_message = nullptr;
  // Ensure memory allocated by the C API is freed.
  absl::Cleanup error_cleanup = [&] {
    if (error_message) {
      api_->free_error_message(error_message);
    }
  };
  bool compile_status = api_->compile(
      tfl_buffer_data, tfl_buffer_size, options_data, options_size,
      compiled_code_data, compiled_code_sizes, num_bytecodes, &error_message);
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

Expected<std::vector<UnsupportedOp>> AdapterAot::GetUnsupportedOps(
    const char* tfl_buffer_data, size_t tfl_buffer_size, const char* options,
    size_t options_size) {
  if (!api_->get_unsupported_ops) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "get_unsupported_ops symbol not loaded");
  }

  GoogleTensorUnsupportedOp* unsupported_ops = nullptr;
  size_t num_unsupported_ops = 0;
  char* error_message = nullptr;
  absl::Cleanup cleanup = [&] {
    if (error_message) {
      api_->free_error_message(error_message);
    }
    if (unsupported_ops) {
      api_->free_unsupported_ops(unsupported_ops, num_unsupported_ops);
    }
  };

  bool success = api_->get_unsupported_ops(
      tfl_buffer_data, tfl_buffer_size, options, options_size, &unsupported_ops,
      &num_unsupported_ops, &error_message);

  if (!success) {
    std::string error_str = "Failed to get unsupported ops";
    if (error_message) {
      absl::StrAppend(&error_str, ": ", error_message);
    }
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, error_str);
  }

  std::vector<UnsupportedOp> result;
  result.reserve(num_unsupported_ops);
  for (size_t i = 0; i < num_unsupported_ops; ++i) {
    result.push_back(UnsupportedOp{
        .op_index = unsupported_ops[i].op_index,
        .reason = unsupported_ops[i].reason ? unsupported_ops[i].reason : "",
    });
  }
  return result;
}

Expected<std::vector<bool>> AdapterAot::AreCompositesSupported(
    absl::Span<const std::string> composite_names, const char* options_data,
    size_t options_size) {
  // TODO(b/564862823): Pass options_data and options_size to
  // GoogleTensorValidateCompositeOps to support per-chip composite op
  // validation.
  if (composite_names.empty()) {
    return std::vector<bool>{};
  }

  if (!api_ || !api_->validate_composite_ops ||
      !api_->free_composite_op_validation_results) {
    return litert::Unexpected(
        kLiteRtStatusErrorNotFound,
        "validate_composite_ops or free_composite_op_validation_results "
        "symbol not loaded");
  }

  std::vector<GoogleTensorCompositeOpDescriptor> descriptors(
      composite_names.size());
  for (size_t i = 0; i < composite_names.size(); ++i) {
    descriptors[i].composite_name = composite_names[i].c_str();
  }
  std::vector<GoogleTensorCompositeOpValidationResult> results(
      composite_names.size());
  if (!api_->validate_composite_ops(
          descriptors.data(), sizeof(GoogleTensorCompositeOpDescriptor),
          descriptors.size(), results.data(),
          sizeof(GoogleTensorCompositeOpValidationResult))) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to validate composite ops");
  }

  absl::Cleanup results_cleanup = [&] {
    if (api_->free_composite_op_validation_results) {
      api_->free_composite_op_validation_results(results.data(),
                                                 results.size());
    }
  };
  // TODO(b/562408779): Surface composite validation failure reasons and
  // categories to the caller instead of only logging and returning booleans.
  std::vector<bool> supported_flags;
  supported_flags.reserve(composite_names.size());
  for (const GoogleTensorCompositeOpValidationResult& result : results) {
    if (result.category == GOOGLE_TENSOR_COMPOSITE_VALIDATION_INTERNAL_ERROR) {
      std::string error_msg = "Internal error during composite op validation";
      if (result.failure_reason != nullptr) {
        absl::StrAppend(&error_msg, ": ", result.failure_reason);
      }
      LITERT_LOG(LITERT_ERROR, "%s", error_msg.c_str());
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, error_msg);
    }
    if (!result.is_supported && result.failure_reason != nullptr) {
      LITERT_LOG(LITERT_INFO, "Composite op validation failed: %s",
                 result.failure_reason);
    }
    supported_flags.push_back(result.is_supported);
  }
  return supported_flags;
}

}  // namespace google_tensor
}  // namespace litert

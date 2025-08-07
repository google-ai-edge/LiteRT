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
#ifndef ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_ADAPTER_H_
#define ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_ADAPTER_H_

#include <memory>
#include <optional>
#include <string>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert::google_tensor {
// Type definition for a function pointer to an ABI stable function
// used to compile a Flatbuffer model.
//
// Functions of this type are expected to:
// @param tfl_buffer_data Pointer to the serialized TFLite model flatbuffer.
// @param tfl_buffer_size Size of the flatbuffer.
// @param soc_model_data Pointer to the string identifying the SOC model
//        (e.g., "g4", "g5", etc.).
// @param soc_model_size Length of the SOC model string.
// @param options LiteRTOpaqueOptions to pass the compiler options.
// @param compiled_code_data On success, will be set to point to a newly
//        allocated buffer containing the compiled code. The caller takes
//        ownership of this buffer and is responsible for freeing it
//        (e.g., using a companion *FreeCompiledCode() function).
// @param compiled_code_size On success, will be set to the size of the
//        buffer pointed to by *compiled_code_data.
// @param out_error_message On failure, may be set to point to a newly allocated
//        NULL-terminated string containing an error message. The caller
//        takes ownership of this string and is responsible for freeing it
//        (e.g., using a companion *FreeErrorMessage() function).
// @return bool indicating whether the compilation was successful or not.
typedef bool (*Compile)(const char* tfl_buffer_data, size_t tfl_buffer_size,
                        const char* soc_model_data, size_t soc_model_size,
                        LiteRtOpaqueOptions options, char** compiled_code_data,
                        size_t* compiled_code_size, char** out_error_message);
typedef void (*CompilerFreeCompiledCode)(char* compiled_code_data);
typedef void (*CompilerFreeErrorMessage)(char* error_message);

// This class adapts the google tensor compiler API for dynamic loading.
class Adapter {
 public:
  // A smart pointer for managing TensorAdapter objects.
  using Ptr = std::unique_ptr<Adapter>;
  struct Api;

  Adapter();
  ~Adapter();

  // Creates a new TensorAdapter and loads the compiler API symbols.
  static litert::Expected<Ptr> Create(
      std::optional<std::string> shared_library_dir);

  // Returns a reference to the loaded API.
  const Api& api() const { return *api_; }

 private:
  // Loads the symbols from the compiler library.
  litert::Expected<void> LoadSymbols(
      std::optional<std::string> shared_library_dir);

  void* dlib_handle_ = nullptr;
  std::unique_ptr<Api> api_;
};

struct Adapter::Api {
  // The function pointer to the compiler wrapper API.
  Compile compile = nullptr;
  CompilerFreeCompiledCode free_compiled_code = nullptr;
  CompilerFreeErrorMessage free_error_message = nullptr;
};

}  // namespace litert::google_tensor

#endif  // ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_ADAPTER_H_

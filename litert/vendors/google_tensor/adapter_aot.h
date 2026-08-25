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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_ADAPTER_AOT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_ADAPTER_AOT_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/google_tensor/adapter.h"

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
// @param options Pointer to the serialized GoogleTensorOptions proto.
// @param options_size Size of the serialized GoogleTensorOptions proto.
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
                        const char* options, size_t options_size,
                        char*** compiled_code_data,
                        size_t** compiled_code_sizes, size_t* num_bytecodes,
                        char** out_error_message);
typedef void (*CompilerFreeCompiledCode)(char** compiled_code_data,
                                         size_t* compiled_code_sizes,
                                         size_t num_bytecodes);
typedef void (*CompilerFreeErrorMessage)(char* error_message);

// Type definition for a function pointer to an ABI stable function
// used to check which operations in the TFLite flatbuffer are unsupported.
//
// Functions of this type are expected to:
// @param tfl_buffer_data Pointer to the serialized TFLite model flatbuffer.
// @param tfl_buffer_size Size of the flatbuffer.
// @param options Pointer to the serialized GoogleTensorOptions proto.
// @param options_size Size of the serialized GoogleTensorOptions proto.
// @param unsupported_op_indices On success, will be set to point to a newly
//        allocated array containing the indices of the unsupported operations.
//        The caller takes ownership of this array and is responsible for
//        freeing it (e.g., using a companion *FreeUnsupportedOps() function).
// @param num_unsupported_ops On success, will be set to the number of elements
//        in the array pointed to by *unsupported_op_indices.
// @param out_error_message On failure, may be set to point to a newly allocated
//        NULL-terminated string containing an error message. The caller
//        takes ownership of this string and is responsible for freeing it
//        (e.g., using a companion *FreeErrorMessage() function).
// @return bool indicating whether the validation was successful or not.
typedef bool (*CompilerGetUnsupportedOps)(
    const char* tfl_buffer_data, size_t tfl_buffer_size, const char* options,
    size_t options_size, int32_t** unsupported_op_indices,
    size_t* num_unsupported_ops, char** out_error_message);

typedef void (*CompilerFreeUnsupportedOps)(int32_t* unsupported_op_indices);

// This class adapts the google tensor compiler API for dynamic loading.
class AdapterAot : public Adapter {
 public:
  // A smart pointer for managing TensorAdapter objects.
  using Ptr = std::unique_ptr<AdapterAot>;

  AdapterAot();
  ~AdapterAot() override;

  // Loads the symbols from the compiler library.
  litert::Expected<void> LoadSymbols(
      std::optional<std::string> shared_library_dir);

  Expected<void> Compile(const char* tfl_buffer_data, size_t tfl_buffer_size,
                         const char* options_data, size_t options_size,
                         char*** compiled_code_data,
                         size_t** compiled_code_sizes,
                         size_t* num_bytecodes) override;

  Expected<std::vector<int32_t>> GetUnsupportedOps(
      const char* tfl_buffer_data, size_t tfl_buffer_size, const char* options,
      size_t options_size) override;

  bool IsAot() const override { return true; }

  void FreeCompiledCode(char** compiled_code_data, size_t* compiled_code_sizes,
                        size_t num_bytecodes) override;

 private:
  struct Api {
    // The function pointer to the compiler wrapper API.
    ::litert::google_tensor::Compile compile = nullptr;
    CompilerFreeCompiledCode free_compiled_code = nullptr;
    CompilerFreeErrorMessage free_error_message = nullptr;
    CompilerGetUnsupportedOps get_unsupported_ops = nullptr;
    CompilerFreeUnsupportedOps free_unsupported_ops = nullptr;
  };

  void* dlib_handle_ = nullptr;
  std::unique_ptr<Api> api_;
};

}  // namespace litert::google_tensor

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_ADAPTER_AOT_H_

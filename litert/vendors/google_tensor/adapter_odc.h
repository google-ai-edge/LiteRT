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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_ADAPTER_ODC_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_ADAPTER_ODC_H_

#include <cstddef>
#include <memory>

#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/google_tensor/adapter.h"
#include "litert/vendors/google_tensor/edgetpu_compiler_api.h"

namespace litert::google_tensor {

// This class adapts the google tensor compiler API for dynamic loading.
class AdapterOdc : public Adapter {
 public:
  // A smart pointer for managing TensorAdapter objects.
  using Ptr = std::unique_ptr<AdapterOdc>;

  AdapterOdc();
  ~AdapterOdc() override;

  // Loads the symbols from the compiler library.
  litert::Expected<void> LoadSymbols();

  Expected<void> Compile(const char* tfl_buffer_data, size_t tfl_buffer_size,
                         const char* soc_model_data, size_t soc_model_size,
                         const char* options, size_t options_size,
                         char*** compiled_code_data,
                         size_t** compiled_code_sizes,
                         size_t* num_bytecodes) override;

  void FreeCompiledCode(char** compiled_code_data, size_t* compiled_code_sizes,
                        size_t num_bytecodes) override;

 private:
  struct Api {
    EdgeTpuCompilerContext* context = nullptr;
    decltype(&EdgeTpuCompilerCreate) create = nullptr;
    decltype(&EdgeTpuCompilerCompileFlatbuffer) compile_flatbuffer = nullptr;
    decltype(&EdgeTpuCompilerFreeCompiledCode) free_compiled_code = nullptr;
    decltype(&EdgeTpuCompilerFreeErrorMessage) free_error_message = nullptr;
    decltype(&EdgeTpuCompilerDestroy) destroy = nullptr;
  };

  SharedLibrary dlib_;
  std::unique_ptr<Api> api_;
};

}  // namespace litert::google_tensor

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_ADAPTER_ODC_H_

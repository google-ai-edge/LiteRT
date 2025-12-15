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

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert::google_tensor {

// This class adapts the google tensor compiler API for dynamic loading.
class Adapter {
 public:
  // A smart pointer for managing TensorAdapter objects.
  using Ptr = std::unique_ptr<Adapter>;

  Adapter() = default;
  virtual ~Adapter() = default;

  // Creates a new TensorAdapter and loads the compiler API symbols.
  static litert::Expected<Ptr> Create(
      std::optional<std::string> shared_library_dir);

  virtual Expected<void> Compile(
      const char* tfl_buffer_data, size_t tfl_buffer_size,
      const char* soc_model_data, size_t soc_model_size, const char* options,
      size_t options_size, char*** compiled_code_data,
      size_t** compiled_code_sizes, size_t* num_bytecodes) = 0;

  virtual void FreeCompiledCode(char** compiled_code_data,
                                size_t* compiled_code_sizes,
                                size_t num_bytecodes) = 0;
};

}  // namespace litert::google_tensor

#endif  // ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_ADAPTER_H_

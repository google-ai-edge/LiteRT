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

#ifndef ODML_LITERT_LITERT_CORE_COMPILATION_OPTIONS_H_
#define ODML_LITERT_LITERT_CORE_COMPILATION_OPTIONS_H_

#include <cstddef>
#include <string>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_op_kernel.h"

// New structure to define a binding between a tensor name and an external
// buffer.
struct LiteRtExternalTensorBinding {
  // The name of the signature in the TFLite model.
  std::string signature_name;
  // The name of the tensor in the TFLite model graph.
  std::string tensor_name;
  // Pointer to the external data buffer. The lifetime of this buffer must
  // exceed the lifetime of the CompiledModel.
  void* data;
  // Size of the external data buffer in bytes. This must match the tensor's
  // expected size.
  size_t size_bytes;
};

struct LiteRtOptionsT {
  struct CustomOpOption {
    std::string op_name;
    int op_version;
    void* user_data;
    LiteRtCustomOpKernel op_kernel;
  };

  // This should be updated every time a field is added/edited.
  //
  // - Renaming a field: increment patch;
  // - Adding or deprecating a field: set patch to 0, increment minor.
  // - Breaking layout compatibility: set patch and minor to 0, increment major.
  //
  // Note: Changing a default value does not impact the version.
  LiteRtApiVersion version = {.major = 1, .minor = 0, .patch = 0};
  LiteRtHwAcceleratorSet hardware_accelerators = kLiteRtHwAcceleratorNone;
  LiteRtOpaqueOptions options = nullptr;
  std::vector<CustomOpOption> custom_op_options;
  std::vector<LiteRtExternalTensorBinding> external_tensor_bindings;
};

#endif  // ODML_LITERT_LITERT_CORE_COMPILATION_OPTIONS_H_

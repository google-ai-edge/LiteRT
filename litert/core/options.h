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

#include <string>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_op_kernel.h"
#include "litert/cc/litert_opaque_options.h"

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
  LiteRtApiVersion version = {.major = 0, .minor = 0, .patch = 1};
  LiteRtHwAcceleratorSet hardware_accelerators = kLiteRtHwAcceleratorNone;
  litert::OpaqueOptions options;
  std::vector<CustomOpOption> custom_op_options;
};

#endif  // ODML_LITERT_LITERT_CORE_COMPILATION_OPTIONS_H_

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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_LITERT_CPU_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_LITERT_CPU_OPTIONS_H_

#include <string>

#include "litert/c/litert_common.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"

// Internal LiteRt CPU options struct. This data structure is used to
// pass CPU options to the interpreter and will be used in the framework
// code.
struct LiteRtCpuOptionsT {
  TfLiteXNNPackDelegateOptions xnn = TfLiteXNNPackDelegateOptionsDefault();
  // We need to keep the string alive because `TfLiteXNNPackDelegateOptions`
  // expects a `const char*` for `weight_cache_file_path` and does not manage
  // its memory.
  std::string weight_cache_file_path_buffer;

  static const char* Identifier() { return "xnnpack"; }
};

namespace litert {
namespace internal {

// Parses the serialized CPU options into the internal struct.
LiteRtStatus ParseLiteRtCpuOptions(const void* data, size_t size,
                                   LiteRtCpuOptionsT* options);

}  // namespace internal
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_LITERT_CPU_OPTIONS_H_

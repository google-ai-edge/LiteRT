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
#include "litert/c/options/litert_cpu_options.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"

// This is essentially the same struct as delegates/ynnpack/ynnpack_delegate.h
// We set the default value to false to avoid unitialized memory access.
struct LiteRtYnnpackOptionsT {
  // The number of threads to use for parallel execution.
  int num_threads = 1;
  // Whether to optimize for static shapes.
  bool static_shape = false;
  // Whether to enable fast math optimizations.
  bool fast_math = false;
  // Whether to ensure consistent arithmetic results across different platforms.
  bool consistent_arithmetic = false;
  // Whether to avoid using excess precision in floating-point calculations.
  bool no_excess_precision = false;
};

// Internal LiteRt CPU options struct. This data structure is used to
// pass CPU options to the interpreter and will be used in the framework
// code.
struct LiteRtCpuOptionsT {
  LiteRtCpuKernelMode kernel_mode = kLiteRtCpuKernelModeDelegate;
  bool enable_ynnpack = false;
  TfLiteXNNPackDelegateOptions xnn = TfLiteXNNPackDelegateOptionsDefault();
  LiteRtYnnpackOptionsT ynn;
  // We need to keep the string alive because `TfLiteXNNPackDelegateOptions`
  // expects a `const char*` for `weight_cache_file_path` and does not manage
  // its memory.
  std::string weight_cache_file_path_buffer;
  bool hint_fully_delegated_to_single_delegate = false;

  static const char* Identifier() { return "cpu_delegate"; }
};

namespace litert {
namespace internal {

// Parses the serialized CPU options into the internal struct.
LiteRtStatus ParseLiteRtCpuOptions(const void* data, size_t size,
                                   LiteRtCpuOptionsT* options);

}  // namespace internal
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_LITERT_CPU_OPTIONS_H_

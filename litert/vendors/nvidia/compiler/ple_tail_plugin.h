// Copyright 2026 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_PLE_TAIL_PLUGIN_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_PLE_TAIL_PLUGIN_H_

#include "NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/ple_tail.h"

namespace litert::nvidia {

// Qwen 4's per-layer embedding past its key and value projections as one
// plugin (trtllm/ple_tail.h; params.rows is the runtime's):
//   inputs  key BF16 [.., streams * hidden], value BF16 [.., hidden], the
//           streams x BF16 like key, the convolution's state FP32 [1, streams
//           * hidden, taps - 1], the layer's FP32 constants
//   outputs x + the layer's result, BF16 like x; the state' like the state
// The runtime binds the state's input and output to one buffer. Null when
// the kernel does not take the shapes.
nvinfer1::IPluginV3* CreatePleTailPlugin(
    const LiteRtNvidiaPleTailParams& params) noexcept;

// Referenced by the dispatch library so the creator's registration object is
// retained when linking the shared library used for engine deserialization.
void EnsurePleTailPluginRegistered() noexcept;

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_PLE_TAIL_PLUGIN_H_

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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_SLOT_PLUGINS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_SLOT_PLUGINS_H_

#include <cstdint>

#include "NvInfer.h"

namespace litert::nvidia {

// The plugin layers of a batched decode step that serve every sequence slot
// in one launch (trtllm/decode_attention_slots.h and its like): the "slots
// twin" of a plugin layer of the single-sequence decode step.
//
// A twin takes the single-sequence layer's inputs once per slot, slot-major
// (slot 0's inputs in that layer's order, then slot 1's, ...), and gives its
// outputs once per slot in the same order. A slot's outputs are what the
// single-sequence layer computes from that slot's inputs, bit for bit. So a
// network that repeats the step per slot replaces the slots' copies of a
// layer by its twin:
//   inputs[s * n + i] = slot s's copy of the layer's input i
//   slot s's copy of the layer's output o = twin output s * m + o
// (n and m the layer's input and output counts). An input every slot shares
// (a constant) is passed once per slot like the others.
//
// The twin of `layer` for `slots` slots (2 to 8), or null when the layer is
// not a plugin with a twin for its shapes:
//   LiteRtNvidiaDecodeAttention  over FP16 or INT8 caches with at most 16
//       rows (not the tensor-core kernels' shapes, new keys or row tokens)
nvinfer1::IPluginV3* CreateSlotsTwin(nvinfer1::IPluginV3Layer& layer,
                                     int32_t slots) noexcept;

// Referenced by the dispatch library so the creators' registration objects
// are retained when linking the shared library used for engine
// deserialization.
void EnsureSlotPluginsRegistered() noexcept;

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_SLOT_PLUGINS_H_

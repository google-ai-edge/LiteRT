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
// in one launch (trtllm/decode_attention_slots.h and its like). A network
// that repeats the single-sequence step's layers per slot replaces the
// slots' copies of a plugin layer by one of these. In every form a slot's
// outputs are what the single-sequence layer computes from that slot's
// inputs, bit for bit.

// --- Layers with a state per slot: the "slots twin" -------------------------
//
// A twin takes the single-sequence layer's inputs once per slot, slot-major
// (slot 0's inputs in that layer's order, then slot 1's, ...), and gives its
// outputs once per slot in the same order:
//   inputs[s * n + i] = slot s's copy of the layer's input i
//   slot s's copy of the layer's output o = twin output s * m + o
// (n and m the layer's input and output counts). An input every slot shares
// (a norm's scale) is passed once per slot like the others; slot 0's is the
// one read.
//
// The twin of `layer` for `slots` slots (2 to 8), or null when the layer is
// not a plugin with a twin for its shapes:
//   LiteRtNvidiaDecodeAttention  over FP16 or INT8 caches with at most 16
//       rows (not the tensor-core kernels' shapes, new keys or row tokens)
//   LiteRtNvidiaQkvNormRope
//   LiteRtNvidiaInt8CacheWrite   every slot's cache written in place where
//       its output buffer is its cache input's, as the layer's own
nvinfer1::IPluginV3* CreateSlotsTwin(nvinfer1::IPluginV3Layer& layer,
                                     int32_t slots) noexcept;

// --- Layers whose only input per slot is the activation row: "slot rows" ----
//
// The form of `plugin` that takes the slots' rows as the rows of its first
// input (the slots' tensors concatenated on the rows axis; the other inputs
// are every slot's) and gives each row the bits the plugin gives it alone,
// its weights read once for all of them; null for a plugin without one:
//   LiteRtNvidiaMoe  a layer block (compiler/moe_plugin.h, slot_rows), up to
//       8 rows
nvinfer1::IPluginV3* CreateSlotRowsPlugin(nvinfer1::IPluginV3& plugin) noexcept;

// --- The cache writes of a step in one launch -------------------------------
//
// A plugin that writes the rows a step adds into `writes` caches (1 to 16:
// the slots' key caches, their value caches, or both when they have one
// shape), in place, what a cast to the cache's type and TensorRT's KV cache
// update write into an FP16 cache. Inputs, once per write: the cache HALF
// [1, H, S, D] (or [1, H, D, S] with `columns`), the rows HALF, BF16 or
// FLOAT [1, H, T, D], and the row position INT32 [1] (clamped to [0, S -
// T]). Outputs: the updated caches. A cache is written in its output
// buffer; where that is not its input's buffer (the binding LiteRT-LM gives a
// cache), the cache is copied there first.
nvinfer1::IPluginV3* CreateCacheWriteSlotsPlugin(int32_t writes,
                                                 bool columns) noexcept;

// Referenced by the dispatch library so the creators' registration objects
// are retained when linking the shared library used for engine
// deserialization.
void EnsureSlotPluginsRegistered() noexcept;

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_SLOT_PLUGINS_H_

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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_MOE_PLUGIN_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_MOE_PLUGIN_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "NvInferRuntime.h"

namespace litert::nvidia {

// The mixture-of-experts feed-forward of Gemma 4 26B-A4B (the "moe" custom
// op) as a TensorRT plugin over the blockwise INT4 experts:
//   inputs  x BF16 [..., model_dim], top_weights FP32 [..., top_k],
//           top_indices INT32 [..., top_k], per_expert_scale FP32 [experts]
//   output  [..., model_dim] BF16
// (trtllm/int4_block_gemv.h, LiteRtNvidiaLaunchMoeInt4Block).
//
// The experts (12.8 GB for the 26B) live outside the engine: the plugin
// records where their weights and scales are in the model file, and loads
// them into an arena, device memory while a reserve stays free and pinned
// host memory the kernels read over PCIe otherwise, the first time an
// execution context attaches it. Arenas are shared by every engine and
// context of the process that names the same pieces (the prefill and decode
// engines of a model), and loaded only once MoeArenaLoadingEnabled() is set,
// which the dispatch library does: an engine build never loads them.
struct MoeExpertSource {
  std::string model_path;
  uint64_t model_size = 0;
  // The gate, up and down INT4 weights and their scales, in that order,
  // then (a fused router only) the router's INT4 rows [experts, model_dim]
  // and their scales, then (a layer block only) the dense feed-forward's
  // gate rows [dense_hidden, model_dim], up rows, gate scales, up scales,
  // down rows [model_dim, dense_hidden] and down scales: bytes [offset,
  // offset + size) of the file. FP32 scales (scale_type 0) are converted to
  // FP16 on the way in; FP16 scales (1) are copied. router_scale_type and
  // dense_scale_type likewise; -1 without a router / dense feed-forward.
  uint64_t offsets[14] = {};
  uint64_t sizes[14] = {};
  int32_t scale_type = 0;
  int32_t router_scale_type = -1;
  int32_t dense_scale_type = -1;
  int32_t dense_hidden = 0;
};

// `routed`: the plugin computes the routing itself from the residual stream
// (LiteRtNvidiaLaunchMoeInt4BlockFused). Its inputs are then x BF16,
// per_expert_scale FP32 [experts], the router norm's scale FP32 [model_dim]
// and the expert norm's scale FP32 [model_dim]; norm_eps is both norms'
// epsilon and router_multiplier the scalar between the router norm and its
// projection; the top weights are renormalized.
// `block` (routed only): the plugin is the layer's whole feed-forward block
// (LiteRtNvidiaLaunchMoeLayerBlock): the dense feed-forward (source's dense
// pieces, dense_hidden wide, GELU dense_gate: 1 tanh, 2 erf) beside the
// experts, the output norms and the residual update, times
// residual_multiplier. Four more FP32 [model_dim] inputs follow: the dense
// input norm's scale, the dense output norm's, the experts' output norm's
// and the sum's norm's. The output is the next residual stream.
nvinfer1::IPluginV3* CreateMoePlugin(
    int32_t experts, int32_t top_k, int32_t model_dim, int32_t hidden_dim,
    int32_t gate, const MoeExpertSource& source, bool routed = false,
    float norm_eps = -1.0f, float router_multiplier = 1.0f, bool block = false,
    int32_t dense_gate = 0, float residual_multiplier = 1.0f) noexcept;

// Referenced by the dispatch library so the creator's registration object is
// retained when linking the shared library used for engine deserialization.
void EnsureMoePluginRegistered() noexcept;

// Lets attached execution contexts load their expert arenas (the dispatch
// library sets it; the compiler never does).
void SetMoeArenaLoadingEnabled(bool enabled) noexcept;
bool MoeArenaLoadingEnabled() noexcept;

// Bytes of expert arenas currently loaded, on the device and in pinned host
// memory, for memory reports.
void MoeArenaBytes(size_t* device_bytes, size_t* host_bytes) noexcept;

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_MOE_PLUGIN_H_

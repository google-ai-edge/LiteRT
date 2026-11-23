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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_HYPER_CONNECTION_PLUGIN_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_HYPER_CONNECTION_PLUGIN_H_

#include <cstdint>

#include "NvInferRuntime.h"

namespace litert::nvidia {

// Where a projection's weights are among the plugin's inputs. With a holder
// (granule != 0) its weights input is the INT64 holder of a weight store
// segment, which starts at the first granule boundary inside it: the packed
// weights are weight_offset bytes into the segment, and the scales
// scale_offset bytes into it when scale_offset >= 0 (the scales input is
// then the holder too).
struct HcMixWeightsPlace {
  int64_t granule = 0;
  int64_t weight_offset = 0;
  int64_t scale_offset = -1;
};

struct HcMixPluginParams {
  int32_t streams = 0;
  int32_t hidden = 0;
  int32_t low_rank = 0;
  float down_scale = 1.0f;
  float eps = 1e-6f;
  // 1: INT4 weights in blocks of 32 with FP16 scales; 0: BF16 weights.
  int32_t int4 = 0;
  // The rows of an extra projection of the normed streams (0: none).
  int32_t extra = 0;
  HcMixWeightsPlace down;
  HcMixWeightsPlace up;
  HcMixWeightsPlace more;  // the extra projection's
};

// Qwen 4's hyper-connection mix as one plugin (trtllm/hyper_connection.h):
// the grouped norm of the streams, the low-rank down projection, silu, the up
// projection, the sigmoid gates and the mean over the streams.
//   inputs  x BF16 [.., streams * hidden] (a few rows), the norm weight FP32
//           [streams * hidden], then
//           INT4: the down weights (INT8 bytes [low_rank * streams * hidden
//           / 2], or a holder), the down scales (FP16 [low_rank * streams *
//           hidden / 32], or the holder), the up weights and the up scales
//           likewise ([streams * hidden] rows of low_rank);
//           BF16: the down weights BF16 [low_rank * streams * hidden] and
//           the up weights BF16 [streams * hidden * low_rank]
//   outputs mixed BF16 [.., hidden], normed BF16 like x
// With `extra` rows, an extra projection of the normed streams runs in the
// same launch (the combine's injection weights read them too): its weights
// (and INT4 scales) [extra, streams * hidden] are the last inputs, in the
// forms of the other two, and its result BF16 [.., extra] the third output.
// Null when the kernel does not take the shapes.
nvinfer1::IPluginV3* CreateHcMixPlugin(
    const HcMixPluginParams& params) noexcept;

// Referenced by the dispatch library so the creator's registration object is
// retained when linking the shared library used for engine deserialization.
void EnsureHyperConnectionPluginRegistered() noexcept;

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_HYPER_CONNECTION_PLUGIN_H_

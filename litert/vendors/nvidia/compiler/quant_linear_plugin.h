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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_QUANT_LINEAR_PLUGIN_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_QUANT_LINEAR_PLUGIN_H_

#include <array>
#include <cstdint>

#include "NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/quant_linear.h"

namespace litert::nvidia {

// The kernels of trtllm/quant_linear.h as one TensorRT plugin, a launch per
// layer. Tensors keep the model's leading dims: [..., n] is `m` rows of n.
//
// A product (LiteRtNvidiaLaunchQuantLinear):
//   inputs  activation [..., segments * columns], INT8 or the float type of
//           params.activation; the packed weights INT8 [bytes]; multipliers
//           FP32 [rows]; then the table INT8 [256] with params.lut, or the
//           row scales FP32 [..., 1] for logits
//   output  [..., rows] INT8, or FP32 logits
// With `holder_granule` the weights input is the INT64 holder of a weight
// store segment and the weights are `weight_offset` bytes into the segment
// (subbyte_gemv_plugin.h).
nvinfer1::IPluginV3* CreateQuantLinearPlugin(
    const LiteRtNvidiaQuantLinearParams& params, int64_t weight_offset = 0,
    int64_t holder_granule = 0) noexcept;

// The outputs of the glue plugin, in this order.
enum QuantGlueOutput : int32_t {
  kQuantGlueOutputX = 1,         // x' FP16 [..., width]
  kQuantGlueOutputZ = 2,         // z [..., width], params.z_type
  kQuantGlueOutputQ = 4,         // q INT8 [..., width]
  kQuantGlueOutputRowScale = 8,  // FP32 [..., 1], the dynamic scales
};

// The float steps between two products (LiteRtNvidiaLaunchQuantGlue):
//   inputs  x [..., width] of params.x_type; y INT8 [..., width] and the
//           branch norm's weights FP32 [width] with params.branch; the
//           norm's weights FP32 [width] with params.norm
//   outputs those of the `outputs` mask (QuantGlueOutput)
nvinfer1::IPluginV3* CreateQuantGluePlugin(
    const LiteRtNvidiaQuantGlueParams& params, int32_t outputs) noexcept;

// The glue in front of a product and the product as one layer
// (LiteRtNvidiaLaunchQuantGlueLinear for few rows; for many, the glue into
// the workspace and the product on the tensor cores):
//   inputs  x as the glue's; the product's weights (or holder), multipliers
//           and table as CreateQuantLinearPlugin's; then the glue's y and
//           branch norm weights (with glue.branch) and norm weights (with
//           glue.norm)
//   outputs the product's INT8 [..., rows], then x' and z of the `outputs`
//           mask (kQuantGlueOutputX, kQuantGlueOutputZ)
nvinfer1::IPluginV3* CreateQuantGlueLinearPlugin(
    const LiteRtNvidiaQuantGlueParams& glue,
    const LiteRtNvidiaQuantLinearParams& linear, int32_t outputs,
    int64_t weight_offset = 0, int64_t holder_granule = 0) noexcept;

// The four products of a factorized MLP on one activation row as one layer
// (LiteRtNvidiaLaunchQuantFactorizedMlp), with `glue` the glue in front of it
// too (LiteRtNvidiaLaunchQuantGlueFactorizedMlp):
//   inputs  activation INT8 [..., segments * columns of stage 0], one row,
//           or with `glue` its x; the packed weights of the four stages,
//           each INT8 [bytes] where weight_offsets[s] is negative, else the
//           INT64 holder of a weight store segment with the weights
//           weight_offsets[s] bytes into it; the multipliers FP32 [rows] of
//           each stage; the table INT8 [256] when a stage has one; then the
//           glue's y and branch norm weights (with glue->branch) and norm
//           weights (with glue->norm)
//   outputs INT8 [..., rows of stage 3], then x' and z of the `outputs`
//           mask (kQuantGlueOutputX, kQuantGlueOutputZ)
nvinfer1::IPluginV3* CreateQuantFactorizedMlpPlugin(
    const LiteRtNvidiaQuantFactorizedMlpParams& params,
    const std::array<int64_t, 4>& weight_offsets, int64_t holder_granule,
    const LiteRtNvidiaQuantGlueParams* glue = nullptr,
    int32_t outputs = 0) noexcept;

// The head norms and the rotation behind the q, k and v projections
// (LiteRtNvidiaLaunchQuantQkvRope; params.tokens is the input's):
//   inputs  qkv INT8 [..., (q_heads + k_heads + v_heads) * depth]; positions
//           INT32, one per token; the norm weights FP32 [depth] of the
//           projections that have heads, in the order q, k, v; the inverse
//           timescales FP32 [depth / 2]
//   outputs FP16, for the projections that have heads in that order:
//           q [1, q_heads, tokens, depth], k [1, k_heads, tokens, depth],
//           v [1, tokens, v_heads, depth]
nvinfer1::IPluginV3* CreateQuantQkvRopePlugin(
    const LiteRtNvidiaQuantQkvRopeParams& params) noexcept;

// Referenced by the dispatch library so the creator's registration object is
// retained when linking the shared library used for engine deserialization.
void EnsureQuantLinearPluginRegistered() noexcept;

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_QUANT_LINEAR_PLUGIN_H_

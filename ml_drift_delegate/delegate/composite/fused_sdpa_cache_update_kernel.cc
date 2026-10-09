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

#include "ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_kernel.h"

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/kernel_info.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tuning_type.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_parser.h"

namespace litert::ml_drift {
namespace {

using TensorHandle = ::ml_drift::GpuModelBuilder::TensorHandle;

// Index of param[0] (start position p0) in the int32 param tensor.
constexpr int kStartParamIndex = 0;

// Ring slot j holds position j until the ring has wrapped, so while p0 < W
// only the slots below p0 hold tokens. The past logits and the P.V product
// are bounded by p0 at runtime (ConvRuntimeCheckDesc), which skips the cache
// entirely for the first prefill chunk (p0 = 0). Depending on the kernel
// chosen, those bounds round p0 up to 4 or to 32 channels, so the softmax
// visits the 32-rounded bound `past_end` (in slices) and gives every empty
// slot (>= `past_filled`) a probability of exactly 0, whatever the mask says.
// Past `past_end`, it writes 0 to the one aligned block of slices that the
// bounded loops may still read (see CreateMaskedSoftmaxOp).
std::string PastEndSliceCode(const std::string& start,
                             const std::string& past_slices) {
  return "  int past_end = max(" +
         ::ml_drift::ConvRuntimeCheckDesc().GetRuntimeEndSlice(start,
                                                               past_slices) +
         ", 0);\n  int past_filled = min(max(" + start + ", 0), " +
         past_slices + " * 4);\n";
}

// Code that reads slice `s` of `SRC` into the float4 `v`. With `apply_mask`,
// it also soft-caps it and applies mask slice `S`.
std::string ReadMaskedLogitsCode(const std::string& src, bool has_softcap,
                                 bool bool_mask, bool apply_mask = true) {
  std::string code = absl::StrReplaceAll(R"(
    float4 v = args.SRC.Read<float>(X, Y, s, 0);
)",
                                         {{"SRC", src}});
  if (apply_mask && has_softcap) {
    code += R"(
    v = ucl::Init<float4>(args.softcap) * tanh(v * ucl::Init<float4>(args.inv_softcap));
)";
  }
  if (apply_mask && bool_mask) {
    // -10000 matches the fill value of the ML Drift SDPA kernels and of the
    // composite's reference decomposition.
    code += R"(
    bool4 m = args.mask.Read<bool>(t, 0, S, 0);
    v.x = m.x ? v.x : -10000.0f;
    v.y = m.y ? v.y : -10000.0f;
    v.z = m.z ? v.z : -10000.0f;
    v.w = m.w ? v.w : -10000.0f;
)";
  } else if (apply_mask) {
    code += R"(
    v += args.mask.Read<float>(t, 0, S, 0);
)";
  }
  if (src == "logits_past") {
    code += R"(
    int c = s * 4;
    v.x = c + 0 < past_filled ? v.x : -3.0e38f;
    v.y = c + 1 < past_filled ? v.y : -3.0e38f;
    v.z = c + 2 < past_filled ? v.z : -3.0e38f;
    v.w = c + 3 < past_filled ? v.w : -3.0e38f;
)";
  }
  if (src == "logits_new") {
    // Channels past T in the last slice do not exist; exp() maps them to 0.
    code += R"(
    int c = s * 4;
    v.y = c + 1 < args.new_len ? v.y : -3.0e38f;
    v.z = c + 2 < args.new_len ? v.z : -3.0e38f;
    v.w = c + 3 < args.new_len ? v.w : -3.0e38f;
)";
  }
  return code;
}

// A GPUOperation with a fixed work group size, for kernels that reduce across
// the work items of a work group.
class FixedWorkGroupOp : public ::ml_drift::GPUOperation {
 public:
  explicit FixedWorkGroupOp(const ::ml_drift::int3& work_group_size) {
    work_group_size_ = work_group_size;
  }

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

  FixedWorkGroupOp(FixedWorkGroupOp&&) = default;
  FixedWorkGroupOp& operator=(FixedWorkGroupOp&&) = default;
  FixedWorkGroupOp(const FixedWorkGroupOp&) = delete;
  FixedWorkGroupOp& operator=(const FixedWorkGroupOp&) = delete;
};

// Soft-caps and masks [logits_past | logits_new] with the [1, 1, T, W + T]
// mask and computes the softmax over all W + T keys. Query rows are packed
// g-major, so row r uses mask row r % T.
//
// Every row is handled by up to 2 work items of one work group: pass 1 reduces
// the row to its max and sum of exponentials (online softmax, F32), pass 2
// re-reads the logits and writes the probabilities. Only the cache slots below
// the runtime bound of PastEndSliceCode() are visited.
//
// The new-token block is produced in one of two ways:
//  * `dst_new` holds probabilities [1, H, R, T] (written in pass 2), or
//  * with `defer_new`, `logits_new` must already be soft-capped and masked,
//    and `dst_new` receives the row statistics [1, H, R, 4] = (1 / sum, max)
//    instead. The consumer applies exp(v - max) / sum while it reads the
//    logits (`src_exp` of the ML Drift matmuls), so the T new-token columns
//    are read once here and never written.
std::unique_ptr<::ml_drift::GPUOperation> CreateMaskedSoftmaxOp(
    const ::ml_drift::TensorDescriptor& logits_past,
    const ::ml_drift::TensorDescriptor& logits_new,
    const ::ml_drift::TensorDescriptor& mask,
    const ::ml_drift::TensorDescriptor& param,
    const ::ml_drift::TensorDescriptor& probs_past,
    const ::ml_drift::TensorDescriptor& dst_new, int cache_size, int new_len,
    const FusedSdpaCacheUpdateAttributes& attr, bool defer_new) {
  const int past_slices = cache_size / 4;
  const int new_slices = (new_len + 3) / 4;
  // Tensors are contiguous along rows within a slice, so a work group spans
  // many rows and splits each row over only a few work items (every work item
  // gets at least 4 slices of its row).
  int threads_per_row = 1;
  while (threads_per_row < 2 &&
         threads_per_row * 8 <= past_slices + new_slices) {
    threads_per_row *= 2;
  }
  const int rows_per_group = 128 / threads_per_row;
  FixedWorkGroupOp op(::ml_drift::int3(rows_per_group, 1, threads_per_row));
  op.tensor_to_grid_ = ::ml_drift::TensorToGrid::kWBToX_HDToY_ZIs1;
  op.AddSrcTensor("logits_past", logits_past);
  op.AddSrcTensor("logits_new", logits_new);
  op.AddSrcTensor("mask", mask);
  op.AddSrcTensor("params", param);
  op.AddDstTensor("probs_past", probs_past);
  op.AddDstTensor(defer_new ? "stats" : "probs_new", dst_new);
  op.args_.AddInt("past_slices", past_slices);
  op.args_.AddInt("new_slices", new_slices);
  op.args_.AddInt("new_len", new_len);
  op.args_.AddInt("zero_slices",
                  ::ml_drift::ConvRuntimeCheckDesc().GetSlicesAlignment());

  const bool has_softcap = attr.softcap.has_value() && *attr.softcap > 0.0f;
  if (has_softcap) {
    op.args_.AddFloat("softcap", *attr.softcap);
    op.args_.AddFloat("inv_softcap", 1.0f / *attr.softcap);
  }
  const bool bool_mask = mask.GetDataType() == ::ml_drift::DataType::kBool;
  const std::string read_past =
      ReadMaskedLogitsCode("logits_past", has_softcap, bool_mask);
  const std::string read_new = ReadMaskedLogitsCode(
      "logits_new", has_softcap, bool_mask, /*apply_mask=*/!defer_new);

  // The trailing batch coordinate is ignored by tensors without a batch axis.
  // Work items of rows past the tensor bounds still take part in the barriers.
  std::string code = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int lz = ucl::GetLocalId<2>();
  int lxy = ucl::GetLocalId<1>() * ROWS_PER_GROUP + ucl::GetLocalId<0>();
  __local float2 loc_mem[THREADS_PER_ROW][ROWS_PER_GROUP];
  bool active = X < args.probs_past.Width() && Y < args.probs_past.Height();
  int t = X % args.new_len;
)" + PastEndSliceCode("args.params.Read(0, 0, 0, 0).x", "args.past_slices") +
                     R"(
  float maximum = -3.0e38f;
  float sum = 0.0f;
  if (active) {
    for (int s = lz; s < past_end; s += THREADS_PER_ROW) {
      int S = s;
)" + read_past + R"(
      float new_max = max(max(max(v.x, v.y), max(v.z, v.w)), maximum);
      sum = sum * exp(maximum - new_max) +
            dot(exp(v - ucl::Init<float4>(new_max)), ucl::Init<float4>(1.0f));
      maximum = new_max;
    }
    for (int s = lz; s < args.new_slices; s += THREADS_PER_ROW) {
      int S = args.past_slices + s;
)" + read_new + R"(
      float new_max = max(max(max(v.x, v.y), max(v.z, v.w)), maximum);
      sum = sum * exp(maximum - new_max) +
            dot(exp(v - ucl::Init<float4>(new_max)), ucl::Init<float4>(1.0f));
      maximum = new_max;
    }
  }
  float2 partial;
  partial.x = sum;
  partial.y = maximum;
  loc_mem[lz][lxy] = partial;
  ucl::SyncThreads<WorkGroup, Local>();
  if (lz == 0) {
    for (int i = 1; i < THREADS_PER_ROW; ++i) {
      float2 other = loc_mem[i][lxy];
      float new_max = max(partial.y, other.y);
      partial.x = partial.x * exp(partial.y - new_max) +
                  other.x * exp(other.y - new_max);
      partial.y = new_max;
    }
    loc_mem[0][lxy] = partial;
  }
  ucl::SyncThreads<WorkGroup, Local>();
  if (!active) {
    return;
  }
  partial = loc_mem[0][lxy];
  float4 row_max = ucl::Init<float4>(partial.y);
  float4 inv_sum = ucl::Init<float4>(1.0f / partial.x);
  for (int s = lz; s < past_end; s += THREADS_PER_ROW) {
    int S = s;
)" + read_past + R"(
    float4 p = exp(v - row_max) * inv_sum;
    args.probs_past.Write(ucl::Convert<args.probs_past::type>(p), X, Y, s);
  }
  // The bounded P.V product steps over its input in blocks of slices that
  // divide the bound alignment, and its do-while loop reads one block even
  // for a bound of 0. So only the first aligned block past the bound needs an
  // explicit probability of 0; later slices are never read.
  int zero_end = min(past_end + args.zero_slices, args.past_slices);
  for (int s = past_end + lz; s < zero_end; s += THREADS_PER_ROW) {
    args.probs_past.Write(ucl::Init<args.probs_past::type>(0.0f), X, Y, s);
  }
)";
  if (defer_new) {
    code += R"(
  if (lz == 0) {
    float4 stats = ucl::Init<float4>(1.0f / partial.x, partial.y, 0.0f, 0.0f);
    args.stats.Write(ucl::Convert<args.stats::type>(stats), X, Y, 0);
  }
}
)";
  } else {
    code += R"(
  for (int s = lz; s < args.new_slices; s += THREADS_PER_ROW) {
    int S = args.past_slices + s;
)" + read_new +
            R"(
    float4 p = exp(v - row_max) * inv_sum;
    args.probs_new.Write(ucl::Convert<args.probs_new::type>(p), X, Y, s);
  }
}
)";
  }
  op.code_ = absl::StrReplaceAll(
      code, {{"ROWS_PER_GROUP", std::to_string(rows_per_group)},
             {"THREADS_PER_ROW", std::to_string(threads_per_row)}});
  return std::make_unique<FixedWorkGroupOp>(std::move(op));
}

// Applies the new-token columns [W, W + T) of the [1, 1, T, W + T] mask to
// `logits_new` [1, H, R, T]: `mask ? logits : -10000` for a bool mask and
// `logits + mask` for an additive one. Row r reads mask row r % T. Being
// elementwise, it is linked into the epilogue of the QK^T matmul.
TensorHandle MaskNewLogits(::ml_drift::GpuModelBuilder* model_builder,
                           const TensorHandle& logits_new,
                           const TensorHandle& mask, int cache_size,
                           int new_len) {
  ::ml_drift::ElementwiseDescriptor op_desc;
  // The mask is read explicitly (no `in2_value`), at a slice offset of W / 4.
  // The trailing batch coordinate is ignored by tensors without a batch axis.
  const std::string read_mask = absl::StrReplaceAll(
      "args.src_tensor_1.Read<TYPE>(X_COORD % NEW_LEN, 0, S_COORD + "
      "PAST_SLICES, 0)",
      {{"NEW_LEN", std::to_string(new_len)},
       {"PAST_SLICES", std::to_string(cache_size / 4)}});
  if (mask.tensor_desc.GetDataType() == ::ml_drift::DataType::kBool) {
    op_desc.args.AddFloat("mask_value", -10000.0f,
                          logits_new.tensor_desc.GetDataType());
    op_desc.code =
        "  bool4 m = " + absl::StrReplaceAll(read_mask, {{"TYPE", "bool"}}) +
        R"(;
  out_value.x = m.x ? in_value.x : args.mask_value;
  out_value.y = m.y ? in_value.y : args.mask_value;
  out_value.z = m.z ? in_value.z : args.mask_value;
  out_value.w = m.w ? in_value.w : args.mask_value;
)";
  } else {
    // `dst_tensor` does not exist once the op is linked, so the sum is formed
    // per component.
    op_desc.code =
        "  float4 m = " + absl::StrReplaceAll(read_mask, {{"TYPE", "float"}}) +
        R"(;
  out_value.x = in_value.x + m.x;
  out_value.y = in_value.y + m.y;
  out_value.z = in_value.z + m.z;
  out_value.w = in_value.w + m.w;
)";
  }
  TensorHandle dst = model_builder->AddTensor(logits_new.tensor_desc);
  ::ml_drift::OperationDef definition;
  definition.src_tensors.push_back(logits_new.tensor_desc);
  definition.src_tensors.push_back(mask.tensor_desc);
  definition.dst_tensors.push_back(dst.tensor_desc);
  auto op = std::make_unique<::ml_drift::GPUOperation>(
      ::ml_drift::CreateGpuOperation(definition, std::move(op_desc)));
  model_builder->AddGpuOperation({logits_new, mask}, {dst}, std::move(op),
                                 "fused_sdpa_mask_new");
  return dst;
}

// A GPUOperation with a fixed grid of work items.
class FixedGridOp : public ::ml_drift::GPUOperation {
 public:
  explicit FixedGridOp(const ::ml_drift::int3& grid_size)
      : fixed_grid_size_(grid_size) {}

  ::ml_drift::int3 GetGridSize() const override { return fixed_grid_size_; }

  FixedGridOp(FixedGridOp&&) = default;
  FixedGridOp& operator=(FixedGridOp&&) = default;
  FixedGridOp(const FixedGridOp&) = delete;
  FixedGridOp& operator=(const FixedGridOp&) = delete;

 private:
  ::ml_drift::int3 fixed_grid_size_;
};

// Writes the last min(valid, W) valid new tokens into the ring buffer, where
// valid = clamp(param[1] - param[0], 0, T) and token X goes to slot
// (param[0] + X) % W. The caches use the packed layouts of AddValuesToCache:
//   key cache   - kOSpatialIOGroupO4I4, O = cache size, I = head dim.
//   value cache - kOSpatialIOGroupI4O4, O = head dim, I = cache size.
// Slots that are not written keep their contents, which relies on the updated
// caches being bound in place to the input caches (as for odml.cache_update).
//
// `attention` is not read. It makes the write data-dependent on the attention
// output so that it can only run after every read of the pre-write caches.
std::unique_ptr<::ml_drift::GPUOperation> CreateRingCacheWriteOp(
    const ::ml_drift::TensorDescriptor& key_new,
    const ::ml_drift::TensorDescriptor& value_new,
    const ::ml_drift::TensorDescriptor& param,
    const ::ml_drift::TensorDescriptor& attention,
    const ::ml_drift::TensorDescriptor& key_cache_out,
    const ::ml_drift::TensorDescriptor& value_cache_out, int heads,
    int cache_size, int head_dim, int new_len) {
  const int head_dim_slices = head_dim / 4;
  FixedGridOp op(::ml_drift::int3(new_len, heads, head_dim_slices));
  op.AddSrcTensor("key_new", key_new);
  op.AddSrcTensor("value_new", value_new);
  op.AddSrcTensor("params", param);
  op.AddSrcTensor("attention", attention);
  op.AddDstTensor("key_cache_out", key_cache_out);
  op.AddDstTensor("value_cache_out", value_cache_out);
  op.args_.AddInt("new_len", new_len);
  op.args_.AddInt("heads", heads);
  op.args_.AddInt("head_dim_slices", head_dim_slices);
  op.args_.AddInt("cache_size", cache_size);
  op.args_.AddInt("cache_slices", cache_size / 4);

  op.code_ = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.new_len || Y >= args.heads || S >= args.head_dim_slices) {
    return;
  }
  // The trailing batch coordinate is ignored by tensors without a batch axis.
  int4 p = args.params.Read(0, 0, 0, 0);
  int start = p.x;
  if (start < 0) {
    return;
  }
  int valid = min(max(p.y - start, 0), args.new_len);
  int first = max(0, valid - args.cache_size);
  if (X < first || X >= valid) {
    return;
  }
  int slot = (start + X) % args.cache_size;
  int slot_slice = slot / 4;
  int slot_lane = slot % 4;

  // Key: new [Hkv, T, D] -> vec4 of 4 consecutive head-dim values.
  args.key_new::type k_val = args.key_new.Read(X, Y, S, 0);
  int k_index = ((Y * args.head_dim_slices + S) * args.cache_slices + slot_slice) * 4 + slot_lane;
  args.key_cache_out.WriteLinear(ucl::Convert<args.key_cache_out::type>(k_val), k_index);

  // Value: new [Hkv, D, T] -> gather 4 consecutive head-dim values of token X.
  float v0;
  float v1;
  float v2;
  float v3;
  args.value_new.ReadPerChannel<float>(v0, S * 4 + 0, Y, X, 0);
  args.value_new.ReadPerChannel<float>(v1, S * 4 + 1, Y, X, 0);
  args.value_new.ReadPerChannel<float>(v2, S * 4 + 2, Y, X, 0);
  args.value_new.ReadPerChannel<float>(v3, S * 4 + 3, Y, X, 0);
  float4 v_val = ucl::Init<float4>(v0, v1, v2, v3);
  int v_index = ((Y * args.cache_slices + slot_slice) * args.head_dim_slices + S) * 4 + slot_lane;
  args.value_cache_out.WriteLinear(ucl::Convert<args.value_cache_out::type>(v_val), v_index);
}
)";
  return std::make_unique<FixedGridOp>(std::move(op));
}

// Copies `src` [1, H, N, C] into a linear buffer as the weights OHWI(N, H, 1,
// C) in the kOSpatialIOGroupO4I4 layout of the key cache, [H][C/4][N/4][4][4]
// with 4 consecutive C values innermost, so that a FullyConnected reads it as
// external weights (src . weights^T per head). This replaces the transpose and
// the weights conversion of a BatchedMatMul, in a single pass.
absl::StatusOr<::ml_drift::GpuModelBuilder::Weights> PackAsWeights(
    ::ml_drift::GpuModelBuilder* model_builder, const TensorHandle& src,
    ::ml_drift::DataType data_type) {
  const ::ml_drift::BHWC shape = src.tensor_desc.GetBHWCShape();
  const ::ml_drift::OHWI weights_shape(shape.w, shape.h, 1, shape.c);
  TensorHandle packed =
      model_builder->AddTensor(::ml_drift::CreateBhwcTensorDescriptor(
          data_type, ::ml_drift::TensorStorageType::kBuffer, shape));
  FixedGridOp op(::ml_drift::int3(shape.w, shape.h, shape.c / 4));
  op.AddSrcTensor("src", src.tensor_desc);
  op.AddDstTensor("dst", packed.tensor_desc);
  op.args_.AddInt("n", shape.w);
  op.args_.AddInt("heads", shape.h);
  op.args_.AddInt("c_slices", shape.c / 4);
  op.code_ = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.n || Y >= args.heads || S >= args.c_slices) {
    return;
  }
  // The trailing batch coordinate is ignored by tensors without a batch axis.
  args.src::type v = args.src.Read(X, Y, S, 0);
  int index = ((Y * args.c_slices + S) * (args.n / 4) + X / 4) * 4 + X % 4;
  args.dst.WriteLinear(ucl::Convert<args.dst::type>(v), index);
}
)";
  model_builder->AddGpuOperation({src}, {packed},
                                 std::make_unique<FixedGridOp>(std::move(op)),
                                 "fused_sdpa_pack_weights");
  ::ml_drift::WeightsDescription desc =
      ::ml_drift::GetFullyConnectedWeightsDesc(data_type, weights_shape);
  desc.layout = ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4;
  return ::ml_drift::CreateExternalWeights(packed, desc, weights_shape);
}

}  // namespace

absl::Status BuildFusedSdpaCacheUpdateGpuGraph(
    const std::vector<uint32_t>& input_ids,
    const std::vector<uint32_t>& output_ids,
    const FusedSdpaCacheUpdateAttributes& attr,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (input_ids.size() != 7) {
    return absl::InvalidArgumentError(
        "odml.fused_sdpa_cache_update expects 7 inputs.");
  }
  if (output_ids.size() != 1 && output_ids.size() != 3) {
    return absl::InvalidArgumentError(
        "odml.fused_sdpa_cache_update expects 1 or 3 outputs.");
  }
  ABSL_ASSIGN_OR_RETURN(TensorHandle query,
                        model_builder->GetTensor(input_ids[0]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle key_cache,
                        model_builder->GetTensor(input_ids[1]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle value_cache,
                        model_builder->GetTensor(input_ids[2]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle key_new,
                        model_builder->GetTensor(input_ids[3]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle value_new,
                        model_builder->GetTensor(input_ids[4]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle mask,
                        model_builder->GetTensor(input_ids[5]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle param,
                        model_builder->GetTensor(input_ids[6]));

  const ::ml_drift::BHWC q_shape = query.tensor_desc.GetBHWCShape();
  const ::ml_drift::BHWC kc_shape = key_cache.tensor_desc.GetBHWCShape();
  const ::ml_drift::BHWC vc_shape = value_cache.tensor_desc.GetBHWCShape();
  const ::ml_drift::BHWC kn_shape = key_new.tensor_desc.GetBHWCShape();
  const ::ml_drift::BHWC vn_shape = value_new.tensor_desc.GetBHWCShape();
  const ::ml_drift::BHWC mask_shape = mask.tensor_desc.GetBHWCShape();
  const int heads = kc_shape.h;
  const int cache_size = kc_shape.w;
  const int head_dim = kc_shape.c;
  const int new_len = kn_shape.w;
  const int rows = q_shape.w;
  if (q_shape != ::ml_drift::BHWC(1, heads, rows, head_dim) ||
      kc_shape.b != 1 ||
      vc_shape != ::ml_drift::BHWC(1, heads, head_dim, cache_size) ||
      kn_shape != ::ml_drift::BHWC(1, heads, new_len, head_dim) ||
      vn_shape != ::ml_drift::BHWC(1, heads, head_dim, new_len) ||
      mask_shape != ::ml_drift::BHWC(1, 1, new_len, cache_size + new_len) ||
      new_len <= 0 || rows % new_len != 0) {
    return absl::InvalidArgumentError(
        "odml.fused_sdpa_cache_update has inconsistent input shapes.");
  }
  if (cache_size % 4 != 0 || head_dim % 4 != 0) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update requires cache size and head dim to be "
        "multiples of 4.");
  }
  // The caches are read as external FullyConnected weights and written with
  // WriteLinear in their packed layouts, both of which need linear buffers.
  if (key_cache.tensor_desc.GetStorageType() !=
          ::ml_drift::TensorStorageType::kBuffer ||
      value_cache.tensor_desc.GetStorageType() !=
          ::ml_drift::TensorStorageType::kBuffer) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update requires BUFFER storage for the KV "
        "cache.");
  }

  const ::ml_drift::DataType data_type = query.tensor_desc.GetDataType();
  ::ml_drift::BatchedMatMulAttributes bmm_attr;
  bmm_attr.transpose_left = false;
  bmm_attr.transpose_right = true;

  // 1. Logits against the pre-write cache: [1, H, R, D] x [W, D]^T per head,
  //    only for the slots below the runtime bound (see PastEndSliceCode).
  ::ml_drift::ConvRuntimeCheckDesc past_logits_check;
  past_logits_check.dst_end_ch_index = kStartParamIndex;
  const ::ml_drift::OHWI k_weights_shape(cache_size, heads, 1, head_dim);
  ::ml_drift::WeightsDescription k_weights_desc =
      ::ml_drift::GetFullyConnectedWeightsDesc(data_type, k_weights_shape);
  k_weights_desc.layout = ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4;
  const ::ml_drift::GpuModelBuilder::Weights k_weights =
      ::ml_drift::CreateExternalWeights(key_cache, k_weights_desc,
                                        k_weights_shape);
  ABSL_ASSIGN_OR_RETURN(TensorHandle logits_past,
                        model_builder->FullyConnectedExternalWeights(
                            query, k_weights, /*biases=*/nullptr,
                            /*src_exp=*/nullptr, past_logits_check, &param));

  // 2. Logits against the new tokens: [1, H, R, D] x [1, H, T, D]^T.
  //
  // For multi-token chunks the new keys and values are packed as weights (see
  // PackAsWeights) for the same FullyConnected kernels as the cache, and the
  // new-token block is not materialized as probabilities: its logits are
  // soft-capped and masked in the epilogue of the QK^T matmul, the softmax op
  // only reduces them to the row max / sum, and the P_new . V_new matmul
  // applies exp(v - max) / sum on load. The single-token (decode) graph keeps
  // the materialized probabilities and the BatchedMatMuls.
  const bool defer_new = new_len > 1 && new_len % 4 == 0;
  TensorHandle logits_new;
  ::ml_drift::GpuModelBuilder::Weights v_new_weights;
  if (defer_new) {
    ABSL_ASSIGN_OR_RETURN(
        const ::ml_drift::GpuModelBuilder::Weights k_new_weights,
        PackAsWeights(model_builder, key_new, data_type));
    ABSL_ASSIGN_OR_RETURN(v_new_weights,
                          PackAsWeights(model_builder, value_new, data_type));
    ABSL_ASSIGN_OR_RETURN(
        logits_new,
        model_builder->FullyConnectedExternalWeights(query, k_new_weights));
    if (attr.softcap.has_value() && *attr.softcap > 0.0f) {
      const float cap = *attr.softcap;
      logits_new = model_builder->Multiplication(logits_new, 1.0f / cap);
      logits_new = model_builder->Elementwise(logits_new,
                                              ::ml_drift::OperationType::kTanh);
      logits_new = model_builder->Multiplication(logits_new, cap);
    }
    logits_new =
        MaskNewLogits(model_builder, logits_new, mask, cache_size, new_len);
  } else {
    ABSL_ASSIGN_OR_RETURN(
        logits_new, model_builder->BatchedMatMul(query, key_new, bmm_attr));
  }

  // 3. Soft-capped, masked softmax over all W + T keys: the cache block as
  //    probabilities, the new-token block as probabilities or statistics.
  TensorHandle probs_past = model_builder->AddTensor(
      ::ml_drift::BHWC(1, heads, rows, cache_size), data_type);
  TensorHandle dst_new = model_builder->AddTensor(
      ::ml_drift::BHWC(1, heads, rows, defer_new ? 4 : new_len), data_type);
  model_builder->AddGpuOperation(
      {logits_past, logits_new, mask, param}, {probs_past, dst_new},
      CreateMaskedSoftmaxOp(logits_past.tensor_desc, logits_new.tensor_desc,
                            mask.tensor_desc, param.tensor_desc,
                            probs_past.tensor_desc, dst_new.tensor_desc,
                            cache_size, new_len, attr, defer_new),
      "fused_sdpa_masked_softmax");

  // 4. out = P_new . V_new + P_past . V_cache. The new-token product is
  //    computed first so that the Add can be linked into the FullyConnected.
  TensorHandle out_new;
  if (defer_new) {
    ABSL_ASSIGN_OR_RETURN(out_new,
                          model_builder->FullyConnectedExternalWeights(
                              logits_new, v_new_weights, /*biases=*/nullptr,
                              /*src_exp=*/&dst_new));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        out_new, model_builder->BatchedMatMul(dst_new, value_new, bmm_attr));
  }
  const ::ml_drift::OHWI v_weights_shape(head_dim, heads, 1, cache_size);
  ::ml_drift::WeightsDescription v_weights_desc =
      ::ml_drift::GetFullyConnectedWeightsDesc(data_type, v_weights_shape);
  v_weights_desc.layout = ::ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  const ::ml_drift::GpuModelBuilder::Weights v_weights =
      ::ml_drift::CreateExternalWeights(value_cache, v_weights_desc,
                                        v_weights_shape);
  ::ml_drift::ConvRuntimeCheckDesc past_values_check;
  past_values_check.src_end_ch_index = kStartParamIndex;
  ABSL_ASSIGN_OR_RETURN(TensorHandle out_past,
                        model_builder->FullyConnectedExternalWeights(
                            probs_past, v_weights, /*biases=*/nullptr,
                            /*src_exp=*/nullptr, past_values_check, &param));
  TensorHandle attention = model_builder->Add(out_past, out_new);

  ABSL_ASSIGN_OR_RETURN(TensorHandle attention_ref,
                        model_builder->GetTensor(output_ids[0]));
  const ::ml_drift::BHWC out_shape = attention_ref.tensor_desc.GetBHWCShape();
  if (attention.tensor_desc.GetBHWCShape() != out_shape) {
    attention = model_builder->Reshape(attention, out_shape);
  }
  ABSL_RETURN_IF_ERROR(
      model_builder->UpdateOutputTensor(attention, output_ids[0]));
  // The output count decides whether the ring buffer is written.
  if (output_ids.size() == 1) {
    return absl::OkStatus();
  }

  // 5. Ring buffer write, strictly after the attention above.
  ABSL_ASSIGN_OR_RETURN(TensorHandle key_cache_ref,
                        model_builder->GetTensor(output_ids[1]));
  ABSL_ASSIGN_OR_RETURN(TensorHandle value_cache_ref,
                        model_builder->GetTensor(output_ids[2]));
  if (key_cache_ref.tensor_desc.GetStorageType() !=
          ::ml_drift::TensorStorageType::kBuffer ||
      value_cache_ref.tensor_desc.GetStorageType() !=
          ::ml_drift::TensorStorageType::kBuffer) {
    return absl::UnimplementedError(
        "odml.fused_sdpa_cache_update requires BUFFER storage for the updated "
        "KV cache.");
  }
  ABSL_ASSIGN_OR_RETURN(TensorHandle attention_out,
                        model_builder->GetTensor(output_ids[0]));
  TensorHandle key_cache_out =
      model_builder->AddTensor(key_cache_ref.tensor_desc);
  TensorHandle value_cache_out =
      model_builder->AddTensor(value_cache_ref.tensor_desc);
  model_builder->AddGpuOperation(
      {key_new, value_new, param, attention_out},
      {key_cache_out, value_cache_out},
      CreateRingCacheWriteOp(key_new.tensor_desc, value_new.tensor_desc,
                             param.tensor_desc, attention_out.tensor_desc,
                             key_cache_out.tensor_desc,
                             value_cache_out.tensor_desc, heads, cache_size,
                             head_dim, new_len),
      "fused_sdpa_ring_cache_write");
  return model_builder->UpdateOutputTensors({key_cache_out, value_cache_out},
                                            {output_ids[1], output_ids[2]});
}

absl::Status CreateFusedSdpaCacheUpdateFromNode(
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  const auto* attr =
      std::any_cast<FusedSdpaCacheUpdateAttributes>(&node.operation.attributes);
  if (attr == nullptr) {
    return absl::InvalidArgumentError(
        "Missing odml.fused_sdpa_cache_update attributes.");
  }
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  std::vector<uint32_t> output_ids;
  output_ids.reserve(outputs.size());
  for (const auto* output : outputs) output_ids.push_back(output->id);
  return BuildFusedSdpaCacheUpdateGpuGraph(input_ids, output_ids, *attr,
                                           model_builder);
}

}  // namespace litert::ml_drift

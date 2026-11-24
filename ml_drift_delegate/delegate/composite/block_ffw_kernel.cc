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

#include "ml_drift_delegate/delegate/composite/block_ffw_kernel.h"

#include <algorithm>
#include <any>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/substitute.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/flops_util.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/kernel_info.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected_util.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/model_hints.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tuning_type.h"  // from @ml_drift
#include "ml_drift/common/task/util.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/block_ffw_parser.h"

namespace litert::ml_drift {
namespace {

using ::ml_drift::Axis;
using ::ml_drift::BHWC;
using ::ml_drift::DataType;
using ::ml_drift::DivideRoundUp;
using ::ml_drift::ElementwiseDescriptor;
using ::ml_drift::ExternalWeights;
using ::ml_drift::GpuInfo;
using ::ml_drift::GpuModelBuilder;
using ::ml_drift::GPUOperation;
using ::ml_drift::GraphFloat32;
using ::ml_drift::Node;
using ::ml_drift::NodeId;
using ::ml_drift::OHWI;
using ::ml_drift::OperationDef;
using ::ml_drift::OperationType;
using ::ml_drift::QuantizeAndDequantizeAttributes;
using ::ml_drift::TensorDescriptor;
using ::ml_drift::TensorStorageType;
using ::ml_drift::Value;
using ::ml_drift::ValueId;
using ::ml_drift::WeightsDescription;
using ::ml_drift::ir::IrOp;
using ::ml_drift::ir::IrTensor;

template <typename TensorT>
std::vector<ValueId> GetValueIds(const std::vector<TensorT*>& values) {
  std::vector<ValueId> ids;
  ids.reserve(values.size());
  for (const auto* v : values) ids.push_back(v->id);
  return ids;
}

void AppendBatchCoordReorder(const OperationDef& op_def, std::string* code) {
  if (op_def.src_tensors[0].HasAxis(Axis::kBatch)) {
    *code += op_def.dst_tensors[0].HasAxis(Axis::kBatch) ? "  SRC_B = DST_B;\n"
                                                         : "  SRC_B = 0;\n";
  }
}

// Returns the maximum number of source part tensors that a single `PackParts`
// dispatch may bind on `gpu_info`, accounting for per-stage descriptor limits
// across buffer and texture/image storage types.
int GetMaxPackPartsInputs(const GpuInfo& gpu_info, const OperationDef& op_def) {
  const int max_src_buffers = gpu_info.GetMaxBufferArguments();
  bool has_non_buffer =
      op_def.dst_tensors[0].GetStorageType() != TensorStorageType::kBuffer;
  for (const auto& src : op_def.src_tensors) {
    if (src.GetStorageType() != TensorStorageType::kBuffer) {
      has_non_buffer = true;
      break;
    }
  }
  int binding_limit = max_src_buffers;
  if (has_non_buffer) {
    // WebGPU binds read-only non-buffer image tensors as storage textures
    // (`GetMaxImageArguments()`), whereas other GPU APIs bind them as sampled
    // textures (`GetMaxTextureArguments()`). Mirrors `GetMaxSrcImages` in
    // `third_party/ml_drift/common/gpu_model_builder.cc`.
    const int max_src_images = gpu_info.IsApiWebGpu()
                                   ? gpu_info.GetMaxImageArguments()
                                   : gpu_info.GetMaxTextureArguments();
    binding_limit = std::min(binding_limit, max_src_images);
  }
  // Mirror GpuModelBuilder::Concat: reserve headroom for dst_tensor + uniforms
  // and cap at 8 on Mali to avoid clEnqueueNDRangeKernel out-of-resources.
  int max_inputs = std::max(2, binding_limit - 4);
  if (gpu_info.IsMali()) {
    max_inputs = std::min(8, max_inputs);
  }
  return max_inputs;
}

// Generates a `GPUOperation` that interleaves `num_parts` branch tensors
// (`[1, 1, T, K2 * S]`) into `[1, K2, T, num_parts * S]`.
//
// Example expanded UCL kernel for `num_parts = 2`, `part_slices = 2`, without
// `Axis::kBatch`:
//   MAIN_FUNCTION($0) {
//     int linear_wb = ucl::GetGlobalId<0>();
//     int Y = ucl::GetGlobalId<1>();
//     int S = ucl::GetGlobalId<2>();
//     int X = linear_wb;
//     if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() ||
//         S >= args.dst_tensor.Slices()) return;
//     args.dst_tensor::type res;
//     int part = S / 2;
//     int src_s = Y * 2 + (S % 2);
//     if (part == 0) {
//       res = args.src_tensor_0.Read(X, 0, src_s);
//     } else {
//       res = args.src_tensor_1.Read(X, 0, src_s);
//     }
//     args.dst_tensor.Write(res, X, Y, S);
//   }
std::unique_ptr<GPUOperation> CreatePackPartsOperation(
    const OperationDef& op_def, int num_parts) {
  auto op = std::make_unique<GPUOperation>();
  op->tensor_to_grid_ = ::ml_drift::TensorToGrid::kWBToX_HDToY_SToZ;
  for (int k = 0; k < num_parts; ++k) {
    op->AddSrcTensor(absl::StrCat("src_tensor_", k), op_def.src_tensors[k]);
  }
  op->AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  const BHWC dst_shape = op_def.dst_tensors[0].GetBHWCShape();
  const int part_slices = (dst_shape.c / num_parts) / 4;
  const bool dst_has_batch = op_def.dst_tensors[0].HasAxis(Axis::kBatch);
  const bool src_has_batch = op_def.src_tensors[0].HasAxis(Axis::kBatch);
  const std::string read_coords =
      src_has_batch ? (dst_has_batch ? "X, 0, src_s, B" : "X, 0, src_s, 0")
                    : "X, 0, src_s";

  std::string c = "MAIN_FUNCTION($0) {\n";
  c += "  int linear_wb = ucl::GetGlobalId<0>();\n";
  c += "  int Y = ucl::GetGlobalId<1>();\n";
  c += "  int S = ucl::GetGlobalId<2>();\n";
  if (dst_has_batch) {
    c += "  int X = linear_wb / args.dst_tensor.Batch();\n";
    c += "  int B = linear_wb % args.dst_tensor.Batch();\n";
  } else {
    c += "  int X = linear_wb;\n";
  }
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) return;\n";
  c += "  args.dst_tensor::type res;\n";
  if (num_parts == 1) {
    absl::StrAppend(&c, "  int src_s = Y * ", part_slices, " + S;\n");
    absl::StrAppend(&c, "  res = args.src_tensor_0.Read(", read_coords, ");\n");
  } else {
    absl::StrAppend(&c, "  int part = S / ", part_slices, ";\n");
    absl::StrAppend(&c, "  int src_s = Y * ", part_slices, " + (S % ",
                    part_slices, ");\n");
    for (int k = 0; k < num_parts; ++k) {
      if (k == 0) {
        c += "  if (part == 0) {\n";
      } else if (k + 1 < num_parts) {
        absl::StrAppend(&c, "  } else if (part == ", k, ") {\n");
      } else {
        c += "  } else {\n";
      }
      absl::StrAppend(&c, "    res = args.src_tensor_", k, ".Read(",
                      read_coords, ");\n");
    }
    c += "  }\n";
  }
  if (dst_has_batch) {
    c += "  args.dst_tensor.Write(res, X, Y, S, B);\n";
  } else {
    c += "  args.dst_tensor.Write(res, X, Y, S);\n";
  }
  c += "}\n";
  op->code_ = std::move(c);
  return op;
}

// Emits one `PackParts` dispatch when `attr.num_parts` fits within the device's
// per-stage binding limit; otherwise chunks inputs into groups of at most
// `max_inputs`, packs each chunk into a partial tensor, and joins the partials
// along channels via `GpuModelBuilder::Concat`.
absl::Status AddPackPartsOperations(const OperationDef& op_def,
                                    const BlockFfwPackPartsAttributes& attr,
                                    const std::vector<ValueId>& src_ids,
                                    const std::vector<ValueId>& dst_ids,
                                    GpuModelBuilder* model_builder) {
  const int max_inputs =
      GetMaxPackPartsInputs(model_builder->gpu_info(), op_def);
  if (attr.num_parts <= max_inputs) {
    model_builder->AddGpuOperation(
        src_ids, dst_ids, CreatePackPartsOperation(op_def, attr.num_parts),
        std::string(kBlockFfwPackPartsType));
    return absl::OkStatus();
  }
  const BHWC dst_shape = op_def.dst_tensors[0].GetBHWCShape();
  const int part_channels = dst_shape.c / attr.num_parts;
  std::vector<GpuModelBuilder::TensorHandle> partials;
  partials.reserve(DivideRoundUp(attr.num_parts, max_inputs));
  for (int start = 0; start < attr.num_parts; start += max_inputs) {
    const int chunk_parts = std::min(max_inputs, attr.num_parts - start);
    TensorDescriptor chunk_dst_desc;
    op_def.dst_tensors[0].CopyWithoutData(&chunk_dst_desc);
    chunk_dst_desc.SetBHWCShape(BHWC(dst_shape.b, dst_shape.h, dst_shape.w,
                                     chunk_parts * part_channels));
    const GpuModelBuilder::TensorHandle chunk_dst =
        model_builder->AddTensor(chunk_dst_desc);

    OperationDef chunk_op_def;
    std::vector<ValueId> chunk_src_ids;
    chunk_src_ids.reserve(chunk_parts);
    chunk_op_def.src_tensors.reserve(chunk_parts);
    for (int k = 0; k < chunk_parts; ++k) {
      chunk_src_ids.push_back(src_ids[start + k]);
      chunk_op_def.src_tensors.push_back(op_def.src_tensors[start + k]);
    }
    chunk_op_def.dst_tensors.push_back(chunk_dst.tensor_desc);
    model_builder->AddGpuOperation(
        chunk_src_ids, {chunk_dst.id},
        CreatePackPartsOperation(chunk_op_def, chunk_parts),
        std::string(kBlockFfwPackPartsType));
    partials.push_back(chunk_dst);
  }
  return model_builder->UpdateOutputTensor(
      model_builder->Concat(partials, Axis::kChannels), dst_ids[0]);
}

std::unique_ptr<GPUOperation> CreateUnpackPartOperation(
    const OperationDef& op_def, const BlockFfwUnpackPartAttributes& attr) {
  const BHWC src_shape = op_def.src_tensors[0].GetBHWCShape();
  const int part_slices = (src_shape.c / attr.num_parts) / 4;
  const int offset_slices = attr.part * part_slices;
  std::string code = "  SRC_X = DST_X;\n";
  if (src_shape.h == 1) {
    code += "  SRC_Y = 0;\n";
    absl::StrAppend(&code, "  SRC_S = ", offset_slices, " + (DST_S);\n");
  } else {
    absl::StrAppend(&code, "  SRC_Y = (DST_S) / ", part_slices, ";\n");
    absl::StrAppend(&code, "  SRC_S = ", offset_slices, " + ((DST_S) % ",
                    part_slices, ");\n");
  }
  AppendBatchCoordReorder(op_def, &code);
  return std::make_unique<GPUOperation>(
      ::ml_drift::CreateReorderGpuOperation(op_def, std::move(code)));
}

// Emits a single-input linkable SiLU (`x / (1 + exp(-x))`) elementwise op.
// While `MergeElementwiseNodes` `TYPE_2` can fold `GpuModelBuilder::SiLU`'s
// two-op `Elementwise(kSigmoid)` + `Multiplication` pattern directly into a
// non-linkable producer (such as `FullyConnected`), a preceding linkable
// `QuantizeAndDequantize` prevents `TYPE_2` fusion
// (`!prev_first_node.gpu_operation->IsLinkable()`) and leaves
// `next_nodes.size() == 2` so `LinkNodes()` cannot fuse it either. A
// single-input `ElementwiseDescriptor` fuses cleanly after QDQ with zero extra
// dispatches and evaluates `exp(-v)` in FP32 to avoid FP16 overflow on negative
// inputs.
std::unique_ptr<GPUOperation> CreateSiluOperation(const OperationDef& op_def) {
  ElementwiseDescriptor op_desc;
  const DataType dst_dtype = op_def.dst_tensors[0].GetDataType();
  const std::string vec_type = ::ml_drift::ToUclDataType(dst_dtype, 4);
  op_desc.code = absl::Substitute(R"(
  float4 v_f32 = ucl::Convert<float4>(in_value);
  float4 act_f32 = v_f32 / (1.0f + exp(-v_f32));
  out_value = ucl::Convert<$0>(act_f32);
)",
                                  vec_type);
  auto op = std::make_unique<GPUOperation>(
      ::ml_drift::CreateGpuOperation(op_def, std::move(op_desc)));
  op->flops_ = op_def.dst_tensors[0].GetBHWDCShape().DimensionsProduct() * 4;
  return op;
}

// Fused single-token (`B == 1, W == 1`) GPU kernel for an `odml.blocked_ffw`
// edge stage (`U1` or `D2`). Combines `num_parts` parallel branches of
// `UnpackPart -> [QDQ] -> Quantized FC -> [QDQ]` plus `PackParts -> [QDQ]` into
// one 3-D dispatch of grid `(dst_slices, work_group_size.y, num_parts)`, where
// workgroup dimension Z selects the partition index `part` in `[0, num_parts)`.
class BlockFfwEdgeFcOp : public GPUOperation {
 public:
  BlockFfwEdgeFcOp(int dst_slices, int num_parts)
      : dst_slices_(dst_slices), num_parts_(num_parts) {}
  BlockFfwEdgeFcOp(BlockFfwEdgeFcOp&&) = default;
  BlockFfwEdgeFcOp& operator=(BlockFfwEdgeFcOp&&) = default;
  BlockFfwEdgeFcOp(const BlockFfwEdgeFcOp&) = delete;
  BlockFfwEdgeFcOp& operator=(const BlockFfwEdgeFcOp&) = delete;

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(dst_slices_, work_group_size_.y, num_parts_);
  }

 private:
  int dst_slices_ = 0;
  int num_parts_ = 0;
};

std::string RowOf(absl::string_view slice, int rows, int row_slices) {
  return rows == 1 ? "0" : absl::StrCat(slice, " / ", row_slices);
}

std::string SliceOfRow(absl::string_view slice, absl::string_view first_slice,
                       int rows, int row_slices) {
  return rows == 1
             ? absl::StrCat(first_slice, " + ", slice)
             : absl::StrCat(first_slice, " + (", slice, " % ", row_slices, ")");
}

struct BindingCounts {
  int buffer_args = 0;
  int src_image_args = 0;
  int dst_image_args = 0;

  void Record(TensorStorageType storage, bool is_dst) {
    if (storage == TensorStorageType::kBuffer ||
        storage == TensorStorageType::kImageBuffer ||
        storage == TensorStorageType::kUnknown) {
      ++buffer_args;
    } else if (is_dst) {
      ++dst_image_args;
    } else {
      ++src_image_args;
    }
  }
};

// Returns true if fusing all `spec.num_parts` weight/scale/zero-point bindings
// plus `src_desc`/`dst_desc` into one dispatch would exceed `gpu_info`'s
// per-stage buffer or texture/image argument limits.
bool ExceedsEdgeFcBindingLimits(const GpuInfo& gpu_info,
                                const TensorDescriptor& src_desc,
                                const TensorDescriptor& dst_desc,
                                const BlockFfwEdgeFcSpec& spec) {
  // Mirror `GetMaxPackPartsInputs`: cap at 8 parts on Mali to avoid
  // `clEnqueueNDRangeKernel` out-of-resources on wide partition counts.
  if (gpu_info.IsMali() && spec.num_parts > 8) {
    return true;
  }
  BindingCounts counts;
  counts.Record(src_desc.GetStorageType(), /*is_dst=*/false);
  counts.Record(dst_desc.GetStorageType(), /*is_dst=*/true);
  for (const auto& pw : spec.part_weights) {
    if (pw.desc.IsLinearLayout()) {
      ++counts.buffer_args;
    } else {
      ++counts.src_image_args;
    }
    if (pw.scale.has_value()) {
      counts.Record(pw.scale->tensor_desc.GetStorageType(), /*is_dst=*/false);
    }
    if (pw.zero_point.has_value()) {
      counts.Record(pw.zero_point->tensor_desc.GetStorageType(),
                    /*is_dst=*/false);
    }
  }
  // Reserve one buffer slot for kernel uniforms on APIs whose uniform buffer
  // shares the buffer argument limit (Metal / OpenCL).
  const int effective_buffer_args =
      gpu_info.IsApiWebGpu() ? counts.buffer_args : counts.buffer_args + 1;
  if (effective_buffer_args > gpu_info.GetMaxBufferArguments()) {
    return true;
  }
  if (gpu_info.IsApiWebGpu() || gpu_info.IsApiMetal()) {
    // WebGPU binds both read-only and write-only non-buffer tensors as storage
    // textures (`GetMaxImageArguments()`), and Metal shares one
    // `[[texture(n)]]` slot table for both read and write textures.
    return counts.src_image_args + counts.dst_image_args >
           gpu_info.GetMaxImageArguments();
  }
  return counts.src_image_args > gpu_info.GetMaxTextureArguments() ||
         counts.dst_image_args > gpu_info.GetMaxImageArguments();
}

// Appends exact ML Drift QuantizeAndDequantize math on `var_name` (of UCL type
// `ToUclDataType(tensor_dtype, 4)`), matching `CreateQuantizeAndOrDequantize`
// and `UseFp32ForQuantizeAndDequantize` across Metal, OpenCL, WebGPU, Vulkan,
// and OpenGL ES.
//
// Example emitted UCL snippet (`var_name = "v0"`, `prefix = "src_qdq0"`,
// `tensor_dtype = kFloat16`, Apple Metal `qdq_type = kFloat32`, `scale > 0`):
//   {
//     float4 q_in = ucl::Convert<float4>(v0);
//     float4 clamped_v = min(
//         ucl::Init<float4>(args.src_qdq0_max),
//         max(ucl::Init<float4>(args.src_qdq0_min), q_in));
//     float4 centered_q = floor(
//         clamped_v * ucl::Init<float4>(args.src_qdq0_inv_scale) +
//         ucl::Init<float4>(args.src_qdq0_zp_frac_half));
//     centered_q = min(
//         ucl::Init<float4>(args.src_qdq0_q_max),
//         max(ucl::Init<float4>(args.src_qdq0_q_min), centered_q));
//     float4 q_out = centered_q * ucl::Init<float4>(args.src_qdq0_scale) +
//                    ucl::Init<float4>(args.src_qdq0_deq_off);
//     v0 = ucl::Convert<half4>(q_out);
//   }
// When `scale == 0`, emits a constant assignment to `args.<prefix>_min`.
void AppendInlineQuantizeAndDequantize(
    std::string* code, GPUOperation* op, absl::string_view prefix,
    absl::string_view var_name, DataType tensor_dtype, const GpuInfo& gpu_info,
    const QuantizeAndDequantizeAttributes& raw_attr) {
  const bool use_fp32_qdq = tensor_dtype == DataType::kFloat16 &&
                            gpu_info.IsApple() && gpu_info.IsApiMetal();
  const DataType qdq_type = use_fp32_qdq ? DataType::kFloat32 : tensor_dtype;
  QuantizeAndDequantizeAttributes attr = raw_attr;
  if (qdq_type == DataType::kFloat16 && attr.scale > 0.0f &&
      attr.scale < 0.000062f) {
    // The smallest positive normal FP16 value is `2^-14 ~= 6.1035e-5`
    // (`~0.000062f`). Smaller positive scales are subnormal in FP16 and would
    // flush to zero or overflow `1.0f / scale` past FP16 max (65504); matches
    // `CreateQuantizeAndOrDequantize` while keeping `scale == 0.0f` exact.
    attr.scale = 0.000062f;
  }
  const std::string q_vec = ::ml_drift::ToUclDataType(qdq_type, 4);
  const std::string t_vec = ::ml_drift::ToUclDataType(tensor_dtype, 4);

  op->args_.AddFloat(absl::StrCat(prefix, "_min"), attr.min, qdq_type);
  if (attr.scale == 0.0f) {
    *code += absl::Substitute(
        "  $0 = ucl::Convert<$1>(ucl::Init<$2>(args.$3_min));\n", var_name,
        t_vec, q_vec, prefix);
    return;
  }

  const float inverse_scale = 1.0f / attr.scale;
  const float zero_point = -attr.min * inverse_scale;
  const float zero_point_int = std::round(zero_point);
  const float zero_point_frac = zero_point - zero_point_int;
  op->args_.AddFloat(absl::StrCat(prefix, "_max"), attr.max, qdq_type);
  op->args_.AddFloat(absl::StrCat(prefix, "_scale"), attr.scale, qdq_type);
  op->args_.AddFloat(absl::StrCat(prefix, "_inv_scale"), inverse_scale,
                     qdq_type);
  op->args_.AddFloat(absl::StrCat(prefix, "_zp_frac_half"),
                     zero_point_frac + 0.5f, qdq_type);
  op->args_.AddFloat(
      absl::StrCat(prefix, "_q_min"),
      std::floor(attr.min * inverse_scale + zero_point_frac + 0.5f), qdq_type);
  op->args_.AddFloat(
      absl::StrCat(prefix, "_q_max"),
      std::floor(attr.max * inverse_scale + zero_point_frac + 0.5f), qdq_type);
  op->args_.AddFloat(absl::StrCat(prefix, "_deq_off"),
                     -zero_point_frac * attr.scale, qdq_type);

  *code += absl::Substitute(R"(  {
    $2 q_in = ucl::Convert<$2>($0);
    $2 clamped_v = min(ucl::Init<$2>(args.$3_max), max(ucl::Init<$2>(args.$3_min), q_in));
    $2 centered_q = floor(clamped_v * ucl::Init<$2>(args.$3_inv_scale) + ucl::Init<$2>(args.$3_zp_frac_half));
    centered_q = min(ucl::Init<$2>(args.$3_q_max), max(ucl::Init<$2>(args.$3_q_min), centered_q));
    $2 q_out = centered_q * ucl::Init<$2>(args.$3_scale) + ucl::Init<$2>(args.$3_deq_off);
    $0 = ucl::Convert<$1>(q_out);
  }
)",
                            var_name, t_vec, q_vec, prefix);
}

void AddIndexedWeightsArguments(const ExternalWeights& weights, int part_idx,
                                GPUOperation* op) {
  int element_size = 0;
  DataType texture_type = DataType::kUnknown;
  switch (weights.desc.type) {
    case DataType::kUint8:
      element_size = 4;
      texture_type = DataType::kUint32;
      break;
    case DataType::kUint4:
      element_size = 2;
      texture_type = DataType::kUint16;
      break;
    case DataType::kUint2:
      element_size = 1;
      texture_type = DataType::kUint8;
      break;
    default:
      ABSL_LOG(FATAL) << "Unsupported BlockFfwEdgeFc quantized weights type: "
                      << ::ml_drift::ToString(weights.desc.type);
  }
  const std::string w_name = absl::StrCat("weights_", part_idx);
  if (weights.desc.IsLinearLayout()) {
    ::ml_drift::BufferDescriptor desc;
    desc.element_type = DataType::kUint32;
    desc.element_size = element_size;
    op->AddSrcBuffer(w_name, desc);
  } else {
    op->AddSrcTensor(w_name,
                     TensorDescriptor(texture_type,
                                      ::ml_drift::TensorStorageType::kTexture2D,
                                      ::ml_drift::Layout::kHW));
  }
  if (weights.scale != nullptr) {
    op->AddSrcTensor(absl::StrCat("weights_scale_", part_idx), *weights.scale);
  }
  if (weights.zero_point != nullptr) {
    op->AddSrcTensor(absl::StrCat("weights_zero_point_", part_idx),
                     *weights.zero_point);
  }
}

// Emits UCL code to load and unpack one 4x4 `(I4, O4)` tile of "edge weights"
// into `w0..w3`. Here "edge weights" refers to one partition's 2-D
// `[O, 1, 1, I]` quantized weight matrix from the `U1` (Stage 1) or `D2`
// (Stage 4) edge stage of `odml.blocked_ffw`, stored in ML Drift's I4O4 linear
// buffer (`kOSpatialIOGroupI4O4`) or 2-D texture
// (`k2DYIsSpatialIOAndXIsOGroupI4O4`) layout with `output_group_size == O / 4`.
std::string GenerateReadEdgeWeightsCode(DataType weights_type,
                                        const WeightsDescription& weights_desc,
                                        absl::string_view weights_obj_name) {
  std::string c;
  if (weights_desc.IsLinearLayout()) {
    c += "    int linear_i4o4 = src_s * args.dst_slices + dst_s;\n";
    if (weights_type == DataType::kInt2) {
      absl::StrAppend(&c, "    uint w = args.", weights_obj_name,
                      ".Read(linear_i4o4);\n");
      c += "    ucl::U32x1ToU2x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    } else if (weights_type == DataType::kInt4) {
      absl::StrAppend(&c, "    uint2 w = args.", weights_obj_name,
                      ".Read(linear_i4o4);\n");
      c += "    ucl::U32x2ToU4x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    } else {
      absl::StrAppend(&c, "    uint4 w = args.", weights_obj_name,
                      ".Read(linear_i4o4);\n");
      c += "    ucl::U32x4ToU8x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    }
  } else {
    const std::string y_c = "src_s";
    if (weights_type == DataType::kInt8) {
      absl::StrAppend(&c, "    uint4 w = args.", weights_obj_name,
                      ".Read(dst_s, ", y_c, ");\n");
      c += "    ucl::U32x4ToU8x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    } else if (weights_type == DataType::kInt4) {
      absl::StrAppend(&c, "    ushort4 w = args.", weights_obj_name,
                      ".Read(dst_s, ", y_c, ");\n");
      c += "    ucl::U16x4ToU4x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    } else {
      absl::StrAppend(&c, "    uchar4 w = args.", weights_obj_name,
                      ".Read(dst_s, ", y_c, ");\n");
      c += "    ucl::U8x4ToU2x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    }
  }
  return c;
}

std::string RewriteIndexedScaleZpNames(std::string code, int part_idx) {
  return absl::StrReplaceAll(
      code,
      {{"args.weights_scale", absl::StrCat("args.weights_scale_", part_idx)},
       {"args.weights_zero_point",
        absl::StrCat("args.weights_zero_point_", part_idx)}});
}

// Parameters needed to generate the per-partition inner reduction loop in
// `BlockFfwEdgeFcOp`.
struct EdgeFcLoopCodegenParams {
  bool is_block_quantized = false;
  bool has_zp = false;
  bool src_has_batch = false;
  bool use_fma = false;
  int src_height = 1;
  int src_row_slices = 0;
  DataType weights_type = DataType::kUnknown;
  ::ml_drift::CalculationsPrecision precision =
      ::ml_drift::CalculationsPrecision::kF32;
  OHWI scale_zp_shape;
  WeightsDescription weights_desc;
  absl::string_view reduction_size;
  absl::string_view src_qdq_code;
};

// Appends the per-partition input-slice reduction loop for partition `part_idx`
// inside `BlockFfwEdgeFcOp`.
//
// Example emitted UCL loop body for `part_idx = 0`, linear per-channel INT2
// buffer weights with zero-point, `src_height = 1`, `src_row_slices = 4`,
// `reduction_size = "8"`, no source QDQ, FP32 accumulation without FMA:
//   w_scale = args.weights_scale_0.Read(dst_s);
//   float4 w_zp = args.weights_zero_point_0.Read(dst_s);
//   w_bias = -w_scale * (ucl::Init<float4>(2) + w_zp);
//   for (int src_s = tid.y; src_s < args.src_slices; src_s += 8) {
//     float4 v0 = args.src_tensor.Read(0, 0, part * 4 + src_s);
//     float4 w0, w1, w2, w3;
//     int linear_i4o4 = src_s * args.dst_slices + dst_s;
//     uint w = args.weights_0.Read(linear_i4o4);
//     ucl::U32x1ToU2x16AsVec4x4<float>(w, w0, w1, w2, w3);
//     w0 = w0 * w_scale + w_bias;
//     w1 = w1 * w_scale + w_bias;
//     w2 = w2 * w_scale + w_bias;
//     w3 = w3 * w_scale + w_bias;
//     r_sp0 += v0.x * w0;
//     r_sp0 += v0.y * w1;
//     r_sp0 += v0.z * w2;
//     r_sp0 += v0.w * w3;
//   }
// For blockwise scales (`is_block_quantized = true`), an outer loop iterates
// over `src_group_id` (reading `weights_scale_<k>` / `weights_zero_point_<k>`
// per block) and an inner loop covers `src_sub_id` in
// `[0, args.src_group_size)`.
void AppendEdgeFcPartLoopBody(const EdgeFcLoopCodegenParams& params,
                              int part_idx, std::string* code) {
  if (!params.is_block_quantized) {
    *code += RewriteIndexedScaleZpNames(
        ::ml_drift::fc::ReadScaleZeroPointLinear(
            "dst_s", params.scale_zp_shape, params.has_zp, params.weights_type),
        part_idx);
    absl::StrAppend(code,
                    "  for (int src_s = tid.y; src_s < args.src_slices; "
                    "src_s += ",
                    params.reduction_size, ") {\n");
  } else {
    absl::StrAppend(code,
                    "  for (int src_group_id = tid.y; src_group_id < "
                    "args.src_groups; src_group_id += ",
                    params.reduction_size, ") {\n");
    *code += RewriteIndexedScaleZpNames(
        ::ml_drift::fc::ReadScaleZeroPointBlock("dst_s", params.scale_zp_shape,
                                                params.has_zp,
                                                params.weights_type),
        part_idx);
    *code +=
        "  for (int src_sub_id = 0; src_sub_id < args.src_group_size; "
        "++src_sub_id) {\n";
    *code +=
        "    int src_s = src_group_id * args.src_group_size + src_sub_id;\n";
  }

  const std::string src_y =
      RowOf("src_s", params.src_height, params.src_row_slices);
  const std::string src_first_s =
      absl::StrCat("part * ", params.src_row_slices);
  const std::string src_s_expr = SliceOfRow(
      "src_s", src_first_s, params.src_height, params.src_row_slices);
  if (params.src_has_batch) {
    absl::StrAppend(code, "    Type v0 = args.src_tensor.Read(0, ", src_y,
                    ", ", src_s_expr, ", 0);\n");
  } else {
    absl::StrAppend(code, "    Type v0 = args.src_tensor.Read(0, ", src_y,
                    ", ", src_s_expr, ");\n");
  }
  absl::StrAppend(code, params.src_qdq_code);
  *code += "    Type w0, w1, w2, w3;\n";
  *code += GenerateReadEdgeWeightsCode(params.weights_type, params.weights_desc,
                                       absl::StrCat("weights_", part_idx));
  *code += ::ml_drift::fc::WeightsScaleAddBias("w_scale", "w_bias",
                                               /*isI4O4=*/true, params.use_fma);
  *code += ::ml_drift::fc::AccumulateFloat("r_sp0", "v0", params.precision,
                                           /*isI4O4=*/true, params.use_fma);
  if (params.is_block_quantized) {
    *code += "  }\n";
  }
  *code += "  }\n";
}

absl::Status EdgeFcNotMatched() {
  return absl::UnimplementedError("Not a fusible BlockFfw edge stage.");
}

bool IsOpName(absl::string_view op_name, OperationType type) {
  return op_name == ToString(type);
}

bool IsQuantizedFcOpName(absl::string_view op_name) {
  return IsOpName(op_name, OperationType::kFullyConnectedInt2) ||
         IsOpName(op_name, OperationType::kFullyConnectedInt4) ||
         IsOpName(op_name, OperationType::kFullyConnectedInt8);
}

template <typename... Ts>
OHWI GetWeightsShape(const std::variant<Ts...>& weights) {
  return std::visit([](const auto& w) { return w.shape; }, weights);
}

template <typename TensorT>
OHWI GetWeightsShape(const TensorT& weights) {
  return weights.shape;
}

struct QuantizedFcAttrSpec {
  OHWI weights_shape;
  OHWI scale_shape;
  bool has_zp = false;
  WeightsDescription weights_desc;
};

template <typename FcAttrT, typename DescFn>
std::optional<QuantizedFcAttrSpec> ExtractFcAttrSpec(
    const std::any& attributes, const GpuInfo& gpu_info, bool prefer_textures,
    DescFn get_weights_desc) {
  const auto* attr = std::any_cast<FcAttrT>(&attributes);
  if (attr == nullptr || !attr->bias.data.empty() || attr->bias.shape.v != 0) {
    return std::nullopt;
  }
  const OHWI weights_shape = GetWeightsShape(attr->weights);
  return QuantizedFcAttrSpec{
      .weights_shape = weights_shape,
      .scale_shape = attr->scale.shape,
      .has_zp = attr->zero_point.shape.o != 0,
      .weights_desc =
          get_weights_desc(gpu_info, weights_shape, prefer_textures),
  };
}

std::optional<GpuModelBuilder::Weights> ExtractQuantizedFcWeightsFromSpec(
    const GpuInfo& gpu_info, bool prefer_textures, absl::string_view op_name,
    const std::any& attributes, const std::vector<ValueId>& input_ids,
    const GpuModelBuilder& builder) {
  if (input_ids.size() != 3 && input_ids.size() != 4) {
    return std::nullopt;
  }

  std::optional<QuantizedFcAttrSpec> fc_spec;
  if (IsOpName(op_name, OperationType::kFullyConnectedInt2)) {
    fc_spec = ExtractFcAttrSpec<::ml_drift::FullyConnectedInt2Attributes>(
        attributes, gpu_info, prefer_textures,
        ::ml_drift::GetFullyConnectedInt2WeightsDesc);
  } else if (IsOpName(op_name, OperationType::kFullyConnectedInt4)) {
    fc_spec = ExtractFcAttrSpec<::ml_drift::FullyConnectedInt4Attributes>(
        attributes, gpu_info, prefer_textures,
        ::ml_drift::GetFullyConnectedInt4WeightsDesc);
  } else if (IsOpName(op_name, OperationType::kFullyConnectedInt8)) {
    fc_spec = ExtractFcAttrSpec<::ml_drift::FullyConnectedInt8Attributes>(
        attributes, gpu_info, prefer_textures,
        ::ml_drift::GetFullyConnectedInt8WeightsDesc);
  } else {
    return std::nullopt;
  }
  if (!fc_spec.has_value()) return std::nullopt;

  if (fc_spec->has_zp ? (input_ids.size() != 4) : (input_ids.size() != 3)) {
    return std::nullopt;
  }
  auto w_handle = builder.GetTensor(input_ids[1]);
  auto s_handle = builder.GetTensor(input_ids[2]);
  if (!w_handle.ok() || !s_handle.ok()) return std::nullopt;

  GpuModelBuilder::TensorHandle zp_handle;
  GpuModelBuilder::TensorHandle* zp_ptr = nullptr;
  if (fc_spec->has_zp) {
    auto zp_tensor = builder.GetTensor(input_ids[3]);
    if (!zp_tensor.ok()) return std::nullopt;
    zp_handle = *std::move(zp_tensor);
    zp_ptr = &zp_handle;
  }

  return ::ml_drift::CreateExternalWeights(
      *w_handle, fc_spec->weights_desc, fc_spec->weights_shape,
      fc_spec->scale_shape, &(*s_handle), zp_ptr, /*sum_i=*/nullptr);
}

std::optional<GpuModelBuilder::Weights> ExtractQuantizedFcWeights(
    const GpuInfo& gpu_info, bool prefer_textures, const GraphFloat32& graph,
    const Node& fc_node, const GpuModelBuilder& builder) {
  const std::vector<Value*> inputs = graph.FindInputs(fc_node.id);
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (graph.FindProducer(inputs[i]->id) != nullptr) return std::nullopt;
  }
  return ExtractQuantizedFcWeightsFromSpec(
      gpu_info, prefer_textures, fc_node.operation.type,
      fc_node.operation.attributes, GetValueIds(inputs), builder);
}

struct GraphFloat32EdgeAdapter {
  using GraphT = GraphFloat32;
  using OpT = Node;
  using OpIdT = NodeId;
  using TensorT = Value;

  static const OpT* GetOp(const GraphT& g, OpIdT id) { return g.GetNode(id); }
  static OpIdT GetOpId(const OpT* op) { return op->id; }
  static absl::string_view OpName(const OpT* op) { return op->operation.type; }
  static const std::any& OpAttributes(const OpT* op) {
    return op->operation.attributes;
  }
  static ValueId TensorValueId(const TensorT* t) { return t->id; }
  static BHWC TensorShape(const TensorT* t) { return t->tensor.shape; }
  static const OpT* FindProducer(const GraphT& g, const TensorT* t) {
    return g.FindProducer(t->id);
  }
  static std::vector<const TensorT*> FindInputs(const GraphT& g,
                                                const OpT* op) {
    const std::vector<Value*> ins = g.FindInputs(op->id);
    return std::vector<const TensorT*>(ins.begin(), ins.end());
  }
  static const TensorT* SingleInput(const GraphT& g, const OpT* op) {
    if (op == nullptr) return nullptr;
    const std::vector<Value*> ins = g.FindInputs(op->id);
    return ins.size() == 1 ? ins[0] : nullptr;
  }
  static const TensorT* SingleOutput(const GraphT& g, const OpT* op) {
    if (op == nullptr) return nullptr;
    const std::vector<Value*> outs = g.FindOutputs(op->id);
    return outs.size() == 1 ? outs[0] : nullptr;
  }
  static const OpT* SingleConsumer(const GraphT& g, const TensorT* t) {
    if (t == nullptr || g.IsGraphOutput(t->id)) return nullptr;
    const std::vector<Node*> consumers = g.FindConsumers(t->id);
    return consumers.size() == 1 ? consumers[0] : nullptr;
  }
  static std::optional<GpuModelBuilder::Weights> ExtractFcWeights(
      const GpuInfo& gpu_info, bool prefer_textures, const GraphT& g,
      const OpT& fc_op, const GpuModelBuilder& builder) {
    return ExtractQuantizedFcWeights(gpu_info, prefer_textures, g, fc_op,
                                     builder);
  }
};

template <typename Adapter>
struct MatchedEdgeBranch {
  const typename Adapter::TensorT* root = nullptr;
  bool saw_first = false;
  GpuModelBuilder::Weights part_weights;
  std::vector<QuantizeAndDequantizeAttributes> src_qdqs;
  std::vector<QuantizeAndDequantizeAttributes> dst_qdqs;
  std::vector<const typename Adapter::OpT*> ops;
  std::vector<const typename Adapter::TensorT*> interm_tensors;
};

// Matches one partition branch `UnpackPart(part_idx, num_parts) -> [QDQ] ->
// Quantized FC -> [QDQ]` feeding `pack_op`'s `part_idx`-th input. Strictly
// read-only (`const GpuModelBuilder&`); produces no side effects on failure.
template <typename Adapter>
std::optional<MatchedEdgeBranch<Adapter>> MatchEdgeBranch(
    const typename Adapter::GraphT& graph, const typename Adapter::OpT* pack_op,
    const typename Adapter::OpT* first_op,
    const typename Adapter::TensorT* branch_tensor, int part_idx, int num_parts,
    const GpuInfo& gpu_info, bool prefer_textures,
    const GpuModelBuilder& model_builder) {
  using OpT = typename Adapter::OpT;
  using TensorT = typename Adapter::TensorT;
  auto is_qdq = [](const OpT* op) {
    return op != nullptr && IsOpName(Adapter::OpName(op),
                                     OperationType::kQuantizeAndDequantize);
  };

  if (Adapter::SingleConsumer(graph, branch_tensor) != pack_op) {
    return std::nullopt;
  }
  MatchedEdgeBranch<Adapter> result;
  result.interm_tensors.push_back(branch_tensor);
  const OpT* prod = Adapter::FindProducer(graph, branch_tensor);
  if (is_qdq(prod)) {
    const auto* qdq_attr = std::any_cast<QuantizeAndDequantizeAttributes>(
        &Adapter::OpAttributes(prod));
    if (qdq_attr == nullptr) return std::nullopt;
    result.dst_qdqs.push_back(*qdq_attr);
    result.ops.push_back(prod);
    branch_tensor = Adapter::SingleInput(graph, prod);
    if (Adapter::SingleConsumer(graph, branch_tensor) != prod) {
      return std::nullopt;
    }
    result.interm_tensors.push_back(branch_tensor);
    prod = Adapter::FindProducer(graph, branch_tensor);
  }

  if (prod == nullptr || !IsQuantizedFcOpName(Adapter::OpName(prod))) {
    return std::nullopt;
  }
  std::optional<GpuModelBuilder::Weights> part_weights =
      Adapter::ExtractFcWeights(gpu_info, prefer_textures, graph, *prod,
                                model_builder);
  if (!part_weights.has_value()) {
    return std::nullopt;
  }
  result.part_weights = *std::move(part_weights);
  result.ops.push_back(prod);
  const std::vector<const TensorT*> fc_ins = Adapter::FindInputs(graph, prod);
  if (fc_ins.empty()) return std::nullopt;
  branch_tensor = fc_ins[0];
  if (Adapter::SingleConsumer(graph, branch_tensor) != prod) {
    return std::nullopt;
  }
  result.interm_tensors.push_back(branch_tensor);
  prod = Adapter::FindProducer(graph, branch_tensor);

  if (is_qdq(prod)) {
    const auto* qdq_attr = std::any_cast<QuantizeAndDequantizeAttributes>(
        &Adapter::OpAttributes(prod));
    if (qdq_attr == nullptr) return std::nullopt;
    result.src_qdqs.push_back(*qdq_attr);
    result.ops.push_back(prod);
    branch_tensor = Adapter::SingleInput(graph, prod);
    if (Adapter::SingleConsumer(graph, branch_tensor) != prod) {
      return std::nullopt;
    }
    result.interm_tensors.push_back(branch_tensor);
    prod = Adapter::FindProducer(graph, branch_tensor);
  }

  if (prod == nullptr || Adapter::OpName(prod) != kBlockFfwUnpackPartType) {
    return std::nullopt;
  }
  const auto* unpack_attr =
      std::any_cast<BlockFfwUnpackPartAttributes>(&Adapter::OpAttributes(prod));
  if (unpack_attr == nullptr || unpack_attr->num_parts != num_parts ||
      unpack_attr->part != part_idx) {
    return std::nullopt;
  }
  result.saw_first = (prod == first_op);
  result.ops.push_back(prod);
  result.root = Adapter::SingleInput(graph, prod);
  if (result.root == nullptr) return std::nullopt;
  return result;
}

absl::Status AddBlockFfwUnpackPartImpl(const OperationDef& op_def,
                                       const std::any& attributes,
                                       const std::vector<ValueId>& src_ids,
                                       const std::vector<ValueId>& dst_ids,
                                       GpuModelBuilder* model_builder) {
  if (model_builder == nullptr) {
    return absl::InvalidArgumentError("Null GpuModelBuilder.");
  }
  if (src_ids.size() != 1 || dst_ids.size() != 1 ||
      op_def.src_tensors.size() != 1 || op_def.dst_tensors.size() != 1) {
    return absl::InvalidArgumentError(
        "BlockFfwUnpackPart expects 1 source and 1 destination tensor.");
  }
  const auto* attr = std::any_cast<BlockFfwUnpackPartAttributes>(&attributes);
  if (attr == nullptr || attr->num_parts <= 0 || attr->part < 0 ||
      attr->part >= attr->num_parts) {
    return absl::InvalidArgumentError(
        "Invalid BlockFfwUnpackPartAttributes.");
  }
  if (op_def.src_tensors[0].GetDataType() !=
      op_def.dst_tensors[0].GetDataType()) {
    return absl::InvalidArgumentError(
        "BlockFfwUnpackPart source and destination data types must match.");
  }
  const BHWC src_shape = op_def.src_tensors[0].GetBHWCShape();
  if (src_shape.h <= 0 || src_shape.c <= 0 ||
      src_shape.c % attr->num_parts != 0 ||
      (src_shape.c / attr->num_parts) % 4 != 0) {
    return absl::InvalidArgumentError(
        "BlockFfwUnpackPart input height must be positive and channels per "
        "part must be a positive multiple of 4.");
  }
  const BHWC dst_shape = op_def.dst_tensors[0].GetBHWCShape();
  const int64_t expected_dst_c =
      static_cast<int64_t>(src_shape.h) * (src_shape.c / attr->num_parts);
  if (dst_shape.b != src_shape.b || dst_shape.h != 1 ||
      dst_shape.w != src_shape.w ||
      static_cast<int64_t>(dst_shape.c) != expected_dst_c) {
    return absl::InvalidArgumentError(
        "BlockFfwUnpackPart destination shape does not match source shape.");
  }
  model_builder->AddGpuOperation(src_ids, dst_ids,
                                 CreateUnpackPartOperation(op_def, *attr),
                                 std::string(kBlockFfwUnpackPartType));
  return absl::OkStatus();
}

absl::Status AddBlockFfwPackPartsImpl(const OperationDef& op_def,
                                      const std::any& attributes,
                                      const std::vector<ValueId>& src_ids,
                                      const std::vector<ValueId>& dst_ids,
                                      GpuModelBuilder* model_builder) {
  if (model_builder == nullptr) {
    return absl::InvalidArgumentError("Null GpuModelBuilder.");
  }
  const auto* attr = std::any_cast<BlockFfwPackPartsAttributes>(&attributes);
  if (attr == nullptr || attr->num_parts <= 0 ||
      static_cast<int>(src_ids.size()) != attr->num_parts ||
      static_cast<int>(op_def.src_tensors.size()) != attr->num_parts ||
      dst_ids.size() != 1 || op_def.dst_tensors.size() != 1) {
    return absl::InvalidArgumentError(
        "Invalid BlockFfwPackPartsAttributes or tensor counts.");
  }
  const BHWC dst_shape = op_def.dst_tensors[0].GetBHWCShape();
  if (dst_shape.h <= 0 || dst_shape.c <= 0 ||
      dst_shape.c % attr->num_parts != 0 ||
      (dst_shape.c / attr->num_parts) % 4 != 0) {
    return absl::InvalidArgumentError(
        "BlockFfwPackParts output height must be positive and channels per "
        "part must be a positive multiple of 4.");
  }
  const int64_t expected_src_c =
      static_cast<int64_t>(dst_shape.h) * (dst_shape.c / attr->num_parts);
  const bool src_has_batch = op_def.src_tensors[0].HasAxis(Axis::kBatch);
  const DataType dst_dtype = op_def.dst_tensors[0].GetDataType();
  for (int k = 0; k < attr->num_parts; ++k) {
    if (op_def.src_tensors[k].HasAxis(Axis::kBatch) != src_has_batch) {
      return absl::InvalidArgumentError(
          "BlockFfwPackParts inputs must have consistent batch axes.");
    }
    if (op_def.src_tensors[k].GetDataType() != dst_dtype) {
      return absl::InvalidArgumentError(
          "BlockFfwPackParts input and output data types must match.");
    }
    const BHWC src_shape = op_def.src_tensors[k].GetBHWCShape();
    if (src_shape.b != dst_shape.b || src_shape.h != 1 ||
        src_shape.w != dst_shape.w ||
        static_cast<int64_t>(src_shape.c) != expected_src_c) {
      return absl::InvalidArgumentError(
          "BlockFfwPackParts source shape does not match destination shape.");
    }
  }
  return AddPackPartsOperations(op_def, *attr, src_ids, dst_ids, model_builder);
}

absl::Status AddBlockFfwSiluImpl(const OperationDef& op_def,
                                 const std::vector<ValueId>& src_ids,
                                 const std::vector<ValueId>& dst_ids,
                                 GpuModelBuilder* model_builder) {
  if (model_builder == nullptr) {
    return absl::InvalidArgumentError("Null GpuModelBuilder.");
  }
  if (src_ids.size() != 1 || dst_ids.size() != 1 ||
      op_def.src_tensors.size() != 1 || op_def.dst_tensors.size() != 1) {
    return absl::InvalidArgumentError(
        "BlockFfwSilu expects 1 source and 1 destination tensor.");
  }
  if (op_def.src_tensors[0].GetDataType() !=
      op_def.dst_tensors[0].GetDataType()) {
    return absl::InvalidArgumentError(
        "BlockFfwSilu source and destination data types must match.");
  }
  if (op_def.src_tensors[0].GetBHWCShape() !=
      op_def.dst_tensors[0].GetBHWCShape()) {
    return absl::InvalidArgumentError(
        "BlockFfwSilu source and destination shapes must match.");
  }
  model_builder->AddGpuOperation(src_ids, dst_ids, CreateSiluOperation(op_def),
                                 std::string(kBlockFfwSiluType));
  return absl::OkStatus();
}

bool SameQdqList(const std::vector<QuantizeAndDequantizeAttributes>& a,
                 const std::vector<QuantizeAndDequantizeAttributes>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].min != b[i].min || a[i].max != b[i].max ||
        a[i].scale != b[i].scale) {
      return false;
    }
  }
  return true;
}

}  // namespace

absl::Status CreateBlockFfwUnpackPartFromNode(
    const OperationDef& op_def, const std::vector<Value*>& inputs,
    const std::vector<Value*>& outputs, const Node& node,
    GpuModelBuilder* model_builder) {
  return AddBlockFfwUnpackPartImpl(op_def, node.operation.attributes,
                                   GetValueIds(inputs), GetValueIds(outputs),
                                   model_builder);
}

absl::Status CreateBlockFfwPackPartsFromNode(
    const OperationDef& op_def, const std::vector<Value*>& inputs,
    const std::vector<Value*>& outputs, const Node& node,
    GpuModelBuilder* model_builder) {
  return AddBlockFfwPackPartsImpl(op_def, node.operation.attributes,
                                  GetValueIds(inputs), GetValueIds(outputs),
                                  model_builder);
}

absl::Status CreateBlockFfwSiluFromNode(const OperationDef& op_def,
                                        const std::vector<Value*>& inputs,
                                        const std::vector<Value*>& outputs,
                                        GpuModelBuilder* model_builder) {
  return AddBlockFfwSiluImpl(op_def, GetValueIds(inputs), GetValueIds(outputs),
                             model_builder);
}

absl::Status CreateBlockFfwUnpackPartFromIrOp(
    const OperationDef& op_def, const std::vector<const IrTensor*>& inputs,
    const std::vector<const IrTensor*>& outputs, const IrOp& op,
    GpuModelBuilder* model_builder) {
  return AddBlockFfwUnpackPartImpl(op_def, op.attr, GetValueIds(inputs),
                                   GetValueIds(outputs), model_builder);
}

absl::Status CreateBlockFfwPackPartsFromIrOp(
    const OperationDef& op_def, const std::vector<const IrTensor*>& inputs,
    const std::vector<const IrTensor*>& outputs, const IrOp& op,
    GpuModelBuilder* model_builder) {
  return AddBlockFfwPackPartsImpl(op_def, op.attr, GetValueIds(inputs),
                                  GetValueIds(outputs), model_builder);
}

absl::Status CreateBlockFfwSiluFromIrOp(
    const OperationDef& op_def, const std::vector<const IrTensor*>& inputs,
    const std::vector<const IrTensor*>& outputs,
    GpuModelBuilder* model_builder) {
  return AddBlockFfwSiluImpl(op_def, GetValueIds(inputs), GetValueIds(outputs),
                             model_builder);
}

absl::Status AddBlockFfwEdgeFc(GpuModelBuilder* builder,
                               const GpuModelBuilder::TensorHandle& src,
                               const GpuModelBuilder::TensorHandle& dst,
                               const BlockFfwEdgeFcSpec& spec) {
  if (builder == nullptr) {
    return absl::InvalidArgumentError("Null GpuModelBuilder.");
  }
  const BHWC src_shape = src.tensor_desc.GetBHWCShape();
  const BHWC dst_shape = dst.tensor_desc.GetBHWCShape();
  const DataType data_type = src.tensor_desc.GetDataType();
  if (src_shape.b != 1 || src_shape.w != 1 || src_shape.h <= 0 ||
      dst_shape.b != 1 || dst_shape.w != 1 || dst_shape.h <= 0 ||
      dst.tensor_desc.GetDataType() != data_type ||
      (data_type != DataType::kFloat16 && data_type != DataType::kFloat32)) {
    return absl::InvalidArgumentError(
        "BlockFfwEdgeFc expects single-token [1, H, 1, C] float16/float32 "
        "tensors of matching type with H > 0.");
  }
  if (spec.num_parts < 1 ||
      static_cast<int>(spec.part_weights.size()) != spec.num_parts) {
    return absl::InvalidArgumentError(
        "BlockFfwEdgeFc expects one weight tensor per part.");
  }
  const GpuModelBuilder::Weights& w0 = spec.part_weights[0];
  const OHWI full_shape = w0.shape;
  if (full_shape.h != 1 || full_shape.w != 1 || full_shape.o <= 0 ||
      full_shape.o % 4 != 0 || full_shape.i <= 0 || full_shape.i % 4 != 0) {
    return absl::InvalidArgumentError(
        "BlockFfwEdgeFc expects quantized weights with O and I divisible by "
        "4.");
  }
  const OHWI part_shape(full_shape.o, 1, 1, full_shape.i);
  const int src_slices = part_shape.i / 4;
  const int dst_slices = part_shape.o / 4;
  if (w0.desc.type != DataType::kUint2 && w0.desc.type != DataType::kUint4 &&
      w0.desc.type != DataType::kUint8) {
    return absl::UnimplementedError(
        "BlockFfwEdgeFc supports only quantized UINT2/UINT4/UINT8 weights.");
  }
  const DataType weights_type =
      ::ml_drift::fc::GetDataTypeForWeights(w0.desc.type);
  if ((w0.desc.layout != ::ml_drift::WeightsLayout::kOSpatialIOGroupI4O4 &&
       w0.desc.layout !=
           ::ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) ||
      w0.desc.GetOutputGroupSize() != dst_slices) {
    return absl::UnimplementedError(
        "BlockFfwEdgeFc requires I4O4 linear or 2-D texture weights layout "
        "with output_group_size == O / 4.");
  }

  for (const auto& pw : spec.part_weights) {
    if (pw.desc != w0.desc || pw.shape != full_shape ||
        pw.scale_zp_shape != w0.scale_zp_shape ||
        pw.scale.has_value() != w0.scale.has_value() ||
        pw.zero_point.has_value() != w0.zero_point.has_value()) {
      return absl::InvalidArgumentError(
          "BlockFfwEdgeFc expects homogeneous weight descriptors across "
          "parts.");
    }
    if (!pw.scale.has_value()) {
      return absl::UnimplementedError(
          "BlockFfwEdgeFc expects tensor scales on quantized weights.");
    }
    if (pw.scale->tensor_desc.GetDataType() != data_type) {
      return absl::InvalidArgumentError(
          "BlockFfwEdgeFc expects weight scale data type to match activation "
          "data type.");
    }
    if (pw.zero_point.has_value() &&
        pw.zero_point->tensor_desc.GetDataType() != data_type) {
      return absl::InvalidArgumentError(
          "BlockFfwEdgeFc expects weight zero-point data type to match "
          "activation data type.");
    }
  }

  auto validate_qdqs =
      [](const std::vector<QuantizeAndDequantizeAttributes>& qdqs)
      -> absl::Status {
    for (const auto& q : qdqs) {
      if (!std::isfinite(q.scale) || q.scale < 0.0f || !std::isfinite(q.min) ||
          !std::isfinite(q.max) || q.min > q.max) {
        return absl::InvalidArgumentError(
            "BlockFfwEdgeFc received invalid QuantizeAndDequantize "
            "attributes.");
      }
    }
    return absl::OkStatus();
  };
  ABSL_RETURN_IF_ERROR(validate_qdqs(spec.src_qdqs));
  ABSL_RETURN_IF_ERROR(validate_qdqs(spec.dst_qdqs));

  if (w0.scale_zp_shape.h != 1 || w0.scale_zp_shape.w != 1 ||
      w0.scale_zp_shape.o != part_shape.o || w0.scale_zp_shape.i <= 0) {
    return absl::UnimplementedError(
        "BlockFfwEdgeFc requires per-output-channel or blockwise scales with "
        "shape (O, 1, 1, G).");
  }
  if (w0.scale_zp_shape.i > 1 && src_slices % w0.scale_zp_shape.i != 0) {
    return absl::UnimplementedError(
        "BlockFfwEdgeFc requires input slices divisible by block scale "
        "groups.");
  }

  if (part_shape.i % src_shape.h != 0 ||
      (part_shape.i / src_shape.h) % 4 != 0 ||
      static_cast<int64_t>(src_shape.c) !=
          static_cast<int64_t>(spec.num_parts) *
              (part_shape.i / src_shape.h)) {
    return absl::InvalidArgumentError(
        "BlockFfwEdgeFc source shape does not match K1 * I.");
  }
  if (part_shape.o % dst_shape.h != 0 ||
      (part_shape.o / dst_shape.h) % 4 != 0 ||
      static_cast<int64_t>(dst_shape.c) !=
          static_cast<int64_t>(spec.num_parts) *
              (part_shape.o / dst_shape.h)) {
    return absl::InvalidArgumentError(
        "BlockFfwEdgeFc destination shape does not match K1 * O.");
  }

  const GpuInfo& gpu_info = builder->gpu_info();
  if (ExceedsEdgeFcBindingLimits(gpu_info, src.tensor_desc, dst.tensor_desc,
                                 spec)) {
    return absl::UnimplementedError(
        "BlockFfwEdgeFc exceeds device per-stage binding limits.");
  }
  const ::ml_drift::CalculationsPrecision precision =
      builder->GetConvPrecision(data_type);

  // Query ML Drift's own FullyConnected workgroup selector for `num_parts`
  // output batches so workgroup reduction matches target GPU occupancy.
  const OHWI ref_weights_shape(part_shape.o, spec.num_parts, 1, part_shape.i);
  const BHWC ref_dst_shape(1, spec.num_parts, 1, part_shape.o);
  TensorDescriptor ref_src_desc = src.tensor_desc;
  ref_src_desc.SetBHWCShape(BHWC(1, spec.num_parts, 1, part_shape.i));
  TensorDescriptor ref_dst_desc = dst.tensor_desc;
  ref_dst_desc.SetBHWCShape(ref_dst_shape);
  ExternalWeights ref_weights;
  ref_weights.desc = w0.desc;
  ref_weights.shape = ref_weights_shape;
  ref_weights.scale_zp_shape =
      OHWI(w0.scale_zp_shape.o, spec.num_parts, 1, w0.scale_zp_shape.i);
  ref_weights.scale = &w0.scale->tensor_desc;
  if (w0.zero_point.has_value()) {
    ref_weights.zero_point = &w0.zero_point->tensor_desc;
  }
  ABSL_ASSIGN_OR_RETURN(
      const ::ml_drift::FullyConnected reference,
      ::ml_drift::CreateFullyConnectedExternalWeights(
          gpu_info, precision, ref_src_desc, ref_dst_desc, ref_weights,
          /*bias=*/nullptr, &ref_dst_shape));
  const int wg_y = reference.work_group_size_.y;
  if (reference.work_group_size_.z != 1 || wg_y <= 1 ||
      (wg_y & (wg_y - 1)) != 0) {
    return absl::UnimplementedError(
        "BlockFfwEdgeFc requires work_group_size.z == 1 and power-of-2 "
        "work_group_size.y > 1.");
  }

  auto op = std::make_unique<BlockFfwEdgeFcOp>(dst_slices, spec.num_parts);
  op->AddSrcTensor("src_tensor", src.tensor_desc);
  op->AddDstTensor("dst_tensor", dst.tensor_desc);
  op->args_.AddInt("src_slices", src_slices);
  op->args_.AddInt("dst_slices", dst_slices);

  const bool is_block_quantized =
      ::ml_drift::fc::IsBlockQuantized(weights_type, w0.scale_zp_shape);
  if (is_block_quantized) {
    op->args_.AddInt("src_groups", w0.scale_zp_shape.i);
    op->args_.AddInt("src_group_size", src_slices / w0.scale_zp_shape.i);
  }

  std::vector<ValueId> input_ids = {src.id};
  for (int k = 0; k < spec.num_parts; ++k) {
    const auto& pw = spec.part_weights[k];
    ExternalWeights ext;
    ext.desc = pw.desc;
    ext.shape = pw.shape;
    ext.scale_zp_shape = pw.scale_zp_shape;
    ext.scale = &pw.scale->tensor_desc;
    if (pw.zero_point.has_value()) {
      ext.zero_point = &pw.zero_point->tensor_desc;
    }
    AddIndexedWeightsArguments(ext, k, op.get());
    input_ids.push_back(pw.weights.id);
    input_ids.push_back(pw.scale->id);
    if (pw.zero_point.has_value()) {
      input_ids.push_back(pw.zero_point->id);
    }
  }

  const int src_row_slices = src_slices / src_shape.h;
  const int dst_row_slices = dst_slices / dst_shape.h;
  const DataType acc_type = precision == ::ml_drift::CalculationsPrecision::kF16
                                ? DataType::kFloat16
                                : DataType::kFloat32;
  const bool use_fma = ::ml_drift::fc::UseFMA(gpu_info);
  const bool has_zp = w0.zero_point.has_value();
  const std::string reduction_size =
      std::to_string(reference.work_group_size_.y);
  const bool src_has_batch = src.tensor_desc.HasAxis(Axis::kBatch);
  const bool dst_has_batch = dst.tensor_desc.HasAxis(Axis::kBatch);

  // Example complete expanded UCL shader for U1 (`num_parts = 2`,
  // `src_shape = [1, 1, 1, 32]`, `dst_shape = [1, 2, 1, 16]`,
  // `work_group_size = (4, 8, 1)`, linear per-channel INT2 buffer weights with
  // zero-point, FP32 precision, no inline QDQ):
  //   MAIN_FUNCTION($0) {
  //     int dst_s = ucl::GetGlobalId<0>();
  //     int part = ucl::GetGroupId<2>();
  //     int dst_end_slice = args.dst_slices;
  //     int dst_s_wg_offset = ucl::GetGroupId<0>() * ucl::GetGroupSize<0>();
  //     if (dst_s_wg_offset >= dst_end_slice) return;
  //     float4 r_sp0 = ucl::Init<float4>(0.0f);
  //     int2 tid;
  //     tid.x = ucl::GetLocalId<0>();
  //     tid.y = ucl::GetLocalId<1>();
  //     if (dst_s < dst_end_slice) {
  //       float4 w_scale;
  //       float4 w_bias;
  //       if (part == 0) {
  //         // `AppendEdgeFcPartLoopBody(..., /*part_idx=*/0)`
  //       } else {
  //         // `AppendEdgeFcPartLoopBody(..., /*part_idx=*/1)`
  //       }
  //     }
  //     __local float4 temp[32];
  //     temp[tid.x * 8 + tid.y] = r_sp0;
  //     for (int stride = 8 / 2; stride > 0; stride /= 2) {
  //       ucl::SyncThreads<WorkGroup, Local>();
  //       if (tid.y < stride) {
  //         r_sp0 += temp[tid.x * 8 + tid.y + stride];
  //         temp[tid.x * 8 + tid.y] = r_sp0;
  //       }
  //     }
  //     if (dst_s >= dst_end_slice) return;
  //     if (tid.y != 0) return;
  //     float4 res_type = ucl::Convert<float4>(r_sp0);
  //     // Optional inline dst QuantizeAndDequantize blocks on `res_type`.
  //     args.dst_tensor::type res_value =
  //         ucl::Convert<args.dst_tensor::type>(res_type);
  //     args.dst_tensor.Write(res_value, 0, dst_s / 2,
  //                           part * 2 + (dst_s % 2));
  //   }
  std::string c = "MAIN_FUNCTION($0) {\n";
  c += "  int dst_s = ucl::GetGlobalId<0>();\n";
  c += "  int part = ucl::GetGroupId<2>();\n";
  c += "  int dst_end_slice = args.dst_slices;\n";
  c += "  int dst_s_wg_offset = ucl::GetGroupId<0>() * "
       "ucl::GetGroupSize<0>();\n";
  c += "  if (dst_s_wg_offset >= dst_end_slice) return;\n";
  absl::StrAppend(&c, "  AccType r_sp0 = ucl::Init<AccType>(",
                  ::ml_drift::GetZeroValue(acc_type), ");\n");
  c += "  int2 tid;\n";
  c += "  tid.x = ucl::GetLocalId<0>();\n";
  c += "  tid.y = ucl::GetLocalId<1>();\n";
  c += "  if (dst_s < dst_end_slice) {\n";
  c += "  Type w_scale;\n";
  c += "  Type w_bias;\n";

  // Pre-emit inline source QDQ uniform arguments once (outside any part loop).
  std::string src_qdq_code;
  for (size_t qi = 0; qi < spec.src_qdqs.size(); ++qi) {
    AppendInlineQuantizeAndDequantize(&src_qdq_code, op.get(),
                                      absl::StrCat("src_qdq", qi), "v0",
                                      data_type, gpu_info, spec.src_qdqs[qi]);
  }

  const EdgeFcLoopCodegenParams loop_params{
      .is_block_quantized = is_block_quantized,
      .has_zp = has_zp,
      .src_has_batch = src_has_batch,
      .use_fma = use_fma,
      .src_height = src_shape.h,
      .src_row_slices = src_row_slices,
      .weights_type = weights_type,
      .precision = precision,
      .scale_zp_shape = w0.scale_zp_shape,
      .weights_desc = w0.desc,
      .reduction_size = reduction_size,
      .src_qdq_code = src_qdq_code,
  };

  if (spec.num_parts == 1) {
    AppendEdgeFcPartLoopBody(loop_params, /*part_idx=*/0, &c);
  } else {
    for (int k = 0; k < spec.num_parts; ++k) {
      if (k == 0) {
        c += "  if (part == 0) {\n";
      } else if (k + 1 < spec.num_parts) {
        absl::StrAppend(&c, "  } else if (part == ", k, ") {\n");
      } else {
        c += "  } else {\n";
      }
      AppendEdgeFcPartLoopBody(loop_params, k, &c);
    }
    c += "  }\n";
  }

  c += "  }\n";  // end if (dst_s < dst_end_slice)
  absl::StrAppend(&c, "  __local AccType temp[",
                  reference.work_group_size_.x * reference.work_group_size_.y,
                  "];\n");
  c += ::ml_drift::fc::GetReductionCode(/*first=*/0, /*last=*/0,
                                        /*local_batch_size=*/1, "tid.x",
                                        "tid.y", reduction_size, "temp");
  c += "  if (dst_s >= dst_end_slice) return;\n";
  c += "  if (tid.y != 0) return;\n";
  c += "  Type res_type = ucl::Convert<Type>(r_sp0);\n";
  for (size_t qi = 0; qi < spec.dst_qdqs.size(); ++qi) {
    AppendInlineQuantizeAndDequantize(&c, op.get(), absl::StrCat("dst_qdq", qi),
                                      "res_type", data_type, gpu_info,
                                      spec.dst_qdqs[qi]);
  }
  c += "  args.dst_tensor::type res_value = "
       "ucl::Convert<args.dst_tensor::type>(res_type);\n";
  const std::string dst_y = RowOf("dst_s", dst_shape.h, dst_row_slices);
  const std::string dst_first_s = absl::StrCat("part * ", dst_row_slices);
  const std::string dst_s_expr =
      SliceOfRow("dst_s", dst_first_s, dst_shape.h, dst_row_slices);
  if (dst_has_batch) {
    absl::StrAppend(&c, "  args.dst_tensor.Write(res_value, 0, ", dst_y, ", ",
                    dst_s_expr, ", 0);\n");
  } else {
    absl::StrAppend(&c, "  args.dst_tensor.Write(res_value, 0, ", dst_y, ", ",
                    dst_s_expr, ");\n");
  }
  c += "}\n";

  op->work_group_size_ = reference.work_group_size_;
  op->compiler_options_ = reference.compiler_options_;
  op->code_ = absl::StrReplaceAll(
      c, {{"SType", ::ml_drift::ToUclDataType(data_type, 1)},
          {"Type", ::ml_drift::ToUclDataType(data_type, 4)},
          {"AccSType", ::ml_drift::ToUclDataType(acc_type, 1)},
          {"AccType", ::ml_drift::ToUclDataType(acc_type, 4)}});
  op->flops_ = spec.num_parts * ::ml_drift::GetConvolutionFlops(
                                    BHWC(1, 1, 1, part_shape.o), part_shape);
  builder->AddGpuOperation(input_ids, std::vector<ValueId>{dst.id},
                           std::move(op), "block_ffw_edge_fc");
  return absl::OkStatus();
}

namespace {

// Matches and fuses `K1` parallel branches of
// `UnpackPart -> [QDQ] -> Quantized FC -> [QDQ]` feeding `PackParts -> [QDQ]`
// into a single `BlockFfwEdgeFcOp` dispatch for single-token decode
// (`B == 1, W == 1`). All inspection before `AddBlockFfwEdgeFc` uses
// `const GpuModelBuilder&`, so a rejected match has zero side effects.
template <typename Adapter, typename ConsumedSetT>
absl::Status TryBlockFfwEdgeFcImpl(
    const ::ml_drift::CreateGpuModelInfo& create_info, const GpuInfo& gpu_info,
    const typename Adapter::GraphT& graph,
    typename Adapter::OpIdT first_op_id, const ConsumedSetT& consumed_ops,
    ConsumedSetT* new_consumed_ops, GpuModelBuilder* model_builder) {
  if (new_consumed_ops == nullptr || model_builder == nullptr) {
    return absl::InvalidArgumentError(
        "Null consumed set or GpuModelBuilder.");
  }
  using OpT = typename Adapter::OpT;
  using TensorT = typename Adapter::TensorT;
  auto is_qdq = [](const OpT* op) {
    return op != nullptr && IsOpName(Adapter::OpName(op),
                                     OperationType::kQuantizeAndDequantize);
  };

  const OpT* first = Adapter::GetOp(graph, first_op_id);
  if (first == nullptr || consumed_ops.count(Adapter::GetOpId(first)) != 0 ||
      Adapter::OpName(first) != kBlockFfwUnpackPartType) {
    return EdgeFcNotMatched();
  }
  const bool prefer_textures =
      create_info.hints.Check(::ml_drift::ModelHints::kPreferTextureWeights);

  const OpT* cur =
      Adapter::SingleConsumer(graph, Adapter::SingleOutput(graph, first));
  if (is_qdq(cur)) {
    cur = Adapter::SingleConsumer(graph, Adapter::SingleOutput(graph, cur));
  }
  if (cur == nullptr || !IsQuantizedFcOpName(Adapter::OpName(cur))) {
    return EdgeFcNotMatched();
  }
  cur = Adapter::SingleConsumer(graph, Adapter::SingleOutput(graph, cur));
  if (is_qdq(cur)) {
    cur = Adapter::SingleConsumer(graph, Adapter::SingleOutput(graph, cur));
  }
  if (cur == nullptr || Adapter::OpName(cur) != kBlockFfwPackPartsType) {
    return EdgeFcNotMatched();
  }

  const OpT* pack_op = cur;
  const auto* pack_attr = std::any_cast<BlockFfwPackPartsAttributes>(
      &Adapter::OpAttributes(pack_op));
  const std::vector<const TensorT*> pack_inputs =
      Adapter::FindInputs(graph, pack_op);
  const TensorT* dst_tensor = Adapter::SingleOutput(graph, pack_op);
  if (pack_attr == nullptr || pack_attr->num_parts < 1 ||
      static_cast<int>(pack_inputs.size()) != pack_attr->num_parts ||
      dst_tensor == nullptr) {
    return EdgeFcNotMatched();
  }

  const int num_parts = pack_attr->num_parts;
  BlockFfwEdgeFcSpec spec;
  spec.num_parts = num_parts;
  spec.part_weights.resize(num_parts);
  std::vector<const OpT*> ops = {pack_op};
  std::vector<const TensorT*> interm_tensors;
  const TensorT* root = nullptr;
  bool saw_first = false;
  const GpuModelBuilder& readonly_builder = *model_builder;

  for (int k = 0; k < num_parts; ++k) {
    std::optional<MatchedEdgeBranch<Adapter>> branch = MatchEdgeBranch<Adapter>(
        graph, pack_op, first, pack_inputs[k], k, num_parts, gpu_info,
        prefer_textures, readonly_builder);
    if (!branch.has_value()) return EdgeFcNotMatched();
    spec.part_weights[k] = std::move(branch->part_weights);
    ops.insert(ops.end(), branch->ops.begin(), branch->ops.end());
    interm_tensors.insert(interm_tensors.end(), branch->interm_tensors.begin(),
                          branch->interm_tensors.end());
    saw_first = saw_first || branch->saw_first;
    if (k == 0) {
      root = branch->root;
      spec.src_qdqs = std::move(branch->src_qdqs);
      spec.dst_qdqs = std::move(branch->dst_qdqs);
    } else if (branch->root != root ||
               !SameQdqList(spec.src_qdqs, branch->src_qdqs) ||
               !SameQdqList(spec.dst_qdqs, branch->dst_qdqs)) {
      return EdgeFcNotMatched();
    }
  }

  if (!saw_first || root == nullptr) {
    return EdgeFcNotMatched();
  }
  const BHWC root_shape = Adapter::TensorShape(root);
  if (root_shape.b != 1 || root_shape.w != 1) {
    return EdgeFcNotMatched();
  }

  // Absorb optional trailing QDQ after PackParts.
  if (const OpT* next_qdq = Adapter::SingleConsumer(graph, dst_tensor);
      is_qdq(next_qdq) &&
      consumed_ops.count(Adapter::GetOpId(next_qdq)) == 0 &&
      Adapter::SingleInput(graph, next_qdq) == dst_tensor) {
    if (const auto* qdq_attr = std::any_cast<QuantizeAndDequantizeAttributes>(
            &Adapter::OpAttributes(next_qdq));
        qdq_attr != nullptr) {
      if (const TensorT* next_out = Adapter::SingleOutput(graph, next_qdq);
          next_out != nullptr) {
        interm_tensors.push_back(dst_tensor);
        ops.push_back(next_qdq);
        spec.dst_qdqs.push_back(*qdq_attr);
        dst_tensor = next_out;
      }
    }
  }

  for (const OpT* op : ops) {
    if (consumed_ops.count(Adapter::GetOpId(op)) != 0) {
      return EdgeFcNotMatched();
    }
  }
  auto src_handle = readonly_builder.GetTensor(Adapter::TensorValueId(root));
  auto dst_handle =
      readonly_builder.GetTensor(Adapter::TensorValueId(dst_tensor));
  if (!src_handle.ok() || !dst_handle.ok()) return EdgeFcNotMatched();
  for (const TensorT* t : interm_tensors) {
    auto th = readonly_builder.GetTensor(Adapter::TensorValueId(t));
    if (!th.ok() || th->tensor_desc.GetDataType() !=
                        src_handle->tensor_desc.GetDataType()) {
      return EdgeFcNotMatched();
    }
  }

  // All matching above is strictly read-only (`const GpuModelBuilder&`), and
  // `AddBlockFfwEdgeFc` only mutates `model_builder` on its final
  // `AddGpuOperation` call once all validation succeeds.
  if (!AddBlockFfwEdgeFc(model_builder, *src_handle, *dst_handle, spec).ok()) {
    return EdgeFcNotMatched();
  }
  for (const OpT* op : ops) {
    new_consumed_ops->insert(Adapter::GetOpId(op));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status TryBlockFfwEdgeFc(
    const ::ml_drift::CreateGpuModelInfo& create_info, const GpuInfo& gpu_info,
    const GraphFloat32& graph, NodeId first_node_id,
    const std::set<NodeId>& consumed_nodes,
    std::set<NodeId>* new_consumed_nodes, GpuModelBuilder* model_builder) {
  return TryBlockFfwEdgeFcImpl<GraphFloat32EdgeAdapter>(
      create_info, gpu_info, graph, first_node_id, consumed_nodes,
      new_consumed_nodes, model_builder);
}

}  // namespace litert::ml_drift

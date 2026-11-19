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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/substitute.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/flops_util.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/kernel_info.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected_util.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
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
using ::ml_drift::Node;
using ::ml_drift::OHWI;
using ::ml_drift::OperationDef;
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

absl::Status AddPackPartsOperations(const OperationDef& op_def,
                                    absl::string_view op_type,
                                    const BlockFfwPackPartsAttributes& attr,
                                    const std::vector<ValueId>& src_ids,
                                    const std::vector<ValueId>& dst_ids,
                                    GpuModelBuilder* model_builder) {
  const int max_inputs =
      GetMaxPackPartsInputs(model_builder->gpu_info(), op_def);
  if (attr.num_parts <= max_inputs) {
    model_builder->AddGpuOperation(
        src_ids, dst_ids, CreatePackPartsOperation(op_def, attr.num_parts),
        std::string(op_type));
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
        std::string(op_type));
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
    absl::StrAppend(&code, "  SRC_S = ", offset_slices, " + DST_S;\n");
  } else {
    absl::StrAppend(&code, "  SRC_Y = DST_S / ", part_slices, ";\n");
    absl::StrAppend(&code, "  SRC_S = ", offset_slices, " + (DST_S % ",
                    part_slices, ");\n");
  }
  AppendBatchCoordReorder(op_def, &code);
  return std::make_unique<GPUOperation>(
      ::ml_drift::CreateReorderGpuOperation(op_def, std::move(code)));
}

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

// Appends exact ML Drift QuantizeAndDequantize math on `var_name` (of UCL type
// `ToUclDataType(tensor_dtype, 4)`), matching `CreateQuantizeAndOrDequantize`
// and `UseFp32ForQuantizeAndDequantize` across Metal, OpenCL, WebGPU, Vulkan,
// and OpenGL ES.
void AppendInlineQdq(std::string* code, GPUOperation* op,
                     absl::string_view prefix, absl::string_view var_name,
                     DataType tensor_dtype, const GpuInfo& gpu_info,
                     const QuantizeAndDequantizeAttributes& raw_attr) {
  const bool use_fp32_qdq = tensor_dtype == DataType::kFloat16 &&
                            gpu_info.IsApple() && gpu_info.IsApiMetal();
  const DataType qdq_type = use_fp32_qdq ? DataType::kFloat32 : tensor_dtype;
  QuantizeAndDequantizeAttributes attr = raw_attr;
  if (qdq_type == DataType::kFloat16 && attr.scale > 0.0f &&
      attr.scale < 0.000062f) {
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
  const std::string w_name = absl::StrCat("weights_", part_idx);
  if (weights.desc.type == DataType::kUint8) {
    if (weights.desc.IsLinearLayout()) {
      ::ml_drift::BufferDescriptor desc;
      desc.element_type = DataType::kUint32;
      desc.element_size = 4;
      op->AddSrcBuffer(w_name, desc);
    } else {
      op->AddSrcTensor(
          w_name, TensorDescriptor(DataType::kUint32,
                                   ::ml_drift::TensorStorageType::kTexture2D,
                                   ::ml_drift::Layout::kHW));
    }
  } else if (weights.desc.type == DataType::kUint4) {
    if (weights.desc.IsLinearLayout()) {
      ::ml_drift::BufferDescriptor desc;
      desc.element_type = DataType::kUint32;
      desc.element_size = 2;
      op->AddSrcBuffer(w_name, desc);
    } else {
      op->AddSrcTensor(
          w_name, TensorDescriptor(DataType::kUint16,
                                   ::ml_drift::TensorStorageType::kTexture2D,
                                   ::ml_drift::Layout::kHW));
    }
  } else if (weights.desc.type == DataType::kUint2) {
    if (weights.desc.IsLinearLayout()) {
      ::ml_drift::BufferDescriptor desc;
      desc.element_type = DataType::kUint32;
      desc.element_size = 1;
      op->AddSrcBuffer(w_name, desc);
    } else {
      op->AddSrcTensor(
          w_name, TensorDescriptor(DataType::kUint8,
                                   ::ml_drift::TensorStorageType::kTexture2D,
                                   ::ml_drift::Layout::kHW));
    }
  }
  if (weights.scale != nullptr) {
    op->AddSrcTensor(absl::StrCat("weights_scale_", part_idx), *weights.scale);
  }
  if (weights.zero_point != nullptr) {
    op->AddSrcTensor(absl::StrCat("weights_zero_point_", part_idx),
                     *weights.zero_point);
  }
}

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

absl::Status CreateBlockFfwNodeOperationImpl(
    const OperationDef& op_def, absl::string_view op_type,
    const std::any& attributes, const std::vector<ValueId>& src_ids,
    const std::vector<ValueId>& dst_ids, GpuModelBuilder* model_builder) {
  if (model_builder == nullptr) {
    return absl::InvalidArgumentError("Null GpuModelBuilder.");
  }

  if (op_type == kBlockFfwUnpackPartType) {
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
    const BHWC src_shape = op_def.src_tensors[0].GetBHWCShape();
    if (src_shape.c <= 0 || src_shape.c % attr->num_parts != 0 ||
        (src_shape.c / attr->num_parts) % 4 != 0) {
      return absl::InvalidArgumentError(
          "BlockFfwUnpackPart input channels per part must be a positive "
          "multiple of 4.");
    }
    model_builder->AddGpuOperation(src_ids, dst_ids,
                                   CreateUnpackPartOperation(op_def, *attr),
                                   std::string(op_type));
    return absl::OkStatus();
  }

  if (op_type == kBlockFfwPackPartsType) {
    const auto* attr = std::any_cast<BlockFfwPackPartsAttributes>(&attributes);
    if (attr == nullptr || attr->num_parts <= 0 ||
        static_cast<int>(src_ids.size()) != attr->num_parts ||
        static_cast<int>(op_def.src_tensors.size()) != attr->num_parts ||
        dst_ids.size() != 1 || op_def.dst_tensors.size() != 1) {
      return absl::InvalidArgumentError(
          "Invalid BlockFfwPackPartsAttributes or tensor counts.");
    }
    const BHWC dst_shape = op_def.dst_tensors[0].GetBHWCShape();
    if (dst_shape.c <= 0 || dst_shape.c % attr->num_parts != 0 ||
        (dst_shape.c / attr->num_parts) % 4 != 0) {
      return absl::InvalidArgumentError(
          "BlockFfwPackParts output channels per part must be a positive "
          "multiple of 4.");
    }
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
    }
    return AddPackPartsOperations(op_def, op_type, *attr, src_ids, dst_ids,
                                  model_builder);
  }

  if (op_type == kBlockFfwSiluType) {
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
    model_builder->AddGpuOperation(src_ids, dst_ids,
                                   CreateSiluOperation(op_def),
                                   std::string(op_type));
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError(
      absl::StrCat("Unexpected BlockFfw node type: ", op_type));
}

}  // namespace

bool IsBlockFfwInternalNodeType(absl::string_view op_type) {
  return op_type == kBlockFfwUnpackPartType ||
         op_type == kBlockFfwPackPartsType || op_type == kBlockFfwSiluType;
}

absl::Status CreateBlockFfwNodeOperation(const OperationDef& op_def,
                                         const std::vector<Value*>& inputs,
                                         const std::vector<Value*>& outputs,
                                         const Node& node,
                                         GpuModelBuilder* model_builder) {
  return CreateBlockFfwNodeOperationImpl(
      op_def, node.operation.type, node.operation.attributes,
      GetValueIds(inputs), GetValueIds(outputs), model_builder);
}

absl::Status CreateBlockFfwNodeOperationFromIrOp(
    const OperationDef& op_def, const std::vector<const IrTensor*>& inputs,
    const std::vector<const IrTensor*>& outputs, const IrOp& op,
    GpuModelBuilder* model_builder) {
  return CreateBlockFfwNodeOperationImpl(op_def, op.name, op.attr,
                                         GetValueIds(inputs),
                                         GetValueIds(outputs), model_builder);
}

absl::Status AddBlockFfwEdgeFc(GpuModelBuilder* builder,
                               const GpuModelBuilder::TensorHandle& src,
                               const GpuModelBuilder::TensorHandle& dst,
                               const BlockFfwEdgeFcSpec& spec) {
  const BHWC src_shape = src.tensor_desc.GetBHWCShape();
  const BHWC dst_shape = dst.tensor_desc.GetBHWCShape();
  const DataType data_type = src.tensor_desc.GetDataType();
  if (src_shape.b != 1 || src_shape.w != 1 || dst_shape.b != 1 ||
      dst_shape.w != 1 || dst.tensor_desc.GetDataType() != data_type ||
      (data_type != DataType::kFloat16 && data_type != DataType::kFloat32)) {
    return absl::InvalidArgumentError(
        "BlockFfwEdgeFc expects single-token [1, H, 1, C] float16/float32 "
        "tensors of matching type.");
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
  const DataType weights_type =
      ::ml_drift::fc::GetDataTypeForWeights(w0.desc.type);
  if (!::ml_drift::fc::IsQuantized(weights_type)) {
    return absl::UnimplementedError(
        "BlockFfwEdgeFc supports only quantized INT2/INT4/INT8 weights.");
  }
  if (!w0.desc.IsI4O4()) {
    return absl::UnimplementedError(
        "BlockFfwEdgeFc requires I4O4 quantized weights layout.");
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
  }

  const int src_slices = part_shape.i / 4;
  const int dst_slices = part_shape.o / 4;
  if (w0.scale_zp_shape.i > 1 && src_slices % w0.scale_zp_shape.i != 0) {
    return absl::UnimplementedError(
        "BlockFfwEdgeFc requires input slices divisible by block scale "
        "groups.");
  }

  if (part_shape.i % src_shape.h != 0 ||
      (part_shape.i / src_shape.h) % 4 != 0 ||
      src_shape.c != spec.num_parts * (part_shape.i / src_shape.h)) {
    return absl::InvalidArgumentError(
        "BlockFfwEdgeFc source shape does not match K1 * I.");
  }
  if (part_shape.o % dst_shape.h != 0 ||
      (part_shape.o / dst_shape.h) % 4 != 0 ||
      dst_shape.c != spec.num_parts * (part_shape.o / dst_shape.h)) {
    return absl::InvalidArgumentError(
        "BlockFfwEdgeFc destination shape does not match K1 * O.");
  }

  const GpuInfo& gpu_info = builder->gpu_info();
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
  if (reference.work_group_size_.y <= 1) {
    return absl::UnimplementedError(
        "BlockFfwEdgeFc requires workgroup reduction (work_group_size.y > 1).");
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
    AppendInlineQdq(&src_qdq_code, op.get(), absl::StrCat("src_qdq", qi), "v0",
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
    AppendInlineQdq(&c, op.get(), absl::StrCat("dst_qdq", qi), "res_type",
                    data_type, gpu_info, spec.dst_qdqs[qi]);
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

}  // namespace litert::ml_drift

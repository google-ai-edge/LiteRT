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
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/substitute.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/block_ffw_parser.h"

namespace litert::ml_drift {
namespace {

using ::ml_drift::Axis;
using ::ml_drift::BHWC;
using ::ml_drift::DataType;
using ::ml_drift::DivideRoundUp;
using ::ml_drift::ElementwiseDescriptor;
using ::ml_drift::GpuInfo;
using ::ml_drift::GpuModelBuilder;
using ::ml_drift::GPUOperation;
using ::ml_drift::Node;
using ::ml_drift::OperationDef;
using ::ml_drift::TensorDescriptor;
using ::ml_drift::TensorStorageType;
using ::ml_drift::Value;
using ::ml_drift::ValueId;

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

}  // namespace litert::ml_drift

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

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/substitute.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
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
using ::ml_drift::GpuModelBuilder;
using ::ml_drift::GPUOperation;
using ::ml_drift::Node;
using ::ml_drift::OperationDef;
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

class BlockFfwPackPartsOp : public GPUOperation {
 public:
  BlockFfwPackPartsOp() = default;
  BlockFfwPackPartsOp(BlockFfwPackPartsOp&&) = default;
  BlockFfwPackPartsOp& operator=(BlockFfwPackPartsOp&&) = default;
  BlockFfwPackPartsOp(const BlockFfwPackPartsOp&) = delete;
  BlockFfwPackPartsOp& operator=(const BlockFfwPackPartsOp&) = delete;

  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(dst_[0]->Width() * dst_[0]->Batch(),
                            dst_[0]->Height(), dst_[0]->Slices());
  }
};

std::unique_ptr<GPUOperation> CreatePackPartsOperation(
    const OperationDef& op_def, int num_parts) {
  auto op = std::make_unique<BlockFfwPackPartsOp>();
  for (int k = 0; k < num_parts; ++k) {
    op->AddSrcTensor(absl::StrCat("src_tensor_", k), op_def.src_tensors[k]);
  }
  op->AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  const BHWC dst_shape = op_def.dst_tensors[0].GetBHWCShape();
  const int part_slices = DivideRoundUp(dst_shape.c / num_parts, 4);
  const bool dst_has_batch = op_def.dst_tensors[0].HasAxis(Axis::kBatch);
  const bool src_has_batch = op_def.src_tensors[0].HasAxis(Axis::kBatch);

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
  absl::StrAppend(&c, "  int part = S / ", part_slices, ";\n");
  absl::StrAppend(&c, "  int src_s = Y * ", part_slices, " + (S % ",
                  part_slices, ");\n");
  c += "  args.dst_tensor::type res;\n";
  for (int k = 0; k < num_parts; ++k) {
    if (k == 0) {
      c += "  if (part == 0) {\n";
    } else if (k + 1 < num_parts) {
      absl::StrAppend(&c, "  } else if (part == ", k, ") {\n");
    } else {
      c += "  } else {\n";
    }
    const std::string read_coords =
        src_has_batch ? (dst_has_batch ? "X, 0, src_s, B" : "X, 0, src_s, 0")
                      : "X, 0, src_s";
    absl::StrAppend(&c, "    res = args.src_tensor_", k, ".Read(", read_coords,
                    ");\n");
  }
  c += "  }\n";
  if (dst_has_batch) {
    c += "  args.dst_tensor.Write(res, X, Y, S, B);\n";
  } else {
    c += "  args.dst_tensor.Write(res, X, Y, S);\n";
  }
  c += "}\n";
  op->code_ = std::move(c);
  return op;
}

std::unique_ptr<GPUOperation> CreateUnpackPartOperation(
    const OperationDef& op_def, const BlockFfwUnpackPartAttributes& attr) {
  const BHWC src_shape = op_def.src_tensors[0].GetBHWCShape();
  const int part_slices = DivideRoundUp(src_shape.c / attr.num_parts, 4);
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
  const DataType dtype = op_def.src_tensors[0].GetDataType();
  const std::string vec_type = ::ml_drift::ToUclDataType(dtype, 4);
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

absl::Status CreateBlockFfwNodeOperationImpl(
    const OperationDef& op_def, absl::string_view op_type,
    const std::any& attributes, const std::vector<ValueId>& src_ids,
    const std::vector<ValueId>& dst_ids, GpuModelBuilder* model_builder) {
  if (op_type == kBlockFfwUnpackPartType) {
    const auto* attr = std::any_cast<BlockFfwUnpackPartAttributes>(&attributes);
    if (attr == nullptr || attr->num_parts <= 0 || attr->part < 0 ||
        attr->part >= attr->num_parts) {
      return absl::InvalidArgumentError(
          "Invalid BlockFfwUnpackPartAttributes.");
    }
    const BHWC src_shape = op_def.src_tensors[0].GetBHWCShape();
    if (src_shape.c <= 0 || src_shape.c % attr->num_parts != 0) {
      return absl::InvalidArgumentError(
          "BlockFfwUnpackPart input channels must be divisible by num_parts.");
    }
    model_builder->AddGpuOperation(src_ids, dst_ids,
                                   CreateUnpackPartOperation(op_def, *attr),
                                   std::string(op_type));
    return absl::OkStatus();
  }

  if (op_type == kBlockFfwPackPartsType) {
    const auto* attr = std::any_cast<BlockFfwPackPartsAttributes>(&attributes);
    if (attr == nullptr || attr->num_parts <= 0 ||
        static_cast<int>(src_ids.size()) != attr->num_parts) {
      return absl::InvalidArgumentError("Invalid BlockFfwPackPartsAttributes.");
    }
    const BHWC dst_shape = op_def.dst_tensors[0].GetBHWCShape();
    if (dst_shape.c <= 0 || dst_shape.c % attr->num_parts != 0) {
      return absl::InvalidArgumentError(
          "BlockFfwPackParts output channels must be divisible by num_parts.");
    }
    model_builder->AddGpuOperation(
        src_ids, dst_ids, CreatePackPartsOperation(op_def, attr->num_parts),
        std::string(op_type));
    return absl::OkStatus();
  }

  if (op_type == kBlockFfwSiluType) {
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

}  // namespace litert::ml_drift

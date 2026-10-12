/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/kernels/elementwise.h"  // from @ml_drift
#include "ml_drift/common/kernels/embedding_lookup.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/kernels/one_hot.h"  // from @ml_drift
#include "ml_drift/common/kernels/prelu.h"  // from @ml_drift
#include "ml_drift/common/kernels/reduce.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/selectors/simple_selectors.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "tensor/arithmetic.h"
#include "tensor/arithmetic_graph.h"
#include "tensor/backends/ml_drift/ml_drift_helpers.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

namespace {

bool IsQuantized(Type type) {
  return type == Type::kI8 || type == Type::kI4 || type == Type::kI2 ||
         type == Type::kU8 || type == Type::kU4;
}

template <typename T>
absl::StatusOr<ml_drift::BHWC> ShapeToBHWC(const T& shape, int fill_value = 1) {
  switch (shape.size()) {
    case 0:
      return ml_drift::BHWC(fill_value, fill_value, fill_value, fill_value);
    case 1:
      return ml_drift::BHWC(fill_value, fill_value, fill_value, shape[0]);
    case 2:
      return ml_drift::BHWC(shape[0], fill_value, fill_value, shape[1]);
    case 3:
      return ml_drift::BHWC(fill_value, shape[0], shape[1], shape[2]);
    case 4:
      return ml_drift::BHWC(shape[0], shape[1], shape[2], shape[3]);
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Cannot convert shape ", ShapeStr(shape), " to MlDrift BHWC."));
  }
}

template <typename T>
absl::StatusOr<ml_drift::OHWI> ShapeToOHWI(const T& shape) {
  switch (shape.size()) {
    case 2:
      return ml_drift::OHWI(shape[0], 1, 1, shape[1]);
    case 4:
      return ml_drift::OHWI(shape[0], shape[1], shape[2], shape[3]);
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Cannot convert shape ", ShapeStr(shape), " to MlDrift OHWI."));
  }
}

absl::StatusOr<ml_drift::Axis> GetMlDriftAxis(const std::vector<int>& shape,
                                              int axis) {
  switch (shape.size()) {
    case 1:
      return ml_drift::Axis::kChannels;
    case 2:
      if (axis == 0) {
        return ml_drift::Axis::kBatch;
      } else if (axis == 1) {
        return ml_drift::Axis::kChannels;
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Cannot convert axis ", axis,
                         " to MlDrift Axis for shape ", ShapeStr(shape)));
      }
    case 3:
      return ml_drift::GetAxis(ml_drift::Layout::kHWC, axis);
    case 4:
      return ml_drift::GetAxis(ml_drift::Layout::kBHWC, axis);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot convert axis ", axis,
                       " to MlDrift Axis for shape ", ShapeStr(shape)));
  }
}

absl::StatusOr<ml_drift::GpuModelBuilder::TensorHandle> ApplyFusedActivation(
    MlDriftBuildContext& ctx, ml_drift::GpuModelBuilder::TensorHandle t,
    litert::tensor::FusedActivation fa) {
  switch (fa) {
    case litert::tensor::FusedActivation::kActNone:
      return t;
    case litert::tensor::FusedActivation::kActRelu: {
      return ctx.gpu_model_builder->ReLU(t, ml_drift::ReLUAttributes());
    }
    default:
      return absl::UnimplementedError("Unsupported activation function.");
  }
}

absl::StatusOr<std::pair<float, float>> GetMinMaxValues(Type type, float scale,
                                                        int64_t zero_point) {
  float min_val, max_val;
  switch (type) {
    case Type::kI32:
      min_val =
          scale * (static_cast<float>(std::numeric_limits<int32_t>::min()) -
                   zero_point);
      max_val =
          scale * (static_cast<float>(std::numeric_limits<int32_t>::max()) -
                   zero_point);
      break;
    case Type::kU32:
      min_val =
          scale * (static_cast<float>(std::numeric_limits<uint32_t>::min()) -
                   zero_point);
      max_val =
          scale * (static_cast<float>(std::numeric_limits<uint32_t>::max()) -
                   zero_point);
      break;
    case Type::kU8:
      min_val =
          scale * (static_cast<float>(std::numeric_limits<uint8_t>::min()) -
                   zero_point);
      max_val =
          scale * (static_cast<float>(std::numeric_limits<uint8_t>::max()) -
                   zero_point);
      break;
    case Type::kI8:
      min_val =
          scale *
          (static_cast<float>(std::numeric_limits<int8_t>::min()) - zero_point);
      max_val =
          scale *
          (static_cast<float>(std::numeric_limits<int8_t>::max()) - zero_point);
      break;
    case Type::kI16:
      min_val =
          scale * (static_cast<float>(std::numeric_limits<int16_t>::min()) -
                   zero_point);
      max_val =
          scale * (static_cast<float>(std::numeric_limits<int16_t>::max()) -
                   zero_point);
      break;
    case Type::kU16:
      min_val =
          scale * (static_cast<float>(std::numeric_limits<uint16_t>::min()) -
                   zero_point);
      max_val =
          scale * (static_cast<float>(std::numeric_limits<uint16_t>::max()) -
                   zero_point);
      break;
    case Type::kI4:
      min_val = scale * (static_cast<float>(-8) - zero_point);
      max_val = scale * (static_cast<float>(7) - zero_point);
      break;
    case Type::kU4:
      min_val = scale * (static_cast<float>(0) - zero_point);
      max_val = scale * (static_cast<float>(15) - zero_point);
      break;
    case Type::kI2:
      min_val = scale * (static_cast<float>(-2) - zero_point);
      max_val = scale * (static_cast<float>(1) - zero_point);
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported quantized type: ", ToString(type)));
  }
  return std::make_pair(min_val, max_val);
}

absl::Status CreateUnaryElementwiseOp(MlDriftBuildContext& ctx,
                                      const graph::Operation& operation,
                                      ml_drift::OperationType op_type,
                                      absl::string_view op_name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(operation));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input,
                              GetTensorHandle(ctx, operation.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto out, GetTensorHandle(ctx, outputs[0]));
  auto gpu_info = ctx.gpu_model_builder->gpu_info();
  ml_drift::OperationDef op_def = {.src_tensors = {input.tensor_desc},
                                   .dst_tensors = {out.tensor_desc}};
  ml_drift::GPUOperation elementwise;
  if (input.tensor_desc.GetBHWCShape() == out.tensor_desc.GetBHWCShape()) {
    elementwise =
        ml_drift::CreateElementwiseOneInput(gpu_info, op_def, op_type);
  } else {
    elementwise = ml_drift::CreateElementwiseOneInputWithBroadcast(
        gpu_info, op_def, op_type, input.tensor_desc.GetBHWCShape(),
        out.tensor_desc.GetBHWCShape());
  }
  ctx.gpu_model_builder->AddGpuOperation(
      {input.id}, {out.id},
      std::make_unique<ml_drift::GPUOperation>(std::move(elementwise)),
      std::string(op_name));
  return CheckGpuOpOk(ctx, operation);
}

absl::Status CreateBinaryElementwiseOp(MlDriftBuildContext& ctx,
                                       const graph::Operation& operation,
                                       ml_drift::OperationType op_type,
                                       absl::string_view op_name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(operation));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto lhs,
                              GetTensorHandle(ctx, operation.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto rhs,
                              GetTensorHandle(ctx, operation.inputs[1]));
  if ((op_type == ml_drift::OperationType::kMul ||
       op_type == ml_drift::OperationType::kAdd) &&
      lhs.tensor_desc.GetBHWCShape() == rhs.tensor_desc.GetBHWCShape()) {
    if (lhs.id < rhs.id) {
      std::swap(lhs, rhs);
    }
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto out, GetTensorHandle(ctx, outputs[0]));
  auto gpu_info = ctx.gpu_model_builder->gpu_info();

  if (lhs.tensor_desc.GetDataType() != rhs.tensor_desc.GetDataType()) {
    if (lhs.tensor_desc.GetDataType() == ml_drift::DataType::kFloat32 &&
        rhs.tensor_desc.GetDataType() == ml_drift::DataType::kFloat16) {
      lhs = ctx.gpu_model_builder->Cast(lhs, ml_drift::DataType::kFloat16);
    } else if (lhs.tensor_desc.GetDataType() == ml_drift::DataType::kFloat16 &&
               rhs.tensor_desc.GetDataType() == ml_drift::DataType::kFloat32) {
      rhs = ctx.gpu_model_builder->Cast(rhs, ml_drift::DataType::kFloat16);
    }
  }

  ml_drift::OperationDef op_def = {
      .src_tensors = {lhs.tensor_desc, rhs.tensor_desc},
      .dst_tensors = {out.tensor_desc}};
  ml_drift::GPUOperation elementwise;
  if (lhs.tensor_desc.GetBHWCShape() == rhs.tensor_desc.GetBHWCShape()) {
    elementwise = ml_drift::CreateElementwiseTwoInput(
        gpu_info, op_def, op_type, rhs.tensor_desc.GetBHWCShape(),
        out.tensor_desc.GetBHWCShape());
  } else {
    elementwise = ml_drift::CreateElementwiseTwoInputWithBroadcast(
        gpu_info, op_def, op_type, lhs.tensor_desc.GetBHWCShape(),
        rhs.tensor_desc.GetBHWCShape(), out.tensor_desc.GetBHWCShape(),
        ml_drift::ElementwiseAttributes());
  }
  ctx.gpu_model_builder->AddGpuOperation(
      {lhs.id, rhs.id}, {out.id},
      std::make_unique<ml_drift::GPUOperation>(std::move(elementwise)),
      std::string(op_name));
  return CheckGpuOpOk(ctx, operation);
}

struct ScaleZpInfo {
  ml_drift::GpuModelBuilder::TensorHandle scale_handle;
  ml_drift::GpuModelBuilder::TensorHandle zero_point_handle;
  ml_drift::OHWI scale_zp_shape;
};

absl::StatusOr<ScaleZpInfo> CreateScaleAndZeroPointToFCTensor(
    MlDriftBuildContext& ctx, const graph::Tensor& weights,
    ml_drift::DataType float_type) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& out_info, graph::GetInfo(weights));
  if (out_info.quantization == nullptr) {
    return absl::InvalidArgumentError("Weights tensor (" + out_info.name +
                                      ")is not quantized.");
  }

  ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kFloat32> scale_values;
  ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kFloat32> zp_values;
  if (auto pcq =
          out_info.quantization->As<const PerChannelAffineQuantization>();
      pcq.ok()) {
    scale_values.data = pcq->scales;
    scale_values.shape = ml_drift::OHWI(pcq->scales.size(), 1, 1, 1);
    if (pcq->zero_points.empty()) {
      zp_values.data.assign(pcq->scales.size(), 0);
      zp_values.shape = ml_drift::OHWI(pcq->scales.size(), 1, 1, 1);
    } else {
      zp_values.data.assign(pcq->zero_points.begin(), pcq->zero_points.end());
      zp_values.shape = ml_drift::OHWI(pcq->zero_points.size(), 1, 1, 1);
    }
  } else if (auto bwq =
                 out_info.quantization->As<const BlockwiseQuantization>();
             bwq.ok()) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto ohwi, ShapeToOHWI(out_info.shape));
    int num_blocks = ohwi.i / bwq->block_size;
    scale_values.data = bwq->scales;
    scale_values.shape = ml_drift::OHWI(ohwi.o, 1, 1, num_blocks);
    if (bwq->zero_points.empty()) {
      zp_values.data.assign(bwq->scales.size(), 0);
      zp_values.shape = ml_drift::OHWI(ohwi.o, 1, 1, num_blocks);
    } else {
      zp_values.data.assign(bwq->zero_points.begin(), bwq->zero_points.end());
      zp_values.shape = ml_drift::OHWI(ohwi.o, 1, 1, num_blocks);
    }
  } else {
    return absl::InvalidArgumentError("Unsupported quantization type");
  }

  auto scale_desc = ScaleOrZeroPointToTensorDesc(
      ctx.gpu_model_builder->gpu_info(), scale_values, float_type);

  auto scale_handle =
      ctx.gpu_model_builder->AddConstantTensor(std::move(scale_desc));
  auto zero_point_desc = ScaleOrZeroPointToTensorDesc(
      ctx.gpu_model_builder->gpu_info(), zp_values, float_type);
  auto zero_point_handle =
      ctx.gpu_model_builder->AddConstantTensor(std::move(zero_point_desc));
  return ScaleZpInfo{scale_handle, zero_point_handle, scale_values.shape};
}

}  // namespace

namespace graph {

absl::Status OpMixin<AbsOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kAbs,
                                  "abs");
}

absl::Status OpMixin<ReluOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));
  auto out = ctx.gpu_model_builder->ReLU(input, ml_drift::ReLUAttributes());
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<Relu6Operation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));
  ml_drift::ReLUAttributes attr;
  attr.activation_max = 6.0f;
  auto out = ctx.gpu_model_builder->ReLU(input, attr);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<LeakyReluOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const LeakyReluOperation& data,
                              op.As<LeakyReluOperation>());
  ml_drift::ReLUAttributes attr;
  attr.alpha = data.alpha;
  auto out = ctx.gpu_model_builder->ReLU(input, attr);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<EluOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kElu,
                                  "elu");
}

absl::Status OpMixin<HardSwishOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kHardSwish,
                                  "hardswish");
}

absl::Status OpMixin<PReluOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& alpha_info,
                              graph::GetInfo(op.inputs[1]));
  if (alpha_info.buffer == nullptr) {
    return absl::InvalidArgumentError(
        "PRelu alpha must be a constant tensor in ML Drift.");
  }

  ml_drift::PReLUAttributes attr;
  auto alpha_data = alpha_info.buffer->Lock().As<const float>();

  if (alpha_info.shape.size() <= 1) {
    ml_drift::Tensor<ml_drift::Linear, ml_drift::DataType::kFloat32>
        alpha_tensor;
    alpha_tensor.shape =
        ml_drift::Linear(alpha_info.shape.empty() ? 1 : alpha_info.shape[0]);
    alpha_tensor.data.assign(alpha_data.begin(), alpha_data.end());
    attr.alpha = alpha_tensor;
  } else {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto bhwc_shape,
                                ShapeToBHWC(alpha_info.shape, 1));
    ml_drift::Tensor<ml_drift::HWC, ml_drift::DataType::kFloat32> alpha_tensor;
    alpha_tensor.shape =
        ml_drift::HWC(bhwc_shape.h, bhwc_shape.w, bhwc_shape.c);
    alpha_tensor.data.assign(alpha_data.begin(), alpha_data.end());
    attr.alpha = alpha_tensor;
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(auto out, GetTensorHandle(ctx, outputs[0]));
  auto gpu_info = ctx.gpu_model_builder->gpu_info();
  ml_drift::OperationDef op_def = {.src_tensors = {input.tensor_desc},
                                   .dst_tensors = {out.tensor_desc}};
  auto prelu_op = ml_drift::CreatePReLU(gpu_info, op_def, attr);
  ctx.gpu_model_builder->AddGpuOperation(
      {input.id}, {out.id},
      std::make_unique<ml_drift::GPUOperation>(std::move(prelu_op)), "prelu");
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<L2NormalizationOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return absl::UnimplementedError(
      "L2Normalization is not supported in ML Drift.");
}

absl::Status OpMixin<AddOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateBinaryElementwiseOp(ctx, op, ml_drift::OperationType::kAdd,
                                   "add");
}

absl::Status OpMixin<SubOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateBinaryElementwiseOp(ctx, op, ml_drift::OperationType::kSub,
                                   "subtract");
}

absl::Status OpMixin<MulOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateBinaryElementwiseOp(ctx, op, ml_drift::OperationType::kMul,
                                   "mul");
}

absl::Status OpMixin<DivOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateBinaryElementwiseOp(ctx, op, ml_drift::OperationType::kDiv,
                                   "div");
}

absl::Status OpMixin<RsqrtOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kRsqrt,
                                  "rsqrt");
}

absl::Status OpMixin<SqrtOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kSqrt,
                                  "sqrt");
}

absl::Status OpMixin<ExpOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kExp,
                                  "exp");
}

absl::Status OpMixin<LogOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kLog,
                                  "log");
}

absl::Status OpMixin<SquareOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kSquare,
                                  "square");
}

absl::Status OpMixin<NegOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kNeg,
                                  "negate");
}

absl::Status OpMixin<CeilOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kCeil,
                                  "ceil");
}

absl::Status OpMixin<FloorOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kFloor,
                                  "floor");
}

absl::Status OpMixin<FloorDivOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateBinaryElementwiseOp(ctx, op, ml_drift::OperationType::kFloorDiv,
                                   "floor_div");
}

absl::Status OpMixin<FloorModOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateBinaryElementwiseOp(ctx, op, ml_drift::OperationType::kFloorMod,
                                   "floor_mod");
}

absl::Status OpMixin<SignOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kSign,
                                  "sign");
}

absl::Status OpMixin<RoundOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kRound,
                                  "round");
}

absl::Status OpMixin<GreaterEqualOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateBinaryElementwiseOp(
      ctx, op, ml_drift::OperationType::kGreaterEqual, "greater_equal");
}

absl::Status OpMixin<LessOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateBinaryElementwiseOp(ctx, op, ml_drift::OperationType::kLess,
                                   "less");
}

absl::Status OpMixin<LogicalAndOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateBinaryElementwiseOp(
      ctx, op, ml_drift::OperationType::kLogicalAnd, "logical_and");
}

absl::Status OpMixin<LogicalNotOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kLogicalNot,
                                  "logical_not");
}

absl::Status OpMixin<MaximumOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateBinaryElementwiseOp(ctx, op, ml_drift::OperationType::kMaximum,
                                   "maximum");
}

absl::Status OpMixin<EqualOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateBinaryElementwiseOp(ctx, op, ml_drift::OperationType::kEqual,
                                   "equal");
}

absl::Status OpMixin<NotEqualOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateBinaryElementwiseOp(ctx, op, ml_drift::OperationType::kNotEqual,
                                   "not_equal");
}

absl::Status OpMixin<QuantizeOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& out_info, graph::GetInfo(outputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const PerChannelAffineQuantization& quantization,
      out_info.quantization->As<const PerChannelAffineQuantization>());
  const float scale = quantization.scales[0];
  const int64_t zero_point = quantization.zero_points[0];
  LRT_TENSOR_ASSIGN_OR_RETURN(
      (auto [min, max]), GetMinMaxValues(out_info.type, scale, zero_point));
  ml_drift::QuantizeAndDequantizeAttributes attr = {.min = min, .max = max};

  auto out = ctx.gpu_model_builder->QuantizeAndDequantize(input, attr);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});

  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<DequantizeOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& in_info,
                              graph::GetInfo(op.inputs[0]));
  // Already dequantized, no need to dequantize again.
  if (in_info.type == Type::kFP32) {
    return absl::OkStatus();
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      const PerChannelAffineQuantization& quantization,
      in_info.quantization->As<const PerChannelAffineQuantization>());
  const float scale = quantization.scales[0];
  const int64_t zero_point = quantization.zero_points[0];
  LRT_TENSOR_ASSIGN_OR_RETURN((auto [min, max]),
                              GetMinMaxValues(in_info.type, scale, zero_point));
  ml_drift::QuantizeAndDequantizeAttributes attr = {
      .min = min, .max = max, .scale = scale};

  auto out = ctx.gpu_model_builder->QuantizeAndDequantize(input, attr);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});

  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<Conv2DOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const Conv2DOperation& data,
                              op.As<Conv2DOperation>());

  ml_drift::Convolution2DAttributes attr;
  attr.strides = ml_drift::HW(data.stride_h, data.stride_w);
  attr.dilations = ml_drift::HW(data.dilation_h_factor, data.dilation_w_factor);

  LRT_TENSOR_ASSIGN_OR_RETURN(auto filter_info, graph::GetInfo(op.inputs[1]));
  ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kFloat32> float_weights;
  float_weights.shape.o = filter_info.shape[0];
  float_weights.shape.h = filter_info.shape[1];
  float_weights.shape.w = filter_info.shape[2];
  float_weights.shape.i = filter_info.shape[3];

  LRT_TENSOR_ASSIGN_OR_RETURN(auto input_info, graph::GetInfo(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input_bhwc, ShapeToBHWC(input_info.shape));

  // Determine if bias is provided and its value
  bool has_bias = op.inputs.size() >= 3;
  graph::TensorInformation bias_info;
  if (has_bias) {
    LRT_TENSOR_ASSIGN_OR_RETURN(bias_info, graph::GetInfo(op.inputs[2]));
  }

  // Need to set `attr.weights` initially to make CalculateSamePadding work
  attr.weights = float_weights;

  if (data.padding == kPaddingSame) {
    attr.padding = ml_drift::CalculateSamePadding(input_bhwc, attr);
  } else {
    attr.padding.prepended = ml_drift::HW(0, 0);
    attr.padding.appended = ml_drift::HW(0, 0);
  }

  ml_drift::GpuModelBuilder::TensorHandle out;
  if (filter_info.buffer != nullptr &&
      (!has_bias || bias_info.buffer != nullptr)) {
    // Both weights and bias are constants, so use static Conv2D
    auto locked_weights = filter_info.buffer->Lock();
    const float* weights_ptr =
        reinterpret_cast<const float*>(locked_weights.data());
    const size_t num_elements = filter_info.shape[0] * filter_info.shape[1] *
                                filter_info.shape[2] * filter_info.shape[3];
    const size_t extra_elements = 16 / sizeof(float);
    float_weights.data.resize(num_elements + extra_elements);
    std::memcpy(float_weights.data.data(), weights_ptr,
                num_elements * sizeof(float));
    attr.weights = std::move(float_weights);

    if (has_bias) {
      auto locked_bias = bias_info.buffer->Lock();
      const float* bias_ptr =
          reinterpret_cast<const float*>(locked_bias.data());
      attr.bias.shape.v = bias_info.shape[0];
      attr.bias.data.resize(bias_info.shape[0] + extra_elements);
      std::memcpy(attr.bias.data.data(), bias_ptr,
                  bias_info.shape[0] * sizeof(float));
    }

    out = ctx.gpu_model_builder->Convolution(input, attr);
  } else {
    // Use dynamic Conv2D
    LRT_TENSOR_ASSIGN_OR_RETURN(auto filter,
                                GetTensorHandle(ctx, op.inputs[1]));

    std::optional<ml_drift::GpuModelBuilder::TensorHandle> bias = std::nullopt;
    if (has_bias) {
      LRT_TENSOR_ASSIGN_OR_RETURN(
          auto bias_ref, GetTensorHandle(ctx, op.inputs[2], true, true));
      bias = bias_ref;
    }
    const ml_drift::GpuModelBuilder::TensorHandle* bias_ptr =
        bias.has_value() ? &bias.value() : nullptr;

    LRT_TENSOR_ASSIGN_OR_RETURN(
        out, ctx.gpu_model_builder->Convolution(input, filter, bias_ptr, attr));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(out,
                              ApplyFusedActivation(ctx, out, data.activation));
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<DepthwiseConv2DOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const DepthwiseConv2DOperation& data,
                              op.As<DepthwiseConv2DOperation>());

  ml_drift::DepthwiseConvolution2DAttributes attr;
  attr.strides = ml_drift::HW(data.stride_h, data.stride_w);
  attr.dilations = ml_drift::HW(data.dilation_h_factor, data.dilation_w_factor);

  LRT_TENSOR_ASSIGN_OR_RETURN(auto filter_info, graph::GetInfo(op.inputs[1]));
  ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kFloat32> float_weights;
  float_weights.shape.o = filter_info.shape[0];
  float_weights.shape.h = filter_info.shape[1];
  float_weights.shape.w = filter_info.shape[2];
  float_weights.shape.i = filter_info.shape[3];

  LRT_TENSOR_ASSIGN_OR_RETURN(auto input_info, graph::GetInfo(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input_bhwc, ShapeToBHWC(input_info.shape));

  // Determine if bias is provided and its value
  bool has_bias = op.inputs.size() >= 3;
  graph::TensorInformation bias_info;
  if (has_bias) {
    LRT_TENSOR_ASSIGN_OR_RETURN(bias_info, graph::GetInfo(op.inputs[2]));
  }

  // Need to set `attr.weights` initially to make CalculateSamePadding work
  attr.weights = float_weights;

  if (data.padding == kPaddingSame) {
    attr.padding = ml_drift::CalculateSamePadding(input_bhwc, attr);
  } else {
    attr.padding.prepended = ml_drift::HW(0, 0);
    attr.padding.appended = ml_drift::HW(0, 0);
  }

  attr.groups = float_weights.shape.o / data.depth_multiplier;

  ml_drift::GpuModelBuilder::TensorHandle out;
  if (filter_info.buffer != nullptr &&
      (!has_bias || bias_info.buffer != nullptr)) {
    // Both weights and bias are constants, so use static DepthwiseConv2D
    auto locked_weights = filter_info.buffer->Lock();
    const float* weights_ptr =
        reinterpret_cast<const float*>(locked_weights.data());
    const size_t num_elements = filter_info.shape[0] * filter_info.shape[1] *
                                filter_info.shape[2] * filter_info.shape[3];
    const size_t extra_elements = 16 / sizeof(float);
    float_weights.data.resize(num_elements + extra_elements);
    std::memcpy(float_weights.data.data(), weights_ptr,
                num_elements * sizeof(float));
    attr.weights = std::move(float_weights);

    if (has_bias) {
      auto locked_bias = bias_info.buffer->Lock();
      const float* bias_ptr =
          reinterpret_cast<const float*>(locked_bias.data());
      attr.bias.shape.v = bias_info.shape[0];
      attr.bias.data.resize(bias_info.shape[0] + extra_elements);
      std::memcpy(attr.bias.data.data(), bias_ptr,
                  bias_info.shape[0] * sizeof(float));
    }

    out = ctx.gpu_model_builder->DepthwiseConvolution(input, attr);
  } else {
    // Use dynamic DepthwiseConv2D
    LRT_TENSOR_ASSIGN_OR_RETURN(auto filter,
                                GetTensorHandle(ctx, op.inputs[1]));

    if (has_bias) {
      return absl::UnimplementedError(
          "Dynamic DepthwiseConv2D with dynamic bias is not supported");
    }

    LRT_TENSOR_ASSIGN_OR_RETURN(
        out, ctx.gpu_model_builder->DepthwiseConvolution(input, filter, attr));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(out,
                              ApplyFusedActivation(ctx, out, data.activation));
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<AveragePool2DOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const AveragePool2DOperation& data,
                              op.As<AveragePool2DOperation>());

  ml_drift::Pooling2DAttributes attr;
  attr.type = ml_drift::PoolingType::kAverage;
  attr.strides = ml_drift::HW(data.stride_h, data.stride_w);
  attr.kernel = ml_drift::HW(data.filter_height, data.filter_width);

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info,
                              graph::GetInfo(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input_bhwc, ShapeToBHWC(input_info.shape));

  if (data.padding == kPaddingSame) {
    attr.padding = ml_drift::CalculateSamePadding(input_bhwc, attr);
  } else {
    attr.padding.prepended = ml_drift::HW(0, 0);
    attr.padding.appended = ml_drift::HW(0, 0);
  }

  auto out = ctx.gpu_model_builder->Pooling(input, attr);
  LRT_TENSOR_ASSIGN_OR_RETURN(out,
                              ApplyFusedActivation(ctx, out, data.activation));
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<MaxPool2DOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const MaxPool2DOperation& data,
                              op.As<MaxPool2DOperation>());

  ml_drift::Pooling2DAttributes attr;
  attr.type = ml_drift::PoolingType::kMax;
  attr.strides = ml_drift::HW(data.stride_h, data.stride_w);
  attr.kernel = ml_drift::HW(data.filter_height, data.filter_width);

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info,
                              graph::GetInfo(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input_bhwc, ShapeToBHWC(input_info.shape));

  if (data.padding == kPaddingSame) {
    attr.padding = ml_drift::CalculateSamePadding(input_bhwc, attr);
  } else {
    attr.padding.prepended = ml_drift::HW(0, 0);
    attr.padding.appended = ml_drift::HW(0, 0);
  }

  auto out = ctx.gpu_model_builder->Pooling(input, attr);
  LRT_TENSOR_ASSIGN_OR_RETURN(out,
                              ApplyFusedActivation(ctx, out, data.activation));
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<FullyConnectedOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  // Reserve the tensor handles for the op.inputs and outputs to avoid rehashing
  // during the conversion, which would invalidate the references.
  ctx.graph_tensor_to_handle.reserve(4);
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto weights,
      GetTensorHandle(ctx, op.inputs[1], /*create_if_not_found=*/false,
                      /*create_linear_tensor=*/true,
                      /*force_buffer_storage=*/true));

  std::optional<ml_drift::GpuModelBuilder::TensorHandle> bias = std::nullopt;
  if (op.inputs.size() >= 3) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto bias_ref,
                                GetTensorHandle(ctx, op.inputs[2], true, true));
    bias = bias_ref;
  }

  ml_drift::DataType float_type = input.tensor_desc.GetDataType();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input_info, graph::GetInfo(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto weights_info, graph::GetInfo(op.inputs[1]));

  LRT_TENSOR_ASSIGN_OR_RETURN(ml_drift::OHWI weights_ohwi,
                              ShapeToOHWI(weights_info.shape));
  const ml_drift::GpuModelBuilder::TensorHandle* bias_ptr =
      bias.has_value() ? &bias.value() : nullptr;
  ml_drift::GpuModelBuilder::TensorHandle out;
  if (weights_info.type == Type::kFP32) {
    auto actual_input = input;
    auto actual_float_type = float_type;
    bool needs_cast_back = false;
    if (weights.tensor_desc.GetDataType() == ml_drift::DataType::kFloat16) {
      actual_float_type = ml_drift::DataType::kFloat16;
    } else if (ctx.default_float_type == ml_drift::DataType::kFloat16 &&
               input.tensor_desc.GetDataType() ==
                   ml_drift::DataType::kFloat16) {
      // When weights are FP32 but the context uses FP16 (e.g.,
      // ctx.default_float_type == ml_drift::DataType::FLOAT16 and input is
      // also FP16), cast the input to FP32 before FullyConnectedExternalWeights
      // and cast the output back to FP16. This is because some
      // FullyConnectedExternalWeights kernels may not support FP32 weights with
      // FP16 inputs directly.
      actual_input =
          ctx.gpu_model_builder->Cast(input, ml_drift::DataType::kFloat32);
      actual_float_type = ml_drift::DataType::kFloat32;
      needs_cast_back = true;
    }
    const ml_drift::GpuModelBuilder::Weights external_weights =
        CreateExternalWeights(
            weights,
            ctx.gpu_model_builder->GetFullyConnectedWeightsDesc(
                actual_float_type, weights_ohwi),
            weights_ohwi);
    LRT_TENSOR_ASSIGN_OR_RETURN(
        out, ctx.gpu_model_builder->FullyConnectedExternalWeights(
                 actual_input, external_weights, bias_ptr));
    if (needs_cast_back) {
      out = ctx.gpu_model_builder->Cast(out, ml_drift::DataType::kFloat16);
    }
  } else if (weights_info.type == Type::kI8) {
    // TODO(piyu): Support caching scale and zero point tensor.
    LRT_TENSOR_ASSIGN_OR_RETURN(
        (auto [scale_handle, zero_point_handle, scale_zp_shape]),
        CreateScaleAndZeroPointToFCTensor(ctx, op.inputs[1], float_type));
    const ml_drift::GpuModelBuilder::Weights external_weights =
        CreateExternalWeights(
            weights,
            ctx.gpu_model_builder->GetFullyConnectedInt8WeightsDesc(
                weights_ohwi),
            weights_ohwi, scale_zp_shape, &scale_handle, &zero_point_handle);
    out = ctx.gpu_model_builder->FullyConnectedInt8ExternalWeights(
        input, external_weights, bias_ptr);
  } else if (weights_info.type == Type::kI4) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        (auto [scale_handle, zero_point_handle, scale_zp_shape]),
        CreateScaleAndZeroPointToFCTensor(ctx, op.inputs[1], float_type));

    const ml_drift::GpuModelBuilder::Weights external_weights =
        CreateExternalWeights(
            weights,
            ctx.gpu_model_builder->GetFullyConnectedInt4WeightsDesc(
                weights_ohwi),
            weights_ohwi, scale_zp_shape, &scale_handle, &zero_point_handle);

    out = ctx.gpu_model_builder->FullyConnectedInt4ExternalWeights(
        input, external_weights, bias_ptr);
  } else if (weights_info.type == Type::kI2) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        (auto [scale_handle, zero_point_handle, scale_zp_shape]),
        CreateScaleAndZeroPointToFCTensor(ctx, op.inputs[1], float_type));
    const ml_drift::GpuModelBuilder::Weights external_weights =
        CreateExternalWeights(
            weights,
            ctx.gpu_model_builder->GetFullyConnectedInt2WeightsDesc(
                weights_ohwi),
            weights_ohwi, scale_zp_shape, &scale_handle, &zero_point_handle);

    out = ctx.gpu_model_builder->FullyConnectedInt2ExternalWeights(
        input, external_weights, bias_ptr);
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(const FullyConnectedOperation& data,
                              op.As<FullyConnectedOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(out,
                              ApplyFusedActivation(ctx, out, data.activation));
  ctx.graph_tensor_to_handle.insert({outputs[0], out});

  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<SoftmaxOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));
  auto out = ctx.gpu_model_builder->Softmax(input);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});

  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<BatchMatMulOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto lhs, GetTensorHandle(ctx, op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto rhs, GetTensorHandle(ctx, op.inputs[1]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& lhs_info,
                              graph::GetInfo(op.inputs[0]));
  // lhs cannot be constant tensor.
  if (lhs_info.buffer != nullptr) {
    return absl::UnimplementedError(
        "BatchMatMul with constant lhs is not supported.");
  }
  // lhs cannot have rank < 2.
  if (lhs_info.shape.size() < 2) {
    return absl::UnimplementedError(
        "BatchMatMul with rank < 2 lhs is not supported.");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& rhs_info,
                              graph::GetInfo(op.inputs[1]));
  // if rhs is constant, it has to be 2D.
  if (rhs_info.buffer != nullptr && rhs_info.shape.size() != 2) {
    return absl::UnimplementedError(
        "BatchMatMul with constant rhs is only supported for 2D tensors.");
  }
  // rhs cannot have rank < 2.
  if (rhs_info.shape.size() < 2) {
    return absl::UnimplementedError(
        "BatchMatMul with rank < 2 rhs is not supported.");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const BatchMatMulOperation& data,
                              op.As<BatchMatMulOperation>());
  ml_drift::BatchedMatMulAttributes attr;
  attr.transpose_left = data.adj_x;
  attr.transpose_right = data.adj_y;
  if (attr.transpose_right) {
    const int k = rhs_info.shape.back();
    if (k % 4 != 0) {
      // Avoid the FC-optimized batched-matmul path which requires K % 4 == 0
      // by explicitly transposing the RHS and disabling transpose_right.
      rhs = ctx.gpu_model_builder->Transpose(rhs, ml_drift::BHWC(0, 1, 3, 2));
      attr.transpose_right = false;
    }
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto out, ctx.gpu_model_builder->BatchedMatMul(lhs, rhs, attr));
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<TransposeOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& perm_info,
                              graph::GetInfo(op.inputs[1]));
  ABSL_CHECK(perm_info.buffer != nullptr);
  auto perm_data = perm_info.buffer->Lock().As<const int32_t>();
  std::vector<int32_t> perm(perm_data.data(),
                            perm_data.data() + perm_data.size());

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info,
                              graph::GetInfo(op.inputs[0]));
  const int rank = input_info.shape.size();
  std::vector<int32_t> adj_perm;
  if (rank > 5) {
    return absl::UnimplementedError(
        "Transpose with rank > 5 is not supported.");
  }
  switch (rank) {
    case 5:
      // Input shape is BHWDC, ML Drift shape is BHWDC.
      adj_perm = perm;
      break;
    case 4:
      // Input shape is BHWC, ML Drift shape is BHWC.
      adj_perm = perm;
      break;
    case 3:
      // Input shape is HWC, ML Drift shape is 1HWC.
      // Permutation values need to be shifted by 1.
      adj_perm = {0, perm[0] + 1, perm[1] + 1, perm[2] + 1};
      break;
    case 2: {
      // Input shape is BC, ML Drift shape is B11C.
      // Permutation is on B and C axes.
      auto map_axis = [](int axis) { return axis == 0 ? 0 : 3; };
      adj_perm = {map_axis(perm[0]), 1, 2, map_axis(perm[1])};
      break;
    }
    default:
      // For rank 1 or 0, no transpose is needed.
      adj_perm = {0, 1, 2, 3};
      break;
  }

  if (rank == 5) {
    if (adj_perm[0] != 0 || adj_perm[1] != 1) {
      return absl::UnimplementedError(
          "5D Transpose only supported if first 2 dims are preserved.");
    }
    // Map 5D perm to 4D: 0, 1, 2, 3, 4 -> 0, (merged), 1, 2, 3
    // adj_perm has indices from 0 to 4.
    // If indices match 0 and 1, we ignore them.
    // For indices 2, 3, 4, we map them to 1, 2, 3? No.
    // BHWC indices: 0(Batch), 1(Height), 2(Width), 3(Channel).
    // Our squashed tensor is [B*H, D2, D3, D4] -> [B_new, H_new, W_new, C_new]
    // B_new corresponds to original dims 0 and 1. Index 0 in 4D.
    // H_new corresponds to original dim 2. Index 1 in 4D.
    // W_new corresponds to original dim 3. Index 2 in 4D.
    // C_new corresponds to original dim 4. Index 3 in 4D.

    // We want output permutation.
    // adj_perm specifies output dim i comes from input dim adj_perm[i].
    // Since output dim 0 and 1 (in 5D) are 0 and 1, the new output dim 0
    // (in 4D) is just 0.
    // New output dim 1 (H_new) corresponds to 5D output dim 2. It comes
    // from 5D input dim adj_perm[2].
    // adj_perm[2] will be 2, 3, or 4 (since 0,1 are fixed).
    // If adj_perm[2] == 2, it maps to 4D index 1.
    // If adj_perm[2] == 3, it maps to 4D index 2.
    // If adj_perm[2] == 4, it maps to 4D index 3.
    // So map: x -> x - 1.
    ml_drift::BHWC perm_shape(0, adj_perm[2] - 1, adj_perm[3] - 1,
                              adj_perm[4] - 1);
    auto out = ctx.gpu_model_builder->Transpose(input, perm_shape);
    ctx.graph_tensor_to_handle.insert({outputs[0], out});
    return CheckGpuOpOk(ctx, op);
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(auto perm_shape, ShapeToBHWC(adj_perm));

  auto out = ctx.gpu_model_builder->Transpose(input, perm_shape);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<SqueezeOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_RETURN_IF_ERROR(op.As<SqueezeOperation>().status());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& output_info,
                              graph::GetInfo(outputs[0]));

  ml_drift::BHWC shape(1, 1, 1, 1);
  if (output_info.shape.size() == 5) {
    shape = ml_drift::BHWC(output_info.shape[0] * output_info.shape[1],
                           output_info.shape[2], output_info.shape[3],
                           output_info.shape[4]);
  } else {
    LRT_TENSOR_ASSIGN_OR_RETURN(shape, ShapeToBHWC(output_info.shape));
  }
  auto out = ctx.gpu_model_builder->Reshape(input, shape);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<ExpandDimsOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));

  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError("ExpandDims expects 2 inputs");
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& axis_info,
                              graph::GetInfo(op.inputs[1]));
  if (axis_info.type != Type::kI32) {
    return absl::InvalidArgumentError("ExpandDims axis must be int32.");
  }
  if (axis_info.buffer == nullptr) {
    return absl::InvalidArgumentError(
        "ExpandDims axis must be a constant tensor");
  }
  auto locked_axis = axis_info.buffer->Lock().As<const int32_t>();
  int real_axis = locked_axis.data()[0];

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info,
                              graph::GetInfo(op.inputs[0]));
  std::vector<int> new_shape = input_info.shape;

  if (real_axis < 0) {
    real_axis += input_info.shape.size() + 1;
  }
  new_shape.insert(new_shape.begin() + real_axis, 1);

  if (new_shape.size() == 5) {
    ml_drift::BHWC shape(new_shape[0] * new_shape[1], new_shape[2],
                         new_shape[3], new_shape[4]);
    auto out = ctx.gpu_model_builder->Reshape(input, shape);
    ctx.graph_tensor_to_handle.insert({outputs[0], out});
    return CheckGpuOpOk(ctx, op);
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(auto shape_bhwc, ShapeToBHWC(new_shape));
  auto out = ctx.gpu_model_builder->Reshape(input, shape_bhwc);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<ReshapeOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(const ReshapeOperation& data,
                              op.As<ReshapeOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));
  if (data.new_shape.size() == 5) {
    // ml_drift::Reshape only supports 4D (BHWC).
    // For 5D shapes [d0, d1, d2, d3, d4], we squash the first two dimensions
    // to map to [d0*d1, d2, d3, d4] in BHWC.
    ml_drift::BHWC shape(data.new_shape[0] * data.new_shape[1],
                         data.new_shape[2], data.new_shape[3],
                         data.new_shape[4]);
    auto out = ctx.gpu_model_builder->Reshape(input, shape);
    ctx.graph_tensor_to_handle.insert({outputs[0], out});
    return CheckGpuOpOk(ctx, op);
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto shape, ShapeToBHWC(data.new_shape));
  auto out = ctx.gpu_model_builder->Reshape(input, shape);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<GeluOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kGelu,
                                  "gelu");
}

absl::Status OpMixin<SinOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kSin,
                                  "sin");
}

absl::Status OpMixin<CosOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kCos,
                                  "cos");
}

absl::Status OpMixin<LogisticOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kSigmoid,
                                  "sigmoid");
}

absl::Status OpMixin<PowOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateBinaryElementwiseOp(ctx, op, ml_drift::OperationType::kPow,
                                   "pow");
}

absl::Status OpMixin<TanhOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  return CreateUnaryElementwiseOp(ctx, op, ml_drift::OperationType::kTanh,
                                  "tanh");
}

absl::Status OpMixin<ProbeOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));
  auto shape = input.tensor_desc.GetBHWCShape();
  auto out = ctx.gpu_model_builder->Reshape(input, shape);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<ResizeBilinearOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& size_info,
                              graph::GetInfo(op.inputs[1]));
  ABSL_CHECK(size_info.buffer != nullptr)
      << "ResizeBilinear requires constant size for ML Drift backend.";
  auto size_data = size_info.buffer->Lock().As<const int32_t>();
  int new_height = size_data.data()[0];
  int new_width = size_data.data()[1];

  LRT_TENSOR_ASSIGN_OR_RETURN(const ResizeBilinearOperation& data,
                              op.As<ResizeBilinearOperation>());

  auto out = ctx.gpu_model_builder->ResizeBilinear(
      input, ml_drift::HW(new_height, new_width), data.align_corners,
      data.half_pixel_centers);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status
OpMixin<ResizeNearestNeighborOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& size_info,
                              graph::GetInfo(op.inputs[1]));
  ABSL_CHECK(size_info.buffer != nullptr)
      << "ResizeNearestNeighbor requires constant size for ML Drift backend.";
  auto size_data = size_info.buffer->Lock().As<const int32_t>();
  int new_height = size_data.data()[0];
  int new_width = size_data.data()[1];

  LRT_TENSOR_ASSIGN_OR_RETURN(const ResizeNearestNeighborOperation& data,
                              op.As<ResizeNearestNeighborOperation>());

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info,
                              graph::GetInfo(op.inputs[0]));
  // Input shape is BHWC (from graph info).
  int input_h = input_info.shape[1];
  int input_w = input_info.shape[2];

  if (new_height % input_h != 0 || new_width % input_w != 0) {
    return absl::UnimplementedError(absl::StrCat(
        "ML Drift ResizeNearestNeighbor only supports integer "
        "scaling. Input: ",
        input_h, "x", input_w, ", Output: ", new_height, "x", new_width));
  }
  int scale_h = new_height / input_h;
  int scale_w = new_width / input_w;
  if (scale_h != scale_w) {
    return absl::UnimplementedError(
        absl::StrCat("ML Drift ResizeNearestNeighbor only supports isotropic "
                     "scaling. Scales: ",
                     scale_h, " vs ", scale_w));
  }

  auto out = ctx.gpu_model_builder->ResizeNearest(
      input, scale_h, data.align_corners, data.half_pixel_centers);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<LstmOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto intermediate,
                              GetTensorHandle(ctx, op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto prev_state,
                              GetTensorHandle(ctx, op.inputs[1]));

  auto dst_new_state =
      ctx.gpu_model_builder->AddTensor(prev_state.tensor_desc.GetBHWCShape(),
                                       prev_state.tensor_desc.GetDataType());
  auto dst_activation =
      ctx.gpu_model_builder->AddTensor(prev_state.tensor_desc.GetBHWCShape(),
                                       prev_state.tensor_desc.GetDataType());

  ml_drift::OperationDef op_def;
  op_def.src_tensors.push_back(intermediate.tensor_desc);
  op_def.src_tensors.push_back(prev_state.tensor_desc);
  op_def.dst_tensors.push_back(dst_new_state.tensor_desc);
  op_def.dst_tensors.push_back(dst_activation.tensor_desc);

  auto gpu_op = ml_drift::SelectLSTM(op_def, ctx.gpu_model_builder->gpu_info());
  ctx.gpu_model_builder->AddGpuOperation({intermediate.id, prev_state.id},
                                         {dst_new_state.id, dst_activation.id},
                                         std::move(gpu_op), "lstm");

  ctx.graph_tensor_to_handle.insert({outputs[0], dst_new_state});
  ctx.graph_tensor_to_handle.insert({outputs[1], dst_activation});

  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<TransposeConvOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  // Inputs: output_shape (0), filter (1), input (2)
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[2]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const TransposeConvOperation& data,
                              op.As<TransposeConvOperation>());

  ml_drift::ConvolutionTransposedAttributes attr;
  attr.stride = ml_drift::HW(data.stride_h, data.stride_w);

  LRT_TENSOR_ASSIGN_OR_RETURN(auto filter_info, graph::GetInfo(op.inputs[1]));
  // TFLite weights for TransposeConv: [O, H, W, I]
  attr.weights.shape.o = filter_info.shape[0];
  attr.weights.shape.h = filter_info.shape[1];
  attr.weights.shape.w = filter_info.shape[2];
  attr.weights.shape.i = filter_info.shape[3];

  LRT_TENSOR_ASSIGN_OR_RETURN(auto input_info, graph::GetInfo(op.inputs[2]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input_bhwc, ShapeToBHWC(input_info.shape));

  if (data.padding == kPaddingSame) {
    attr.padding = ml_drift::CalculateSamePadding(input_bhwc, attr);
  } else {
    attr.padding.prepended = ml_drift::HW(0, 0);
    attr.padding.appended = ml_drift::HW(0, 0);
  }

  // Use output_shape tensor (input 0) to calculate adjacent.
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& out_shape_info,
                              graph::GetInfo(op.inputs[0]));
  if (out_shape_info.buffer != nullptr) {
    auto out_shape_data = out_shape_info.buffer->Lock().As<const int32_t>();
    int expected_h = out_shape_data.data()[1];
    int expected_w = out_shape_data.data()[2];

    ml_drift::BHWC calculated_out =
        ml_drift::CalculateOutputShape(input_bhwc, attr);
    attr.adjacent.h = std::max(0, expected_h - calculated_out.h);
    attr.adjacent.w = std::max(0, expected_w - calculated_out.w);
  }

  ml_drift::GpuModelBuilder::TensorHandle out;
  if (filter_info.buffer != nullptr) {
    auto locked_weights = filter_info.buffer->Lock();
    const float* weights_ptr =
        reinterpret_cast<const float*>(locked_weights.data());
    const size_t num_elements = filter_info.shape[0] * filter_info.shape[1] *
                                filter_info.shape[2] * filter_info.shape[3];
    // XNNPACK requires extra bytes at the end of weights.
    const size_t extra_elements = 16 / sizeof(float);
    attr.weights.data.resize(num_elements + extra_elements);
    std::memcpy(attr.weights.data.data(), weights_ptr,
                num_elements * sizeof(float));
    out = ctx.gpu_model_builder->ConvolutionTransposed(input, attr);
  } else {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto filter,
                                GetTensorHandle(ctx, op.inputs[1]));
    out = ctx.gpu_model_builder->ConvolutionTransposed(input, filter, attr);
  }

  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<TransposeConv2DOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  // Inputs: output_shape (0), filter (1), input (2)
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[2]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const TransposeConv2DOperation& data,
                              op.As<TransposeConv2DOperation>());

  ml_drift::ConvolutionTransposedAttributes attr;
  attr.stride = ml_drift::HW(data.stride_h, data.stride_w);

  LRT_TENSOR_ASSIGN_OR_RETURN(auto filter_info, graph::GetInfo(op.inputs[1]));
  // TFLite weights for TransposeConv: [O, H, W, I]
  attr.weights.shape.o = filter_info.shape[0];
  attr.weights.shape.h = filter_info.shape[1];
  attr.weights.shape.w = filter_info.shape[2];
  attr.weights.shape.i = filter_info.shape[3];

  LRT_TENSOR_ASSIGN_OR_RETURN(auto input_info, graph::GetInfo(op.inputs[1]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input_bhwc, ShapeToBHWC(input_info.shape));

  if (data.padding == kPaddingSame) {
    attr.padding = ml_drift::CalculateSamePadding(input_bhwc, attr);
  } else {
    attr.padding.prepended = ml_drift::HW(0, 0);
    attr.padding.appended = ml_drift::HW(0, 0);
  }

  // Use output_shape tensor (input 0) to calculate adjacent.
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& out_shape_info,
                              graph::GetInfo(op.inputs[0]));
  if (out_shape_info.buffer != nullptr) {
    auto out_shape_data = out_shape_info.buffer->Lock().As<const int32_t>();
    int expected_h = out_shape_data.data()[1];
    int expected_w = out_shape_data.data()[2];

    ml_drift::BHWC calculated_out =
        ml_drift::CalculateOutputShape(input_bhwc, attr);
    attr.adjacent.h = std::max(0, expected_h - calculated_out.h);
    attr.adjacent.w = std::max(0, expected_w - calculated_out.w);
  }

  ml_drift::GpuModelBuilder::TensorHandle out;
  if (filter_info.buffer != nullptr) {
    auto locked_weights = filter_info.buffer->Lock();
    const float* weights_ptr =
        reinterpret_cast<const float*>(locked_weights.data());
    const size_t num_elements = filter_info.shape[0] * filter_info.shape[1] *
                                filter_info.shape[2] * filter_info.shape[3];
    // XNNPACK requires extra bytes at the end of weights.
    const size_t extra_elements = 16 / sizeof(float);
    attr.weights.data.resize(num_elements + extra_elements);
    std::memcpy(attr.weights.data.data(), weights_ptr,
                num_elements * sizeof(float));
    out = ctx.gpu_model_builder->ConvolutionTransposed(input, attr);
  } else {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto filter,
                                GetTensorHandle(ctx, op.inputs[1]));
    out = ctx.gpu_model_builder->ConvolutionTransposed(input, filter, attr);
  }

  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<CastOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const CastOperation& data,
                              op.As<CastOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto to_type, GetMlDriftDataType(data.to, ctx.default_float_type));
  auto out = ctx.gpu_model_builder->Cast(input, to_type);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});

  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<OneHotOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(const OneHotOperation& data,
                              op.As<OneHotOperation>());

  LRT_TENSOR_ASSIGN_OR_RETURN(auto indices, GetTensorHandle(ctx, op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto out, GetTensorHandle(ctx, outputs[0]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& output_info,
                              graph::GetInfo(outputs[0]));
  int rank = output_info.shape.size();
  int resolved_axis = data.axis;
  if (resolved_axis < 0) {
    resolved_axis += rank;
  }
  if (resolved_axis != rank - 1) {
    return absl::UnimplementedError(
        "ML Drift OneHot only supports axis = -1 (last dimension).");
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& on_value_info,
                              graph::GetInfo(op.inputs[2]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& off_value_info,
                              graph::GetInfo(op.inputs[3]));

  if (on_value_info.buffer == nullptr || off_value_info.buffer == nullptr) {
    return absl::InvalidArgumentError(
        "OneHot on_value and off_value must be constants in ML Drift.");
  }
  if (on_value_info.type != Type::kFP32 || off_value_info.type != Type::kFP32) {
    return absl::UnimplementedError(
        "ML Drift OneHot currently only supports float32 values.");
  }

  LockedBufferSpan<const float> on_lock =
      on_value_info.buffer->Lock().As<const float>();
  float on_value = on_lock.data()[0];
  LockedBufferSpan<const float> off_lock =
      off_value_info.buffer->Lock().As<const float>();
  float off_value = off_lock.data()[0];

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& indices_info,
                              graph::GetInfo(op.inputs[0]));
  int n = 1;
  for (int dim : indices_info.shape) {
    n *= dim;
  }
  int depth = output_info.shape.back();
  auto indices_flat =
      ctx.gpu_model_builder->Reshape(indices, ml_drift::BHWC(n, 1, 1, 1));
  auto out_flat = ctx.gpu_model_builder->AddTensor(
      ml_drift::BHWC(n, 1, 1, depth), out.tensor_desc.GetDataType());

  ml_drift::OneHotAttributes attr;
  attr.on_value = on_value;
  attr.off_value = off_value;

  auto gpu_info = ctx.gpu_model_builder->gpu_info();
  ml_drift::OperationDef op_def = {.src_tensors = {indices_flat.tensor_desc},
                                   .dst_tensors = {out_flat.tensor_desc}};

  auto gpu_op = ml_drift::CreateOneHot(op_def, attr);
  ctx.gpu_model_builder->AddGpuOperation(
      {indices_flat.id}, {out_flat.id},
      std::make_unique<ml_drift::GPUOperation>(std::move(gpu_op)), "one_hot");

  auto out_reshaped =
      ctx.gpu_model_builder->Reshape(out_flat, out.tensor_desc.GetBHWCShape());
  ctx.graph_tensor_to_handle[outputs[0]] = out_reshaped;

  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<GatherOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto indices, GetTensorHandle(ctx, op.inputs[1]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const GatherOperation& data,
                              op.As<GatherOperation>());

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info,
                              graph::GetInfo(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto mld_axis,
                              GetMlDriftAxis(input_info.shape, data.axis));

  if (data.batch_dims != 0) {
    return absl::UnimplementedError(
        "Batch dims != 0 is not supported for Gather in ML Drift.");
  }

  auto out = ctx.gpu_model_builder->Gather(input, indices, mld_axis);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<CumsumOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& axis_info,
                              graph::GetInfo(op.inputs[1]));
  ABSL_CHECK(axis_info.buffer != nullptr);
  LockedBufferSpan<const int32_t> axis_lock =
      axis_info.buffer->Lock().As<const int32_t>();
  const int axis = axis_lock.data()[0];
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info,
                              graph::GetInfo(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto mld_axis,
                              GetMlDriftAxis(input_info.shape, axis));
  auto out = ctx.gpu_model_builder->Cumsum(input, mld_axis);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<DepthToSpaceOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const DepthToSpaceOperation& data,
                              op.As<DepthToSpaceOperation>());

  auto out = ctx.gpu_model_builder->DepthToSpace(input, data.block_size);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<ReverseOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto out, GetTensorHandle(ctx, outputs[0]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& axes_info,
                              graph::GetInfo(op.inputs[1]));
  ABSL_CHECK(axes_info.buffer != nullptr)
      << "Reverse requires constant axes for ML Drift backend.";

  auto axes_data = axes_info.buffer->Lock().As<const int32_t>();
  std::set<ml_drift::Axis> mld_axes;
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info,
                              graph::GetInfo(op.inputs[0]));

  for (int i = 0; i < axes_data.size(); ++i) {
    int axis = axes_data.data()[i];
    if (axis < 0) {
      axis += input_info.shape.size();
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(auto mld_axis,
                                GetMlDriftAxis(input_info.shape, axis));
    mld_axes.insert(mld_axis);
  }

  ml_drift::ReverseAttributes attr;
  attr.axes = mld_axes;

  ml_drift::OperationDef op_def = {.src_tensors = {input.tensor_desc},
                                   .dst_tensors = {out.tensor_desc}};

  auto gpu_op = ml_drift::SelectReverse(attr, op_def);
  ctx.gpu_model_builder->AddGpuOperation({input.id}, {out.id},
                                         std::move(gpu_op), "reverse");

  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<SumOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& axes_info,
                              graph::GetInfo(op.inputs[1]));
  ABSL_CHECK(axes_info.buffer != nullptr);
  auto axes_data = axes_info.buffer->Lock().As<const int32_t>();
  std::vector<int32_t> axes(axes_data.data(),
                            axes_data.data() + axes_data.size());

  std::set<ml_drift::Axis> mld_axes;
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info,
                              graph::GetInfo(op.inputs[0]));
  for (int axis : axes) {
    if (axis < 0) {
      axis += input_info.shape.size();
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(auto mld_axis,
                                GetMlDriftAxis(input_info.shape, axis));
    mld_axes.insert(mld_axis);
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(const SumOperation& data, op.As<SumOperation>());

  auto out = ctx.gpu_model_builder->Reduce(input, ml_drift::Reduce::Type::kSum,
                                           mld_axes);
  if (!data.keep_dims) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto& out_info,
                                graph::GetInfo(outputs[0]));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto final_shape_bhwc,
                                ShapeToBHWC(out_info.shape));
    out = ctx.gpu_model_builder->Reshape(out, final_shape_bhwc);
  }
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<ReduceMaxOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(const ReduceMaxOperation& data,
                              op.As<ReduceMaxOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& axes_info,
                              graph::GetInfo(op.inputs[1]));
  ABSL_CHECK(axes_info.buffer != nullptr);
  auto axes_data = axes_info.buffer->Lock().As<const int32_t>();
  std::vector<int32_t> axes(axes_data.data(),
                            axes_data.data() + axes_data.size());

  std::set<ml_drift::Axis> mld_axes;
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info,
                              graph::GetInfo(op.inputs[0]));
  for (int axis : axes) {
    if (axis < 0) {
      axis += input_info.shape.size();
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(auto mld_axis,
                                GetMlDriftAxis(input_info.shape, axis));
    mld_axes.insert(mld_axis);
  }

  auto out = ctx.gpu_model_builder->Reduce(
      input, ml_drift::Reduce::Type::kMaximum, mld_axes);
  if (!data.keep_dims) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto& out_info,
                                graph::GetInfo(outputs[0]));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto final_shape_bhwc,
                                ShapeToBHWC(out_info.shape));
    out = ctx.gpu_model_builder->Reshape(out, final_shape_bhwc);
  }
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<ArgMaxOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& axis_info,
                              graph::GetInfo(op.inputs[1]));
  if (axis_info.buffer == nullptr) {
    return absl::InvalidArgumentError("ArgMax axis must be a constant tensor.");
  }
  auto axis_data = axis_info.buffer->Lock().As<const int32_t>();
  int32_t axis = axis_data.data()[0];

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info,
                              graph::GetInfo(op.inputs[0]));
  if (axis < 0) {
    axis += input_info.shape.size();
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto mld_axis,
                              GetMlDriftAxis(input_info.shape, axis));

  auto out = ctx.gpu_model_builder->Reduce(
      input, ml_drift::Reduce::Type::kMaximumIndex, {mld_axis});

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& out_info, graph::GetInfo(outputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto final_shape_bhwc,
                              ShapeToBHWC(out_info.shape));
  if (out.tensor_desc.GetBHWCShape() != final_shape_bhwc) {
    out = ctx.gpu_model_builder->Reshape(out, final_shape_bhwc);
  }

  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<MeanOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(const MeanOperation& data,
                              op.As<MeanOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& axes_info,
                              graph::GetInfo(op.inputs[1]));
  ABSL_CHECK(axes_info.buffer != nullptr);
  auto axes_data = axes_info.buffer->Lock().As<const int32_t>();
  std::vector<int32_t> axes(axes_data.data(),
                            axes_data.data() + axes_data.size());

  std::set<ml_drift::Axis> mld_axes;
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info,
                              graph::GetInfo(op.inputs[0]));
  for (int axis : axes) {
    if (axis < 0) {
      axis += input_info.shape.size();
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(auto mld_axis,
                                GetMlDriftAxis(input_info.shape, axis));
    mld_axes.insert(mld_axis);
  }

  auto out = ctx.gpu_model_builder->Reduce(input, ml_drift::Reduce::Type::kMean,
                                           mld_axes);
  if (!data.keep_dims) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto& out_info,
                                graph::GetInfo(outputs[0]));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto final_shape_bhwc,
                                ShapeToBHWC(out_info.shape));
    out = ctx.gpu_model_builder->Reshape(out, final_shape_bhwc);
  }
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<SelectOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto condition,
                              GetTensorHandle(ctx, op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto t, GetTensorHandle(ctx, op.inputs[1]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto e, GetTensorHandle(ctx, op.inputs[2]));
  auto out = ctx.gpu_model_builder->SelectV2(condition, t, e);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});

  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<SelectV2Operation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto condition,
                              GetTensorHandle(ctx, op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto t, GetTensorHandle(ctx, op.inputs[1]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto e, GetTensorHandle(ctx, op.inputs[2]));
  auto out = ctx.gpu_model_builder->SelectV2(condition, t, e);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});

  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<SliceOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& begin_info,
                              graph::GetInfo(op.inputs[1]));
  ABSL_CHECK(begin_info.buffer != nullptr);
  auto begin_data = begin_info.buffer->Lock().As<const int32_t>();
  std::vector<int> begin(begin_data.data(),
                         begin_data.data() + begin_data.size());

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& size_info,
                              graph::GetInfo(op.inputs[2]));
  ABSL_CHECK(size_info.buffer != nullptr);
  auto size_data = size_info.buffer->Lock().As<const int32_t>();
  std::vector<int> size(size_data.data(), size_data.data() + size_data.size());

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info,
                              graph::GetInfo(op.inputs[0]));
  for (int i = 0; i < size.size(); ++i) {
    if (size[i] == -1) {
      size[i] = input_info.shape[i] - begin[i];
    }
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto start_bhwc, ShapeToBHWC(begin, 0));
  std::vector<int> end;
  end.reserve(begin.size());
  for (int i = 0; i < begin.size(); ++i) {
    end.push_back(begin[i] + size[i]);
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto end_bhwc, ShapeToBHWC(end, 1));

  ml_drift::SliceAttributes attr;
  attr.starts = start_bhwc;
  attr.ends = end_bhwc;
  attr.strides = ml_drift::BHWC(1, 1, 1, 1);

  auto out = ctx.gpu_model_builder->StridedSlice(input, attr);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<ConcatenationOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  std::vector<ml_drift::GpuModelBuilder::TensorHandle> input_handles;
  for (const auto& input : op.inputs) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto handle, GetTensorHandle(ctx, input));
    input_handles.push_back(handle);
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(const ConcatenationOperation& data,
                              op.As<ConcatenationOperation>());
  const int axis = data.axis;
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& output_info,
                              graph::GetInfo(outputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto axis_info,
                              GetMlDriftAxis(output_info.shape, axis));
  auto out = ctx.gpu_model_builder->Concat(input_handles, axis_info);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});

  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<PackOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  std::vector<ml_drift::GpuModelBuilder::TensorHandle> reshaped_handles;
  LRT_TENSOR_ASSIGN_OR_RETURN(const PackOperation& data,
                              op.As<PackOperation>());
  for (const auto& input : op.inputs) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto handle, GetTensorHandle(ctx, input));
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, graph::GetInfo(input));
    std::vector<int> new_shape = info.shape;
    new_shape.insert(new_shape.begin() + data.axis, 1);
    LRT_TENSOR_ASSIGN_OR_RETURN(auto new_bhwc, ShapeToBHWC(new_shape));
    reshaped_handles.push_back(
        ctx.gpu_model_builder->Reshape(handle, new_bhwc));
  }

  const int axis = data.axis;
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& output_info,
                              graph::GetInfo(outputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto axis_info,
                              GetMlDriftAxis(output_info.shape, axis));
  auto out = ctx.gpu_model_builder->Concat(reshaped_handles, axis_info);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});

  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<UnpackOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const UnpackOperation& data,
                              op.As<UnpackOperation>());
  const int axis = data.axis;
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info,
                              graph::GetInfo(op.inputs[0]));
  const int size = input_info.shape[axis] / data.num;
  LRT_TENSOR_ASSIGN_OR_RETURN(auto axis_info,
                              GetMlDriftAxis(input_info.shape, axis));

  auto split_outputs = ctx.gpu_model_builder->Split(input, axis_info, size);

  for (int i = 0; i < split_outputs.size(); ++i) {
    ctx.graph_tensor_to_handle.insert({outputs[i], split_outputs[i]});
  }

  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<SplitOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));

  LRT_TENSOR_ASSIGN_OR_RETURN(const SplitOperation& data,
                              op.As<SplitOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& axis_info,
                              graph::GetInfo(op.inputs[1]));
  ABSL_CHECK(axis_info.buffer != nullptr);
  LockedBufferSpan<const int32_t> axis_lock =
      axis_info.buffer->Lock().As<const int32_t>();
  const int axis_val = axis_lock.data()[0];

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_graph_info,
                              graph::GetInfo(op.inputs[0]));
  const int split_dim_size = input_graph_info.shape[axis_val];
  const int size = split_dim_size / data.num_splits;

  LRT_TENSOR_ASSIGN_OR_RETURN(auto mld_axis,
                              GetMlDriftAxis(input_graph_info.shape, axis_val));

  auto split_outputs = ctx.gpu_model_builder->Split(input, mld_axis, size);

  for (int i = 0; i < split_outputs.size(); ++i) {
    ctx.graph_tensor_to_handle.insert({outputs[i], split_outputs[i]});
  }

  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<DynamicUpdateSliceOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto operand, GetTensorHandle(ctx, op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto update, GetTensorHandle(ctx, op.inputs[1]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto start_indices,
                              GetTensorHandle(ctx, op.inputs[2]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto out, GetTensorHandle(ctx, outputs[0]));
  auto gpu_info = ctx.gpu_model_builder->gpu_info();
  ml_drift::OperationDef op_def = {
      .src_tensors = {{operand.tensor_desc.GetDataType(),
                       operand.tensor_desc.GetStorageType(),
                       operand.tensor_desc.GetBHWDCShape().b == 1
                           ? ml_drift::Layout::kHWC
                           : ml_drift::Layout::kBHWC},
                      {update.tensor_desc.GetDataType(),
                       update.tensor_desc.GetStorageType(),
                       update.tensor_desc.GetBHWDCShape().b == 1
                           ? ml_drift::Layout::kHWC
                           : ml_drift::Layout::kBHWC},
                      {start_indices.tensor_desc.GetDataType(),
                       start_indices.tensor_desc.GetStorageType(),
                       start_indices.tensor_desc.GetBHWDCShape().b == 1
                           ? ml_drift::Layout::kHWC
                           : ml_drift::Layout::kBHWC}},
      .dst_tensors = {
          {out.tensor_desc.GetDataType(), out.tensor_desc.GetStorageType(),
           out.tensor_desc.GetBHWDCShape().b == 1 ? ml_drift::Layout::kHWC
                                                  : ml_drift::Layout::kBHWC}}};
  auto dynamic_update_slice =
      ml_drift::SelectDynamicUpdateSlice(op_def, gpu_info);
  ctx.gpu_model_builder->AddGpuOperation(
      {operand.id, update.id, start_indices.id}, {out.id},
      std::move(dynamic_update_slice), "dynamic_update_slice");

  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<EmbeddingLookupOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  ml_drift::DataType float_type = ctx.default_float_type;
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  auto ids_or =
      GetTensorHandle(ctx, op.inputs[0], /*create_if_not_found=*/false);
  ml_drift::GpuModelBuilder::TensorHandle ids;
  int batch = 1;
  int seq_len = 1;
  if (!ids_or.ok()) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto& ids_info,
                                graph::GetInfo(op.inputs[0]));
    batch = ids_info.shape.size() > 1 ? ids_info.shape[0] : 1;
    seq_len = ids_info.shape.size() > 1 ? ids_info.shape[1] : ids_info.shape[0];
    // ml drift requires the context length on the third dimension.
    ml_drift::BHWC ids_bhwc(batch, 1, seq_len, 1);
    ml_drift::Layout ids_layout =
        batch == 1 ? ml_drift::Layout::kHWC : ml_drift::Layout::kBHWC;
    LRT_TENSOR_ASSIGN_OR_RETURN(auto ids_type,
                                GetMlDriftDataType(ids_info.type, float_type));
    ml_drift::TensorDescriptor ids_tensor_desc(
        ids_type, ctx.gpu_model_builder->default_storage(), ids_layout);
    ids_tensor_desc.SetBHWCShape(ids_bhwc);
    ids = ctx.gpu_model_builder->AddTensor(ids_tensor_desc);
    ctx.graph_tensor_to_handle.insert({op.inputs[0], ids});
  } else {
    ids = *ids_or;
    auto ids_shape = ids.tensor_desc.GetBHWCShape();
    batch = ids_shape.b;
    seq_len = ids_shape.w;
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto value,
      GetTensorHandle(ctx, op.inputs[1], /*create_if_not_found=*/false,
                      /*create_linear_tensor=*/true,
                      /*force_buffer_storage=*/true));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& value_info,
                              graph::GetInfo(op.inputs[1]));

  // Manually construct output shape: BHWC(batch, 1, seq_len, emb_dim)
  LRT_TENSOR_ASSIGN_OR_RETURN(ml_drift::OHWI weights_ohwi,
                              ShapeToOHWI(value_info.shape));
  ml_drift::BHWC new_shape(batch, 1, seq_len, weights_ohwi.i);
  auto out = ctx.gpu_model_builder->AddTensor(new_shape, float_type);

  bool is_quantized = IsQuantized(value_info.type);
  if (is_quantized) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        (auto [scale_handle, zero_point_handle, scale_zp_shape]),
        CreateScaleAndZeroPointToFCTensor(ctx, op.inputs[1], float_type));

    ml_drift::WeightsDescription weights_desc;
    if (value_info.type == Type::kI8) {
      weights_desc =
          ctx.gpu_model_builder->GetFullyConnectedInt8WeightsDesc(weights_ohwi);
    } else if (value_info.type == Type::kI4) {
      weights_desc =
          ctx.gpu_model_builder->GetFullyConnectedInt4WeightsDesc(weights_ohwi);
    } else if (value_info.type == Type::kI2) {
      weights_desc =
          ctx.gpu_model_builder->GetFullyConnectedInt2WeightsDesc(weights_ohwi);
    } else {
      return absl::UnimplementedError(
          absl::StrCat("Unsupported quantized type for EmbeddingLookup: ",
                       ToString(value_info.type)));
    }
    const ml_drift::GpuModelBuilder::Weights external_weights =
        ml_drift::CreateExternalWeights(value, weights_desc, weights_ohwi,
                                        scale_zp_shape, &scale_handle,
                                        &zero_point_handle);
    out = ctx.gpu_model_builder->EmbeddingLookup(
        ids, external_weights, float_type, ml_drift::Axis::kWidth);
    ctx.graph_tensor_to_handle.insert({outputs[0], out});
    return CheckGpuOpOk(ctx, op);
  }

  ml_drift::WeightsDescription weights_desc =
      ctx.gpu_model_builder->GetFullyConnectedWeightsDesc(float_type,
                                                          weights_ohwi);

  const ml_drift::TensorDescriptor* scale_desc = nullptr;
  const ml_drift::TensorDescriptor* zero_point_desc = nullptr;

  auto gpu_op = ml_drift::CreateEmbeddingLookupExternalWeights(
      ids.tensor_desc, out.tensor_desc, value.tensor_desc, weights_desc,
      weights_ohwi, scale_desc, zero_point_desc, ml_drift::Axis::kWidth);

  std::vector<ml_drift::ValueId> src_ids;
  src_ids.reserve(2);
  src_ids.push_back(ids.id);
  src_ids.push_back(value.id);
  gpu_op.read_size_ = ids.tensor_desc.GetMemorySizeInBytes();
  gpu_op.read_size_ += out.tensor_desc.GetMemorySizeInBytes() /
                       SizeInBitsOf(float_type) *
                       SizeInBitsOf(weights_desc.type);

  ctx.gpu_model_builder->AddGpuOperation(
      src_ids, {out.id},
      std::make_unique<ml_drift::GPUOperation>(std::move(gpu_op)),
      "embedding_lookup");

  ctx.graph_tensor_to_handle.insert({outputs[0], out});

  return CheckGpuOpOk(ctx, op);
}

absl::Status OpMixin<TileOperation, MlDriftMixinTag>::ToMlDrift(
    const graph::Operation& op, MlDriftBuildContext& ctx) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto input, GetTensorHandle(ctx, op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& multipler,
                              graph::GetInfo(op.inputs[1]));
  auto multiplier_data = multipler.buffer->Lock().As<const int32_t>();
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info,
                              graph::GetInfo(op.inputs[0]));
  const auto& input_shape = input_info.shape;
  bool needs_tiling = false;
  for (int i = 0; i < input_shape.size(); ++i) {
    if (multiplier_data.data()[i] > 1) {
      needs_tiling = true;
      break;
    }
  }
  if (!needs_tiling) {
    // no tiling needed, just return the input.
    auto out = input;
    ctx.graph_tensor_to_handle.insert({outputs[0], out});
    return CheckGpuOpOk(ctx, op);
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& output_info,
                              graph::GetInfo(outputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto final_shape_bhwc,
                              ShapeToBHWC(output_info.shape));
  auto out = ctx.gpu_model_builder->Tile(input, final_shape_bhwc);
  ctx.graph_tensor_to_handle.insert({outputs[0], out});
  return CheckGpuOpOk(ctx, op);
}

}  // namespace graph
}  // namespace litert::tensor

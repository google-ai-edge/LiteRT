/* Copyright 2026 Google LLC.

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

#include "tensor/backends/ynnpack/arithmetic.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "ynnpack/composites/composites.h"  // from @XNNPACK
#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/arithmetic_graph.h"
#include "tensor/backends/common_nnpack/utils.h"
#include "tensor/backends/ynnpack/conversion.h"
#include "tensor/backends/ynnpack/utils.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::graph {

namespace {

// YNNPACK creates the value backing a node output when the in/out id is
// `YNN_INVALID_VALUE_ID`. Intermediate values introduced by a lowering use this
// so that YNNPACK infers their shape and type.
constexpr uint32_t kInferredValueId = YnnpackBuildContext::kInferredValueId;

struct BinaryIOIds {
  uint32_t lhs;
  uint32_t rhs;
  uint32_t output;
};

struct UnaryIOIds {
  uint32_t input;
  uint32_t output;
};

// Returns the single output value id of `op`.
absl::StatusOr<uint32_t> DefineSingleOutput(const Operation& op,
                                            YnnpackBuildContext& ctx,
                                            absl::string_view op_name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> outputs,
                              graph::GetOutputs(op));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  return ctx.DefineValue(outputs.front());
}

absl::StatusOr<BinaryIOIds> PrepareBinaryIO(const Operation& op,
                                            YnnpackBuildContext& ctx,
                                            absl::string_view op_name) {
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects two inputs", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t lhs_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t rhs_id, ctx.DefineValue(op.inputs[1]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));
  return BinaryIOIds{lhs_id, rhs_id, output_id};
}

absl::StatusOr<UnaryIOIds> PrepareUnaryIO(const Operation& op,
                                          YnnpackBuildContext& ctx,
                                          absl::string_view op_name) {
  if (op.inputs.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects one input", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));
  return UnaryIOIds{input_id, output_id};
}

// YNNPACK nodes have no fused activation parameters. Clamping is expressed with
// explicit max/min nodes, which the subgraph fusion pass folds back into the
// producing node.
bool NeedsClamp(const NnpackActivationBounds& bounds) {
  return bounds.min > -std::numeric_limits<float>::infinity() ||
         bounds.max < std::numeric_limits<float>::infinity();
}

absl::StatusOr<uint32_t> DefineScalarConstant(YnnpackBuildContext& ctx,
                                              float value) {
  return ctx.DefineConstant(&value, sizeof(value), Type::kFP32, /*shape=*/{});
}

// Emits `output = min(max(input, bounds.min), bounds.max)`, skipping the
// clamps whose bound is infinite.
//
// Warning: this should only be called when `NeedsClamp` holds, otherwise
// nothing would write to `output_id`.
absl::Status ApplyActivationBounds(YnnpackBuildContext& ctx, uint32_t input_id,
                                   const NnpackActivationBounds& bounds,
                                   uint32_t output_id,
                                   absl::string_view op_name) {
  const bool has_min = bounds.min > -std::numeric_limits<float>::infinity();
  const bool has_max = bounds.max < std::numeric_limits<float>::infinity();
  uint32_t current_id = input_id;
  if (has_min) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const uint32_t min_id,
                                DefineScalarConstant(ctx, bounds.min));
    uint32_t clamped_id = has_max ? kInferredValueId : output_id;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_binary(ctx.subgraph(), ynn_binary_max,
                                                 current_id, min_id,
                                                 &clamped_id, /*flags=*/0))
        << op_name;
    current_id = clamped_id;
  }
  if (has_max) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const uint32_t max_id,
                                DefineScalarConstant(ctx, bounds.max));
    uint32_t clamped_id = output_id;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_binary(ctx.subgraph(), ynn_binary_min,
                                                 current_id, max_id,
                                                 &clamped_id, /*flags=*/0))
        << op_name;
  }
  return absl::OkStatus();
}

absl::Status AddBinaryNode(ynn_binary_operator op_type, const BinaryIOIds& io,
                           FusedActivation activation, YnnpackBuildContext& ctx,
                           absl::string_view op_name) {
  const NnpackActivationBounds bounds = GetActivationBounds(activation);
  const bool needs_clamp = NeedsClamp(bounds);
  uint32_t result_id = needs_clamp ? kInferredValueId : io.output;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_binary(ctx.subgraph(), op_type, io.lhs,
                                               io.rhs, &result_id,
                                               /*flags=*/0))
      << op_name;
  if (needs_clamp) {
    return ApplyActivationBounds(ctx, result_id, bounds, io.output, op_name);
  }
  return absl::OkStatus();
}

absl::Status AddUnaryNode(ynn_unary_operator op_type, const UnaryIOIds& io,
                          YnnpackBuildContext& ctx, absl::string_view op_name) {
  uint32_t output_id = io.output;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_unary(ctx.subgraph(), op_type, io.input,
                                              &output_id, /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

// Reads a constant int32 tensor into a vector.
absl::StatusOr<std::vector<int32_t>> GetConstantInt32Vector(
    const graph::Tensor& tensor, absl::string_view op_name,
    absl::string_view field_name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& info,
                              graph::GetInfo(tensor));
  if (info.type != Type::kI32) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: %s must be int32. Got type id %d.", op_name,
                        field_name, static_cast<int>(info.type)));
  }
  if (info.buffer == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s: %s must be a constant tensor", op_name, field_name));
  }
  LockedBufferSpan<const int32_t> locked =
      info.buffer->Lock().As<const int32_t>();
  return std::vector<int32_t>(locked.begin(), locked.end());
}

// Emits a transpose of the two innermost dimensions of `value_id`.
//
// Returns the transpose output id.
absl::StatusOr<uint32_t> TransposeInnermostDims(YnnpackBuildContext& ctx,
                                                uint32_t value_id,
                                                absl::string_view op_name) {
  static constexpr int32_t kSwapLastTwo[] = {-1, -2};
  uint32_t transposed_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_transpose(
      ctx.subgraph(), /*num_axes=*/2, kSwapLastTwo, value_id, &transposed_id,
      YNN_NODE_FLAG_KEEP_DIMS))
      << op_name;
  return transposed_id;
}

// Emits a reshape of `input_id` to `output_id`.
absl::Status ReshapeTo(YnnpackBuildContext& ctx, uint32_t input_id,
                       const std::vector<size_t>& new_shape, uint32_t output_id,
                       absl::string_view op_name) {
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_reshape(
      ctx.subgraph(), new_shape.size(),
      new_shape.empty() ? nullptr : new_shape.data(), input_id, &output_id,
      /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

}  // namespace

absl::Status OpMixin<AddOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const BinaryIOIds io,
                              PrepareBinaryIO(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(const AddOperation& op_data,
                              op.As<AddOperation>());
  return AddBinaryNode(ynn_binary_add, io, op_data.activation, ctx, op_name);
}

absl::Status OpMixin<MulOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const BinaryIOIds io,
                              PrepareBinaryIO(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(const MulOperation& op_data,
                              op.As<MulOperation>());
  return AddBinaryNode(ynn_binary_multiply, io, op_data.activation, ctx,
                       op_name);
}

absl::Status OpMixin<SubOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const BinaryIOIds io,
                              PrepareBinaryIO(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(const SubOperation& op_data,
                              op.As<SubOperation>());
  return AddBinaryNode(ynn_binary_subtract, io, op_data.activation, ctx,
                       op_name);
}

absl::Status OpMixin<DivOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const BinaryIOIds io,
                              PrepareBinaryIO(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(const DivOperation& op_data,
                              op.As<DivOperation>());
  return AddBinaryNode(ynn_binary_divide, io, op_data.activation, ctx, op_name);
}

absl::Status OpMixin<MaximumOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const BinaryIOIds io,
                              PrepareBinaryIO(op, ctx, op_name));
  return AddBinaryNode(ynn_binary_max, io, kActNone, ctx, op_name);
}

absl::Status OpMixin<MinimumOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const BinaryIOIds io,
                              PrepareBinaryIO(op, ctx, op_name));
  return AddBinaryNode(ynn_binary_min, io, kActNone, ctx, op_name);
}

absl::Status OpMixin<PowOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const BinaryIOIds io,
                              PrepareBinaryIO(op, ctx, op_name));
  return AddBinaryNode(ynn_binary_pow, io, kActNone, ctx, op_name);
}

absl::Status OpMixin<PReluOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const BinaryIOIds io,
                              PrepareBinaryIO(op, ctx, op_name));
  return AddBinaryNode(ynn_binary_leaky_relu, io, kActNone, ctx, op_name);
}

absl::Status OpMixin<AbsOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_abs, io, ctx, op_name);
}

absl::Status OpMixin<SquareOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_square, io, ctx, op_name);
}

absl::Status OpMixin<RsqrtOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_rsqrt, io, ctx, op_name);
}

absl::Status OpMixin<SqrtOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_sqrt, io, ctx, op_name);
}

absl::Status OpMixin<ExpOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_exp, io, ctx, op_name);
}

absl::Status OpMixin<LogOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_log, io, ctx, op_name);
}

absl::Status OpMixin<CeilOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_ceil, io, ctx, op_name);
}

absl::Status OpMixin<FloorOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_floor, io, ctx, op_name);
}

absl::Status OpMixin<SignOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_sign, io, ctx, op_name);
}

absl::Status OpMixin<RoundOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_round, io, ctx, op_name);
}

absl::Status OpMixin<NegOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_negate, io, ctx, op_name);
}

absl::Status OpMixin<TanhOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_tanh, io, ctx, op_name);
}

absl::Status OpMixin<LogisticOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_sigmoid, io, ctx, op_name);
}

absl::Status OpMixin<CosOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_cos, io, ctx, op_name);
}

absl::Status OpMixin<SinOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_sin, io, ctx, op_name);
}

absl::Status OpMixin<HardSwishOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_hardswish, io, ctx, op_name);
}

absl::Status OpMixin<CastOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return AddUnaryNode(ynn_unary_convert, io, ctx, op_name);
}

absl::Status OpMixin<DequantizeOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  // Quantization parameters are attached to the value by the conversion layer,
  // which already emits the dequantize node. What is left here is a plain type
  // conversion.
  return AddUnaryNode(ynn_unary_convert, io, ctx, op_name);
}

absl::Status OpMixin<ReluOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return ApplyActivationBounds(ctx, io.input, GetActivationBounds(kActRelu),
                               io.output, op_name);
}

absl::Status OpMixin<Relu6Operation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  return ApplyActivationBounds(ctx, io.input, GetActivationBounds(kActRelu6),
                               io.output, op_name);
}

absl::Status OpMixin<LeakyReluOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(const LeakyReluOperation& op_data,
                              op.As<LeakyReluOperation>());
  uint32_t output_id = io.output;
  LRT_TENSOR_RETURN_IF_ERROR(ynn::define_leaky_relu(ctx.subgraph(), io.input,
                                                    op_data.alpha, output_id))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<EluOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  uint32_t output_id = io.output;
  LRT_TENSOR_RETURN_IF_ERROR(
      ynn::define_elu(ctx.subgraph(), io.input, /*alpha=*/1.0f, output_id))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<GeluOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(const GeluOperation& op_data,
                              op.As<GeluOperation>());
  uint32_t output_id = io.output;
  if (op_data.approximate) {
    LRT_TENSOR_RETURN_IF_ERROR(
        ynn::define_approx_gelu(ctx.subgraph(), io.input, output_id))
        << op_name;
  } else {
    LRT_TENSOR_RETURN_IF_ERROR(
        ynn::define_gelu(ctx.subgraph(), io.input, output_id))
        << op_name;
  }
  return absl::OkStatus();
}

absl::Status OpMixin<SoftmaxOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const SoftmaxOperation& op_data,
                              op.As<SoftmaxOperation>());
  if (op_data.beta != 1.0f) {
    return absl::UnimplementedError(absl::StrFormat(
        "%s: YNNPACK softmax only supports beta == 1", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  uint32_t output_id = io.output;
  LRT_TENSOR_RETURN_IF_ERROR(
      ynn::define_softmax(ctx.subgraph(), io.input, op_data.beta, output_id))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<L2NormalizationOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError(
      "L2Normalization is not supported in YNNPACK.");
}

absl::Status OpMixin<AveragePool2DOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "AveragePool2D";
  LRT_TENSOR_ASSIGN_OR_RETURN(const AveragePool2DOperation& op_data,
                              op.As<AveragePool2DOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnaryIOIds io,
                              PrepareUnaryIO(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& input_info,
                              graph::GetInfo(op.inputs[0]));
  const ynn_type type = ToYnnType(input_info.type);
  if (type == ynn_type_invalid) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: unsupported input type", op_name));
  }
  const NnpackActivationBounds bounds = GetActivationBounds(op_data.activation);
  const bool needs_clamp = NeedsClamp(bounds);
  uint32_t result_id = needs_clamp ? kInferredValueId : io.output;
  LRT_TENSOR_RETURN_IF_ERROR(ynn::define_average_pool_2d(
      ctx.subgraph(), io.input, type, op_data.padding == kPaddingSame,
      op_data.filter_height, op_data.filter_width, op_data.stride_h,
      op_data.stride_w, result_id))
      << op_name;
  if (!needs_clamp) {
    return absl::OkStatus();
  }
  return ApplyActivationBounds(ctx, result_id, bounds, io.output, op_name);
}

absl::Status OpMixin<MaxPool2DOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError("MaxPool2D is not supported in YNNPACK.");
}

absl::Status OpMixin<Conv2DOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError("Conv2D is not supported in YNNPACK.");
}

absl::Status OpMixin<DepthwiseConv2DOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError(
      "DepthwiseConv2D is not supported in YNNPACK.");
}

absl::Status OpMixin<TransposeConvOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError("TransposeConv is not supported in YNNPACK.");
}

absl::Status OpMixin<TransposeConv2DOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError(
      "TransposeConv2D is not supported in YNNPACK.");
}

absl::Status OpMixin<FullyConnectedOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  const absl::string_view op_name = op.GetName();
  LRT_TENSOR_ASSIGN_OR_RETURN(const FullyConnectedOperation& op_data,
                              op.As<FullyConnectedOperation>());
  if (op.inputs.size() < 2 || op.inputs.size() > 3) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expects 2 or 3 inputs (input, weights[, bias])", op_name));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t weights_id,
                              ctx.DefineValue(op.inputs[1]));
  uint32_t bias_id = YNN_INVALID_VALUE_ID;
  if (op.inputs.size() == 3) {
    LRT_TENSOR_ASSIGN_OR_RETURN(bias_id, ctx.DefineValue(op.inputs[2]));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));

  // `ynn_define_dot` expects `b` with shape [..., K, N], while FullyConnected
  // stores the weights as [N, K].
  LRT_TENSOR_ASSIGN_OR_RETURN(const uint32_t transposed_weights_id,
                              TransposeInnermostDims(ctx, weights_id, op_name));

  const NnpackActivationBounds bounds = GetActivationBounds(op_data.activation);
  const bool needs_clamp = NeedsClamp(bounds);
  uint32_t result_id = needs_clamp ? kInferredValueId : output_id;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_dot(ctx.subgraph(), /*num_k_dims=*/1,
                                            input_id, transposed_weights_id,
                                            bias_id, &result_id, /*flags=*/0))
      << op_name;
  if (!needs_clamp) {
    return absl::OkStatus();
  }
  return ApplyActivationBounds(ctx, result_id, bounds, output_id, op_name);
}

absl::Status OpMixin<BatchMatMulOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "BatchMatMul";
  LRT_TENSOR_ASSIGN_OR_RETURN(const BatchMatMulOperation& op_data,
                              op.As<BatchMatMulOperation>());
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs", op_name));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t lhs_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t rhs_id, ctx.DefineValue(op.inputs[1]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));

  // `ynn_define_dot` computes `a[..., M, K] . b[..., K, N]`.
  if (op_data.adj_x) {
    LRT_TENSOR_ASSIGN_OR_RETURN(lhs_id,
                                TransposeInnermostDims(ctx, lhs_id, op_name));
  }
  if (op_data.adj_y) {
    LRT_TENSOR_ASSIGN_OR_RETURN(rhs_id,
                                TransposeInnermostDims(ctx, rhs_id, op_name));
  }

  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_dot(
      ctx.subgraph(), /*num_k_dims=*/1, lhs_id, rhs_id,
      /*input_c_id=*/YNN_INVALID_VALUE_ID, &output_id, /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<TransposeOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Transpose";
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs (input, perm)", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const std::vector<int32_t> perm,
      GetConstantInt32Vector(op.inputs[1], op_name, "permutation"));

  LRT_TENSOR_RETURN_IF_ERROR(
      ynn_define_static_transpose(ctx.subgraph(), perm.size(), perm.data(),
                                  input_id, &output_id, /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<MeanOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Mean";
  LRT_TENSOR_ASSIGN_OR_RETURN(const MeanOperation& op_data,
                              op.As<MeanOperation>());
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs (input, axes)", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> outputs,
                              graph::GetOutputs(op));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              ctx.DefineValue(outputs.front()));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& output_info,
                              graph::GetInfo(outputs.front()));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const std::vector<int32_t> axes,
      GetConstantInt32Vector(op.inputs[1], op_name, "axes"));

  const ynn_type output_type = ToYnnType(output_info.type);
  if (output_type == ynn_type_invalid) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: unsupported output type", op_name));
  }
  // Note: we don't currently support emitting a quantized mean.
  LRT_TENSOR_RETURN_IF_ERROR(ynn::define_reduce_sum(
      ctx.subgraph(), axes.size(), axes.data(), input_id,
      /*input_zero_point_id=*/YNN_INVALID_VALUE_ID,
      /*input_scale_id=*/YNN_INVALID_VALUE_ID, op_data.keep_dims,
      /*mean=*/true, /*squared=*/false, output_type,
      /*output_zero_point_id=*/YNN_INVALID_VALUE_ID,
      /*output_scale_id=*/YNN_INVALID_VALUE_ID, output_id))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<SliceOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Slice";
  if (op.inputs.size() != 3) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 3 inputs (input, begin, size)", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const std::vector<int32_t> begins,
      GetConstantInt32Vector(op.inputs[1], op_name, "begin"));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const std::vector<int32_t> sizes,
      GetConstantInt32Vector(op.inputs[2], op_name, "size"));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& input_info,
                              graph::GetInfo(op.inputs[0]));

  if (begins.size() != sizes.size() ||
      begins.size() != input_info.shape.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s: begin (%d) and size (%d) must have one entry per input dimension "
        "(%d)",
        op_name, begins.size(), sizes.size(), input_info.shape.size()));
  }

  const size_t num_axes = begins.size();
  std::vector<int32_t> axes(num_axes);
  std::iota(axes.begin(), axes.end(), 0);
  std::vector<int64_t> slice_begins(num_axes);
  std::vector<int64_t> slice_ends(num_axes);
  for (size_t i = 0; i < num_axes; ++i) {
    slice_begins[i] = begins[i];
    if (sizes[i] < 0) {
      // A negative size means "up to the end of this dimension". Spelling that
      // out as the extent the graph happens to have been built with would
      // freeze the dimension, which breaks graphs that are re-run with a
      // different shape, e.g. attention over a KV cache that grows by one
      // token per decode step. YNNPACK interprets an end of 0 or less as
      // relative to the (possibly symbolic) extent, so 0 keeps it dynamic.
      slice_ends[i] = 0;
    } else {
      slice_ends[i] = begins[i] + sizes[i];
    }
  }

  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_slice(
      ctx.subgraph(), num_axes, axes.data(), slice_begins.data(),
      slice_ends.data(), /*strides=*/nullptr, input_id, &output_id,
      /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<ConcatenationOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Concatenation";
  LRT_TENSOR_ASSIGN_OR_RETURN(const ConcatenationOperation& op_data,
                              op.As<ConcatenationOperation>());
  if (op.inputs.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects at least 1 input", op_name));
  }

  std::vector<uint32_t> input_ids;
  input_ids.reserve(op.inputs.size());
  for (const graph::Tensor& input : op.inputs) {
    LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(input));
    input_ids.push_back(input_id);
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));

  LRT_TENSOR_RETURN_IF_ERROR(
      ynn_define_concatenate(ctx.subgraph(), op_data.axis, input_ids.size(),
                             input_ids.data(), &output_id, /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<SqueezeOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Squeeze";
  if (op.inputs.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 1 input", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> outputs,
                              graph::GetOutputs(op));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              ctx.DefineValue(outputs.front()));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& output_info,
                              graph::GetInfo(outputs.front()));

  const std::vector<size_t> new_shape(output_info.shape.begin(),
                                      output_info.shape.end());
  return ReshapeTo(ctx, input_id, new_shape, output_id, op_name);
}

absl::Status OpMixin<ExpandDimsOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "ExpandDims";
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const std::vector<int32_t> axes,
      GetConstantInt32Vector(op.inputs[1], op_name, "axis"));
  if (axes.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: axis tensor must not be empty", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& input_info,
                              graph::GetInfo(op.inputs[0]));

  const int rank = static_cast<int>(input_info.shape.size());
  int axis = axes[0];
  if (axis < 0) {
    axis += rank + 1;
  }
  if (axis < 0 || axis > rank) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: axis %d is out of range for input of rank %d",
                        op_name, axes[0], rank));
  }

  const int32_t new_axis = axis;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_expand_dims(
      ctx.subgraph(), /*num_new_axes=*/1, &new_axis, input_id, &output_id,
      /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<ReshapeOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Reshape";
  LRT_TENSOR_ASSIGN_OR_RETURN(const ReshapeOperation& op_data,
                              op.As<ReshapeOperation>());
  if (op.inputs.size() != 1 && op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 1 or 2 inputs", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));

  std::vector<size_t> new_shape;
  new_shape.reserve(op_data.new_shape.size());
  for (const int dim : op_data.new_shape) {
    new_shape.push_back(dim < 0 ? 0 : static_cast<size_t>(dim));
  }
  if (op_data.inferred_axis >= 0) {
    if (op_data.inferred_axis >= new_shape.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s: inferred axis (%v) is out of range for shape of rank %v",
          op_name, op_data.inferred_axis, new_shape.size()));
    }
    new_shape[op_data.inferred_axis] = 0;
  }
  return ReshapeTo(ctx, input_id, new_shape, output_id, op_name);
}

absl::Status OpMixin<TileOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Tile";
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs (input, multiples)", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              DefineSingleOutput(op, ctx, op_name));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const std::vector<int32_t> multiples,
      GetConstantInt32Vector(op.inputs[1], op_name, "multiples"));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& input_info,
                              graph::GetInfo(op.inputs[0]));

  if (input_info.shape.size() != multiples.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: input rank (%d) != multiples size (%d)", op_name,
                        input_info.shape.size(), multiples.size()));
  }

  // `ynn_define_static_broadcast` treats a zero entry in `new_dims` as "pass
  // this dimension through unchanged". We must use that for untiled
  // dimensions rather than spelling out their current extent: an extent that
  // is only known symbolically (e.g. a KV cache sequence length that grows on
  // every decode step) would otherwise be frozen at graph-build time and the
  // broadcast would reject the tensor on the next reshape.
  std::vector<size_t> new_shape(multiples.size());
  for (size_t i = 0; i < multiples.size(); ++i) {
    const int dim = input_info.shape[i];
    if (multiples[i] > 1 && dim != 1) {
      return absl::UnimplementedError(absl::StrFormat(
          "%s: YNNPACK broadcast can only tile dimensions of size 1. "
          "Dimension %d has size %d but multiples[%d]=%d",
          op_name, i, dim, i, multiples[i]));
    }
    new_shape[i] =
        multiples[i] == 1 ? 0 : static_cast<size_t>(dim * multiples[i]);
  }

  // The broadcast on its own would be enough, but we force the result to be
  // materialized with a copy.
  //
  // A bare `static_broadcast` result is a virtual (stride-0) view. When such a
  // view is later consumed as the right-hand operand of a dot, YNNPACK clones
  // the operand's transpose into an *aliasing* transpose and folds it into the
  // weight packer (`always_alias_transpose` in ynnpack/subgraph/dot.cc), and
  // the optimizer additionally commutes the transpose with the broadcast
  // (`rewrite_transpose_broadcast` in ynnpack/subgraph/fusion.cc). The
  // resulting subgraph reads the broadcast dimension with the wrong strides:
  // GQA attention scores come out constant along the key axis. Inserting a
  // copy keeps the broadcast away from the packer and produces correct
  // results.
  //
  // TODO(b/557248107): Drop the copy once the underlying YNNPACK issue is
  // fixed; it costs a materialization of the tiled tensor.
  uint32_t broadcast_id = YNN_INVALID_VALUE_ID;
  LRT_TENSOR_RETURN_IF_ERROR(
      ynn_define_static_broadcast(ctx.subgraph(), new_shape.size(),
                                  new_shape.data(), input_id, &broadcast_id,
                                  /*flags=*/0))
      << op_name;
  LRT_TENSOR_RETURN_IF_ERROR(
      ynn_define_copy(ctx.subgraph(), broadcast_id, &output_id, /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<SplitOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Split";
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> outputs,
                              graph::GetOutputs(op));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const SplitOperation& op_data,
                              op.As<SplitOperation>());
  if (outputs.size() != op_data.num_splits) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects %d outputs, but got %d", op_name,
                        op_data.num_splits, outputs.size()));
  }

  std::vector<uint32_t> output_ids(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    LRT_TENSOR_ASSIGN_OR_RETURN(output_ids[i], ctx.DefineValue(outputs[i]));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      const std::vector<int32_t> axes,
      GetConstantInt32Vector(op.inputs[1], op_name, "axis"));
  if (axes.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: axis tensor must not be empty", op_name));
  }

  LRT_TENSOR_RETURN_IF_ERROR(
      ynn_define_even_split(ctx.subgraph(), axes[0], input_id,
                            output_ids.size(), output_ids.data(), /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<GatherOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  constexpr absl::string_view op_name = "Gather";
  LRT_TENSOR_ASSIGN_OR_RETURN(const GatherOperation& op_data,
                              op.As<GatherOperation>());
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects 2 inputs (input, indices)", op_name));
  }
  if (op_data.batch_dims != 0) {
    return absl::UnimplementedError(absl::StrFormat(
        "%s: batch_dims != 0 is not supported in YNNPACK.", op_name));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t index_id, ctx.DefineValue(op.inputs[1]));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> outputs,
                              graph::GetOutputs(op));
  if (outputs.empty()) {
    return absl::NotFoundError(absl::StrFormat("%s missing outputs", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id,
                              ctx.DefineValue(outputs.front()));
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& output_info,
                              graph::GetInfo(outputs.front()));

  const int32_t axis = op_data.axis;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_gather(
      ctx.subgraph(), /*num_axes=*/1, &axis, output_info.shape.size(), input_id,
      index_id, &output_id, /*flags=*/0))
      << op_name;
  return absl::OkStatus();
}

absl::Status OpMixin<SpaceToDepthOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError("SpaceToDepth is not supported in YNNPACK.");
}

absl::Status OpMixin<DepthToSpaceOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError("DepthToSpace is not supported in YNNPACK.");
}

absl::Status OpMixin<ResizeBilinearOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError(
      "ResizeBilinear is not supported in YNNPACK.");
}

absl::Status
OpMixin<ResizeNearestNeighborOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError(
      "ResizeNearestNeighbor is not supported in YNNPACK.");
}

absl::Status OpMixin<RopeOperation, YnnpackMixinTag>::ToYnnpack(
    const graph::Operation& op, YnnpackBuildContext& ctx) const {
  return absl::UnimplementedError("Rope is not supported in YNNPACK.");
}

}  // namespace litert::tensor::graph

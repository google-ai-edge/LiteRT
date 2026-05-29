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
#include "tensor/backends/tflite/arithmetic_tflite.h"

#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/arithmetic_graph.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/utils/macros.h"
#include "tflite/schema/mutable/schema_generated.h"

namespace litert::tensor::graph {

namespace {

inline absl::StatusOr<tflite::ActivationFunctionType> ToTflite(
    FusedActivation activation_function) {
  switch (activation_function) {
    case FusedActivation::kActNone:
      return tflite::ActivationFunctionType_NONE;
    case FusedActivation::kActRelu:
      return tflite::ActivationFunctionType_RELU;
    case FusedActivation::kActReluN1To1:
      return tflite::ActivationFunctionType_RELU_N1_TO_1;
    case FusedActivation::kActRelu6:
      return tflite::ActivationFunctionType_RELU6;
    case FusedActivation::kActTanh:
      return tflite::ActivationFunctionType_TANH;
    case FusedActivation::kActSignBit:
      return tflite::ActivationFunctionType_SIGN_BIT;
    default:
      return absl::InvalidArgumentError("Invalid activation function.");
  }
}

absl::StatusOr<tflite::Padding> ToTflite(const Padding padding) {
  switch (padding) {
    case kPaddingSame:
      return tflite::Padding_SAME;
    case kPaddingValid:
      return tflite::Padding_VALID;
    default:
      return absl::UnimplementedError("Unknown padding type.");
  }
}

absl::StatusOr<tflite::TensorType> ToTfLiteArgMaxOutputTensorType(Type type) {
  switch (type) {
    case Type::kI32:
      return tflite::TensorType_INT32;
    case Type::kI64:
      return tflite::TensorType_INT64;
    default:
      return absl::InvalidArgumentError(
          "Unsupported output type for ArgMax. Only kI32 and kI64 are "
          "supported by TFLite.");
  }
}

}  // namespace

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    AddOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const AddOperation& data, op.As<AddOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data.activation));
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_ADD,
      tflite::AddOptionsT{.fused_activation_function = tflite_activation});
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    MulOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const MulOperation& data, op.As<MulOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data.activation));
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_MUL,
      tflite::MulOptionsT{.fused_activation_function = tflite_activation});
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    AbsOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_ABS);
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    ReluOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_RELU);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<Relu6Operation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_RELU6);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ReluN1To1Operation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_RELU_N1_TO_1);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ZerosLikeOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_ZEROS_LIKE);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<Relu0To1Operation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_RELU_0_TO_1);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<LeakyReluOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const LeakyReluOperation& data,
                              op.As<LeakyReluOperation>());
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LEAKY_RELU,
                           tflite::LeakyReluOptionsT{.alpha = data.alpha});
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    EluOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_ELU);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<HardSwishOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_HARD_SWISH);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<PReluOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_PRELU);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<L2NormalizationOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_L2_NORMALIZATION,
      tflite::L2NormOptionsT{.fused_activation_function =
                                 tflite::ActivationFunctionType_NONE});
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    SubOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const SubOperation& data, op.As<SubOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data.activation));
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_SUB,
      tflite::SubOptionsT{.fused_activation_function = tflite_activation});
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    DivOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const DivOperation& data, op.As<DivOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data.activation));
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_DIV,
      tflite::DivOptionsT{.fused_activation_function = tflite_activation});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SquareOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SQUARE);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<RsqrtOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_RSQRT);
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    PowOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_POW);
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    NegOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_NEG);
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    SqrtOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SQRT);
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    ExpOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_EXP);
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    LogOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LOG);
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    CeilOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_CEIL);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<FloorOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_FLOOR);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<FloorDivOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_FLOOR_DIV);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<FloorModOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_FLOOR_MOD);
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    SignOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SIGN);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<RoundOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_ROUND);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<TransposeOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_TRANSPOSE);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<LogisticOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LOGISTIC);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<EmbeddingLookupOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_EMBEDDING_LOOKUP);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<DynamicUpdateSliceOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_DYNAMIC_UPDATE_SLICE);
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    TileOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_TILE);
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    GeluOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const GeluOperation& data,
                              op.As<GeluOperation>());
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_GELU,
      tflite::GeluOptionsT{.approximate = data.approximate});
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    TanhOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_TANH);
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    CastOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_CAST);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SelectOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SELECT);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SelectV2Operation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SELECT_V2);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SliceOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SLICE);
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    LessOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LESS);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<GreaterOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_GREATER);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<GreaterEqualOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_GREATER_EQUAL);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<EqualOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_EQUAL);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<NotEqualOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_NOT_EQUAL);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<MinimumOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_MINIMUM);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<MaximumOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_MAXIMUM);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<LogicalAndOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LOGICAL_AND);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<LogicalOrOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LOGICAL_OR);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<LogicalNotOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LOGICAL_NOT);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<BitwiseXorOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_BITWISE_XOR);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<RightShiftOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_RIGHT_SHIFT);
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    CosOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_COS);
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    SinOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SIN);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SoftmaxOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const SoftmaxOperation& data,
                              op.As<SoftmaxOperation>());
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SOFTMAX,
                           tflite::SoftmaxOptionsT{.beta = data.beta});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<LogSoftmaxOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LOG_SOFTMAX);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ReshapeOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const ReshapeOperation& data,
                              op.As<ReshapeOperation>());
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_RESHAPE,
      tflite::ReshapeOptionsT{.new_shape = data.new_shape});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SqueezeOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const SqueezeOperation& data,
                              op.As<SqueezeOperation>());
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_SQUEEZE,
      tflite::SqueezeOptionsT{.squeeze_dims = data.squeeze_dims});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ExpandDimsOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_EXPAND_DIMS);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<BatchMatMulOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const BatchMatMulOperation& data,
                              op.As<BatchMatMulOperation>());
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_BATCH_MATMUL,
                           tflite::BatchMatMulOptionsT{
                               .adj_x = data.adj_x,
                               .adj_y = data.adj_y,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<FullyConnectedOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const FullyConnectedOperation& data,
                              op.As<FullyConnectedOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data.activation));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_FULLY_CONNECTED,
                           tflite::FullyConnectedOptionsT{
                               .fused_activation_function = tflite_activation,
                               .keep_num_dims = data.keep_num_dims,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ConcatenationOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const ConcatenationOperation& data,
                              op.As<ConcatenationOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data.activation));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_CONCATENATION,
                           tflite::ConcatenationOptionsT{
                               .axis = data.axis,
                               .fused_activation_function = tflite_activation,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    PackOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const PackOperation& data,
                              op.As<PackOperation>());
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_PACK,
      tflite::PackOptionsT{.values_count = static_cast<int>(op.inputs.size()),
                           .axis = data.axis});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<UnpackOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const UnpackOperation& data,
                              op.As<UnpackOperation>());
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_UNPACK,
      tflite::UnpackOptionsT{.num = data.num, .axis = data.axis});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SplitOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const SplitOperation& data,
                              op.As<SplitOperation>());

  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_SPLIT,
      tflite::SplitOptionsT{.num_splits = data.num_splits});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<CustomOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const CustomOperation& data,
                              op.As<CustomOperation>());
  TfLiteOpBuildInfo info(::tflite::BuiltinOperator_CUSTOM);
  info.custom_code = &data.custom_code;
  info.custom_options = &data.custom_options;
  return info;
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<AveragePool2DOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const AveragePool2DOperation& data,
                              op.As<AveragePool2DOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data.activation));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_padding, ToTflite(data.padding));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_AVERAGE_POOL_2D,
                           tflite::Pool2DOptionsT{
                               .padding = tflite_padding,
                               .stride_w = data.stride_w,
                               .stride_h = data.stride_h,
                               .filter_width = data.filter_width,
                               .filter_height = data.filter_height,
                               .fused_activation_function = tflite_activation,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<MaxPool2DOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const MaxPool2DOperation& data,
                              op.As<MaxPool2DOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data.activation));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_padding, ToTflite(data.padding));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_MAX_POOL_2D,
                           tflite::Pool2DOptionsT{
                               .padding = tflite_padding,
                               .stride_w = data.stride_w,
                               .stride_h = data.stride_h,
                               .filter_width = data.filter_width,
                               .filter_height = data.filter_height,
                               .fused_activation_function = tflite_activation,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<Conv2DOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const Conv2DOperation& data,
                              op.As<Conv2DOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data.activation));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_padding, ToTflite(data.padding));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_CONV_2D,
                           tflite::Conv2DOptionsT{
                               .padding = tflite_padding,
                               .stride_w = data.stride_w,
                               .stride_h = data.stride_h,
                               .fused_activation_function = tflite_activation,
                               .dilation_w_factor = data.dilation_w_factor,
                               .dilation_h_factor = data.dilation_h_factor,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<DepthwiseConv2DOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const DepthwiseConv2DOperation& data,
                              op.As<DepthwiseConv2DOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data.activation));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_padding, ToTflite(data.padding));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                           tflite::DepthwiseConv2DOptionsT{
                               .padding = tflite_padding,
                               .stride_w = data.stride_w,
                               .stride_h = data.stride_h,
                               .depth_multiplier = data.depth_multiplier,
                               .fused_activation_function = tflite_activation,
                               .dilation_w_factor = data.dilation_w_factor,
                               .dilation_h_factor = data.dilation_h_factor,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    PadOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_PAD);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<PadV2Operation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_PADV2);
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    SumOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const SumOperation& data, op.As<SumOperation>());
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_SUM,
      tflite::ReducerOptionsT{.keep_dims = data.keep_dims});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ReduceMaxOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const ReduceMaxOperation& data,
                              op.As<ReduceMaxOperation>());
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_REDUCE_MAX,
      tflite::ReducerOptionsT{.keep_dims = data.keep_dims});
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    MeanOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const MeanOperation& data,
                              op.As<MeanOperation>());
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_MEAN,
      tflite::ReducerOptionsT{.keep_dims = data.keep_dims});
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    TopKOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_TOPK_V2);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<CumsumOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const CumsumOperation& data,
                              op.As<CumsumOperation>());
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_CUMSUM,
                           tflite::CumsumOptionsT{.exclusive = data.exclusive,
                                                  .reverse = data.reverse});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ReverseOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_REVERSE_V2);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SpaceToDepthOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const SpaceToDepthOperation& data,
                              op.As<SpaceToDepthOperation>());
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_SPACE_TO_DEPTH,
      tflite::SpaceToDepthOptionsT{.block_size = data.block_size});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<DepthToSpaceOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const DepthToSpaceOperation& data,
                              op.As<DepthToSpaceOperation>());
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_DEPTH_TO_SPACE,
      tflite::DepthToSpaceOptionsT{.block_size = data.block_size});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<GatherOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const GatherOperation& data,
                              op.As<GatherOperation>());
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_GATHER,
      tflite::GatherOptionsT{.axis = data.axis, .batch_dims = data.batch_dims});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<GatherNdOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_GATHER_ND,
                           tflite::GatherNdOptionsT{});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<OneHotOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const OneHotOperation& data,
                              op.As<OneHotOperation>());
  tflite::OneHotOptionsT options;
  options.axis = data.axis;
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_ONE_HOT, options);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<QuantizeOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_QUANTIZE);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<DequantizeOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_DEQUANTIZE);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ProbeOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, GetInfo(op.inputs[0]));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_RESHAPE,
                           tflite::ReshapeOptionsT{.new_shape = info.shape});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ResizeBilinearOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const ResizeBilinearOperation& data,
                              op.As<ResizeBilinearOperation>());
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_RESIZE_BILINEAR,
                           tflite::ResizeBilinearOptionsT{
                               .align_corners = data.align_corners,
                               .half_pixel_centers = data.half_pixel_centers,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ResizeNearestNeighborOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const ResizeNearestNeighborOperation& data,
                              op.As<ResizeNearestNeighborOperation>());
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                           tflite::ResizeNearestNeighborOptionsT{
                               .align_corners = data.align_corners,
                               .half_pixel_centers = data.half_pixel_centers,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<NonMaxSuppressionV5Operation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_NON_MAX_SUPPRESSION_V5);
}

absl::StatusOr<TfLiteOpBuildInfo> OpMixin<
    LstmOperation, TfLiteMixinTag>::ToTfLite(const graph::Operation& op) const {
  static const std::string* kCustomLstmCode = new std::string("LSTM_BASIC");
  TfLiteOpBuildInfo info(::tflite::BuiltinOperator_CUSTOM);
  info.custom_code = kCustomLstmCode;
  return info;
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<TransposeConvOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const TransposeConvOperation& data,
                              op.As<TransposeConvOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_padding, ToTflite(data.padding));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_TRANSPOSE_CONV,
                           tflite::TransposeConvOptionsT{
                               .padding = tflite_padding,
                               .stride_w = data.stride_w,
                               .stride_h = data.stride_h,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<TransposeConv2DOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const TransposeConv2DOperation& data,
                              op.As<TransposeConv2DOperation>());
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_padding, ToTflite(data.padding));
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_TRANSPOSE_CONV,
      tflite::TransposeConvOptionsT{.padding = tflite_padding,
                                    .stride_w = data.stride_w,
                                    .stride_h = data.stride_h});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ArgMaxOperation, TfLiteMixinTag>::ToTfLite(
    const graph::Operation& op) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(const ArgMaxOperation& data,
                              op.As<ArgMaxOperation>());
  tflite::ArgMaxOptionsT options;
  LRT_TENSOR_ASSIGN_OR_RETURN(options.output_type,
                              ToTfLiteArgMaxOutputTensorType(data.output_type));
  return TfLiteOpBuildInfo(tflite::BuiltinOperator_ARG_MAX, options);
}

}  // namespace litert::tensor::graph

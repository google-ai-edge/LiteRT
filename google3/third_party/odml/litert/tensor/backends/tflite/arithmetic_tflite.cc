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
#include "third_party/odml/litert/tensor/backends/tflite/arithmetic_tflite.h"

#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/arithmetic_graph.h"
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/internal/graph.h"
#include "third_party/odml/litert/tensor/utils/macros.h"
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

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<AddOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const BinaryOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data->activation));
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_ADD,
      tflite::AddOptionsT{.fused_activation_function = tflite_activation});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<MulOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const BinaryOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data->activation));
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_MUL,
      tflite::MulOptionsT{.fused_activation_function = tflite_activation});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<AbsOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_ABS);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ReluOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_RELU);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<Relu6OperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_RELU6);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<LeakyReluOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const LeakyReluOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_LEAKY_RELU,
      tflite::LeakyReluOptionsT{.alpha = data->alpha});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<EluOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_ELU);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<HardSwishOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_HARD_SWISH);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<PReluOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_PRELU);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<L2NormalizationOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_L2_NORMALIZATION,
      tflite::L2NormOptionsT{.fused_activation_function =
                                 tflite::ActivationFunctionType_NONE});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SubOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const BinaryOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data->activation));
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_SUB,
      tflite::SubOptionsT{.fused_activation_function = tflite_activation});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<DivOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const BinaryOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data->activation));
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_DIV,
      tflite::DivOptionsT{.fused_activation_function = tflite_activation});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SquareOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SQUARE);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<RsqrtOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_RSQRT);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<PowOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_POW);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<NegOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_NEG);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SqrtOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SQRT);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ExpOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_EXP);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<LogOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LOG);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<CeilOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_CEIL);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<FloorOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_FLOOR);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<FloorDivOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_FLOOR_DIV);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<FloorModOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_FLOOR_MOD);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SignOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SIGN);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<RoundOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_ROUND);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<TransposeOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_TRANSPOSE);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<LogisticOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LOGISTIC);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<EmbeddingLookupOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_EMBEDDING_LOOKUP);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<DynamicUpdateSliceOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_DYNAMIC_UPDATE_SLICE);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<TileOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_TILE);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<GeluOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const GeluOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_GELU,
      tflite::GeluOptionsT{.approximate = data->approximate});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<TanhOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_TANH);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<CastOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_CAST);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SelectOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SELECT);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SelectV2OperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SELECT_V2);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SliceOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SLICE);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<LessOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LESS);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<GreaterOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_GREATER);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<GreaterEqualOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_GREATER_EQUAL);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<EqualOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_EQUAL);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<NotEqualOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_NOT_EQUAL);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<MinimumOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_MINIMUM);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<MaximumOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_MAXIMUM);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<LogicalAndOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LOGICAL_AND);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<LogicalOrOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LOGICAL_OR);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<LogicalNotOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LOGICAL_NOT);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<BitwiseXorOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_BITWISE_XOR);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<RightShiftOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_RIGHT_SHIFT);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<CosOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_COS);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SinOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SIN);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SoftmaxOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const SoftmaxOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SOFTMAX,
                           tflite::SoftmaxOptionsT{.beta = data->beta});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<LogSoftmaxOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LOG_SOFTMAX);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ReshapeOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const ReshapeOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_RESHAPE,
      tflite::ReshapeOptionsT{.new_shape = data->new_shape});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SqueezeOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const SqueezeOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_SQUEEZE,
      tflite::SqueezeOptionsT{.squeeze_dims = data->squeeze_dims});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ExpandDimsOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_EXPAND_DIMS);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<BatchMatMulOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const BatchMatMulOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_BATCH_MATMUL,
                           tflite::BatchMatMulOptionsT{
                               .adj_x = data->adj_x,
                               .adj_y = data->adj_y,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<FullyConnectedOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const FullyConnectedOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data->activation));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_FULLY_CONNECTED,
                           tflite::FullyConnectedOptionsT{
                               .fused_activation_function = tflite_activation,
                               .keep_num_dims = data->keep_num_dims,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ConcatenationOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const ConcatenationOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data->activation));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_CONCATENATION,
                           tflite::ConcatenationOptionsT{
                               .axis = data->axis,
                               .fused_activation_function = tflite_activation,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<PackOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const PackOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  auto op = dynamic_cast<const Operation*>(this);
  if (op == nullptr) {
    return absl::InternalError("Failed to cast to Operation base.");
  }
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_PACK,
      tflite::PackOptionsT{.values_count = static_cast<int>(op->inputs.size()),
                           .axis = data->axis});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<UnpackOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const UnpackOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_UNPACK,
      tflite::UnpackOptionsT{.num = data->num, .axis = data->axis});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SplitOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const SplitOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  auto op = dynamic_cast<const Operation*>(this);
  if (op == nullptr) {
    return absl::InternalError("Failed to cast to Operation base.");
  }

  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_SPLIT,
      tflite::SplitOptionsT{.num_splits = data->num_splits});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<CustomOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const CustomOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  TfLiteOpBuildInfo info(::tflite::BuiltinOperator_CUSTOM);
  info.custom_code = &data->custom_code;
  info.custom_options = &data->custom_options;
  return info;
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<AveragePool2DOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const AveragePool2DOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data->activation));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_padding, ToTflite(data->padding));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_AVERAGE_POOL_2D,
                           tflite::Pool2DOptionsT{
                               .padding = tflite_padding,
                               .stride_w = data->stride_w,
                               .stride_h = data->stride_h,
                               .filter_width = data->filter_width,
                               .filter_height = data->filter_height,
                               .fused_activation_function = tflite_activation,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<MaxPool2DOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const MaxPool2DOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data->activation));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_padding, ToTflite(data->padding));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_MAX_POOL_2D,
                           tflite::Pool2DOptionsT{
                               .padding = tflite_padding,
                               .stride_w = data->stride_w,
                               .stride_h = data->stride_h,
                               .filter_width = data->filter_width,
                               .filter_height = data->filter_height,
                               .fused_activation_function = tflite_activation,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<Conv2DOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const Conv2DOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data->activation));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_padding, ToTflite(data->padding));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_CONV_2D,
                           tflite::Conv2DOptionsT{
                               .padding = tflite_padding,
                               .stride_w = data->stride_w,
                               .stride_h = data->stride_h,
                               .fused_activation_function = tflite_activation,
                               .dilation_w_factor = data->dilation_w_factor,
                               .dilation_h_factor = data->dilation_h_factor,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<DepthwiseConv2DOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const DepthwiseConv2DOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_activation,
                              ToTflite(data->activation));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_padding, ToTflite(data->padding));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                           tflite::DepthwiseConv2DOptionsT{
                               .padding = tflite_padding,
                               .stride_w = data->stride_w,
                               .stride_h = data->stride_h,
                               .depth_multiplier = data->depth_multiplier,
                               .fused_activation_function = tflite_activation,
                               .dilation_w_factor = data->dilation_w_factor,
                               .dilation_h_factor = data->dilation_h_factor,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<PadOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_PAD);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<PadV2OperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_PADV2);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SumOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const SumOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_SUM,
      tflite::ReducerOptionsT{.keep_dims = data->keep_dims});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ReduceMaxOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const ReduceMaxOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_REDUCE_MAX,
      tflite::ReducerOptionsT{.keep_dims = data->keep_dims});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<MeanOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const MeanOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_MEAN,
      tflite::ReducerOptionsT{.keep_dims = data->keep_dims});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<TopKOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_TOPK_V2);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<CumsumOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const CumsumOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_CUMSUM,
                           tflite::CumsumOptionsT{.exclusive = data->exclusive,
                                                  .reverse = data->reverse});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ReverseOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_REVERSE_V2);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<SpaceToDepthOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const SpaceToDepthOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_SPACE_TO_DEPTH,
      tflite::SpaceToDepthOptionsT{.block_size = data->block_size});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<DepthToSpaceOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const DepthToSpaceOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_DEPTH_TO_SPACE,
      tflite::DepthToSpaceOptionsT{.block_size = data->block_size});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<GatherOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const GatherOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_GATHER,
      tflite::GatherOptionsT{.axis = data->axis,
                             .batch_dims = data->batch_dims});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<GatherNdOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_GATHER_ND,
                           tflite::GatherNdOptionsT{});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<OneHotOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const OneHotOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  tflite::OneHotOptionsT options;
  options.axis = data->axis;
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_ONE_HOT, options);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<QuantizeOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_QUANTIZE);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<DequantizeOperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_DEQUANTIZE);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ProbeOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto op = dynamic_cast<const Operation*>(this);
  if (op == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, GetInfo(op->inputs[0]));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_RESHAPE,
                           tflite::ReshapeOptionsT{.new_shape = info.shape});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ResizeBilinearOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const ResizeBilinearOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_RESIZE_BILINEAR,
                           tflite::ResizeBilinearOptionsT{
                               .align_corners = data->align_corners,
                               .half_pixel_centers = data->half_pixel_centers,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ResizeNearestNeighborOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const ResizeNearestNeighborOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                           tflite::ResizeNearestNeighborOptionsT{
                               .align_corners = data->align_corners,
                               .half_pixel_centers = data->half_pixel_centers,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<NonMaxSuppressionV5OperationTag, TfLiteMixinTag>::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_NON_MAX_SUPPRESSION_V5);
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<LstmOperationTag, TfLiteMixinTag>::ToTfLite() const {
  static const std::string* kCustomLstmCode = new std::string("LSTM_BASIC");
  TfLiteOpBuildInfo info(::tflite::BuiltinOperator_CUSTOM);
  info.custom_code = kCustomLstmCode;
  return info;
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<TransposeConvOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const TransposeConvOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_padding, ToTflite(data->padding));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_TRANSPOSE_CONV,
                           tflite::TransposeConvOptionsT{
                               .padding = tflite_padding,
                               .stride_w = data->stride_w,
                               .stride_h = data->stride_h,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<TransposeConv2DOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const TransposeConv2DOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError("Could not access data base.");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tflite_padding, ToTflite(data->padding));
  return TfLiteOpBuildInfo(
      ::tflite::BuiltinOperator_TRANSPOSE_CONV,
      tflite::TransposeConvOptionsT{.padding = tflite_padding,
                                    .stride_w = data->stride_w,
                                    .stride_h = data->stride_h});
}

absl::StatusOr<TfLiteOpBuildInfo>
OpMixin<ArgMaxOperationTag, TfLiteMixinTag>::ToTfLite() const {
  auto data = dynamic_cast<const ArgMaxOperationData*>(this);
  if (data == nullptr) {
    return absl::FailedPreconditionError(
        "Could not access ArgMaxOperationData.");
  }
  tflite::ArgMaxOptionsT options;
  LRT_TENSOR_ASSIGN_OR_RETURN(
      options.output_type, ToTfLiteArgMaxOutputTensorType(data->output_type));
  return TfLiteOpBuildInfo(tflite::BuiltinOperator_ARG_MAX, options);
}

}  // namespace litert::tensor::graph

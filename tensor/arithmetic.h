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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_ARITHMETIC_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_ARITHMETIC_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/types/optional.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic_graph.h"  // IWYU pragma: export
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/arithmetic_helpers.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/source_location.h"

namespace litert::tensor {

template <class... Mixins>
absl::Status CheckUnaryElementwiseOp(const graph::Tensor& a) {
  LRT_TENSOR_RETURN_IF_ERROR(graph::GetStatus(a));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& a_info, graph::GetInfo(a));
  if (a_info.shape.empty()) {
    return absl::InvalidArgumentError("Input tensor must not be a scalar.");
  }
  return absl::OkStatus();
}

template <class... Mixins>
Tensor<Mixins...> Add(
    Tensor<Mixins...> a, Tensor<Mixins...> b,
    FusedActivation fused_activation = FusedActivation::kActNone,
    source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::AddOperation<Mixins...>>(loc, a, b);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Add(
    Tensor<Mixins...> a, float b,
    FusedActivation fused_activation = FusedActivation::kActNone,
    source_location loc = source_location::current()) {
  Tensor<Mixins...> b_tensor(
      {.type = Type::kFP32,
       .shape = {},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({b})});
  return Add(a, b_tensor, fused_activation, loc);
}

template <class... Mixins>
Tensor<Mixins...> Mul(
    Tensor<Mixins...> a, Tensor<Mixins...> b,
    FusedActivation fused_activation = FusedActivation::kActNone,
    source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::MulOperation<Mixins...>>(loc, a, b);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Mul(
    Tensor<Mixins...> a, float b,
    FusedActivation fused_activation = FusedActivation::kActNone,
    source_location loc = source_location::current()) {
  Tensor<Mixins...> b_tensor(
      {.type = Type::kFP32,
       .shape = {},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({b})});
  return Mul(a, b_tensor, fused_activation, loc);
}

template <class... Mixins>
Tensor<Mixins...> Sub(
    Tensor<Mixins...> a, Tensor<Mixins...> b,
    FusedActivation fused_activation = FusedActivation::kActNone,
    source_location loc = source_location::current()) {
  return ElementwiseOp<graph::SubOperation<Mixins...>>(loc, a, b);
}

template <class... Mixins>
Tensor<Mixins...> Sub(
    float a, Tensor<Mixins...> b,
    FusedActivation fused_activation = FusedActivation::kActNone,
    source_location loc = source_location::current()) {
  Tensor<Mixins...> a_tensor(
      {.type = Type::kFP32,
       .shape = {},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({a})});
  return Sub(a_tensor, b, fused_activation, loc);
}

template <class... Mixins>
Tensor<Mixins...> Div(
    Tensor<Mixins...> a, Tensor<Mixins...> b,
    FusedActivation fused_activation = FusedActivation::kActNone,
    source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::DivOperation<Mixins...>>(loc, a, b);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Div(
    Tensor<Mixins...> a, float b,
    FusedActivation fused_activation = FusedActivation::kActNone,
    source_location loc = source_location::current()) {
  Tensor<Mixins...> b_tensor(
      {.type = Type::kFP32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({b})});
  return Div(a, b_tensor, fused_activation, loc);
}

template <class... Mixins>
Tensor<Mixins...> Abs(Tensor<Mixins...> a,
                      source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::AbsOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Relu(Tensor<Mixins...> a,
                       source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::ReluOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Relu6(Tensor<Mixins...> a,
                        source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::Relu6Operation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> ReluN1To1(Tensor<Mixins...> a,
                            source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::ReluN1To1Operation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> ZerosLike(Tensor<Mixins...> a,
                            source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::ZerosLikeOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Relu0To1(Tensor<Mixins...> a,
                           source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::Relu0To1Operation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> LeakyRelu(Tensor<Mixins...> a, float alpha = 0.2f,
                            source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::LeakyReluOperation<Mixins...>>();
  op->alpha = alpha;
  AddInputs(op, a);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& a_info = *GetInfo(a.GetRaw());
  graph::TensorInformation& o_info = *GetInfo(output.GetRaw());
  o_info.shape = a_info.shape;
  o_info.type = a_info.type;
  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Elu(Tensor<Mixins...> a,
                      source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::EluOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> HardSwish(Tensor<Mixins...> a,
                            source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::HardSwishOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> PRelu(Tensor<Mixins...> a, Tensor<Mixins...> alpha,
                        source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::PReluOperation<Mixins...>>();
  AddInputs(op, a, alpha);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& a_info = *GetInfo(a.GetRaw());
  graph::TensorInformation& o_info = *GetInfo(output.GetRaw());
  o_info.shape = a_info.shape;
  o_info.type = a_info.type;
  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> L2Normalization(
    Tensor<Mixins...> input, source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::L2NormalizationOperation<Mixins...>>();
  AddInputs(op, input);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = input_info.shape;
  output_info.type = input_info.type;
  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Square(Tensor<Mixins...> a,
                         source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::SquareOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Rsqrt(Tensor<Mixins...> a,
                        source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::RsqrtOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Pow(Tensor<Mixins...> a, Tensor<Mixins...> b,
                      source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::PowOperation<Mixins...>>(loc, a, b);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Neg(Tensor<Mixins...> a,
                      source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::NegOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Sqrt(Tensor<Mixins...> a,
                       source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::SqrtOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Less(Tensor<Mixins...> a, Tensor<Mixins...> b,
                       source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::LessOperation<Mixins...>>(loc, a, b);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Greater(Tensor<Mixins...> a, Tensor<Mixins...> b,
                          source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::GreaterOperation<Mixins...>>(loc, a, b);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> GreaterEqual(
    Tensor<Mixins...> a, Tensor<Mixins...> b,
    source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::GreaterEqualOperation<Mixins...>>(loc, a, b);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Equal(Tensor<Mixins...> a, Tensor<Mixins...> b,
                        source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::EqualOperation<Mixins...>>(loc, a, b);
  graph::TensorInformation& o_info = *GetInfo(output.GetRaw());
  o_info.type = Type::kBOOL;
  return output;
}

template <class... Mixins>
Tensor<Mixins...> NotEqual(Tensor<Mixins...> a, Tensor<Mixins...> b,
                           source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::NotEqualOperation<Mixins...>>(loc, a, b);
  graph::TensorInformation& o_info = *GetInfo(output.GetRaw());
  o_info.type = Type::kBOOL;
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Minimum(Tensor<Mixins...> a, Tensor<Mixins...> b,
                          source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::MinimumOperation<Mixins...>>(loc, a, b);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Maximum(Tensor<Mixins...> a, Tensor<Mixins...> b,
                          source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::MaximumOperation<Mixins...>>(loc, a, b);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> LogicalAnd(Tensor<Mixins...> a, Tensor<Mixins...> b,
                             source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::LogicalAndOperation<Mixins...>>(loc, a, b);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> LogicalOr(Tensor<Mixins...> a, Tensor<Mixins...> b,
                            source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::LogicalOrOperation<Mixins...>>(loc, a, b);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> LogicalNot(Tensor<Mixins...> a,
                             source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::LogicalNotOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> BitwiseXor(Tensor<Mixins...> a, Tensor<Mixins...> b,
                             source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::BitwiseXorOperation<Mixins...>>(loc, a, b);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> RightShift(Tensor<Mixins...> a, Tensor<Mixins...> b,
                             source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::RightShiftOperation<Mixins...>>(loc, a, b);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Cos(Tensor<Mixins...> a,
                      source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::CosOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Sin(Tensor<Mixins...> a,
                      source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::SinOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Exp(Tensor<Mixins...> a,
                      source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::ExpOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Log(Tensor<Mixins...> a,
                      source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::LogOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Ceil(Tensor<Mixins...> a,
                       source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::CeilOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Floor(Tensor<Mixins...> a,
                        source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::FloorOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> FloorDiv(Tensor<Mixins...> a, Tensor<Mixins...> b,
                           source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::FloorDivOperation<Mixins...>>(loc, a, b);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> FloorMod(Tensor<Mixins...> a, Tensor<Mixins...> b,
                           source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::FloorModOperation<Mixins...>>(loc, a, b);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Sign(Tensor<Mixins...> a,
                       source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::SignOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Round(Tensor<Mixins...> a,
                        source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::RoundOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Logistic(Tensor<Mixins...> a,
                           source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::LogisticOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Tanh(Tensor<Mixins...> a,
                       source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::TanhOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Pad(Tensor<Mixins...> a, Tensor<Mixins...> b,
                      source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::PadOperation<Mixins...>>();
  AddInputs(op, a, b);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& a_info = *GetInfo(a.GetRaw());
  const graph::TensorInformation& b_info = *GetInfo(b.GetRaw());
  graph::TensorInformation& o_info = *GetInfo(output.GetRaw());
  o_info.type = a_info.type;
  o_info.shape = a_info.shape;
  if (b_info.buffer) {
    const auto b_data = b_info.buffer->Lock().As<const int32_t>().data();
    for (int i = 0; i < o_info.shape.size(); ++i) {
      o_info.shape[i] += b_data[i * 2] + b_data[i * 2 + 1];
    }
  } else {
    return Tensor<Mixins...>(graph::ErrorTensor(
        absl::InvalidArgumentError("The padding tensor must have a buffer.")));
  }
  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> PadV2(Tensor<Mixins...> a, Tensor<Mixins...> b,
                        Tensor<Mixins...> c,
                        source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::PadV2Operation<Mixins...>>();
  AddInputs(op, a, b, c);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& a_info = *GetInfo(a.GetRaw());
  const graph::TensorInformation& b_info = *GetInfo(b.GetRaw());
  graph::TensorInformation& o_info = *GetInfo(output.GetRaw());
  o_info.type = a_info.type;
  o_info.shape = a_info.shape;
  if (b_info.buffer) {
    const auto b_data = b_info.buffer->Lock().As<const int32_t>().data();
    for (int i = 0; i < o_info.shape.size(); ++i) {
      o_info.shape[i] += b_data[i * 2] + b_data[i * 2 + 1];
    }
  } else {
    return Tensor<Mixins...>(graph::ErrorTensor(
        absl::InvalidArgumentError("The padding tensor must have a buffer.")));
  }
  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> ExpandDims(Tensor<Mixins...> input, Tensor<Mixins...> axis,
                             source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::ExpandDimsOperation<Mixins...>>();
  AddInputs(op, input, axis);
  Tensor<Mixins...> output = AddOutput(op, loc);

  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  const graph::TensorInformation& axis_info = *GetInfo(axis.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());

  output_info.shape = input_info.shape;
  output_info.type = input_info.type;

  if (axis_info.buffer) {
    int real_axis =
        axis_info.buffer->Lock().template As<const int32_t>().data()[0];
    if (real_axis < 0) {
      real_axis += input_info.shape.size() + 1;
    }
    output_info.shape.insert(output_info.shape.begin() + real_axis, 1);
  }

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> ExpandDims(Tensor<Mixins...> input, int axis,
                             source_location loc = source_location::current()) {
  Tensor<Mixins...> axis_tensor(
      {.type = Type::kI32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>({axis})});
  return ExpandDims(input, axis_tensor, loc);
}

template <class... Mixins>
Tensor<Mixins...> Squeeze(Tensor<Mixins...> input,
                          std::vector<int> squeeze_dims = {},
                          source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::SqueezeOperation<Mixins...>>();
  op->squeeze_dims = squeeze_dims;
  AddInputs(op, input);
  Tensor<Mixins...> output = AddOutput(op, loc);

  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());

  output_info.type = input_info.type;

  int rank = input_info.shape.size();
  std::vector<int> actual_squeeze_dims = squeeze_dims;
  for (int& dim : actual_squeeze_dims) {
    if (dim < 0) {
      dim += rank;
    }
  }

  if (actual_squeeze_dims.empty()) {
    for (int i = 0; i < rank; ++i) {
      if (input_info.shape[i] != 1) {
        output_info.shape.push_back(input_info.shape[i]);
      }
    }
  } else {
    for (int i = 0; i < rank; ++i) {
      if (absl::c_find(actual_squeeze_dims, i) == actual_squeeze_dims.end()) {
        output_info.shape.push_back(input_info.shape[i]);
      }
    }
  }

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Reshape(Tensor<Mixins...> input, std::vector<int> new_shape,
                          source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::ReshapeOperation<Mixins...>>();
  op->new_shape = new_shape;
  AddInputs(op, input);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = op->new_shape;
  output_info.type = input_info.type;
  if (input_info.GetSize() != output_info.GetSize()) {
    return Tensor<Mixins...>(graph::ErrorTensor(absl::InvalidArgumentError(
        absl::StrCat("The output size must be the same as the input size. "
                     "input_size: ",
                     input_info.GetSize(), " output_size: ",
                     output_info.GetSize(), " input_name: ", input.GetName(),
                     " output_name: ", output.GetName()))));
  }

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Softmax(Tensor<Mixins...> a, float beta = 1,
                          source_location loc = source_location::current()) {
  auto operation = std::make_shared<graph::SoftmaxOperation<Mixins...>>();
  operation->beta = beta;
  AddInputs(operation, a);
  Tensor<Mixins...> output = AddOutput(operation, loc);
  const graph::TensorInformation& a_info = *GetInfo(a.GetRaw());
  graph::TensorInformation& o_info = *GetInfo(output.GetRaw());
  o_info.shape = a_info.shape;
  o_info.type = a_info.type;
  graph::OpDebugger::DebugOp(*operation);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> LogSoftmax(Tensor<Mixins...> a,
                             source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::LogSoftmaxOperation<Mixins...>>(loc, a);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Sum(Tensor<Mixins...> a, std::vector<int> axes,
                      bool keep_dims,
                      source_location loc = source_location::current()) {
  Tensor<Mixins...> axes_tensor(
      {.type = Type::kI32,
       .shape = {static_cast<int>(axes.size())},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>(axes)});
  return Sum(a, axes_tensor, keep_dims, loc);
}

template <class... Mixins>
Tensor<Mixins...> Sum(Tensor<Mixins...> a, Tensor<Mixins...> b, bool keep_dims,
                      source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::SumOperation<Mixins...>>();
  op->keep_dims = keep_dims;
  AddInputs(op, a, b);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& a_info = *GetInfo(a.GetRaw());
  const graph::TensorInformation& b_info = *GetInfo(b.GetRaw());
  graph::TensorInformation& o_info = *GetInfo(output.GetRaw());
  o_info.type = a_info.type;
  if (b_info.buffer == nullptr) {
    return Tensor<Mixins...>(graph::ErrorTensor(absl::InvalidArgumentError(
        "The reduction tensor must have a buffer.")));
  }
  const auto b_data = b_info.buffer->Lock().As<const int32_t>().data();
  if (op->keep_dims) {
    o_info.shape = a_info.shape;
    if (b_info.shape.empty()) {
      o_info.shape.push_back(o_info.shape[b_data[0]]);
    } else {
      for (int i = 0; i < b_info.shape[0]; ++i) {
        o_info.shape[b_data[i]] = 1;
      }
    }
  } else {
    o_info.shape = {};
    for (int i = 0; i < a_info.shape.size(); ++i) {
      bool found = false;
      for (int j = 0; j < b_info.shape[0]; ++j) {
        if (i == b_data[j]) {
          found = true;
          break;
        }
      }
      if (!found) {
        o_info.shape.push_back(a_info.shape[i]);
      }
    }
  }

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> ReduceMax(Tensor<Mixins...> a, std::vector<int> axes,
                            bool keep_dims,
                            source_location loc = source_location::current()) {
  Tensor<Mixins...> axes_tensor(
      {.type = Type::kI32,
       .shape = {static_cast<int>(axes.size())},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>(axes)});
  return ReduceMax(a, axes_tensor, keep_dims, loc);
}

template <class... Mixins>
Tensor<Mixins...> ReduceMax(Tensor<Mixins...> a, Tensor<Mixins...> b,
                            bool keep_dims,
                            source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::ReduceMaxOperation<Mixins...>>();
  op->keep_dims = keep_dims;
  AddInputs(op, a, b);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& a_info = *GetInfo(a.GetRaw());
  const graph::TensorInformation& b_info = *GetInfo(b.GetRaw());
  graph::TensorInformation& o_info = *GetInfo(output.GetRaw());
  o_info.type = a_info.type;
  if (b_info.buffer == nullptr) {
    return Tensor<Mixins...>(graph::ErrorTensor(absl::InvalidArgumentError(
        "The reduction tensor must have a buffer.")));
  }
  const auto b_data = b_info.buffer->Lock().As<const int32_t>().data();
  if (op->keep_dims) {
    o_info.shape = a_info.shape;
    if (b_info.shape.empty()) {
      o_info.shape.push_back(o_info.shape[b_data[0]]);
    } else {
      for (int i = 0; i < b_info.shape[0]; ++i) {
        o_info.shape[b_data[i]] = 1;
      }
    }
  } else {
    o_info.shape = {};
    for (int i = 0; i < a_info.shape.size(); ++i) {
      bool found = false;
      for (int j = 0; j < b_info.shape[0]; ++j) {
        if (i == b_data[j]) {
          found = true;
          break;
        }
      }
      if (!found) {
        o_info.shape.push_back(a_info.shape[i]);
      }
    }
  }

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Mean(Tensor<Mixins...> a, std::vector<int> axes,
                       bool keep_dims,
                       source_location loc = source_location::current()) {
  Tensor<Mixins...> axes_tensor(
      {.type = Type::kI32,
       .shape = {static_cast<int>(axes.size())},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>(axes)});
  return Mean(a, axes_tensor, keep_dims, loc);
}

template <class... Mixins>
Tensor<Mixins...> Mean(Tensor<Mixins...> a, Tensor<Mixins...> b, bool keep_dims,
                       source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::MeanOperation<Mixins...>>();
  op->keep_dims = keep_dims;
  AddInputs(op, a, b);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& a_info = *GetInfo(a.GetRaw());
  const graph::TensorInformation& b_info = *GetInfo(b.GetRaw());
  graph::TensorInformation& o_info = *GetInfo(output.GetRaw());
  o_info.type = a_info.type;
  if (b_info.buffer == nullptr) {
    return Tensor<Mixins...>(graph::ErrorTensor(absl::InvalidArgumentError(
        "The reduction tensor must have a buffer.")));
  }
  const auto b_data = b_info.buffer->Lock().As<const int32_t>().data();
  if (op->keep_dims) {
    o_info.shape = a_info.shape;
    if (b_info.shape.empty()) {
      o_info.shape.push_back(o_info.shape[b_data[0]]);
    } else {
      for (int i = 0; i < b_info.shape[0]; ++i) {
        o_info.shape[b_data[i]] = 1;
      }
    }
  } else {
    o_info.shape = {};
    for (int i = 0; i < a_info.shape.size(); ++i) {
      bool found = false;
      for (int j = 0; j < b_info.shape[0]; ++j) {
        if (i == b_data[j]) {
          found = true;
          break;
        }
      }
      if (!found) {
        o_info.shape.push_back(a_info.shape[i]);
      }
    }
  }

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> ArgMax(Tensor<Mixins...> a, int axis,
                         Type output_type = Type::kI64,
                         source_location loc = source_location::current()) {
  Tensor<Mixins...> axis_tensor(
      {.type = Type::kI32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>({axis})});
  return ArgMax(a, axis_tensor, output_type, loc);
}

template <class... Mixins>
Tensor<Mixins...> ArgMax(Tensor<Mixins...> a, Tensor<Mixins...> b,
                         Type output_type = Type::kI64,
                         source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::ArgMaxOperation<Mixins...>>();
  op->output_type = output_type;
  AddInputs(op, a, b);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& a_info = *GetInfo(a.GetRaw());
  const graph::TensorInformation& b_info = *GetInfo(b.GetRaw());
  graph::TensorInformation& o_info = *GetInfo(output.GetRaw());
  o_info.type = output_type;
  if (b_info.buffer == nullptr) {
    return Tensor<Mixins...>(graph::ErrorTensor(absl::InvalidArgumentError(
        "The reduction tensor must have a buffer.")));
  }
  const auto b_data = b_info.buffer->Lock().As<const int32_t>().data();
  int axis = b_data[0];
  if (axis < 0) {
    axis += a_info.shape.size();
  }
  o_info.shape = {};
  for (int i = 0; i < a_info.shape.size(); ++i) {
    if (i != axis) {
      o_info.shape.push_back(a_info.shape[i]);
    }
  }

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> BatchMatMul(
    Tensor<Mixins...> x, Tensor<Mixins...> y, bool adj_x = false,
    bool adj_y = false, source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::BatchMatMulOperation<Mixins...>>();
  op->adj_x = adj_x;
  op->adj_y = adj_y;
  AddInputs(op, x, y);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& x_info = *GetInfo(x.GetRaw());
  const graph::TensorInformation& y_info = *GetInfo(y.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());

  std::vector<int> x_shape = x_info.shape;
  std::vector<int> y_shape = y_info.shape;
  if (adj_x) {
    std::swap(x_shape[x_shape.size() - 2], x_shape[x_shape.size() - 1]);
  }
  if (adj_y) {
    std::swap(y_shape[y_shape.size() - 2], y_shape[y_shape.size() - 1]);
  }

  if (x_shape.back() != y_shape[y_shape.size() - 2]) {
    std::string x_shape_str = absl::StrJoin(x_shape, ",");
    std::string y_shape_str = absl::StrJoin(y_shape, ",");
    return Tensor<Mixins...>(
        graph::ErrorTensor(absl::InvalidArgumentError(absl::StrCat(
            "The inner dimensions of the input tensors must match. x_name: ",
            x.GetName(), " y_name: ", y.GetName(), " x_shape: ", x_shape_str,
            " y_shape: ", y_shape_str, " adj_x: ", adj_x, " adj_y: ", adj_y))));
  }

  // Batch dimensions should be broadcastable.
  if (x_shape.size() != y_shape.size()) {
    // For now, we only support same rank BatchMatMul.
    // TODO(piyu): Support different ranks.
  }
  output_info.shape.reserve(x_shape.size());
  for (size_t i = 0; i < x_shape.size() - 2; ++i) {
    const auto x_dim = x_shape[i];
    const auto y_dim = y_shape[i];
    if (x_dim != y_dim && x_dim != 1 && y_dim != 1) {
      return Tensor<Mixins...>(graph::ErrorTensor(absl::InvalidArgumentError(
          absl::StrCat("The batch dimensions of the input tensors must be "
                       "broadcastable. x_shape: ",
                       absl::StrJoin(x_info.shape, ","),
                       " y_shape: ", absl::StrJoin(y_info.shape, ",")))));
    }
    output_info.shape.push_back(std::max(x_dim, y_dim));
  }
  output_info.shape.push_back(x_shape[x_shape.size() - 2]);
  output_info.shape.push_back(y_shape[y_shape.size() - 1]);

  output_info.type = x_info.type;

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> FullyConnected(
    Tensor<Mixins...> input, Tensor<Mixins...> weights,
    std::optional<Tensor<Mixins...>> bias,
    FusedActivation activation = kActNone, bool keep_num_dims = true,
    source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::FullyConnectedOperation<Mixins...>>();
  op->activation = activation;
  op->keep_num_dims = keep_num_dims;
  AddInputs(op, input, weights);
  if (bias.has_value()) {
    AddInputs(op, bias.value());
  }
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  const graph::TensorInformation& weights_info = *GetInfo(weights.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  if (keep_num_dims) {
    output_info.shape = input_info.shape;
    output_info.shape.back() = weights_info.shape[0];
  } else {
    output_info.shape = {input_info.shape[0], weights_info.shape[0]};
  }
  output_info.type = input_info.type;

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> FullyConnected(
    Tensor<Mixins...> input, Tensor<Mixins...> weights, Tensor<Mixins...> bias,
    FusedActivation activation = kActNone, bool keep_num_dims = true,
    source_location loc = source_location::current()) {
  return FullyConnected(input, weights, std::optional(std::move(bias)),
                        activation, keep_num_dims, loc);
}

template <class... Mixins>
Tensor<Mixins...> FullyConnected(
    Tensor<Mixins...> input, Tensor<Mixins...> weights,
    FusedActivation activation = kActNone, bool keep_num_dims = true,
    source_location loc = source_location::current()) {
  return FullyConnected(input, weights,
                        /*bias=*/std::optional<Tensor<Mixins...>>(std::nullopt),
                        activation, keep_num_dims, loc);
}

template <class... Mixins>
Tensor<Mixins...> AveragePool2D(
    Tensor<Mixins...> input, int filter_height, int filter_width, int stride_h,
    int stride_w, Padding padding, FusedActivation activation = kActNone,
    source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::AveragePool2DOperation<Mixins...>>();
  op->filter_height = filter_height;
  op->filter_width = filter_width;
  op->stride_h = stride_h;
  op->stride_w = stride_w;
  op->padding = padding;
  op->activation = activation;
  AddInputs(op, input);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.type = input_info.type;

  const int input_h = input_info.shape[1];
  const int input_w = input_info.shape[2];

  int output_h = 0;
  int output_w = 0;
  if (padding == kPaddingSame) {
    output_h = (input_h + stride_h - 1) / stride_h;
    output_w = (input_w + stride_w - 1) / stride_w;
  } else if (padding == kPaddingValid) {
    output_h = (input_h - filter_height) / stride_h + 1;
    output_w = (input_w - filter_width) / stride_w + 1;
  }

  output_info.shape = {input_info.shape[0], output_h, output_w,
                       input_info.shape[3]};

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> MaxPool2D(Tensor<Mixins...> input, int filter_height,
                            int filter_width, int stride_h, int stride_w,
                            Padding padding,
                            FusedActivation activation = kActNone,
                            source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::MaxPool2DOperation<Mixins...>>();
  op->filter_height = filter_height;
  op->filter_width = filter_width;
  op->stride_h = stride_h;
  op->stride_w = stride_w;
  op->padding = padding;
  op->activation = activation;
  AddInputs(op, input);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.type = input_info.type;

  const int input_h = input_info.shape[1];
  const int input_w = input_info.shape[2];

  int output_h = 0;
  int output_w = 0;
  if (padding == kPaddingSame) {
    output_h = (input_h + stride_h - 1) / stride_h;
    output_w = (input_w + stride_w - 1) / stride_w;
  } else if (padding == kPaddingValid) {
    output_h = (input_h - filter_height) / stride_h + 1;
    output_w = (input_w - filter_width) / stride_w + 1;
  }

  output_info.shape = {input_info.shape[0], output_h, output_w,
                       input_info.shape[3]};

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
TensorHandle Conv2DImpl(Tensor<Mixins...> input, Tensor<Mixins...> filter,
                        absl::optional<Tensor<Mixins...>> bias, int stride_h,
                        int stride_w, Padding padding, int dilation_h_factor,
                        int dilation_w_factor, FusedActivation activation,
                        source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::Conv2DOperation<Mixins...>>();
  op->stride_h = stride_h;
  op->stride_w = stride_w;
  op->padding = padding;
  op->dilation_h_factor = dilation_h_factor;
  op->dilation_w_factor = dilation_w_factor;
  op->activation = activation;
  if (bias.has_value()) {
    AddInputs(op, input, filter, *bias);
  } else {
    AddInputs(op, input, filter);
  }
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  const graph::TensorInformation& filter_info = *GetInfo(filter.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.type = input_info.type;

  const int input_h = input_info.shape[1];
  const int input_w = input_info.shape[2];
  const int filter_h = filter_info.shape[1];
  const int filter_w = filter_info.shape[2];
  const int output_channels = filter_info.shape[0];

  int output_h = 0;
  int output_w = 0;
  if (padding == kPaddingSame) {
    output_h = (input_h + stride_h - 1) / stride_h;
    output_w = (input_w + stride_w - 1) / stride_w;
  } else if (padding == kPaddingValid) {
    output_h =
        (input_h - (filter_h - 1) * dilation_h_factor - 1) / stride_h + 1;
    output_w =
        (input_w - (filter_w - 1) * dilation_w_factor - 1) / stride_w + 1;
  }

  output_info.shape = {input_info.shape[0], output_h, output_w,
                       output_channels};

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Conv2D(Tensor<Mixins...> input, Tensor<Mixins...> filter,
                         Tensor<Mixins...> bias, int stride_h, int stride_w,
                         Padding padding, int dilation_h_factor = 1,
                         int dilation_w_factor = 1,
                         FusedActivation activation = kActNone,
                         source_location loc = source_location::current()) {
  return Conv2DImpl(input, filter, absl::optional(bias), stride_h, stride_w,
                    padding, dilation_h_factor, dilation_w_factor, activation,
                    loc);
}

template <class... Mixins>
Tensor<Mixins...> Conv2D(Tensor<Mixins...> input, Tensor<Mixins...> filter,
                         int stride_h, int stride_w, Padding padding,
                         int dilation_h_factor = 1, int dilation_w_factor = 1,
                         FusedActivation activation = kActNone,
                         source_location loc = source_location::current()) {
  return Conv2DImpl(input, filter, absl::nullopt, stride_h, stride_w, padding,
                    dilation_h_factor, dilation_w_factor, activation, loc);
}

template <class... Mixins>
Tensor<Mixins...> DepthwiseConv2DImpl(
    Tensor<Mixins...> input, Tensor<Mixins...> filter,
    absl::optional<Tensor<Mixins...>> bias, int stride_h, int stride_w,
    Padding padding, int dilation_h_factor, int dilation_w_factor,
    int depth_multiplier, FusedActivation activation, source_location loc) {
  auto op = std::make_shared<graph::DepthwiseConv2DOperation<Mixins...>>();
  op->stride_h = stride_h;
  op->stride_w = stride_w;
  op->padding = padding;
  op->dilation_h_factor = dilation_h_factor;
  op->dilation_w_factor = dilation_w_factor;
  op->depth_multiplier = depth_multiplier;
  op->activation = activation;
  if (bias.has_value()) {
    AddInputs(op, input, filter, *bias);
  } else {
    AddInputs(op, input, filter);
  }
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  const graph::TensorInformation& filter_info = *GetInfo(filter.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.type = input_info.type;

  const int input_h = input_info.shape[1];
  const int input_w = input_info.shape[2];
  const int filter_h = filter_info.shape[1];
  const int filter_w = filter_info.shape[2];
  const int output_channels = filter_info.shape[3];

  int output_h = 0;
  int output_w = 0;
  if (padding == kPaddingSame) {
    output_h = (input_h + stride_h - 1) / stride_h;
    output_w = (input_w + stride_w - 1) / stride_w;
  } else if (padding == kPaddingValid) {
    output_h =
        (input_h - (filter_h - 1) * dilation_h_factor - 1) / stride_h + 1;
    output_w =
        (input_w - (filter_w - 1) * dilation_w_factor - 1) / stride_w + 1;
  }

  output_info.shape = {input_info.shape[0], output_h, output_w,
                       output_channels};

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> DepthwiseConv2D(
    Tensor<Mixins...> input, Tensor<Mixins...> filter, Tensor<Mixins...> bias,
    int stride_h, int stride_w, Padding padding, int dilation_h_factor = 1,
    int dilation_w_factor = 1, int depth_multiplier = 1,
    FusedActivation activation = kActNone,
    source_location loc = source_location::current()) {
  return DepthwiseConv2DImpl(
      input, filter, absl::optional(bias), stride_h, stride_w, padding,
      dilation_h_factor, dilation_w_factor, depth_multiplier, activation, loc);
}

template <class... Mixins>
Tensor<Mixins...> DepthwiseConv2D(
    Tensor<Mixins...> input, Tensor<Mixins...> filter, int stride_h,
    int stride_w, Padding padding, int dilation_h_factor = 1,
    int dilation_w_factor = 1, int depth_multiplier = 1,
    FusedActivation activation = kActNone,
    source_location loc = source_location::current()) {
  return DepthwiseConv2DImpl(input, filter, absl::nullopt, stride_h, stride_w,
                             padding, dilation_h_factor, dilation_w_factor,
                             depth_multiplier, activation, loc);
}

template <class... Mixins>
Tensor<Mixins...> Concatenation(
    absl::Span<Tensor<Mixins...>> inputs, int axis,
    FusedActivation activation = kActNone,
    source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::ConcatenationOperation<Mixins...>>();
  op->axis = axis;
  op->activation = activation;
  AddInputs(op, inputs);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& first_input_info =
      *GetInfo(inputs[0].GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.type = first_input_info.type;
  output_info.shape = first_input_info.shape;
  for (size_t i = 1; i < inputs.size(); ++i) {
    const graph::TensorInformation& input_info = *GetInfo(inputs[i].GetRaw());
    output_info.shape[axis] += input_info.shape[axis];
  }

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Concatenation(
    std::initializer_list<Tensor<Mixins...>> inputs, int axis,
    FusedActivation activation = kActNone,
    source_location loc = source_location::current()) {
  std::vector input_storage(inputs);
  return Concatenation(absl::MakeSpan(input_storage), axis, activation, loc);
}

template <class... Mixins>
Tensor<Mixins...> Pack(absl::Span<Tensor<Mixins...>> inputs, int axis,
                       source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::PackOperation<Mixins...>>();
  op->axis = axis;
  AddInputs(op, inputs);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& first_input_info =
      *GetInfo(inputs[0].GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.type = first_input_info.type;
  output_info.shape = first_input_info.shape;
  output_info.shape.insert(output_info.shape.begin() + axis, inputs.size());

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Pack(std::initializer_list<Tensor<Mixins...>> inputs,
                       int axis,
                       source_location loc = source_location::current()) {
  std::vector input_storage(inputs);
  return Pack(absl::MakeSpan(input_storage), axis, loc);
}

template <class... Mixins>
std::vector<Tensor<Mixins...>> Unpack(
    Tensor<Mixins...> input, int num, int axis,
    source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::UnpackOperation<Mixins...>>();
  op->num = num;
  op->axis = axis;
  AddInputs(op, input);
  auto out_group = graph::NewTensorGroup(num, loc);
  op->outputs_group = out_group;
  out_group->producer = op;
  std::vector<Tensor<Mixins...>> outputs;
  outputs.reserve(num);
  const graph::TensorInformation& input_info = GetInfo(input.GetRaw()).value();
  std::vector<int> output_shape = input_info.shape;
  output_shape.erase(output_shape.begin() + axis);
  for (int i = 0; i < num; ++i) {
    auto& output_info = out_group->tensor_infos[i];
    output_info.type = input_info.type;
    output_info.shape = output_shape;
    outputs.emplace_back(graph::GetTensor(i, out_group));
  }

  graph::OpDebugger::DebugOp(*op);
  return outputs;
}

template <class... Mixins>
std::vector<Tensor<Mixins...>> Split(
    Tensor<Mixins...> input, Tensor<Mixins...> axis, int num_splits,
    source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::SplitOperation<Mixins...>>();
  op->num_splits = num_splits;
  AddInputs(op, input, axis);

  auto out_group = graph::NewTensorGroup(num_splits, loc);
  op->outputs_group = out_group;
  out_group->producer = op;
  std::vector<Tensor<Mixins...>> outputs;
  outputs.reserve(num_splits);
  const graph::TensorInformation& input_info = GetInfo(input.GetRaw()).value();
  const graph::TensorInformation& axis_info = GetInfo(axis.GetRaw()).value();
  int axis_val = 0;
  if (axis_info.buffer) {
    axis_val = axis_info.buffer->Lock().As<const int32_t>().data()[0];
  } else {
    // TODO(b/269489748): Support dynamic axis for shape inference.
    return {Tensor<Mixins...>(graph::ErrorTensor(absl::InvalidArgumentError(
        "Axis must be a constant tensor for Split.")))};
  }

  if (axis_val < 0) {
    axis_val += input_info.shape.size();
  }

  if (input_info.shape[axis_val] % num_splits != 0) {
    return {Tensor<Mixins...>(graph::ErrorTensor(absl::InvalidArgumentError(
        "Number of splits must evenly divide the dimension.")))};
  }

  std::vector<int> output_shape = input_info.shape;
  output_shape[axis_val] /= num_splits;

  for (int i = 0; i < num_splits; ++i) {
    auto& output_info = out_group->tensor_infos[i];
    output_info.type = input_info.type;
    output_info.shape = output_shape;
    outputs.emplace_back(graph::GetTensor(i, out_group));
  }

  graph::OpDebugger::DebugOp(*op);
  return outputs;
}

template <class... Mixins>
std::vector<Tensor<Mixins...>> Split(
    Tensor<Mixins...> input, int axis, int num_splits,
    source_location loc = source_location::current()) {
  Tensor<Mixins...> axis_tensor(
      {.type = Type::kI32,
       .shape = {},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>({axis})});
  return Split(input, axis_tensor, num_splits, loc);
}

template <class... Mixins>
Tensor<Mixins...> SpaceToDepth(
    Tensor<Mixins...> input, int block_size,
    source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::SpaceToDepthOperation<Mixins...>>();
  op->block_size = block_size;
  AddInputs(op, input);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.type = input_info.type;

  // TFLite space_to_depth supports 4D input [batch, height, width, depth]
  if (input_info.shape.size() == 4) {
    output_info.shape = {input_info.shape[0], input_info.shape[1] / block_size,
                         input_info.shape[2] / block_size,
                         input_info.shape[3] * block_size * block_size};
  } else {
    // Basic fallback to same shape if it doesn't match 4D
    output_info.shape = input_info.shape;
  }

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> DepthToSpace(
    Tensor<Mixins...> input, int block_size,
    source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::DepthToSpaceOperation<Mixins...>>();
  op->block_size = block_size;
  AddInputs(op, input);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.type = input_info.type;

  if (input_info.shape.size() == 4) {
    output_info.shape = {input_info.shape[0], input_info.shape[1] * block_size,
                         input_info.shape[2] * block_size,
                         input_info.shape[3] / (block_size * block_size)};
  } else {
    output_info.shape = input_info.shape;
  }

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Transpose(Tensor<Mixins...> input, Tensor<Mixins...> perm,
                            source_location loc = source_location::current()) {
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  const graph::TensorInformation& perm_info = *GetInfo(perm.GetRaw());
  ABSL_CHECK_EQ(perm_info.type, Type::kI32)
      << "Transpose only supports I32 permutation types.";
  auto op = std::make_shared<graph::TransposeOperation<Mixins...>>();
  AddInputs(op, input, perm);
  Tensor<Mixins...> output = AddOutput(op, loc);
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  if (perm_info.buffer) {
    const auto perm_data = perm_info.buffer->Lock().As<const int32_t>();
    const auto& input_shape = input_info.shape;
    output_info.shape.resize(input_shape.size());
    for (size_t i = 0; i < perm_data.size(); ++i) {
      output_info.shape[i] = input_shape[perm_data.data()[i]];
    }
  } else {
    // If perm is not a constant, we cannot infer the shape at this time.
    // TODO(piyu): Support dynamic shape inference.
    output_info.shape = input_info.shape;
  }
  output_info.type = input_info.type;

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Transpose(Tensor<Mixins...> input,
                            const std::vector<int>& perm,
                            source_location loc = source_location::current()) {
  Tensor<Mixins...> perm_tensor(
      {.type = Type::kI32,
       .shape = {static_cast<int>(perm.size())},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>(perm)});
  return Transpose(input, perm_tensor, loc);
}

template <class... Mixins>
Tensor<Mixins...> Tile(Tensor<Mixins...> input, Tensor<Mixins...> multiples,
                       source_location loc = source_location::current()) {
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  const graph::TensorInformation& multiples_info = *GetInfo(multiples.GetRaw());
  ABSL_CHECK_EQ(multiples_info.type, Type::kI32)
      << "Tile only supports I32 multiples types.";
  auto op = std::make_shared<graph::TileOperation<Mixins...>>();
  AddInputs(op, input, multiples);
  Tensor<Mixins...> output = AddOutput(op, loc);
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  if (multiples_info.buffer) {
    const auto multiples_data =
        multiples_info.buffer->Lock().As<const int32_t>();
    const auto& input_shape = input_info.shape;
    output_info.shape.resize(input_shape.size());
    for (size_t i = 0; i < multiples_data.size(); ++i) {
      output_info.shape[i] = input_shape[i] * multiples_data.data()[i];
    }
  } else {
    // If multiples is not a constant, we cannot infer the shape at this time.
    // TODO(piyu): Support dynamic shape inference.
    output_info.shape = input_info.shape;
  }
  output_info.type = input_info.type;

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Tile(Tensor<Mixins...> input,
                       const std::vector<int>& multiples,
                       source_location loc = source_location::current()) {
  Tensor<Mixins...> multiples_tensor(
      {.type = Type::kI32,
       .shape = {static_cast<int>(multiples.size())},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>(multiples)});
  return Tile(input, multiples_tensor, loc);
}

template <class... Mixins>
Tensor<Mixins...> Gelu(Tensor<Mixins...> input, bool approximate = false,
                       source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::GeluOperation<Mixins...>>();
  op->approximate = approximate;
  AddInputs(op, input);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = input_info.shape;
  output_info.type = input_info.type;

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Cast(Tensor<Mixins...> input, Type to,
                       source_location loc = source_location::current()) {
  if (input.GetType() == to) {
    return input;
  }
  auto op = std::make_shared<graph::CastOperation<Mixins...>>();
  op->to = to;
  AddInputs(op, input);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = input_info.shape;
  output_info.type = to;

  graph::OpDebugger::DebugOp(*op);
  return output;
}

inline absl::StatusOr<std::vector<int>> BroadcastShapes(
    const std::vector<int>& shape1, const std::vector<int>& shape2) {
  if (shape1.empty()) return shape2;
  if (shape2.empty()) return shape1;

  const auto& s1 = shape1.size() > shape2.size() ? shape1 : shape2;
  const auto& s2 = shape1.size() > shape2.size() ? shape2 : shape1;
  std::vector<int> result_shape(s1.size());
  const int rank_diff = s1.size() - s2.size();

  for (int i = s1.size() - 1; i >= 0; --i) {
    const int dim1 = s1[i];
    const int dim2 = (i >= rank_diff) ? s2[i - rank_diff] : 1;

    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Shapes are not broadcast-compatible: ", absl::StrJoin(shape1, ","),
          " vs ", absl::StrJoin(shape2, ",")));
    }
    result_shape[i] = std::max(dim1, dim2);
  }
  return result_shape;
}

template <class... Mixins>
Tensor<Mixins...> Select(Tensor<Mixins...> condition, Tensor<Mixins...> a,
                         Tensor<Mixins...> b,
                         source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::SelectOperation<Mixins...>>();
  AddInputs(op, condition, a, b);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& condition_info = *GetInfo(condition.GetRaw());
  const graph::TensorInformation& a_info = *GetInfo(a.GetRaw());
  const graph::TensorInformation& b_info = *GetInfo(b.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());

  if (a_info.shape != b_info.shape) {
    return Tensor<Mixins...>(graph::ErrorTensor(absl::InvalidArgumentError(
        absl::StrCat("Shapes of a and b must match for Select. a: ",
                     absl::StrJoin(a_info.shape, ","),
                     " b: ", absl::StrJoin(b_info.shape, ",")))));
  }
  // In TFLite, Select condition can be a 1D tensor with length matching the
  // first dimension of the inputs, or match the inputs completely.
  if (condition_info.shape != a_info.shape) {
    if (condition_info.shape.size() != 1 ||
        condition_info.shape[0] != a_info.shape[0]) {
      return Tensor<Mixins...>(graph::ErrorTensor(absl::InvalidArgumentError(
          absl::StrCat("Shape of condition must match a and b or be 1D with "
                       "length matching the first dimension. condition: ",
                       absl::StrJoin(condition_info.shape, ","),
                       " a: ", absl::StrJoin(a_info.shape, ",")))));
    }
  }

  output_info.shape = a_info.shape;
  output_info.type = a_info.type;

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> SelectV2(Tensor<Mixins...> condition, Tensor<Mixins...> a,
                           Tensor<Mixins...> b,
                           source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::SelectV2Operation<Mixins...>>();
  AddInputs(op, condition, a, b);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& condition_info = *GetInfo(condition.GetRaw());
  const graph::TensorInformation& a_info = *GetInfo(a.GetRaw());
  const graph::TensorInformation& b_info = *GetInfo(b.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());

  auto broadcast_ab = BroadcastShapes(a_info.shape, b_info.shape);
  if (!broadcast_ab.ok()) {
    return Tensor<Mixins...>(graph::ErrorTensor(broadcast_ab.status()));
  }
  auto final_shape = BroadcastShapes(condition_info.shape, *broadcast_ab);
  if (!final_shape.ok()) {
    return Tensor<Mixins...>(graph::ErrorTensor(final_shape.status()));
  }

  output_info.shape = *final_shape;
  output_info.type = a_info.type;

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Slice(Tensor<Mixins...> input, Tensor<Mixins...> begin,
                        Tensor<Mixins...> size,
                        source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::SliceOperation<Mixins...>>();
  AddInputs(op, input, begin, size);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  const graph::TensorInformation& size_info = *GetInfo(size.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.type = input_info.type;
  if (size_info.buffer) {
    const auto size_data = size_info.buffer->Lock().As<const int32_t>();
    output_info.shape.assign(size_data.begin(), size_data.end());
    for (size_t i = 0; i < output_info.shape.size(); ++i) {
      if (output_info.shape[i] == -1) {
        output_info.shape[i] = input_info.shape[i];
      }
    }
  } else {
    // If size is not a constant, we cannot infer the shape at this time.
    // TODO(b/269489748): Support dynamic shape inference.
    output_info.shape = input_info.shape;
  }

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Slice(Tensor<Mixins...> input, const std::vector<int>& begin,
                        const std::vector<int>& size,
                        source_location loc = source_location::current()) {
  Tensor<Mixins...> begin_tensor(
      {.type = Type::kI32,
       .shape = {static_cast<int>(begin.size())},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>(begin)});
  Tensor<Mixins...> size_tensor(
      {.type = Type::kI32,
       .shape = {static_cast<int>(size.size())},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>(size)});
  return Slice(input, begin_tensor, size_tensor, loc);
}

template <class... Mixins>
Tensor<Mixins...> EmbeddingLookup(
    Tensor<Mixins...> ids, Tensor<Mixins...> value,
    Type output_type = Type::kFP32,
    source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::EmbeddingLookupOperation<Mixins...>>();
  AddInputs(op, ids, value);
  Tensor<Mixins...> output = AddOutput(op, loc);

  const graph::TensorInformation& value_info = *GetInfo(value.GetRaw());
  const graph::TensorInformation& ids_info = *GetInfo(ids.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());

  output_info.type = output_type;

  output_info.shape = ids_info.shape;
  output_info.shape.push_back(value_info.shape.back());

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> EmbeddingLookup(
    const std::vector<int>& ids, Tensor<Mixins...> value,
    source_location loc = source_location::current()) {
  Tensor<Mixins...> ids_tensor(
      {.type = Type::kI32,
       .shape = {static_cast<int>(ids.size())},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>(ids)});
  return EmbeddingLookup(ids_tensor, value, Type::kFP32, loc);
}

template <class... Mixins>
Tensor<Mixins...> DynamicUpdateSlice(
    Tensor<Mixins...> operand, Tensor<Mixins...> update,
    Tensor<Mixins...> start_indices,
    source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::DynamicUpdateSliceOperation<Mixins...>>();
  AddInputs(op, operand, update, start_indices);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& operand_info = *GetInfo(operand.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = operand_info.shape;
  output_info.type = operand_info.type;

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> DynamicUpdateSlice(
    Tensor<Mixins...> operand, Tensor<Mixins...> update,
    const std::vector<int>& start_indices,
    source_location loc = source_location::current()) {
  Tensor<Mixins...> start_indices_tensor(
      {.type = Type::kI32,
       .shape = {static_cast<int>(start_indices.size())},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>(start_indices)});
  return DynamicUpdateSlice(operand, update, start_indices_tensor, loc);
}

template <class... Mixins>
std::vector<Tensor<Mixins...>> Custom(
    absl::Span<Tensor<Mixins...>> inputs, std::string custom_code,
    std::vector<uint8_t> custom_options,
    const std::vector<std::vector<int>>& output_shapes,
    const std::vector<Type>& output_types,
    source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::CustomOperation<Mixins...>>();
  op->custom_code = std::move(custom_code);
  op->custom_options = std::move(custom_options);
  for (auto& input : inputs) {
    AddInputs(op, input);
  }

  std::vector<Tensor<Mixins...>> outputs;
  outputs.reserve(output_shapes.size());
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    Tensor<Mixins...> output = AddOutput(op, loc);
    graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
    output_info.shape = output_shapes[i];
    output_info.type = output_types[i];
    outputs.push_back(output);
  }
  graph::OpDebugger::DebugOp(*op);
  return outputs;
}

template <class... Mixins>
std::vector<Tensor<Mixins...>> Custom(
    std::initializer_list<Tensor<Mixins...>> inputs, std::string custom_code,
    std::vector<uint8_t> custom_options,
    const std::vector<std::vector<int>>& output_shapes,
    const std::vector<Type>& output_types,
    source_location loc = source_location::current()) {
  std::vector input_storage(inputs);
  return Custom(absl::MakeSpan(input_storage), std::move(custom_code),
                std::move(custom_options), output_shapes, output_types, loc);
}

template <class... Mixins>
std::vector<Tensor<Mixins...>> TopK(
    Tensor<Mixins...> input, Tensor<Mixins...> k,
    source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::TopKOperation<Mixins...>>();
  AddInputs(op, input, k);

  std::vector<Tensor<Mixins...>> outputs;
  outputs.reserve(2);

  Tensor<Mixins...> values = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& values_info = *GetInfo(values.GetRaw());
  values_info.type = input_info.type;
  values_info.shape = input_info.shape;
  values_info.shape.back() = GetInfo(k.GetRaw())
                                 ->buffer->Lock()
                                 .template As<const int32_t>()
                                 .data()[0];
  outputs.push_back(values);

  Tensor<Mixins...> indices = AddOutput(op, loc);
  graph::TensorInformation& indices_info = *GetInfo(indices.GetRaw());
  indices_info.type = Type::kI32;
  indices_info.shape = input_info.shape;
  indices_info.shape.back() = GetInfo(k.GetRaw())
                                  ->buffer->Lock()
                                  .template As<const int32_t>()
                                  .data()[0];
  outputs.push_back(indices);

  graph::OpDebugger::DebugOp(*op);
  return outputs;
}

template <class... Mixins>
std::vector<Tensor<Mixins...>> TopK(
    Tensor<Mixins...> input, int k,
    source_location loc = source_location::current()) {
  Tensor<Mixins...> k_tensor(
      {.type = Type::kI32,
       .shape = {},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>({k})});
  return TopK(input, k_tensor, loc);
}

template <class... Mixins>
Tensor<Mixins...> Cumsum(Tensor<Mixins...> input, Tensor<Mixins...> axis,
                         bool exclusive = false, bool reverse = false,
                         source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::CumsumOperation<Mixins...>>();
  op->exclusive = exclusive;
  op->reverse = reverse;
  AddInputs(op, input, axis);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = input_info.shape;
  output_info.type = input_info.type;

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Cumsum(Tensor<Mixins...> input, int axis,
                         bool exclusive = false, bool reverse = false,
                         source_location loc = source_location::current()) {
  Tensor<Mixins...> axis_tensor(
      {.type = Type::kI32,
       .shape = {},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>({axis})});
  return Cumsum(input, axis_tensor, exclusive, reverse, loc);
}

template <class... Mixins>
Tensor<Mixins...> Reverse(Tensor<Mixins...> input, Tensor<Mixins...> axes,
                          source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::ReverseOperation<Mixins...>>();
  AddInputs(op, input, axes);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = input_info.shape;
  output_info.type = input_info.type;

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Gather(Tensor<Mixins...> input, Tensor<Mixins...> indices,
                         int axis, int batch_dims = 0,
                         source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::GatherOperation<Mixins...>>();
  op->axis = axis;
  op->batch_dims = batch_dims;
  AddInputs(op, input, indices);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  const graph::TensorInformation& indices_info = *GetInfo(indices.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.type = input_info.type;
  output_info.shape.clear();
  int i = 0;
  for (; i < axis; ++i) {
    output_info.shape.push_back(input_info.shape[i]);
  }
  output_info.shape.insert(output_info.shape.end(),
                           indices_info.shape.begin() + batch_dims,
                           indices_info.shape.end());
  for (i = axis + 1; i < input_info.shape.size(); ++i) {
    output_info.shape.push_back(input_info.shape[i]);
  }

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> OneHot(Tensor<Mixins...> indices, Tensor<Mixins...> depth,
                         Tensor<Mixins...> on_value,
                         Tensor<Mixins...> off_value, int axis = -1,
                         source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::OneHotOperation<Mixins...>>();
  op->axis = axis;
  AddInputs(op, indices, depth, on_value, off_value);
  Tensor<Mixins...> output = AddOutput(op, loc);

  const graph::TensorInformation& indices_info = *GetInfo(indices.GetRaw());
  const graph::TensorInformation& depth_info = *GetInfo(depth.GetRaw());
  const graph::TensorInformation& on_value_info = *GetInfo(on_value.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());

  output_info.type = on_value_info.type;

  int resolved_axis = axis;
  if (resolved_axis < 0) {
    resolved_axis += indices_info.shape.size() + 1;
  }

  output_info.shape = indices_info.shape;
  int depth_val = -1;
  if (depth_info.buffer) {
    depth_val =
        depth_info.buffer->Lock().template As<const int32_t>().data()[0];
  }
  output_info.shape.insert(output_info.shape.begin() + resolved_axis,
                           depth_val);

  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> GatherNd(Tensor<Mixins...> input, Tensor<Mixins...> indices,
                           source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::GatherNdOperation<Mixins...>>();
  AddInputs(op, input, indices);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  const graph::TensorInformation& indices_info = *GetInfo(indices.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());

  output_info.type = input_info.type;
  output_info.shape.clear();

  int indices_ndims = indices_info.shape.size();
  int input_ndims = input_info.shape.size();
  int index_depth = indices_info.shape.back();
  int outer_dims = indices_ndims - 1;

  for (int i = 0; i < outer_dims; ++i) {
    output_info.shape.push_back(indices_info.shape[i]);
  }
  for (int i = index_depth; i < input_ndims; ++i) {
    output_info.shape.push_back(input_info.shape[i]);
  }

  graph::OpDebugger::DebugOp(*op);
  return output;
}

// Performs an element-wise `Quantize` operation.
template <class... Mixins>
Tensor<Mixins...> Quantize(Tensor<Mixins...> a, Type type,
                           std::vector<float> scale,
                           std::vector<int64_t> zero_point,
                           source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::QuantizeOperation<Mixins...>>();
  auto status = CheckUnaryElementwiseOp(a.GetRaw());
  if (!status.ok()) {
    return Tensor<Mixins...>(graph::ErrorTensor(status, loc));
  }
  op->inputs = {a.GetRaw()};
  auto out_group = graph::NewTensorGroup(1, loc);
  op->outputs_group = out_group;
  out_group->producer = std::move(op);
  auto& out_info = out_group->tensor_infos[0];
  auto& a_info = graph::GetInfo(a.GetRaw()).value();
  out_info = a_info;
  out_info.type = type;
  auto quantization = std::make_shared<graph::PerChannelAffineQuantization>();
  quantization->scales = scale;
  quantization->zero_points = zero_point;
  quantization->quantized_dimension = 0;
  out_info.quantization = quantization;

  graph::OpDebugger::DebugOp(*out_group->producer);
  a_info.consumers.push_back(out_group->producer);
  Tensor<Mixins...> output(graph::GetTensor(0, out_group));
  return output;
}

// Performs an element-wise `Dequantize` operation.
template <class... Mixins>
Tensor<Mixins...> Dequantize(Tensor<Mixins...> a,
                             source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::DequantizeOperation<Mixins...>>();
  auto status = CheckUnaryElementwiseOp(a.GetRaw());
  if (!status.ok()) {
    return Tensor<Mixins...>(graph::ErrorTensor(status, loc));
  }
  op->inputs = {a.GetRaw()};
  auto out_group = graph::NewTensorGroup(1, loc);
  op->outputs_group = out_group;
  out_group->producer = std::move(op);
  auto& out_info = out_group->tensor_infos[0];
  auto& a_info = graph::GetInfo(a.GetRaw()).value();
  out_info = a_info;
  out_info.type = Type::kFP32;

  graph::OpDebugger::DebugOp(*out_group->producer);
  a_info.consumers.push_back(out_group->producer);
  Tensor<Mixins...> output(graph::GetTensor(0, out_group));
  return output;
}

template <class... Mixins>
Tensor<Mixins...> Probe(Tensor<Mixins...> a,
                        source_location loc = source_location::current()) {
  Tensor<Mixins...> output =
      ElementwiseOp<graph::ProbeOperation<Mixins...>>(loc, a);
  if (auto producer = graph::GetProducer(output.GetRaw());
      producer.ok() && *producer) {
    graph::OpDebugger::DebugOp(**producer);
  }
  return output;
}

template <class... Mixins>
Tensor<Mixins...> ResizeBilinear(
    Tensor<Mixins...> input, Tensor<Mixins...> size, bool align_corners = false,
    bool half_pixel_centers = false,
    source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::ResizeBilinearOperation<Mixins...>>();
  op->align_corners = align_corners;
  op->half_pixel_centers = half_pixel_centers;
  AddInputs(op, input, size);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  const graph::TensorInformation& size_info = *GetInfo(size.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.type = input_info.type;
  output_info.shape = input_info.shape;
  if (size_info.buffer) {
    const auto size_data = size_info.buffer->Lock().As<const int32_t>();
    if (output_info.shape.size() == 4 && size_data.size() == 2) {
      output_info.shape[1] = size_data.data()[0];
      output_info.shape[2] = size_data.data()[1];
    } else {
      output_info.shape = std::vector<int>(size_data.begin(), size_data.end());
    }
  }
  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> ResizeBilinear(
    Tensor<Mixins...> input, const std::vector<int>& size,
    bool align_corners = false, bool half_pixel_centers = false,
    source_location loc = source_location::current()) {
  Tensor<Mixins...> size_tensor(
      {.type = Type::kI32,
       .shape = {static_cast<int>(size.size())},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>(size)});
  return ResizeBilinear(input, size_tensor, align_corners, half_pixel_centers,
                        loc);
}

template <class... Mixins>
Tensor<Mixins...> ResizeNearestNeighbor(
    Tensor<Mixins...> input, Tensor<Mixins...> size, bool align_corners = false,
    bool half_pixel_centers = false,
    source_location loc = source_location::current()) {
  auto op =
      std::make_shared<graph::ResizeNearestNeighborOperation<Mixins...>>();
  op->align_corners = align_corners;
  op->half_pixel_centers = half_pixel_centers;
  AddInputs(op, input, size);
  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  const graph::TensorInformation& size_info = *GetInfo(size.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.type = input_info.type;
  output_info.shape = input_info.shape;
  if (size_info.buffer) {
    const auto size_data = size_info.buffer->Lock().As<const int32_t>();
    if (output_info.shape.size() == 4 && size_data.size() == 2) {
      output_info.shape[1] = size_data.data()[0];
      output_info.shape[2] = size_data.data()[1];
    }
  }
  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> ResizeNearestNeighbor(
    Tensor<Mixins...> input, const std::vector<int>& size,
    bool align_corners = false, bool half_pixel_centers = false,
    source_location loc = source_location::current()) {
  Tensor<Mixins...> size_tensor(
      {.type = Type::kI32,
       .shape = {static_cast<int>(size.size())},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>(size)});
  return ResizeNearestNeighbor(input, size_tensor, align_corners,
                               half_pixel_centers, loc);
}

template <class... Mixins>
std::vector<Tensor<Mixins...>> NonMaxSuppressionV5(
    Tensor<Mixins...> boxes, Tensor<Mixins...> scores,
    Tensor<Mixins...> max_output_size, Tensor<Mixins...> iou_threshold,
    Tensor<Mixins...> score_threshold, Tensor<Mixins...> soft_nms_sigma,
    source_location loc = source_location::current()) {
  // max_output_size must be a constant tensor to ensure static output shapes.
  if (graph::GetInfo(max_output_size.GetRaw())->buffer == nullptr) {
    auto error = absl::InvalidArgumentError(
        "max_output_size must be a constant tensor.");
    return {Tensor<Mixins...>(graph::ErrorTensor(error)),
            Tensor<Mixins...>(graph::ErrorTensor(error)),
            Tensor<Mixins...>(graph::ErrorTensor(error))};
  }

  auto op = std::make_shared<graph::NonMaxSuppressionV5Operation<Mixins...>>();
  AddInputs(op, boxes, scores, max_output_size, iou_threshold, score_threshold,
            soft_nms_sigma);

  auto out_group = graph::NewTensorGroup(3, loc);
  op->outputs_group = out_group;
  out_group->producer = op;

  std::vector<Tensor<Mixins...>> outputs;
  outputs.reserve(3);

  // Output 0: selected_indices (Int32)
  auto& indices_info = out_group->tensor_infos[0];
  indices_info.type = Type::kI32;

  // Output 1: selected_scores (Float32)
  auto& scores_info = out_group->tensor_infos[1];
  scores_info.type = Type::kFP32;

  // Infer shape from max_output_size.
  if (auto* max_output_size_buffer =
          graph::GetInfo(max_output_size.GetRaw())->buffer.get()) {
    const int max_output_size_val =
        *max_output_size_buffer->Lock().template As<const int32_t>().data();
    indices_info.shape = {max_output_size_val};
    scores_info.shape = {max_output_size_val};
  }

  // Output 2: valid_outputs (Int32)
  auto& valid_outputs_info = out_group->tensor_infos[2];
  valid_outputs_info.type = Type::kI32;
  valid_outputs_info.shape = {};

  for (int i = 0; i < 3; ++i) {
    outputs.emplace_back(graph::GetTensor(i, out_group));
  }

  graph::OpDebugger::DebugOp(*op);
  return outputs;
}

template <class... Mixins>
std::vector<Tensor<Mixins...>> NonMaxSuppressionV5(
    Tensor<Mixins...> boxes, Tensor<Mixins...> scores, int max_output_size,
    Tensor<Mixins...> iou_threshold, Tensor<Mixins...> score_threshold,
    Tensor<Mixins...> soft_nms_sigma,
    source_location loc = source_location::current()) {
  Tensor<Mixins...> max_output_size_tensor(
      {.type = Type::kI32,
       .shape = {},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>({max_output_size})});
  return NonMaxSuppressionV5(boxes, scores, max_output_size_tensor,
                             iou_threshold, score_threshold, soft_nms_sigma,
                             loc);
}

template <class... Mixins>
std::vector<Tensor<Mixins...>> Lstm(
    Tensor<Mixins...> intermediate, Tensor<Mixins...> prev_state,
    source_location loc = source_location::current()) {
  auto op = std::make_shared<graph::LstmOperation<Mixins...>>();
  AddInputs(op, intermediate, prev_state);

  auto out_group = graph::NewTensorGroup(2, loc);
  op->outputs_group = out_group;
  out_group->producer = op;

  std::vector<Tensor<Mixins...>> outputs;
  outputs.reserve(2);

  const graph::TensorInformation& prev_state_info =
      *GetInfo(prev_state.GetRaw());

  // Output 0: new_state
  auto& new_state_info = out_group->tensor_infos[0];
  new_state_info.type = prev_state_info.type;
  new_state_info.shape = prev_state_info.shape;
  outputs.emplace_back(graph::GetTensor(0, out_group));

  // Output 1: activation
  auto& activation_info = out_group->tensor_infos[1];
  activation_info.type = prev_state_info.type;
  activation_info.shape = prev_state_info.shape;
  outputs.emplace_back(graph::GetTensor(1, out_group));

  graph::OpDebugger::DebugOp(*op);
  return outputs;
}

template <class... Mixins>
Tensor<Mixins...> TransposeConv(
    Tensor<Mixins...> filter, Tensor<Mixins...> input, Tensor<Mixins...> bias,
    const std::vector<int>& output_shape, Padding padding, int stride_h,
    int stride_w, FusedActivation activation = kActNone,
    source_location loc = source_location::current()) {
  Tensor<Mixins...> output_shape_tensor(
      {.type = Type::kI32,
       .shape = {static_cast<int>(output_shape.size())},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>(output_shape)});

  auto op = std::make_shared<graph::TransposeConvOperation<Mixins...>>();
  op->padding = padding;
  op->stride_h = stride_h;
  op->stride_w = stride_w;

  // TFLite TransposeConv inputs: output_shape, weights, input
  AddInputs(op, output_shape_tensor, filter, input);

  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.type = input_info.type;
  output_info.shape = output_shape;

  graph::OpDebugger::DebugOp(*op);
  // Handle Bias (Add)
  output = Add(output, bias, activation, loc);

  return output;
}

template <class... Mixins>
Tensor<Mixins...> TransposeConv2D(
    Tensor<Mixins...> filter, Tensor<Mixins...> input, Tensor<Mixins...> bias,
    const std::vector<int>& output_shape, Padding padding, int stride_h,
    int stride_w, FusedActivation activation = kActNone,
    source_location loc = source_location::current()) {
  Tensor<Mixins...> output_shape_tensor(
      {.type = Type::kI32,
       .shape = {static_cast<int>(output_shape.size())},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>(output_shape)});

  auto op = std::make_shared<graph::TransposeConv2DOperation<Mixins...>>();
  op->padding = padding;
  op->stride_h = stride_h;
  op->stride_w = stride_w;

  AddInputs(op, output_shape_tensor, filter, input);

  Tensor<Mixins...> output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.type = input_info.type;
  output_info.shape = output_shape;

  graph::OpDebugger::DebugOp(*op);
  // Handle Bias (Add)
  output = Add(output, bias, activation, loc);

  return output;
}

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_ARITHMETIC_H_

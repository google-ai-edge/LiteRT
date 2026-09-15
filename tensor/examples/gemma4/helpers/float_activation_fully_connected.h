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

#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_HELPERS_FLOAT_ACTIVATION_FULLY_CONNECTED_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_HELPERS_FLOAT_ACTIVATION_FULLY_CONNECTED_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/internal/arithmetic_helpers.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/mixin.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/source_location.h"

namespace litert::tensor::graph {

struct FloatActivationFullyConnectedOperation : Operation {
  absl::string_view GetName() const override {
    return "FloatActivationFullyConnected";
  }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

template <>
class OpMixin<FloatActivationFullyConnectedOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const Operation& op,
                         XnnpackBuildContext& ctx) const override;
};

}  // namespace litert::tensor::graph

namespace litert::tensor::examples::gemma4 {

absl::Status ValidateFloatActivationFullyConnected(const TensorHandle& input,
                                                   const TensorHandle& weights);

// Applies bias-free FC with FP32 activations and symmetric channelwise INT4 or
// INT8 constant weights. Input/output leading dimensions are preserved. Unlike
// the core XNNPACK FC lowering, this operation never quantizes the input.
// Weights require one positive scale per output channel, axis 0, and zero point
// 0. At least two output channels are required by the current value conversion.
// INT4 input channels must be even for portable XNNPACK kernels. Each compiled
// graph retains a private packed INT4 copy in the kernel's offset-binary
// format. Both weight types require a readable constant buffer and a locked
// span large enough for the complete logical matrix.
template <class... Mixins>
Tensor<Mixins...> FloatActivationFullyConnected(
    Tensor<Mixins...> input, Tensor<Mixins...> weights,
    source_location loc = source_location::current()) {
  LRT_TENSOR_RETURN_IF_ERROR(
      ValidateFloatActivationFullyConnected(input, weights));
  auto op = std::make_shared<graph::FloatActivationFullyConnectedOperation>();
  RegisterMixins<Mixins...>(op);
  AddInputs(op, input, weights);
  Tensor<Mixins...> output = AddOutput(op, loc);
  LRT_TENSOR_ASSIGN_OR_RETURN(auto& output_info,
                              graph::GetInfo(output.GetRaw()));
  output_info.shape = input.GetShape();
  output_info.shape.back() = weights.GetShape()[0];
  output_info.type = Type::kFP32;
  graph::OpDebugger::DebugOp(*op);
  return output;
}

}  // namespace litert::tensor::examples::gemma4

#endif  // LITERT_TENSOR_EXAMPLES_GEMMA4_HELPERS_FLOAT_ACTIVATION_FULLY_CONNECTED_H_

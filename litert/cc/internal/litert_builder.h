// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_BUILDER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_BUILDER_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_builder.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_ranked_tensor_type.h"

/// @file
/// @brief Defines the C++ equivalent of `LiteRtBuilder` for model
/// modification.

namespace litert {

/// @brief Specification for a ranked tensor.
/// @todo Reuse the logic for generating `RankedTensorType` in testing.
struct RankedTensorSpec {
  RankedTensorType ranked_tensor_type;
  std::optional<Weights> weights;
  std::optional<LiteRtQuantizationPerTensor> per_tensor_quantization;
  std::optional<LiteRtQuantizationPerChannel> per_channel_quantization;
  std::optional<std::string> tensor_name;
};

/// @brief A builder for `RankedTensorSpec`.
///
/// This class is necessary as LiteRT is pinned to C++17, which does not
/// support designated initializers for structs.
class RankedTensorSpecBuilder {
 public:
  explicit RankedTensorSpecBuilder(RankedTensorType type) {
    ranked_tensor_type_ = std::move(type);
  };

  RankedTensorSpecBuilder&& WithWeights(Weights w) && {
    weights_ = std::move(w);
    return std::move(*this);
  }

  RankedTensorSpecBuilder&& WithPerTensorQuantization(
      LiteRtQuantizationPerTensor q) && {
    per_tensor_quantization_ = std::move(q);
    return std::move(*this);
  }

  RankedTensorSpecBuilder&& WithPerChannelQuantization(
      LiteRtQuantizationPerChannel q) && {
    per_channel_quantization_ = std::move(q);
    return std::move(*this);
  }

  RankedTensorSpecBuilder&& WithTensorName(std::string name) && {
    tensor_name_ = std::move(name);
    return std::move(*this);
  }

  RankedTensorSpec Build() && {
    return RankedTensorSpec{
        *std::move(ranked_tensor_type_), std::move(weights_),
        std::move(per_tensor_quantization_),
        std::move(per_channel_quantization_), std::move(tensor_name_)};
  }

 private:
  std::optional<RankedTensorType> ranked_tensor_type_;
  std::optional<Weights> weights_;
  std::optional<LiteRtQuantizationPerTensor> per_tensor_quantization_;
  std::optional<LiteRtQuantizationPerChannel> per_channel_quantization_;
  std::optional<std::string> tensor_name_;
};

/// @brief Helper to create a RankedTensorSpec with a specific element type.
template <typename T>
RankedTensorSpec TensorType(const std::vector<int32_t>& dims) {
  return RankedTensorSpecBuilder(
             RankedTensorType(
                 GetElementType<T>(),
                 Layout(BuildLayout(dims.data(), dims.data() + dims.size()))))
      .Build();
}

class Builder : public internal::NonOwnedHandle<LiteRtBuilder> {
 public:
  explicit Builder(LiteRtBuilder builder)
      : internal::NonOwnedHandle<LiteRtBuilder>(builder) {}
  /// @brief Builds a tensor from a `RankedTensorSpec`.
  Expected<Tensor> BuildTensor(const RankedTensorSpec& spec) const {
    LiteRtTensor tensor;
    LiteRtRankedTensorType ranked_tensor_type_litert =
        static_cast<LiteRtRankedTensorType>(spec.ranked_tensor_type);

    LiteRtWeights litert_weights;
    if (spec.weights.has_value()) {
      litert_weights = spec.weights->Get();
    } else {
      litert_weights = nullptr;
    }

    LiteRtQuantizationTypeId quantization_type_id = kLiteRtQuantizationNone;
    LiteRtQuantizationPerTensor litert_per_tensor_quantization;
    if (spec.per_tensor_quantization.has_value()) {
      litert_per_tensor_quantization = *spec.per_tensor_quantization;
      quantization_type_id = kLiteRtQuantizationPerTensor;
    }
    LiteRtQuantizationPerChannel litert_per_channel_quantization;
    if (spec.per_channel_quantization.has_value()) {
      litert_per_channel_quantization = *spec.per_channel_quantization;
      quantization_type_id = kLiteRtQuantizationPerChannel;
    }
    const char* name = spec.tensor_name ? spec.tensor_name->c_str() : nullptr;
    internal::AssertOk(LiteRtBuilderBuildTensor, Get(), kLiteRtRankedTensorType,
                       ranked_tensor_type_litert, LiteRtUnrankedTensorType(),
                       litert_weights, quantization_type_id,
                       litert_per_tensor_quantization,
                       litert_per_channel_quantization, name, &tensor);
    return Tensor(tensor);
  }

  /// @brief Builds a tensor similar to the given tensor.
  Expected<Tensor> CloneTensor(const Tensor& src) const {
    if (src.TypeId() != kLiteRtRankedTensorType) {
      return Unexpected(Status::kErrorUnsupported);
    }
    auto ranked_type = src.RankedTensorType();
    if (!ranked_type) {
      return ranked_type.Error();
    }

    RankedTensorSpecBuilder spec_builder(*ranked_type);

    if (src.HasWeights()) {
      spec_builder = std::move(spec_builder).WithWeights(src.Weights());
    }

    if (src.HasQuantization()) {
      auto q_type = src.QTypeId();
      if (q_type == kLiteRtQuantizationPerTensor) {
        spec_builder =
            std::move(spec_builder)
                .WithPerTensorQuantization(src.PerTensorQuantization());
      } else if (q_type == kLiteRtQuantizationPerChannel) {
        spec_builder =
            std::move(spec_builder)
                .WithPerChannelQuantization(src.PerChannelQuantization());
      }
    }

    return BuildTensor(std::move(spec_builder)
                           .WithTensorName(std::string(src.Name()))
                           .Build());
  }

  /// @brief Builds weights for a tensor.
  template <typename T>
  Expected<Weights> BuildWeights(absl::Span<const T> data,
                                 Tensor& tensor) const {
    const uint8_t* data_uint8 = reinterpret_cast<const uint8_t*>(data.data());
    size_t size_uint8 = data.size() * sizeof(T);
    LiteRtWeights weights;
    internal::AssertOk(LiteRtBuilderBuildWeights, this->Get(), data_uint8,
                       size_uint8, tensor.Get(), &weights);
    return Weights(weights);
  }

  /// @brief A trait for building scalars.
  Expected<Tensor> BuildScalar(
      LiteRtElementType element_type,
      std::optional<std::string> name = std::nullopt) const {
    LiteRtTensor tensor;
    LiteRtUnrankedTensorType unranked_tensor_type;
    unranked_tensor_type.element_type = element_type;
    const char* name_ptr = name ? name->c_str() : nullptr;
    internal::AssertOk(LiteRtBuilderBuildTensor, Get(),
                       kLiteRtUnrankedTensorType, LiteRtRankedTensorType(),
                       unranked_tensor_type, LiteRtWeights(),
                       kLiteRtQuantizationNone, LiteRtQuantizationPerTensor(),
                       LiteRtQuantizationPerChannel(), name_ptr, &tensor);
    return Tensor(tensor);
  }

  Op BuildOp(LiteRtOpCode op_code, const std::vector<Tensor>& inputs,
             const std::vector<Tensor>& outputs) const {
    LiteRtOp litert_op;
    std::vector<LiteRtTensor> input_tensors;
    input_tensors.reserve(inputs.size());
    for (const auto& input : inputs) {
      input_tensors.push_back(input.Get());
    }
    std::vector<LiteRtTensor> output_tensors;
    output_tensors.reserve(outputs.size());
    for (const auto& output : outputs) {
      output_tensors.push_back(output.Get());
    }
    internal::AssertOk(LiteRtBuilderBuildOp, Get(), op_code,
                       input_tensors.size(), input_tensors.data(),
                       output_tensors.size(), output_tensors.data(),
                       &litert_op);
    return Op(litert_op);
  }

  /// @brief Clones the given op.
  Op BuildOp(Op& src, const std::vector<Tensor>& inputs,
             const std::vector<Tensor>& outputs) {
    return BuildOp(src.Code(), inputs, outputs);
  };

  /// @brief Sets the op options for the given op.
  ///
  /// The options must be a subclass of `OpOptions`; otherwise, an error is
  /// returned.
  template <typename T>
  Expected<void> SetOpOptions(Op& op, T&& options) const {
    if constexpr (!std::is_base_of_v<OpOptions, T>) {
      return Unexpected(Status::kErrorInvalidArgument);
    }
    options.op = op.Get();
    options.SetOpOptions(this->Get());

    return Expected<void>();
  }

  /// @brief Records the op to be erased.
  void EraseOp(Op& op) const {
    internal::AssertOk(LiteRtBuilderEraseOp, this->Get(), op.Get());
  }

  // --- Extended API ---

  /// @brief Create an Op with the given code, inputs, and output types.
  /// Returns all output tensors.
  Expected<std::vector<Tensor>> CreateOpWithOutputSpec(
      LiteRtOpCode code, const std::vector<Tensor>& inputs,
      const std::vector<RankedTensorSpec>& output_specs) const {
    std::vector<Tensor> outputs;
    outputs.reserve(output_specs.size());

    std::vector<Tensor> results;
    results.reserve(output_specs.size());

    for (const auto& spec : output_specs) {
      auto t_res = BuildTensor(spec);
      if (!t_res) {
        return t_res.Error();
      }
      outputs.push_back(Tensor(*t_res));
      results.push_back(std::move(*t_res));
    }

    BuildOp(code, inputs, outputs);
    return results;
  }

  /// @brief Overload for single-output Ops.
  Expected<Tensor> CreateOpWithOutputSpec(LiteRtOpCode code,
                                          const std::vector<Tensor>& inputs,
                                          RankedTensorSpec output_spec) const {
    std::vector<RankedTensorSpec> specs;
    specs.push_back(std::move(output_spec));
    auto results = CreateOpWithOutputSpec(code, inputs, specs);
    if (!results) {
      return results.Error();
    }
    return std::move((*results)[0]);
  }

  /// @brief Replaces the given op with a new op.
  /// The outputs of the old op are reused for the new op.
  /// The old op is erased.
  Op ReplaceOp(Op& op, LiteRtOpCode new_code,
               const std::vector<Tensor>& inputs) const {
    OpOutputs op_outputs = op.Outputs();
    std::vector<Tensor> outputs;
    outputs.reserve(op_outputs.size());
    for (const auto& out : op_outputs) {
      outputs.push_back(Tensor(out));
    }
    Op new_op = BuildOp(new_code, inputs, outputs);
    EraseOp(op);
    return new_op;
  }
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_BUILDER_H_

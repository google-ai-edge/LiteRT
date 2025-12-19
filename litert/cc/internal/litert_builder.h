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

// Model Builder. C++ equivalent of LiteRtBuilder.
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_builder.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_ranked_tensor_type.h"

namespace litert {

// TODO(yunandrew): Reuse the logic for generating RankedTensorType in testing.
struct RankedTensorSpec {
  RankedTensorType ranked_tensor_type;
  std::optional<Weights> weights;
  std::optional<LiteRtQuantizationPerTensor> per_tensor_quantization;
  std::optional<LiteRtQuantizationPerChannel> per_channel_quantization;
  std::optional<std::string> tensor_name;
};

// Builder for RankedTensorSpec. We need this class since LiteRT is pinned with
// c++ 17, a version before designated initializers of structs were introduced.
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

class Builder : public internal::NonOwnedHandle<LiteRtBuilder> {
 public:
  explicit Builder(LiteRtBuilder builder)
      : internal::NonOwnedHandle<LiteRtBuilder>(builder) {}
  // For ranked tensors.
  Expected<Tensor> BuildTensor(const RankedTensorSpec& spec) const;

  // Build weights for a tensor.
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

  // Trait for building scalars.
  Expected<Tensor> BuildScalar(
      LiteRtElementType element_type,
      std::optional<std::string> name = std::nullopt) const;

  Op BuildOp(LiteRtOpCode op_code, OpInputs& inputs, OpOutputs& outputs) const;
  // Clone the given op.
  Op BuildOp(Op& src, OpInputs& inputs, OpOutputs& outputs) {
    return BuildOp(src.Code(), inputs, outputs);
  };

  // Set the op options for the given op.
  // The options must be a subclass of OpOptions, will return error otherwise.
  template <typename T>
  Expected<void> SetOpOptions(Op& op, T&& options) const {
    if constexpr (!std::is_base_of_v<OpOptions, T>) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument);
    }
    options.op = op.Get();
    options.SetOpOptions(this->Get());

    return Expected<void>();
  }

  // Record the op to be erased.
  void EraseOp(Op& op) const {
    internal::AssertOk(LiteRtBuilderEraseOp, this->Get(), op.Get());
  }
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_BUILDER_H_

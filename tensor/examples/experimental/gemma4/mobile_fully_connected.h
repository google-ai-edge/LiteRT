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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_MOBILE_FULLY_CONNECTED_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_MOBILE_FULLY_CONNECTED_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::examples::experimental::gemma4::cpu {

// Lightweight lookup helper for preloaded weight maps (used instead of the
// safetensor-backed `TensorMapping` interface because `.litertlm` bundles
// supply all tensors upfront in a flat map). Returns `weights[name]` verbatim
// (preserving quantized types and quantization parameters), or synthesizes an
// external placeholder tensor with `placeholder_type` and `placeholder_shape`.
template <class... Mixins>
Tensor<Mixins...> GetWeightOrPlaceholder(
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights,
    absl::string_view name, Type placeholder_type,
    absl::Span<const int> placeholder_shape) {
  const auto it = weights.find(name);
  if (it != weights.end()) {
    return it->second;
  }
  return Tensor<Mixins...>(
      {.name = std::string(name),
       .type = placeholder_type,
       .shape = std::vector<int>(placeholder_shape.begin(),
                                 placeholder_shape.end())});
}

// Applies RMSNorm along the last axis of `x` using primitive arithmetic ops to
// match the CPU reference graphs. Pass `TensorHandle::Invalid()` for `scale`
// to omit the learned scale multiply; non-OK error tensors on `x`, `epsilon`,
// or a non-Invalid `scale` are propagated rather than silently ignored.
template <class... Mixins>
Tensor<Mixins...> PrimitiveRmsNorm(const Tensor<Mixins...>& x,
                                   const Tensor<Mixins...>& scale,
                                   const Tensor<Mixins...>& epsilon) {
  LRT_TENSOR_RETURN_IF_ERROR(x);
  LRT_TENSOR_RETURN_IF_ERROR(epsilon);
  if (scale != TensorHandle::Invalid()) {
    LRT_TENSOR_RETURN_IF_ERROR(scale);
  }
  if (x.GetShape().empty()) {
    return absl::InvalidArgumentError(
        "PrimitiveRmsNorm expects input with rank >= 1, got scalar.");
  }
  Tensor<Mixins...> mean =
      Mean(Square(x), {static_cast<int>(x.GetShape().size()) - 1},
           /*keep_dims=*/true);
  Tensor<Mixins...> result = Mul(x, Rsqrt(Add(mean, epsilon)));
  return scale != TensorHandle::Invalid() ? Mul(result, scale) : result;
}

// Reads a single finite, positive FP32 activation scale from a scalar (rank 0
// or shape `{1}`) constant tensor.
inline absl::StatusOr<float> ReadMobileActivationScale(
    const TensorHandle& tensor) {
  LRT_TENSOR_ASSIGN_OR_RETURN(Buffer & buffer, tensor.GetBuffer());
  LRT_TENSOR_ASSIGN_OR_RETURN(size_t size, buffer.ByteSize());
  if (tensor.GetType() != Type::kFP32 || size != sizeof(float) ||
      !(tensor.GetShape().empty() || tensor.GetShape() == Shape{1})) {
    return absl::InvalidArgumentError(
        "Mobile activation scale must be one FP32 scalar");
  }
  LockedBufferSpan<const std::byte> lock = buffer.Lock();
  if (lock.data() == nullptr || lock.size() != sizeof(float)) {
    return absl::FailedPreconditionError(
        "Could not lock mobile activation scale");
  }
  float scale;
  std::memcpy(&scale, lock.data(), sizeof(scale));
  if (!std::isfinite(scale) || scale <= 0) {
    return absl::InvalidArgumentError(
        "Mobile activation scale must be finite and positive");
  }
  return scale;
}

// Google mobile-CT linear layers have symmetric, static per-tensor INT8 input
// and output scales. Attach those scales to real quantized graph values, so
// the delegate performs quantize -> integer FC -> requantize -> dequantize.
// Layers without activation scales (including the head) retain FP32
// activations. This avoids adding quantization that is absent from the
// checkpoint scheme.
template <class... Mixins>
Tensor<Mixins...> MobileFullyConnected(
    const Tensor<Mixins...>& input, const Tensor<Mixins...>& weight,
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>* absl_nullable
        weights = nullptr) {
  LRT_TENSOR_RETURN_IF_ERROR(input);
  LRT_TENSOR_RETURN_IF_ERROR(weight);
  const auto unquantized_fc = [&]() -> Tensor<Mixins...> {
    if (weight.GetType() == Type::kI2) {
      return absl::InvalidArgumentError(
          "Original static INT2 weights require original activation scales");
    }
    return FullyConnected(input, weight);
  };
  if (weights == nullptr) {
    return unquantized_fc();
  }
  absl::string_view module = weight.GetName();
  if (!absl::ConsumeSuffix(&module, ".weight")) {
    return unquantized_fc();
  }
  const auto in = weights->find(absl::StrCat(module, ".input_scale"));
  const auto out = weights->find(absl::StrCat(module, ".output_scale"));
  if (in == weights->end() && out == weights->end()) {
    return unquantized_fc();
  }
  if (in == weights->end() || out == weights->end()) {
    return absl::InvalidArgumentError(
        absl::StrCat(module, ": input/output scales must both be present"));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(float input_scale,
                              ReadMobileActivationScale(in->second));
  LRT_TENSOR_ASSIGN_OR_RETURN(float output_scale,
                              ReadMobileActivationScale(out->second));
  if (input.GetType() != Type::kFP32 ||
      (weight.GetType() != Type::kI2 && weight.GetType() != Type::kI4 &&
       weight.GetType() != Type::kI8) ||
      !weight.GetQuantization()) {
    return absl::InvalidArgumentError(
        "Mobile static FC requires FP32 input and "
        "quantized INT2/INT4/INT8 weights");
  }
  Tensor<Mixins...> quantized_input =
      Quantize(input, Type::kI8, {input_scale}, {0});
  LRT_TENSOR_RETURN_IF_ERROR(quantized_input);
  Tensor<Mixins...> quantized_output = FullyConnected(quantized_input, weight);
  LRT_TENSOR_RETURN_IF_ERROR(quantized_output);
  quantized_output.SetQuantization(
      std::make_shared<PerChannelAffineQuantization>(
          std::vector<float>{output_scale}, std::vector<int64_t>{0}));
  return Dequantize(quantized_output);
}

}  // namespace litert::tensor::examples::experimental::gemma4::cpu

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_MOBILE_FULLY_CONNECTED_H_

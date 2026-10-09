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

#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_MOBILE_FULLY_CONNECTED_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_MOBILE_FULLY_CONNECTED_H_

#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensor/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::gemma4::cpu {

// Match the primitive RMSNorm expansion used by the CPU reference graphs.
template <class... M>
Tensor<M...> PrimitiveRmsNorm(const Tensor<M...>& x, const Tensor<M...>& scale,
                              const Tensor<M...>& epsilon) {
  Tensor<M...> mean = Mean(Square(x), {int(x.GetShape().size()) - 1}, true);
  Tensor<M...> result = Mul(x, Rsqrt(Add(mean, epsilon)));
  return scale.GetStatus().ok() ? Mul(result, scale) : result;
}

inline absl::StatusOr<float> ReadMobileActivationScale(
    const TensorHandle& tensor) {
  absl::StatusOr<Buffer&> buffer = tensor.GetBuffer();
  if (!buffer.ok()) {
    return buffer.status();
  }
  absl::StatusOr<size_t> size = buffer->ByteSize();
  if (!size.ok()) {
    return size.status();
  }
  if (tensor.GetType() != Type::kFP32 || *size != sizeof(float) ||
      !(tensor.GetShape().empty() || tensor.GetShape() == Shape{1})) {
    return absl::InvalidArgumentError(
        "Mobile activation scale must be one FP32 scalar");
  }
  LockedBufferSpan<const std::byte> lock = buffer->Lock();
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
    Tensor<Mixins...> input, Tensor<Mixins...> weight,
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>* weights =
        nullptr) {
  if (weights == nullptr) {
    if (weight.GetType() == Type::kI2) {
      return Tensor<Mixins...>(graph::ErrorTensor(absl::InvalidArgumentError(
          "Original static INT2 weights require original activation scales")));
    }
    return FullyConnected(input, weight);
  }
  std::string module(weight.GetName());
  constexpr size_t suffix_size = sizeof(".weight") - 1;
  if (module.size() < suffix_size ||
      module.compare(module.size() - suffix_size, suffix_size, ".weight") !=
          0) {
    return FullyConnected(input, weight);
  }
  module.resize(module.size() - suffix_size);
  typename absl::flat_hash_map<std::string, Tensor<Mixins...>>::const_iterator
      in = weights->find(absl::StrCat(module, ".input_scale"));
  typename absl::flat_hash_map<std::string, Tensor<Mixins...>>::const_iterator
      out = weights->find(absl::StrCat(module, ".output_scale"));
  if (in == weights->end() && out == weights->end()) {
    if (weight.GetType() == Type::kI2) {
      return Tensor<Mixins...>(graph::ErrorTensor(absl::InvalidArgumentError(
          "Original static INT2 weights require original activation scales")));
    }
    if ((weight.GetType() == Type::kI4 || weight.GetType() == Type::kI8) &&
        weight.GetQuantization() &&
        weight.GetQuantization()
            ->template As<PerChannelAffineQuantization>()
            .ok()) {
      return FullyConnected(input, weight);
    }
    return FullyConnected(input, weight);
  }
  const auto error = [](absl::Status status) {
    return Tensor<Mixins...>(graph::ErrorTensor(std::move(status)));
  };
  if (in == weights->end() || out == weights->end()) {
    return error(absl::InvalidArgumentError(
        absl::StrCat(module, ": input/output scales must both be present")));
  }
  absl::StatusOr<float> input_scale = ReadMobileActivationScale(in->second);
  absl::StatusOr<float> output_scale = ReadMobileActivationScale(out->second);
  if (!input_scale.ok()) {
    return error(input_scale.status());
  }
  if (!output_scale.ok()) {
    return error(output_scale.status());
  }
  if (input.GetType() != Type::kFP32 ||
      (weight.GetType() != Type::kI2 && weight.GetType() != Type::kI4 &&
       weight.GetType() != Type::kI8) ||
      !weight.GetQuantization()) {
    return error(
        absl::InvalidArgumentError("Mobile static FC requires FP32 input and "
                                   "quantized INT2/INT4/INT8 weights"));
  }
  Tensor<Mixins...> quantized_input =
      Quantize(input, Type::kI8, {*input_scale}, {0});
  Tensor<Mixins...> quantized_output = FullyConnected(quantized_input, weight);
  quantized_output.SetQuantization(
      std::make_shared<PerChannelAffineQuantization>(
          std::vector<float>{*output_scale}, std::vector<int64_t>{0}));
  return Dequantize(quantized_output);
}

}  // namespace litert::tensor::examples::gemma4::cpu

#endif  // LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_MOBILE_FULLY_CONNECTED_H_

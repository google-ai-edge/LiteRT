/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tflite/experimental/shlo/ops/sine.h"

#include <cmath>

#include "absl/status/status.h"
#include "tflite/experimental/shlo/bf16.h"
#include "tflite/experimental/shlo/data_type.h"
#include "tflite/experimental/shlo/dispatch.h"
#include "tflite/experimental/shlo/f16.h"
#include "tflite/experimental/shlo/ops/unary_elementwise.h"
#include "tflite/experimental/shlo/ops/util.h"
#include "tflite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct Sine {
  template <class T>
  T operator()(T v) const {
    return std::sin(v);
  }
};

template <>
F16 Sine::operator()<F16>(F16 val) const {
  return F16(operator()(static_cast<float>(val)));
}

template <>
BF16 Sine::operator()<BF16>(BF16 val) const {
  return BF16(operator()(static_cast<float>(val)));
}

SineOp Create(SineOp::Attributes) { return {}; }

absl::Status Prepare(SineOp& op, const Tensor& input, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(CheckSupportedTypes(
      CheckCtx("sine"), input, IsFloatTensor, IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("sine"), input, output));
  return absl::OkStatus();
}

absl::Status Evaluate(SineOp& op, const Tensor& input, Tensor& output) {
  Sine sine;
  if (input.IsPerTensorQuantized()) {
    DISPATCH_QUANTIZED(
        detail::DequantizeOpQuantizePerTensor,
        input.quantized_per_tensor_element_type().StorageType(),
        input.quantized_per_tensor_element_type().ExpressedType(), sine, input,
        output)
  } else if (!input.IsQuantized() && IsFloat(input.StorageType())) {
    DISPATCH_FLOAT(detail::EvaluateNoQuantization, input.tensor_element_type(),
                   sine, input, output);
  }
  return absl::FailedPreconditionError("Unsupported tensor type.");
}

};  // namespace shlo_ref

// Copyright 2026 Google LLC.
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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_SIMPLE_UNARY_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_SIMPLE_UNARY_H_

#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

// Infers the output shape for an identity op (output shape == input shape).
inline LiteRtStatus InferIdentity(absl::Span<Dims>& input_shapes,
                                  std::vector<Dims>& output_shapes) {
  if (output_shapes.size() != 1) {
    LITERT_LOG(LITERT_ERROR, "Invalid number of output shapes for unary op.");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  output_shapes[0] = input_shapes[0];
  return kLiteRtStatusOk;
}

#define DEFINE_SIMPLE_UNARY_INFER(name)                               \
  inline LiteRtStatus Infer##name(const LiteRtOpT& op,                \
                                  absl::Span<Dims>& input_shapes,     \
                                  std::vector<Dims>& output_shapes) { \
    return InferIdentity(input_shapes, output_shapes);                \
  }

DEFINE_SIMPLE_UNARY_INFER(Abs)
DEFINE_SIMPLE_UNARY_INFER(Cast)
DEFINE_SIMPLE_UNARY_INFER(Ceil)
DEFINE_SIMPLE_UNARY_INFER(Cos)
DEFINE_SIMPLE_UNARY_INFER(Cumsum)
DEFINE_SIMPLE_UNARY_INFER(Dequantize)
DEFINE_SIMPLE_UNARY_INFER(Elu)
DEFINE_SIMPLE_UNARY_INFER(Exp)
DEFINE_SIMPLE_UNARY_INFER(Floor)
DEFINE_SIMPLE_UNARY_INFER(Gelu)
DEFINE_SIMPLE_UNARY_INFER(HardSwish)
DEFINE_SIMPLE_UNARY_INFER(L2Normalization)
DEFINE_SIMPLE_UNARY_INFER(LeakyRelu)
DEFINE_SIMPLE_UNARY_INFER(Log)
DEFINE_SIMPLE_UNARY_INFER(LogicalNot)
DEFINE_SIMPLE_UNARY_INFER(Logistic)
DEFINE_SIMPLE_UNARY_INFER(Neg)
DEFINE_SIMPLE_UNARY_INFER(Quantize)
DEFINE_SIMPLE_UNARY_INFER(Relu)
DEFINE_SIMPLE_UNARY_INFER(Relu0To1)
DEFINE_SIMPLE_UNARY_INFER(Relu6)
DEFINE_SIMPLE_UNARY_INFER(ReluN1To1)
DEFINE_SIMPLE_UNARY_INFER(ReverseV2)
DEFINE_SIMPLE_UNARY_INFER(Round)
DEFINE_SIMPLE_UNARY_INFER(Rsqrt)
DEFINE_SIMPLE_UNARY_INFER(Sign)
DEFINE_SIMPLE_UNARY_INFER(Sin)
DEFINE_SIMPLE_UNARY_INFER(Softmax)
DEFINE_SIMPLE_UNARY_INFER(Sqrt)
DEFINE_SIMPLE_UNARY_INFER(Square)
DEFINE_SIMPLE_UNARY_INFER(Tanh)

#undef DEFINE_SIMPLE_UNARY_INFER

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_SIMPLE_UNARY_H_

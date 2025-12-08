// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/hadamard_transform_op_builder.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "QnnOpDef.h"  // from @qairt
#include "absl/numeric/bits.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

std::optional<float> GetSylvesterHadamardScale(const TensorWrapper& weight) {
  if (weight.GetRank() != 2) return std::nullopt;
  if (weight.GetDimension(0) != weight.GetDimension(1)) return std::nullopt;
  const std::uint32_t n = weight.GetDimension(0);

  // Ensure the size is power-of-two.
  if (n == 0 || (n & (n - 1)) != 0) return std::nullopt;

  // Get weight data.
  if (weight.GetDataType() != Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_8) {
    return std::nullopt;
  }
  auto hadamard = weight.GetTensorData<std::int8_t>();
  if (!hadamard.has_value()) return std::nullopt;

  std::int8_t hadamard_value = hadamard.value()[0];
  // Reject all-zero weight tensors: the Sylvester check below would trivially
  // succeed and return scale = 0, miscompiling the op into a zero output.
  if (hadamard_value == 0) return std::nullopt;

  // Ensure the matrix adheres Sylvester's construction.
  for (std::uint32_t i = 0; i < n; ++i) {
    for (std::uint32_t j = 0; j < n; ++j) {
      int bits = absl::popcount(i & j);
      std::int8_t val = ((bits & 1) == 0) ? +hadamard_value : -hadamard_value;
      if (hadamard.value()[i * n + j] != val) {
        return std::nullopt;
      }
    }
  }

  // QNN HadamardTransform: out = HadamardTransform(in) * scale
  //
  // Dequantized weight is defined as: S * (q - Z),
  // where q is the stored int8 value and Z is the zero-point.
  //
  // For a valid Hadamard matrix, the zero-point Z must be 0.
  // Otherwise, the dequantized weights will not preserve the
  // Hadamard property (i.e., symmetric +/- structure).
  //
  // Specifically, elements must satisfy:
  //   S * (q - Z) == -S * (-q - Z)
  // => S * (q - Z) == S * (q + Z)
  // => Z == 0
  //
  // Therefore, if Z != 0, the weight cannot represent a Hadamard matrix.
  if (const auto* p = std::get_if<BwScaleOffsetQuantizeParamsWrapper>(
          &weight.GetQuantParams())) {
    if (p->GetZeroPoint() != 0) return std::nullopt;
    return p->GetScale() * hadamard_value * std::sqrt(n);
  } else if (const auto* p = std::get_if<ScaleOffsetQuantizeParamsWrapper>(
                 &weight.GetQuantParams())) {
    if (p->GetZeroPoint() != 0) return std::nullopt;
    return p->GetScale() * hadamard_value * std::sqrt(n);
  }
  return std::nullopt;
}

OpWrapper CreateHadamardTransformOp(const TensorWrapper& input,
                                    const TensorWrapper& output, float scale) {
  OpWrapper op(GetUniqueOpName(QNN_OP_HADAMARD_TRANSFORM),
               QNN_OP_HADAMARD_TRANSFORM, QnnOpCode::kHadamardTransform);
  op.AddInputTensor(input);
  op.AddOutputTensor(output);
  op.AddScalarParam<float>(QNN_OP_HADAMARD_TRANSFORM_PARAM_SCALE, scale);
  return op;
}

}  // namespace qnn

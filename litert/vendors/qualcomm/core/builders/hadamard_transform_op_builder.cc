// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/hadamard_transform_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cmath>
#include "QnnOpDef.h"  // from @qairt
#include "absl/numeric/bits.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
namespace qnn {

namespace {
constexpr std::size_t kInputIndex = 0;
constexpr std::size_t kWeightIndex = 1;
constexpr std::size_t kOutputIndex = 0;
}  // namespace

std::optional<float> GetSylvesterHadamardScale(const TensorWrapper& weight) {
  if (weight.GetRank() != 2) return std::nullopt;
  if (weight.GetDim(0) != weight.GetDim(1)) return std::nullopt;
  const std::uint32_t n = weight.GetDim(0);

  // Ensure the size is power-of-two.
  if (n == 0 || (n & (n - 1)) != 0) return std::nullopt;

  // Get weight data.
  if (weight.GetDataType() != Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_8) {
    return std::nullopt;
  }
  auto hadamard = weight.GetTensorData<std::int8_t>();
  if (!hadamard.has_value()) return std::nullopt;

  std::int8_t hadamard_value = hadamard.value()[0];

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
  // HadamardTransform() applies the Hadamard matrix with a normalization factor
  // of 1/sqrt(n), where n is the size of the last dimension of the input and
  // output tensors.
  //
  // As the dequantized weight is S * (hadamard_value - Z), to match the
  // normalized form, the required scale is:
  //     scale = S * (hadamard_value - Z) * sqrt(n)
  // where S and Z are the quantization scale and zero-point.
  if (const auto* p = std::get_if<BwScaleOffsetQuantizeParamsWrapper>(
          &weight.GetQuantParams())) {
    return p->GetScale() * (hadamard_value - p->GetZeroPoint()) * std::sqrt(n);
  } else if (const auto* p = std::get_if<ScaleOffsetQuantizeParamsWrapper>(
                 &weight.GetQuantParams())) {
    return p->GetScale() * (hadamard_value + p->GetZeroPoint()) * std::sqrt(n);
  }
  return std::nullopt;
}

std::vector<OpWrapper> BuildHadamardTransformOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, float scale) {
  std::vector<OpWrapper> res;
  auto& op = CreateOpWrapper(res, QNN_OP_HADAMARD_TRANSFORM);
  op.AddScalarParam<float>(QNN_OP_HADAMARD_TRANSFORM_PARAM_SCALE, scale);
  op.AddInputTensor(inputs[kInputIndex]);
  op.AddOutputTensor(outputs[kOutputIndex]);
  return res;
}

}  // namespace qnn

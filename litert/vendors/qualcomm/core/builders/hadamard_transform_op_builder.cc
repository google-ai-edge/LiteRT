// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/hadamard_transform_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "QnnOpDef.h"  // from @qairt
#include "absl/numeric/bits.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
namespace qnn {

namespace {
constexpr std::size_t kInputIndex = 0;
constexpr std::size_t kWeightIndex = 1;
constexpr std::size_t kOutputIndex = 0;

bool IsSylvesterHadamard(const TensorWrapper& weight) {
  if (weight.GetRank() != 2) return false;
  if (weight.GetDim(0) != weight.GetDim(1)) return false;
  uint32_t n = weight.GetDim(0);
  if (n == 0 || n % 2) return false;

  // TODO: Support more Hadamard weight since we only support int8 for now.
  // Get weight data.
  if (weight.GetDataType() != Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_8) {
    return false;
  }
  auto hadamard = weight.GetTensorData<std::int8_t>();
  if (!hadamard.has_value()) return false;

  auto hadamard_value = hadamard.value();
  std::int8_t scale = hadamard_value[0];
  // Ensure all entries are ±scale.
  for (int i = 0; i < n * n; ++i) {
    if (hadamard_value[i] != scale && hadamard_value[i] != -scale) return false;
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      // (-1)^(popcount(i & j))
      int bits = absl::popcount(static_cast<uint64_t>(i & j));
      std::int8_t val = (bits % 2 == 0) ? +scale : -scale;
      if (hadamard_value[i * n + j] != val) {
        return false;
      }
    }
  }
  return true;
}
}  // namespace

std::vector<OpWrapper> BuildHadamardTransformOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  // If this Hadamard transform is derived from a fully connected (FC) layer,
  // verify that the weight tensor already represents a valid Hadamard matrix
  if (inputs.size() > 1 && !IsSylvesterHadamard(inputs[kWeightIndex])) {
    return {};
  };
  std::vector<OpWrapper> res;
  auto& op = CreateOpWrapper(res, QNN_OP_HADAMARD_TRANSFORM);
  op.AddInputTensor(inputs[kInputIndex]);
  op.AddOutputTensor(outputs[kOutputIndex]);
  return res;
}

}  // namespace qnn

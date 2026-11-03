// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/layer_norm_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {

static constexpr size_t kInputIndex = 0;
static constexpr size_t kAxesIndex = 3;

std::vector<OpWrapper> BuildLayerNormOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, float epsilon) {
  std::vector<OpWrapper> res;

  // HTP LayerNorm has no int16 kernel, only uint16: bracket it with converts
  // (signed and unsigned 16-bit differ only by the zero point).
  TensorWrapper* input = &inputs[kInputIndex].get();
  TensorWrapper* output = &outputs[0].get();
  const bool convert_i16 = input->IsQuantI16() && output->IsQuantI16();
  if (convert_i16) {
    input = &tensor_pool.CloneNativeTensorFrom(*input);
    input->ConvertFromQuantI16ToQuantU16();
    auto& convert_in = CreateOpWrapper(res, QNN_OP_CONVERT);
    convert_in.AddInputTensor(inputs[kInputIndex]);
    convert_in.AddOutputTensor(*input);
    output = &tensor_pool.CloneNativeTensorFrom(*output);
    output->ConvertFromQuantI16ToQuantU16();
  }

  auto& layer_norm_op = CreateOpWrapper(res, QNN_OP_LAYER_NORM);
  // inputs: input, gamma, beta, axes
  layer_norm_op.AddInputTensor(*input);
  for (size_t i = 1; i < kAxesIndex && i < inputs.size(); ++i) {
    layer_norm_op.AddInputTensor(inputs[i]);
  }
  layer_norm_op.AddOutputTensor(*output);

  layer_norm_op.AddScalarParam<float>(QNN_OP_LAYER_NORM_PARAM_EPSILON, epsilon);
  if (inputs.size() > kAxesIndex) {
    layer_norm_op.AddTensorParam(QNN_OP_LAYER_NORM_PARAM_AXES,
                                 inputs[kAxesIndex]);
  } else {
    std::vector<std::uint32_t> axis_data = {
        inputs[kInputIndex].get().GetRank() - 1};
    TensorWrapper& axis_tensor = tensor_pool.CreateStaticTensor(
        QNN_DATATYPE_UINT_32, inputs[kInputIndex].get().GetQuantParams(),
        {static_cast<uint32_t>(axis_data.size())},
        sizeof(std::uint32_t) * axis_data.size(), axis_data.data());
    layer_norm_op.AddTensorParam(QNN_OP_LAYER_NORM_PARAM_AXES, axis_tensor);
  }

  if (convert_i16) {
    auto& convert_out = CreateOpWrapper(res, QNN_OP_CONVERT);
    convert_out.AddInputTensor(*output);
    convert_out.AddOutputTensor(outputs[0]);
  }
  return res;
}

}  // namespace qnn

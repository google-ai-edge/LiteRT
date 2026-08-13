// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include <cmath>
#include <cstdint>
#include <limits>
#include <variant>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {

namespace {

constexpr size_t kInputIndex = 0;
constexpr size_t kAlphaIndex = 1;
constexpr size_t kOutputIndex = 0;

TensorWrapper* CreateU16AlphaTensor(TensorPool& tensor_pool,
                                    const TensorWrapper& alpha_tensor) {
  const auto alpha_data = alpha_tensor.GetTensorData<std::int8_t>();
  if (!alpha_data.has_value()) {
    QNN_LOG_ERROR("Failed to read static PRelu alpha tensor data.");
    return nullptr;
  }

  const auto* src_quant =
      std::get_if<ScaleOffsetQuantizeParamsWrapper>(
          &alpha_tensor.GetQuantParams());
  if (src_quant == nullptr) {
    QNN_LOG_ERROR("PRelu int8 alpha tensor must use per-tensor quantization.");
    return nullptr;
  }

  const float src_scale = src_quant->GetScale();
  const std::int32_t src_zp = src_quant->GetZeroPoint();
  std::vector<float> dq(alpha_data->size());
  float min_v = std::numeric_limits<float>::infinity();
  float max_v = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < alpha_data->size(); ++i) {
    const float f =
        (static_cast<std::int32_t>((*alpha_data)[i]) - src_zp) * src_scale;
    dq[i] = f;
    if (f < min_v) min_v = f;
    if (f > max_v) max_v = f;
  }

  // Include zero in the range so the encoding remains asymmetric-valid, and
  // enforce a minimum span to avoid a zero-scale encoding when alpha is
  // constant.
  if (min_v > 0.0f) min_v = 0.0f;
  if (max_v < 0.0f) max_v = 0.0f;
  float range = max_v - min_v;
  if (range < 1e-8f) range = 1e-8f;

  const float dst_scale = range / 65535.0f;
  std::int32_t dst_zp =
      static_cast<std::int32_t>(std::lround(-min_v / dst_scale));
  if (dst_zp < 0) dst_zp = 0;
  if (dst_zp > 65535) dst_zp = 65535;

  std::vector<std::uint16_t> u16_alpha(dq.size());
  for (size_t i = 0; i < dq.size(); ++i) {
    std::int32_t q =
        static_cast<std::int32_t>(std::lround(dq[i] / dst_scale)) + dst_zp;
    if (q < 0) q = 0;
    if (q > 65535) q = 65535;
    u16_alpha[i] = static_cast<std::uint16_t>(q);
  }

  QuantizeParamsWrapperVariant dst_quant;
  dst_quant.emplace<ScaleOffsetQuantizeParamsWrapper>(dst_scale, dst_zp);

  return &tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UFIXED_POINT_16, dst_quant, alpha_tensor.GetDimensions(),
      sizeof(std::uint16_t) * u16_alpha.size(), u16_alpha.data());
}

}  // namespace

std::vector<OpWrapper> BuildPreluOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  if (inputs.size() != 2) {
    QNN_LOG_ERROR("Prelu op must have exactly two input tensors.");
    return {};
  }
  if (outputs.size() != 1) {
    QNN_LOG_ERROR("Prelu op must have exactly one output tensor.");
    return {};
  }

  TensorWrapper& input_tensor = inputs[kInputIndex];
  TensorWrapper& alpha_tensor = inputs[kAlphaIndex];
  TensorWrapper& output_tensor = outputs[kOutputIndex];

  if (input_tensor.IsQuantU16() && alpha_tensor.IsTensorStatic() &&
      alpha_tensor.IsQuantI8()) {
    TensorWrapper* u16_alpha =
        CreateU16AlphaTensor(tensor_pool, alpha_tensor);
    if (u16_alpha == nullptr) {
      return {};
    }

    auto& prelu_op = CreateOpWrapper(res, QNN_OP_PRELU);
    prelu_op.AddInputTensor(input_tensor);
    prelu_op.AddInputTensor(*u16_alpha);
    prelu_op.AddOutputTensor(output_tensor);

    return res;
  }

  auto& prelu_op = CreateOpWrapper(res, QNN_OP_PRELU);
  for (const auto& input : inputs) {
    prelu_op.AddInputTensor(input);
  }
  prelu_op.AddOutputTensor(output_tensor);

  return res;
}

}  // namespace qnn

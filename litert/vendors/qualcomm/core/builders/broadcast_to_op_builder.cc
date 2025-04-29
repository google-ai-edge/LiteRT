// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/broadcast_to_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <variant>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {
namespace {
std::vector<std::uint32_t> GetStaticTensorDimention(
    const std::vector<std::uint32_t>& input_dimensions,
    const std::vector<std::uint32_t>& output_dimensions) {
  std::uint32_t input_size = input_dimensions.size();
  std::uint32_t output_size = output_dimensions.size();
  std::vector<std::uint32_t> final_dimensions;

  if (input_size < output_size) {
    int padding_size = output_size - input_size;
    final_dimensions = std::vector<std::uint32_t>(padding_size, 1);
    final_dimensions.insert(final_dimensions.end(), input_dimensions.begin(),
                            input_dimensions.end());
  } else {
    final_dimensions = input_dimensions;
  }

  for (std::size_t i = 0; i < output_size; ++i) {
    final_dimensions[i] =
        (output_dimensions[i] > final_dimensions[i]) ? output_dimensions[i] : 1;
  }

  return final_dimensions;
}

template <typename T>
TensorWrapper& CreateStaticTensor(TensorPool& tensor_pool,
                                  const Qnn_DataType_t data_type,
                                  TensorWrapper& input, TensorWrapper& output,
                                  bool is_quant) {
  std::vector<std::uint32_t> static_dims =
      GetStaticTensorDimention(input.GetDims(), output.GetDims());
  std::uint32_t static_size =
      std::accumulate(static_dims.begin(), static_dims.end(), 1,
                      std::multiplies<std::uint32_t>());

  T static_value{0};
  QuantizeParamsWrapperVariant quant_param{};
  if (is_quant) {
    auto input_quant_param =
        std::get<ScaleOffsetQuantizeParamsWrapper>(input.GetQuantParams());
    quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(input_quant_param);
    static_value = -input_quant_param.GetZeroPoint();
  }
  std::vector<T> static_data(static_size, static_value);

  return tensor_pool.CreateStaticTensor(data_type, quant_param, static_dims,
                                        sizeof(T) * static_size,
                                        static_data.data());
}
}  // namespace

std::vector<OpWrapper> BuildBroadcastToOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  // TODO: handle per-channel case
  if (std::holds_alternative<AxisScaleOffsetQuantizeParamsWrapper>(
          inputs[0].get().GetQuantParams())) {
    return res;
  }

  const char* qnn_op = nullptr;
  if (inputs[0].get().GetDataType() == QNN_DATATYPE_BOOL_8) {
    qnn_op = QNN_OP_ELEMENT_WISE_OR;
  } else {
    qnn_op = QNN_OP_ELEMENT_WISE_ADD;
  }

  auto& broadcast_op = CreateOpWrapper(res, qnn_op);
  broadcast_op.AddInputTensor(inputs[0]);

  switch (inputs[0].get().GetDataType()) {
    case QNN_DATATYPE_BOOL_8: {
      TensorWrapper& static_tensor = CreateStaticTensor<std::uint8_t>(
          tensor_pool, inputs[0].get().GetDataType(), inputs[0], outputs[0],
          false);
      broadcast_op.AddInputTensor(static_tensor);
      break;
    }
    case QNN_DATATYPE_UFIXED_POINT_8: {
      TensorWrapper& static_tensor = CreateStaticTensor<std::uint8_t>(
          tensor_pool, inputs[0].get().GetDataType(), inputs[0], outputs[0],
          true);
      broadcast_op.AddInputTensor(static_tensor);
      break;
    }
    case QNN_DATATYPE_SFIXED_POINT_8: {
      TensorWrapper& static_tensor = CreateStaticTensor<std::int8_t>(
          tensor_pool, inputs[0].get().GetDataType(), inputs[0], outputs[0],
          true);
      broadcast_op.AddInputTensor(static_tensor);
      break;
    }
    case QNN_DATATYPE_UFIXED_POINT_16: {
      TensorWrapper& static_tensor = CreateStaticTensor<std::uint16_t>(
          tensor_pool, inputs[0].get().GetDataType(), inputs[0], outputs[0],
          true);
      broadcast_op.AddInputTensor(static_tensor);
      break;
    }
    case QNN_DATATYPE_SFIXED_POINT_16: {
      TensorWrapper& static_tensor = CreateStaticTensor<std::int16_t>(
          tensor_pool, inputs[0].get().GetDataType(), inputs[0], outputs[0],
          true);
      broadcast_op.AddInputTensor(static_tensor);
      break;
    }
    case QNN_DATATYPE_FLOAT_32: {
      TensorWrapper& static_tensor =
          CreateStaticTensor<float>(tensor_pool, inputs[0].get().GetDataType(),
                                    inputs[0], outputs[0], false);
      broadcast_op.AddInputTensor(static_tensor);
      break;
    }
    default: {
      QNN_LOG_ERROR("Unsupported QNN data type when creating static tensor");
      break;
    }
  }

  broadcast_op.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn

// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/embedding_lookup_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/quantize_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {
namespace {
constexpr int kTableIdx = 1;
constexpr int kIndicesIdx = 0;
constexpr int kOutputIdx = 0;
constexpr std::int32_t kGatherDefaultAxis = 0;
}  // namespace

OpWrapper CreateGatherOp(const TensorWrapper& table,
                         const TensorWrapper& indices,
                         const TensorWrapper& output, std::int32_t axis) {
  OpWrapper op(GetUniqueOpName(QNN_OP_GATHER), QNN_OP_GATHER,
               QnnOpCode::kGather);
  op.AddInputTensor(table);
  op.AddInputTensor(indices);
  op.AddOutputTensor(output);
  op.AddScalarParam<std::int32_t>(QNN_OP_GATHER_PARAM_AXIS, axis);
  return op;
}

std::vector<OpWrapper> BuildEmbeddingLookupOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  const TensorWrapper* table_tensor = &(inputs[kTableIdx].get());
  const TensorWrapper& indices_tensor = inputs[kIndicesIdx];
  const TensorWrapper& output_tensor = outputs[kOutputIdx];

  // Case: QInt8 table with QInt16 output
  if (table_tensor->IsQuantI8() && output_tensor.IsQuantI16()) {
    QNN_LOG_WARNING(
        "The data type of embedding lookup table is int8, but output data type "
        "is int16. Int8 table will be cast to int16.");
    std::vector<std::int16_t> int16_data;
    size_t data_len = table_tensor->GetTensorNumElements();
    auto int8_data = table_tensor->GetTensorData<std::int8_t>();
    if (!int8_data.has_value()) {
      QNN_LOG_ERROR("Embedding lookup get int8 table failed.");
      return {};
    }
    int16_data.reserve(data_len);
    for (size_t i = 0; i < data_len; ++i) {
      int16_data.emplace_back(static_cast<std::int16_t>((*int8_data)[i]));
    }

    table_tensor = &tensor_pool.CreateStaticTensor(
        output_tensor.GetDataType(), table_tensor->GetQuantParams(),
        table_tensor->GetDimensions(),
        sizeof(decltype(int16_data)::value_type) * int16_data.size(),
        reinterpret_cast<void*>(int16_data.data()));
  }

  const auto& table_quant_params = table_tensor->GetQuantParams();
  if (table_quant_params == output_tensor.GetQuantParams()) {
    return MakeVector(CreateGatherOp(*table_tensor, indices_tensor,
                                     output_tensor, kGatherDefaultAxis));
  }
  QNN_LOG_WARNING(
      "Add a Convert op after the Gather op since the table's quant params do "
      "not match the output's.");
  const auto& gather_output =
      tensor_pool.CloneNativeTensorFrom(output_tensor, table_quant_params);
  return MakeVector(CreateGatherOp(*table_tensor, indices_tensor, gather_output,
                                   kGatherDefaultAxis),
                    CreateConvertOp(gather_output, output_tensor));
}

}  // namespace qnn

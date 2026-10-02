// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/relu_min_max.h"

#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {

size_t MinMaxToReLUMinMax(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_index, TensorPool& tensor_pool,
    size_t pattern_size) {
    QNN_LOG_INFO("[G2G] MinMaxToReLUMinMax");
    auto& elementwise_binary_max = ops[start_index];
    auto& elementwise_binary_min = ops[start_index + 1];
    // Connection check
    if (elementwise_binary_max.GetOutputTensor(0) != elementwise_binary_min.GetInputTensor(0) &&
        elementwise_binary_max.GetOutputTensor(0) != elementwise_binary_min.GetInputTensor(1)) {
        QNN_LOG_WARNING("[G2G] Skip MinMaxToReLUMinMax after connection check");
        return 1;
    }

    // Static check
    if (!elementwise_binary_min.GetInputTensor(0).IsTensorStatic() &&
        !elementwise_binary_min.GetInputTensor(1).IsTensorStatic()) {
        QNN_LOG_WARNING("[G2G] Skip MinMaxToReLUMinMax after Tensor Static check on Min");
        return 1;
    }

    if (!elementwise_binary_max.GetInputTensor(0).IsTensorStatic() &&
        !elementwise_binary_max.GetInputTensor(1).IsTensorStatic()) {
        QNN_LOG_WARNING("[G2G] Skip MinMaxToReLUMinMax after Tensor Static check on Max");
        return 1;
    }

    QNN_LOG_INFO("[G2G] MinMaxToReLUMinMax pass check");

    auto elementwise_binary_min_const_idx = elementwise_binary_min.GetInputTensor(0).IsTensorStatic()? 0 : 1;
    auto elementwise_binary_max_const_idx = elementwise_binary_max.GetInputTensor(0).IsTensorStatic()? 0 : 1;

    if (elementwise_binary_min.GetInputTensor(elementwise_binary_min_const_idx).GetDataType()
        != Qnn_DataType_t::QNN_DATATYPE_FLOAT_32 ||
        elementwise_binary_max.GetInputTensor(elementwise_binary_max_const_idx).GetDataType()
        != Qnn_DataType_t::QNN_DATATYPE_FLOAT_32) {
        QNN_LOG_WARNING("[G2G] Skip MinMaxToReLUMinMax since currently only support float32 constant");
        return 1;
    }

    auto min_const = elementwise_binary_min.GetInputTensor(elementwise_binary_min_const_idx).GetTensorData<float_t>();
    auto max_const = elementwise_binary_max.GetInputTensor(elementwise_binary_max_const_idx).GetTensorData<float_t>();

    if (!min_const.has_value()) {
        QNN_LOG_WARNING("[G2G] Skip MinMaxToReLUMinMax since no value on Min const");
        return 1;
    }
    if (!max_const.has_value()) {
        QNN_LOG_WARNING("[G2G] Skip MinMaxToReLUMinMax since no value on Max const");
        return 1;
    }

    std::vector<OpWrapper> new_ops;
    OpWrapper& relu_min_max_op = CreateOpWrapper(new_ops, QNN_OP_ELEMENT_WISE_NEURON);
    relu_min_max_op.AddInputTensor(elementwise_binary_max.GetInputTensor(1-elementwise_binary_max_const_idx));
    relu_min_max_op.AddOutputTensor(elementwise_binary_min.GetOutputTensor(0));
    relu_min_max_op.AddScalarParam<std::uint32_t>(
        QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION,
        QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX);
    relu_min_max_op.AddScalarParam<float>(QNN_OP_ELEMENT_WISE_NEURON_PARAM_MIN_VALUE,
                                          max_const.value()[0]);
    relu_min_max_op.AddScalarParam<float>(QNN_OP_ELEMENT_WISE_NEURON_PARAM_MAX_VALUE,
                                          min_const.value()[0]);

    const bool is_valid =
        std::all_of(new_ops.begin(), new_ops.end(),
                    [validate_op_config](OpWrapper& op_wrapper) -> bool {
                      return validate_op_config(op_wrapper);
                    });
    if (is_valid) {
        ops.insert(ops.begin() + start_index + pattern_size,
                   std::make_move_iterator(new_ops.begin()),
                   std::make_move_iterator(new_ops.end()));
        ops.erase(ops.begin() + start_index,
                  ops.begin() + start_index + pattern_size);
        QNN_LOG_INFO("[G2G] MinMaxToReLUMinMax done!");
        return new_ops.size();
    }

    QNN_LOG_INFO("[G2G] MinMaxToReLUMinMax skip due to validation");
    return 1;
}
}
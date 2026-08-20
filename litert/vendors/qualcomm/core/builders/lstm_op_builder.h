// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_LSTM_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_LSTM_OP_BUILDER_H_

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

// Lowers TFLite LSTM and UnidirectionalSequenceLSTM to QNN_OP_LSTM.
//
// Supports the canonical FULL kernel with float or quantized 2D (single step)
// or 3D (sequence) input. The 20-slot form omits layer-normalization
// coefficients; the 24-slot form maps them to QNN inputs 12 through 15.
// The BASIC kernel is not supported.

std::vector<OpWrapper> BuildLstmOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, float cell_clip,
    float proj_clip, bool time_major);

OpWrapper CreateLstmOp(const std::vector<ConstTensorWrapperRef>& inputs,
                       const std::vector<ConstTensorWrapperRef>& outputs,
                       float cell_clip, float proj_clip, bool time_major,
                       std::uint32_t direction);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_LSTM_OP_BUILDER_H_

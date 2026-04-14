// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

static constexpr char kCustomOptionsParamName[] = "CustomInitialData";

std::vector<OpWrapper> BuildCustomOp(
    TensorPool& tensor_pool, const char* package_name, const char* op_type,
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    absl::Span<const uint8_t> custom_options) {
  std::vector<OpWrapper> res;
  auto& op = res.emplace_back(GetUniqueOpName(op_type), package_name, op_type,
                              QnnOpCode::kUnknown);
  for (auto& input : inputs) {
    op.AddInputTensor(input);
  }
  for (auto& output : outputs) {
    op.AddOutputTensor(output);
  }
  if (!custom_options.empty()) {
    std::uint32_t options_bytes = custom_options.size();

    op.AddTensorParam(
        kCustomOptionsParamName,
        tensor_pool.CreateStaticTensor(QNN_DATATYPE_UINT_8, {}, {options_bytes},
                                       options_bytes, custom_options.data()));
  }
  return res;
}

}  // namespace qnn
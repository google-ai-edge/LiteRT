// Copyright (c) 2025 MediaTek Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/vendors/mediatek/compiler/legalizations/common_op_legalization.h"

#include <cstdint>
#include <string>
#include <vector>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/mediatek/compiler/legalizations/neuron_utils.h"
#include "litert/vendors/mediatek/compiler/legalizations/operand_map.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

bool VerifyCommonOp(const litert::Op& op, LiteRtOpCode op_code) {
  // Do some common check
  auto check_tensor_types = [&](const auto& tensors) {
    for (const auto& tensor : tensors) {
      auto mtk_type = GetNeuronTensorType(tensor);
      if (!mtk_type) {
        LITERT_LOG(LITERT_ERROR, "%s", mtk_type.Error().Message().c_str());
        return false;
      }
    }
    return true;
  };

  if (!check_tensor_types(op.Inputs()) || !check_tensor_types(op.Outputs())) {
    return false;
  }

  if (op.Code() == kLiteRtOpCodeShloComposite) {
    const char* op_name;
    if (LiteRtGetSHLOCompositeOpName(op.Get(), &op_name) != kLiteRtStatusOk) {
      return false;
    }
    if (std::string(op_name) == "odml.rms_norm" ||
        std::string(op_name) == "odml.l2_norm") {
      return true;
    }
    return false;
  }

  return true;
}

Expected<void> LegalizeCommonOp(const NeuronAdapterApi& neuron_adapter_api,
                                NeuronModel* model, OperandMap& operand_map,
                                const litert::Op& op,
                                NeuronOperationType mtk_operation_type) {
  LITERT_LOG(LITERT_INFO, "Legalize Op: %d", mtk_operation_type);
  std::vector<uint32_t> input_indices;
  int32_t tensor_flags = 0;
  if (mtk_operation_type == NEURON_UNKNOWN) {
    tensor_flags |= NN_TENSOR_FLAG_USE_INVALID_TENSOR_TYPE;
  }
  for (auto& input : op.Inputs()) {
    auto id = operand_map.GetOperandIndex(input, tensor_flags);
    if (!id) {
      return id.Error();
    }
    input_indices.push_back(*id);
  }

  std::vector<uint32_t> output_indices;
  for (auto& output : op.Outputs()) {
    auto id = operand_map.GetOperandIndex(output, tensor_flags);
    if (!id) {
      return id.Error();
    }
    output_indices.push_back(*id);
  }

  if (ModelAddOperation(neuron_adapter_api, model, /*type=*/mtk_operation_type,
                        input_indices, output_indices) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure, "Failed to add operation");
  }

  return {};
}

}  // namespace litert::mediatek

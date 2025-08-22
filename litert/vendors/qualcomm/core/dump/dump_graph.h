// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_DUMP_DUMP_GRAPH_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_DUMP_DUMP_GRAPH_H_
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @com_github_nlohmann_json
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {
nlohmann::json SerializeTensorToJson(const Qnn_TensorV1_t& qnn_tensor);
nlohmann::json SerializeQuantParamToJson(
    const Qnn_QuantizeParams_t& quant_params);
nlohmann::json SerializeScalarParamToJson(const Qnn_Scalar_t& scalar);
nlohmann::json SerializeTensorParamToJson(const Qnn_TensorV1_t& qnn_tensor);
nlohmann::json SerializeOpToJson(const Qnn_OpConfig_t& op_config);

void DumpIrJson(
    const absl::flat_hash_set<const TensorWrapper*>& tensor_wrappers,
    std::vector<OpWrapper>& graph_op_wrappers, std::string_view json_dir,
    std::string_view graph_name);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_DUMP_DUMP_GRAPH_H_

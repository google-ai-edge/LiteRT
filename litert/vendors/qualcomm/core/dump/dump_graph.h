// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_DUMP_DUMP_GRAPH_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_DUMP_DUMP_GRAPH_H_
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
namespace qnn {
void DumpQnnJson(const absl::flat_hash_set<const TensorWrapper*>& tensors,
                 std::vector<OpWrapper>& graph_op_wrappers,
                 const char* json_path);
}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_DUMP_DUMP_GRAPH_H_

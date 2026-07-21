// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_QNN_COMPOSE_GRAPH_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_QNN_COMPOSE_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "QnnCommon.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_element_type.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "litert/vendors/qualcomm/qnn_manager.h"

namespace litert::qnn {

LiteRtStatus ConvertDataType(const litert::ElementType litert_type,
                             const bool is_quantized, Qnn_DataType_t& qnn_type);

LiteRtStatus ConvertTensor(
    const litert::compiler::Tensor& litert_tensor,
    ::qnn::TensorPool& tensor_pool, ::qnn::TensorWrapper*& tensor_wrapper,
    const absl::flat_hash_set<std::int32_t>& ids_to_dump = {},
    bool is_tensor_output = false);

// `op_index` is the topological index of the op within its subgraph; it is
// included in the "unsupported op" error message so a specific op can be
// located (and cross-referenced with the "_LiteRt_OpId_<n>" suffix added to
// the QNN op name).
LiteRtStatus ConvertOp(const ::qnn::Options& options,
                       const litert::compiler::Op& litert_op,
                       ::qnn::TensorPool& tensor_pool,
                       std::vector<::qnn::TensorWrapperRef>& input_tensors,
                       std::vector<::qnn::TensorWrapperRef>& output_tensors,
                       std::vector<::qnn::OpWrapper>& op_wrappers,
                       size_t op_index, ::qnn::SdkVersion sdk_version);

// Composes a new QNN Graph from given LiteRt Graph. Qnn Graph is written to
// context behind "qnn". Uses given graph_name to name entry point.
LiteRtStatus ComposeGraph(const LiteRtCompilerContext* ctx, QnnManager& qnn,
                          ::qnn::QnnBackend& qnn_backend,
                          Qnn_ContextHandle_t context_handle,
                          Qnn_ProfileHandle_t profile_handle,
                          LiteRtSubgraph subgraph,
                          absl::string_view qnn_graph_name,
                          const ::qnn::Options& options,
                          std::vector<::qnn::TensorWrapper>* inputs = nullptr,
                          std::vector<::qnn::TensorWrapper>* outputs = nullptr);

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_QNN_COMPOSE_GRAPH_H_

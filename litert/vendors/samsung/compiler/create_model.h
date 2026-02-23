// Copyright (C) 2026 Samsung Electronics Co. LTD.
// SPDX-License-Identifier: Apache-2.0
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
#ifndef ODML_LITERT_LITERT_VENDORS_SAMSUNG_COMPILER_CREATE_MODEL_H_
#define ODML_LITERT_LITERT_VENDORS_SAMSUNG_COMPILER_CREATE_MODEL_H_

#include "absl/container/flat_hash_map.h" // from @com_google_absl
#include "absl/container/inlined_vector.h"

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"

#include "litert/vendors/samsung/ai_litecore_manager.h"
#include "litert/vendors/samsung/compiler/builders/op_wrapper.h"

namespace litert::samsung {

class GraphCreator {
public:
  GraphCreator(AiLiteCoreManager::Ptr ai_lite_core, graph_handler_t handler);

  GraphCreator(const GraphCreator &) = delete;
  GraphCreator &operator=(const GraphCreator &) = delete;

  LiteRtStatus CreateTensor(const Tensor &t);

  LiteRtStatus CreateOpNode(const OpWrapper &op_wrapper);

  LiteRtStatus AddInput(const Tensor &t_input);

  LiteRtStatus AddOutput(const Tensor &t_output);

  LiteRtStatus Finish() const;

  Expected<std::vector<char>> Release() const;

private:
  AiLiteCoreManager::Ptr ai_lite_core_;
  graph_handler_t handler_;
  // tensor map record the LiteRtTensor which is already registered
  absl::flat_hash_map<LiteRtTensor, TENSOR_ID_T> tensors_map_;
  // graph inputs/outputs
  absl::InlinedVector<TENSOR_ID_T, kExpectedMaxNumOfSubgraphInputs>
      input_indices_;
  absl::InlinedVector<TENSOR_ID_T, kExpectedMaxNumOfSubgraphOutputs>
      output_indices_;

  /* private function */
  LiteRtStatus CreateQParam(const Tensor &t);
};

Expected<std::vector<char>> CreateModel(AiLiteCoreManager::Ptr ai_lite_core,
                                        const Subgraph &partition);
} // namespace litert::samsung
#endif

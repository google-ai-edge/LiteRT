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

#ifndef ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_CREATE_MODEL_H_
#define ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_CREATE_MODEL_H_

#include <string>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/mediatek/compiler/legalizations/operand_map.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

// Create a new NeuronModel Graph from given LiteRt Graph.
Expected<void> CreateModel(
    const NeuronAdapterApi& neuron_adapter_api, const Subgraph& partition,
    const std::string& model_name, NeuronModel* model, OperandMap* operand_map,
    std::unordered_set<int>* unknown_op_indices = nullptr);

}  // namespace litert::mediatek

#endif  // ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_CREATE_MODEL_H_

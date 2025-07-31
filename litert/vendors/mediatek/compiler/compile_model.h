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

#ifndef ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_COMPILE_MODEL_H_
#define ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_COMPILE_MODEL_H_

#include <optional>
#include <string>

#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_mediatek_options.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

Expected<NeuronCompilationPtr> CompileModel(
    const NeuronAdapterApi& neuron_adapter_api, NeuronModel* model,
    std::optional<std::string> soc_model,
    ::litert::Expected<litert::mediatek::MediatekOptions>& mediatek_opts,
    int subgraph_index, bool get_supported_mode = false);

Expected<void> GetSupportedOperations(
    const NeuronAdapterApi& neuron_adapter_api, NeuronModel* model,
    std::optional<std::string> soc_model,
    ::litert::Expected<litert::mediatek::MediatekOptions>& mediatek_opts,
    const int subgraph_index, bool* support_flags, int num_ops);

}  // namespace litert::mediatek

#endif  // ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_COMPILE_MODEL_H_

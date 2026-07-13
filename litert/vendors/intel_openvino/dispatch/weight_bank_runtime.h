// Copyright (C) 2026 Intel Corporation
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

#ifndef ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_DISPATCH_WEIGHT_BANK_RUNTIME_H_
#define ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_DISPATCH_WEIGHT_BANK_RUNTIME_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/intel_openvino/compiler/global_graph.h"

namespace litert::openvino {

// Binds one weight-Parameter to a view into the shared buffer.
struct BoundWeight {
  size_t input_index;  // port on the compiled model
  ov::Tensor view;     // view into the shared usm-host buffer (zero-copy)
};

// Allocates one shared usm-host buffer holding the pool (once per process) and
// returns a view into it for each of |compiled_model|'s weight-Parameters named
// in |const_map| (input_index -> BufferId). The caller sets each view on the
// infer request and keeps the views alive for its lifetime.
litert::Expected<std::vector<BoundWeight>> BindSharedWeightsGpu(
    ov::Core& core, const OpenVinoGlobalGraph& global_graph,
    const ov::CompiledModel& compiled_model,
    const std::map<uint32_t, uint32_t>& const_map);

}  // namespace litert::openvino

#endif  // ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_DISPATCH_WEIGHT_BANK_RUNTIME_H_

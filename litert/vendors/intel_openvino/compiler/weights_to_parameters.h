// Copyright 2026 Google LLC.
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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_COMPILER_WEIGHTS_TO_PARAMETERS_H_
#define LITERT_VENDORS_INTEL_OPENVINO_COMPILER_WEIGHTS_TO_PARAMETERS_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "openvino/core/model.hpp"
#include "litert/vendors/intel_openvino/compiler/weight_bank.h"

namespace litert::openvino {

// Replaces each large weight Constant in |model| that |bank| knows (matched by
// friendly_name == LiteRt tensor name) with an ov::op::v0::Parameter of the same
// element type and shape, rewiring all consumers. Constants at or below
// kMinConvertBytes are left unchanged.
//
// |const_map| (must be non-null) is populated with friendly_name -> BufferId for
// every converted weight. The dispatcher resolves each weight-Parameter to its
// pool buffer by friendly_name (robust to input reordering across import).
// Returns the number of weights converted.
size_t ConvertWeightsToParameters(const std::shared_ptr<ov::Model>& model,
                                  const WeightBank& bank,
                                  std::map<std::string, uint32_t>* const_map);

}  // namespace litert::openvino

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_WEIGHTS_TO_PARAMETERS_H_

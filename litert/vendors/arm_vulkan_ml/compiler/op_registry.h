// Copyright 2026 The LiteRT Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_OP_REGISTRY_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_OP_REGISTRY_H_

#include <vector>

#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"

namespace litert::arm_vulkan_ml {

enum class LoweringRoute { kGeneral, kPortable, kTyped, kExtended, kDecomposed };

// Types describe the principal data input and first result. Individual analyzers
// still validate every operand/result, quantization, shape and operator option.
struct OperationVariant {
  LiteRtElementType input_type;
  LiteRtElementType output_type;
  LoweringRoute route;
  const char* constraints;
};

struct OperationRegistration {
  LiteRtOpCode code;
  int data_input;
  std::vector<OperationVariant> variants;
};

// The compiler selects its analysis path from this registry. Diagnostics link
// the same library; registration is a supported subset, not device acceptance.
const std::vector<OperationRegistration>& GetOperationRegistrations();
const OperationRegistration* FindOperationRegistration(LiteRtOpCode code);
const OperationVariant* FindOperationVariant(
    const OperationRegistration& registration, LiteRtElementType input_type,
    LiteRtElementType output_type);
const char* LoweringRouteName(LoweringRoute route);
const char* RegisteredElementTypeName(LiteRtElementType type);

}  // namespace litert::arm_vulkan_ml
#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_OP_REGISTRY_H_

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

#include <string>
#include <vector>

#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"

namespace litert::arm_vulkan_ml {

// Compiler lowering subsystem responsible for an operation variant.
enum class LoweringRoute {
  kGeneral,
  kPortable,
  kTyped,
  kExtended,
  kDecomposed
};

// Types describe the principal data input and first result. Individual
// analyzers still validate every operand/result, quantization, shape and
// operator option.
struct OperationVariant {
  LiteRtElementType input_type;
  LiteRtElementType output_type;
  LoweringRoute route;
  const char* constraints;
};

// Supported variants and principal data input index for a LiteRT opcode.
struct OperationRegistration {
  LiteRtOpCode code;
  int data_input;
  std::vector<OperationVariant> variants;
};

// Supported custom normalization composite registration.
struct NormalizationRegistration {
  const char* name;
  int version;
  const char* constraints;
};

// Returns all registered custom normalization composites.
const std::vector<NormalizationRegistration>& GetNormalizationRegistrations();

// Looks up a custom normalization composite by name, or returns nullptr.
const NormalizationRegistration* FindNormalizationRegistration(
    const std::string& name);

// The compiler selects its analysis path from this registry. Diagnostics link
// the same library; registration is a supported subset, not device acceptance.
const std::vector<OperationRegistration>& GetOperationRegistrations();

// Looks up an `OperationRegistration` by `code`, or returns nullptr.
const OperationRegistration* FindOperationRegistration(LiteRtOpCode code);

// Looks up an `OperationVariant` matching `(input_type, output_type)` in
// `registration`, or returns nullptr.
const OperationVariant* FindOperationVariant(
    const OperationRegistration& registration, LiteRtElementType input_type,
    LiteRtElementType output_type);

// Returns the lowercase string name for `route`.
const char* LoweringRouteName(LoweringRoute route);

// Returns the uppercase string name for `type`.
const char* RegisteredElementTypeName(LiteRtElementType type);

}  // namespace litert::arm_vulkan_ml
#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_OP_REGISTRY_H_

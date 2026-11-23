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

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_TOSA_REQUIREMENTS_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_TOSA_REQUIREMENTS_H_

#include <string>

#include "litert/vendors/arm_vulkan_ml/common/arm_vulkan_ml_jit_executable.h"

namespace tfsc::spirv {
class Module;
}  // namespace tfsc::spirv

namespace litert::arm_vulkan_ml {

// Summarizes operator counts and largest intermediate tensor byte sizes in
// `module`.
GraphMetrics CollectGraphMetrics(const tfsc::spirv::Module& module);

// Inspects final TFSC instructions in `module` so generated tensors and
// operators are included in `requirements`, or populates `reason` on failure.
bool CollectTosaLevelRequirements(const tfsc::spirv::Module& module,
                                  TosaLevelRequirements& requirements,
                                  std::string& reason);

}  // namespace litert::arm_vulkan_ml
#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_TOSA_REQUIREMENTS_H_

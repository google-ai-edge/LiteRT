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

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_PARTITION_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_PARTITION_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "litert/cc/internal/litert_extended_model.h"
#include "litert/vendors/arm_vulkan_ml/common/arm_vulkan_ml_jit_executable.h"
#include "litert/vendors/arm_vulkan_ml/compiler/graph_ir.h"
#include "litert/vendors/arm_vulkan_ml/compiler/softmax.h"

namespace litert::arm_vulkan_ml {

// Optional diagnostic attribution. IDs shared across operators are recorded at
// their first creation; graph input/output scaffolding is not attributed.
struct LoweringRecord {
  size_t operator_index;
  std::vector<uint32_t> result_ids;
  std::vector<size_t> source_operators;
  std::string composite;
};

// Compiles `graph` into a `CompiledPartition` containing serialized TOSA
// SPIR-V and constant descriptors, or populates `reason` and returns nullptr.
std::shared_ptr<const CompiledPartition> CompilePartition(
    const Subgraph& graph, std::string& reason,
    std::vector<LoweringRecord>* diagnostics = nullptr,
    OptimizationReport* optimization = nullptr,
    SoftmaxLutType softmax_lut = SoftmaxLutType::kAuto,
    int rescale_multiplier_bits = 32);

}  // namespace litert::arm_vulkan_ml
#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_PARTITION_H_

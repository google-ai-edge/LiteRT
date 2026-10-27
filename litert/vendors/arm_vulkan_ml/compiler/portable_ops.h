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

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_PORTABLE_OPS_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_PORTABLE_OPS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "litert/vendors/arm_vulkan_ml/compiler/operation.h"

namespace litert::arm_vulkan_ml {

// Validated lowering parameters for a portable data-layout or FP32 operation.
struct PortablePlan {
  LiteRtOpCode code;
  bool quantized = false;
  tfsc::tosa::DataType data_type = tfsc::tosa::DataType::float32_t;
  std::vector<int> inputs;
  std::vector<std::vector<uint32_t>> shapes, outputs;
  std::vector<int32_t> parameters, sizes, strides, padding, kernel;
  bool adj_x = false, adj_y = false;
  int32_t axis = 0;
  uint32_t activation = 0;
  float scalar = 0;
};

// Validates and lowers `op` into a `PortablePlan`, or populates `reason` and
// returns nullptr.
std::shared_ptr<PortablePlan> AnalyzePortableOperation(const Op& op,
                                                       std::string& reason);

// Emits TOSA instructions for a validated `PortablePlan` onto `graph`.
std::vector<tfsc::tosa::ResId> EmitPortableOperation(
    tfsc::tosa::Graph& graph, std::vector<tfsc::tosa::ResId> inputs,
    const PortablePlan& plan, GraphConstantPool& constants);

}  // namespace litert::arm_vulkan_ml
#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_PORTABLE_OPS_H_

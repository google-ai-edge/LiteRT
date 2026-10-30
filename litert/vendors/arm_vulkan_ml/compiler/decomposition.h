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

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_DECOMPOSITION_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_DECOMPOSITION_H_

#include <memory>
#include <string>
#include <vector>

#include "litert/cc/internal/litert_extended_model.h"
#include "litert/vendors/arm_vulkan_ml/compiler/conv2d.h"

namespace litert::arm_vulkan_ml {

struct DecompositionTensor {
  tfsc::tosa::DataType type = tfsc::tosa::DataType::int8_t;
  std::vector<uint32_t> shape;
  double scale = 1;
  int32_t zero_point = 0;
  bool scalar = false;
  bool complex = false;
};

// Owns analysis results. Constants/controls are copied; model objects are never
// consulted during emission. Runtime operands remain in original operand order.
struct DecompositionPlan {
  LiteRtOpCode code;
  DecompositionTensor data, output;
  std::vector<DecompositionTensor> inputs;
  std::vector<int> runtime_inputs;
  std::vector<uint32_t> intermediate_shape;
  std::vector<int32_t> parameters, padding, indices;
  std::vector<float> coefficients;
  std::vector<int8_t> table;
  std::vector<Scale32> scales;
  std::vector<AccumulatorRange> ranges;
  std::vector<int8_t> weights;
  std::vector<int32_t> bias, strides, dilations;
  std::vector<uint32_t> weight_shape;
  Int8Range activation{-128, 127};
  std::vector<std::string> extensions;
  int32_t axis = 0;
  int32_t block = 0;
  bool flag = false;
};

std::shared_ptr<DecompositionPlan> AnalyzeDecomposition(
    const Op& op, std::string& reason);
tfsc::tosa::ResId EmitDecomposition(tfsc::tosa::Graph& graph,
    const std::vector<tfsc::tosa::ResId>& inputs,
    const DecompositionPlan& plan, GraphConstantPool& constants);

}  // namespace litert::arm_vulkan_ml
#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_DECOMPOSITION_H_

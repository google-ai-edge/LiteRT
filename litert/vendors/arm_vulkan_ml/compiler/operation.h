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

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_OPERATION_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_OPERATION_H_
#include <optional>
#include <string>
#include <vector>
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/vendors/arm_vulkan_ml/compiler/conv2d.h"
#include "litert/vendors/arm_vulkan_ml/compiler/op_registry.h"
#include "litert/vendors/arm_vulkan_ml/compiler/softmax.h"
namespace litert::arm_vulkan_ml {
struct QuantizedTensor {
  std::vector<uint32_t> shape;
  double scale;
  int32_t zero_point;
  bool floating = false;
};
struct PortablePlan;
struct DecompositionPlan;
struct OperationPlan {
  std::shared_ptr<PortablePlan> portable;
  std::shared_ptr<DecompositionPlan> decomposition;
  LiteRtOpCode code;
  LoweringRoute route = LoweringRoute::kGeneral;
  std::vector<std::string> required_extensions;
  bool typed = false;
  std::vector<tfsc::tosa::DataType> input_types;
  tfsc::tosa::DataType output_type = tfsc::tosa::DataType::int8_t;
  std::vector<int> runtime_inputs;
  std::vector<QuantizedTensor> inputs;
  QuantizedTensor output;
  std::optional<Conv2dPlan> convolution;
  std::optional<QuantizedSoftmaxPlan> softmax;
  std::vector<int32_t> parameters;
  std::vector<int32_t> kernel, stride;
  std::vector<uint32_t> cropped_shape;
  Int8Range activation{-128,127};
  std::vector<Scale32> scales;
  std::vector<AccumulatorRange> ranges;
  std::vector<int8_t> table;
  std::vector<float> float_weights, float_bias;
  std::vector<uint32_t> weight_shape;
  std::vector<int32_t> padding, dilation;
  uint32_t float_activation = 0;
  int32_t axis = 0;
  int8_t pad_value = 0;
};
std::optional<OperationPlan> AnalyzeOperation(const Op& op, std::string& reason,
    SoftmaxLutType softmax_lut = kLrtArmVulkanMLSoftmaxLutAuto);
tfsc::tosa::ResId EmitOperation(tfsc::tosa::Graph& graph,
    const std::vector<tfsc::tosa::ResId>& inputs, const OperationPlan& plan,
    GraphConstantPool& constants);
}  // namespace litert::arm_vulkan_ml
#endif

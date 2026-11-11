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

#ifndef LITERT_ARM_VULKAN_ML_SPECIAL_OPS_H_
#define LITERT_ARM_VULKAN_ML_SPECIAL_OPS_H_

#include "litert/vendors/arm_vulkan_ml/compiler/operation.h"

namespace litert::arm_vulkan_ml {
// Owned plans: no model handles or FlexBuffer references survive analysis.
struct SpecialPlan {
  enum class Kind {
    kFullyConnected,
    kLayerNorm,
    kRmsNorm,
    kL2Norm,
    kQuantize,
    kDequantize,
    kMinimum,
    kMaximum
  };
  Kind kind;
  tfsc::tosa::DataType input_type, output_type;
  std::vector<tfsc::tosa::DataType> input_types;
  std::vector<QuantizedTensor> inputs;
  QuantizedTensor output;
  std::vector<int8_t> weights;
  std::vector<int64_t> bias;
  std::vector<Scale32> scales;
  std::vector<int64_t> minimum, maximum;
  uint32_t batch = 0, channels = 0, depth = 0;
  Int8Range activation{-128, 127};
  double epsilon = 0;
};
bool IsSpecialOperation(const Op& op);
std::optional<OperationPlan> AnalyzeSpecialOperation(const Op& op,
                                                     std::string& reason);
tfsc::tosa::ResId EmitSpecialOperation(
    tfsc::tosa::Graph& g, const std::vector<tfsc::tosa::ResId>& v,
    const SpecialPlan& p, GraphConstantPool& pool);
}  // namespace litert::arm_vulkan_ml
#endif

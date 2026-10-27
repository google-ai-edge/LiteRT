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

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_CONV2D_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_CONV2D_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "litert/vendors/arm_vulkan_ml/compiler/quantization.h"
#include "tosa/Graph.hpp"

namespace litert::arm_vulkan_ml {

// Normalized NHWC activation / OHWI weight description. Owns constants and
// quantization metadata; no model pointers survive analysis.
struct Conv2dSpec {
  std::array<uint32_t, 4> input_shape;
  std::array<uint32_t, 4> weight_shape;
  std::array<uint32_t, 4> output_shape;
  std::array<int32_t, 4> padding;  // top, bottom, left, right
  std::array<int32_t, 2> stride;
  std::array<int32_t, 2> dilation;
  std::vector<int8_t> weights;
  std::vector<int32_t> bias;
  double input_scale;
  double output_scale;
  int32_t input_zero_point;
  int32_t output_zero_point;
  std::vector<double> weight_scales;
  std::vector<double> bias_scales;
  Activation activation;
  bool depthwise = false;  // Initial path supports depth_multiplier=1.
};

struct Conv2dPlan {
  Conv2dSpec spec;
  std::vector<Scale32> scales;
  std::vector<AccumulatorRange> accumulator_ranges;
  Int8Range activation_range;
  std::array<uint32_t, 4> convolution_input_shape;
  std::array<int32_t, 4> convolution_padding;
};

// Shared legality analysis for partitioning and compilation. Weight/bias zero
// points must already have been checked to be zero by the model adapter.
std::optional<Conv2dPlan> AnalyzeConv2d(Conv2dSpec spec, std::string& reason);

// Constants are ordered exactly as OpGraphConstantARM IDs. The caller retains
// their storage until execution; no TFSC IR is needed to consume these bytes.
struct GraphConstant {
  tfsc::tosa::Tensor tensor;
  std::vector<uint8_t> bytes;
};

// Range-proven requantization; composes base TOSA operations for LiteRT
// rounding, and isolates channels requiring the single-round path.
tfsc::tosa::ResId EmitRescale(tfsc::tosa::Graph& graph, tfsc::tosa::ResId input,
    const std::vector<uint32_t>& shape, int32_t output_zero_point,
    const std::vector<Scale32>& scales, const std::vector<AccumulatorRange>& ranges,
    std::vector<GraphConstant>& constants);

// Whether both stages of the base-profile rounding composition are legal.
bool UsesStagedRounding(const Conv2dPlan& plan);

// Emits CONV2D -> RESCALE -> CLAMP, or a TFSC error ResId. The module owner
// must check both this result and WriteToBinary before publishing an artifact.
// `accumulator` optionally exposes the pre-rescale value for numerical tests.
tfsc::tosa::ResId EmitConv2d(tfsc::tosa::Graph& graph,
                            tfsc::tosa::ResId input,
                            const Conv2dPlan& plan,
                            std::vector<GraphConstant>& constants,
                            tfsc::tosa::ResId* accumulator = nullptr);

}  // namespace litert::arm_vulkan_ml
#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_CONV2D_H_

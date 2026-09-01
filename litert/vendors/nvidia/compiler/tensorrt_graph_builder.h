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

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_TENSORRT_GRAPH_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_TENSORRT_GRAPH_BUILDER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/vendors/nvidia/bytecode.h"

namespace litert::nvidia {

struct TensorRtLlmHeadBuildData {
  uint32_t hidden_output_port = 0;
  uint32_t logits_output_port = 0;
  uint32_t k = 0;
  uint32_t n = 0;
  float soft_cap = 0.0f;
  TensorRtLlmHeadWeightFormat weight_format =
      TensorRtLlmHeadWeightFormat::kInvalid;
  // Aliases the partition model's constant buffer. The compiler serializes
  // this span before the model is released, avoiding a second ~96 MiB host
  // copy for Gemma4's packed INT2 vocabulary head.
  absl::Span<const uint8_t> packed_weights;
  std::vector<uint8_t> bf16_scales;
};

struct TensorRtRefitWeightBuildData {
  std::string name;
  TensorRtWeightDataType data_type = TensorRtWeightDataType::kFloat;
  uint64_t count = 0;
  std::vector<uint8_t> data;
};

struct TensorRtBuildResult {
  std::vector<uint8_t> engine;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::optional<TensorRtLlmHeadBuildData> trtllm_head;
  std::vector<TensorRtRefitWeightBuildData> refit_weights;
  bool is_stripped_plan = false;
};

// Shared-weight mode builds stripped, refittable plans. The compiler plugin
// then stores each unique named weight once in a multi-engine bytecode bundle.
bool TensorRtSharedWeightsEnabled();

bool IsTensorRtOpSupported(const litert::compiler::Op& op);

Expected<TensorRtBuildResult> BuildTensorRtEngine(
    const litert::compiler::Subgraph& subgraph);

}  // namespace litert::nvidia

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_TENSORRT_GRAPH_BUILDER_H_

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

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_BYTECODE_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_BYTECODE_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "litert/cc/litert_expected.h"

namespace litert::nvidia {

inline constexpr uint32_t kTensorRtBytecodeVersion = 1;
inline constexpr uint32_t kTensorRtBytecodeVersionWithTrtLlmHead = 2;
inline constexpr uint32_t kTensorRtBytecodeVersionWithTypedHead = 3;

enum class TensorRtLlmHeadWeightFormat : uint32_t {
  kInvalid = 0,
  kInt4ColumnMajorInterleaved = 1,
  kInt2TfliteRowMajor = 2,
};

// TensorRT-LLM's ColumnMajorInterleaved W4A16 layout is shared by the
// upstream SM80 and SM120 kernels. Hopper and datacenter Blackwell use
// different layouts, so they must fall back to the ordinary TensorRT engine.
inline constexpr bool IsTensorRtLlmHeadComputeCapabilitySupported(
    int compute_capability) {
  return (compute_capability >= 80 && compute_capability < 90) ||
         compute_capability >= 120;
}

// The local row-major W2 kernel carries compute_80 PTX and does not depend on
// TensorRT-LLM's architecture-specific interleaved weight layout.
inline constexpr bool IsInt2GemvComputeCapabilitySupported(
    int compute_capability) {
  return compute_capability >= 80;
}

// Optional TensorRT-LLM weight-only LM-head payload. The pointers alias the
// enclosing bytecode buffer, just like TensorRtBytecode::engine_data, and
// remain valid only while that buffer is alive. BF16 scales are serialized as
// little-endian uint16 bit patterns and exposed as bytes to avoid imposing an
// alignment requirement on the bytecode buffer.
struct TensorRtLlmHead {
  uint32_t hidden_output_port = 0;
  uint32_t logits_output_port = 0;
  uint32_t k = 0;
  uint32_t n = 0;
  float soft_cap = 0.0f;
  TensorRtLlmHeadWeightFormat weight_format =
      TensorRtLlmHeadWeightFormat::kInvalid;
  const uint8_t* packed_weights = nullptr;
  size_t packed_weights_size = 0;
  const uint8_t* bf16_scales = nullptr;
  size_t bf16_scales_size = 0;
};

struct TensorRtBytecode {
  uint32_t version = 0;
  std::string function_name;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  const uint8_t* engine_data = nullptr;
  size_t engine_size = 0;
  std::optional<TensorRtLlmHead> trtllm_head;
};

Expected<std::vector<uint8_t>> PackTensorRtBytecode(
    const std::string& function_name, const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs, const void* engine_data,
    size_t engine_size, const TensorRtLlmHead* trtllm_head = nullptr);

Expected<TensorRtBytecode> ParseTensorRtBytecode(const void* data, size_t size);

}  // namespace litert::nvidia

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_BYTECODE_H_

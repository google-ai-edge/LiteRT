/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_SAFETENSOR_LOADER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_SAFETENSOR_LOADER_H_

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/examples/utils/safetensor_loader.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples {

enum class Gemma3QuantizedLoadMode {
  // Convert INT8 tensors with quantization metadata or side tensors
  // (.scale/.zero_point) to FP32 immediately.
  kDequantizeToFp32,
  // Keep quantized tensors in their quantized storage type and attach Tensor
  // API quantization metadata.
  kPreserveQuantized,
};

inline bool AbslParseFlag(absl::string_view text,
                          Gemma3QuantizedLoadMode* mode,
                          std::string* error) {
  if (text == "fp32" || text == "float") {
    *mode = Gemma3QuantizedLoadMode::kDequantizeToFp32;
    return true;
  }
  if (text == "preserve" || text == "quantized") {
    *mode = Gemma3QuantizedLoadMode::kPreserveQuantized;
    return true;
  }
  *error = "unknown value for enumeration";
  return false;
}

// Returns a textual flag value corresponding to `mode`.
inline std::string AbslUnparseFlag(Gemma3QuantizedLoadMode mode) {
  switch (mode) {
    case Gemma3QuantizedLoadMode::kDequantizeToFp32:
      return "fp32";
    case Gemma3QuantizedLoadMode::kPreserveQuantized:
      return "preserve";
  }
  return "unknown";
}

// Loads a single Gemma3 tensor from `loader`, applying Gemma3 checkpoint
// conventions:
// - BF16 and FP16 tensors are converted to FP32.
// - Zero-centered RMSNorm weights (`*layernorm*`, `*norm.weight`) receive a
//   +1.0f offset.
// - Legacy INT8 side tensors (`.scale`/`.scales`/`.weight_scales` and optional
//   `.zero_point`) are attached or dequantized according to
//   `quantized_load_mode`.
absl::StatusOr<TensorHandle> LoadGemma3Tensor(
    const SafetensorLoader& loader, absl::string_view name,
    Gemma3QuantizedLoadMode quantized_load_mode);

// Loads all tensors into a map using `LoadGemma3Tensor`.
absl::StatusOr<absl::flat_hash_map<std::string, TensorHandle>>
LoadAllGemma3Tensors(const SafetensorLoader& loader,
                     Gemma3QuantizedLoadMode quantized_load_mode);

// Loads weights with HuggingFace-to-model name mapping using
// `LoadGemma3Tensor`.
absl::StatusOr<absl::flat_hash_map<std::string, TensorHandle>>
LoadGemma3WeightsWithMapping(
    const SafetensorLoader& loader,
    const absl::flat_hash_map<std::string, std::string>& name_mapping,
    Gemma3QuantizedLoadMode quantized_load_mode);

// Creates the HuggingFace to model weight name mapping for Gemma3.
absl::flat_hash_map<std::string, std::string> GetGemma3WeightMapping(
    int n_layers);

}  // namespace litert::tensor::examples

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_SAFETENSOR_LOADER_H_

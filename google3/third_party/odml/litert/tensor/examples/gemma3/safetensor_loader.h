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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
// copybara:uncomment_begin(google_only)
// #include "third_party/odml/litert/tensor/backends/xnnpack/arithmetic.h"
// copybara:uncomment_end
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/examples/gemma3/safetensors.h"
#include "third_party/odml/litert/tensor/tensor.h"

namespace litert::tensor::examples {

struct TensorStorageInfo {
  std::shared_ptr<void> file_data;
  const std::byte* data_base = nullptr;
  size_t data_size = 0;
};

// Tensor metadata from safetensor file.
struct SafetensorTensorInfo {
  std::string name;
  safetensors::dtype dtype;
  std::vector<int64_t> shape;
  size_t data_start;  // Offset in data section
  size_t data_end;    // End offset in data section
  std::shared_ptr<TensorStorageInfo> storage;
};

// SafeTensor file loader using safetensors-cpp library.
// Supports loading tensors from HuggingFace safetensor format.
class SafetensorLoader {
 public:
  enum class QuantizedLoadMode {
    // Convert INT8 tensors with quantization side tensors (.scale/.zero_point)
    // to FP32 immediately.
    kDequantizeToFp32,
    // Keep quantized tensors as INT8 and attach Tensor API quantization
    // metadata.
    kPreserveQuantized,
  };

  // Loads a safetensor file or a directory of safetensor files.
  static absl::StatusOr<SafetensorLoader> Load(const std::string& path);

  // Gets list of all tensor names.
  std::vector<std::string> GetTensorNames() const;

  // Gets tensor info by name.
  absl::StatusOr<SafetensorTensorInfo> GetTensorInfo(
      const std::string& name) const;

  // Loads a tensor.
  //
  // BF16 tensors are automatically converted to FP32.
  absl::StatusOr<TensorHandle> LoadTensor(
      const std::string& name, QuantizedLoadMode quantized_load_mode) const;

  // Loads all tensors into a map.
  absl::StatusOr<absl::flat_hash_map<std::string, TensorHandle>> LoadAllTensors(
      QuantizedLoadMode quantized_load_mode) const;

  // Loads weights with name mapping.
  // Converts HuggingFace weight names to model weight names.
  absl::StatusOr<absl::flat_hash_map<std::string, TensorHandle>>
  LoadWeightsWithMapping(
      const absl::flat_hash_map<std::string, std::string>& name_mapping,
      QuantizedLoadMode quantized_load_mode) const;

 private:
  SafetensorLoader() = default;

  // Loads a single safetensor file and appends its tensors.
  absl::Status AddSafetensorFile(const std::string& path);

  // Convert safetensor dtype enum to Type enum.
  static absl::StatusOr<Type> DtypeToType(safetensors::dtype dtype);

  // Map of tensor name to metadata.
  absl::flat_hash_map<std::string, SafetensorTensorInfo> tensor_infos_;
};

inline bool AbslParseFlag(absl::string_view text,
                          SafetensorLoader::QuantizedLoadMode* mode,
                          std::string* error) {
  if (text == "fp32" || text == "float") {
    *mode = SafetensorLoader::QuantizedLoadMode::kDequantizeToFp32;
    return true;
  }
  if (text == "preserve" || text == "quantized") {
    *mode = SafetensorLoader::QuantizedLoadMode::kPreserveQuantized;
    return true;
  }
  *error = "unknown value for enumeration";
  return false;
}

// AbslUnparseFlag converts from an OutputMode to a string.
// Must be in same namespace as OutputMode.

// Returns a textual flag value corresponding to the OutputMode `mode`.
inline std::string AbslUnparseFlag(SafetensorLoader::QuantizedLoadMode mode) {
  switch (mode) {
    case SafetensorLoader::QuantizedLoadMode::kDequantizeToFp32:
      return "fp32";
    case SafetensorLoader::QuantizedLoadMode::kPreserveQuantized:
      return "preserve";
  }
}

// Creates the HuggingFace to model weight name mapping for Gemma3.
absl::flat_hash_map<std::string, std::string> GetGemma3WeightMapping(
    int n_layers);

}  // namespace litert::tensor::examples

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_SAFETENSOR_LOADER_H_

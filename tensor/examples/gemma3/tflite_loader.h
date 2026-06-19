/* Copyright 2026 Google LLC.

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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_TFLITE_LOADER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_TFLITE_LOADER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tflite/core/model_builder.h"

namespace litert::tensor::examples {

struct TfliteTensorInfo {
  std::string name;
  Type type;
  std::vector<int64_t> shape;
  const std::byte* data = nullptr;
  size_t data_size = 0;
  std::shared_ptr<Quantization> quantization;
  std::shared_ptr<tflite::FlatBufferModel> model_keep_alive;
};

struct TfliteWeightMappingEntry {
  std::string tflite_tensor_name;
  // If not empty, specifies the range of rows to slice [start_row, end_row].
  // Slicing is only supported along the first dimension (dim 0).
  std::vector<int> slice_range;
};

using TfliteWeightMapping =
    absl::flat_hash_map<std::string, TfliteWeightMappingEntry>;

class TfliteLoader {
 public:
  enum class QuantizedLoadMode {
    // Convert quantized tensors to FP32 immediately.
    kDequantizeToFp32,
    // Keep quantized tensors as is and attach quantization metadata.
    kPreserveQuantized,
  };

  // Loads a TFLite file.
  static absl::StatusOr<TfliteLoader> Load(const std::string& path);

  // Gets list of all tensor names.
  std::vector<std::string> GetTensorNames() const;

  // Gets tensor info by name.
  absl::StatusOr<TfliteTensorInfo> GetTensorInfo(const std::string& name) const;

  // Loads a tensor.
  absl::StatusOr<TensorHandle> LoadTensor(
      const std::string& name, QuantizedLoadMode quantized_load_mode) const;

  // Loads a slice of a tensor along the first dimension (dim 0).
  absl::StatusOr<TensorHandle> LoadTensorWithSlice(
      const std::string& name, int start_row, int end_row,
      QuantizedLoadMode quantized_load_mode) const;

  // Loads all tensors into a map.
  absl::StatusOr<absl::flat_hash_map<std::string, TensorHandle>> LoadAllTensors(
      QuantizedLoadMode quantized_load_mode) const;

  // Loads weights with name mapping and optional slicing.
  absl::StatusOr<absl::flat_hash_map<std::string, TensorHandle>>
  LoadWeightsWithMapping(const TfliteWeightMapping& name_mapping,
                         QuantizedLoadMode quantized_load_mode) const;

 private:
  TfliteLoader() = default;

  std::shared_ptr<tflite::FlatBufferModel> model_;
  absl::flat_hash_map<std::string, TfliteTensorInfo> tensor_infos_;
};

inline bool AbslParseFlag(absl::string_view text,
                          TfliteLoader::QuantizedLoadMode* mode,
                          std::string* error) {
  if (text == "fp32" || text == "float") {
    *mode = TfliteLoader::QuantizedLoadMode::kDequantizeToFp32;
    return true;
  }
  if (text == "preserve" || text == "quantized") {
    *mode = TfliteLoader::QuantizedLoadMode::kPreserveQuantized;
    return true;
  }
  *error = "unknown value for enumeration";
  return false;
}

inline std::string AbslUnparseFlag(TfliteLoader::QuantizedLoadMode mode) {
  switch (mode) {
    case TfliteLoader::QuantizedLoadMode::kDequantizeToFp32:
      return "fp32";
    case TfliteLoader::QuantizedLoadMode::kPreserveQuantized:
      return "preserve";
  }
}

// Creates the TFLite to model weight name mapping for Gemma3 1B.
TfliteWeightMapping GetGemma3TfliteWeightMapping(int n_layers);

}  // namespace litert::tensor::examples

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_TFLITE_LOADER_H_

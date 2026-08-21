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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_SAFETENSOR_LOADER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_SAFETENSOR_LOADER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/datatypes.h"
#include "tensor/examples/utils/safetensors.h"
#include "tensor/tensor.h"

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

// Quantization configuration extracted from header metadata.
struct QuantizationConfig {
  enum class Method {
    kUnknown = 0,
    kCompressedTensors,
  };

  enum class Format {
    kUnknown = 0,
    kPackQuantized,
    kIntQuantized,
  };

  Method quant_method = Method::kUnknown;
  Format format = Format::kUnknown;
  int num_bits = 0;    // Required: 4 or 8 (must be read from metadata)
  int group_size = 0;  // Required for pack-quantized: block size (must be read
                       // from metadata)
  bool symmetric = true;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, Method method) {
    switch (method) {
      case Method::kCompressedTensors:
        sink.Append("compressed-tensors");
        break;
      default:
        sink.Append("unknown");
        break;
    }
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, Format format) {
    switch (format) {
      case Format::kPackQuantized:
        sink.Append("pack-quantized");
        break;
      case Format::kIntQuantized:
        sink.Append("int-quantized");
        break;
      default:
        sink.Append("unknown");
        break;
    }
  }
};

// SafeTensor file loader using safetensors-cpp library.
// Supports loading tensors from HuggingFace safetensor format.
class SafetensorLoader {
 public:
  // Loads a safetensor file or a directory of safetensor files.
  static absl::StatusOr<SafetensorLoader> Load(const std::string& path);

  // Gets list of all tensor names.
  std::vector<std::string> GetTensorNames() const;

  // Gets tensor info by name.
  absl::StatusOr<SafetensorTensorInfo> GetTensorInfo(
      absl::string_view name) const;

  // Gets quantization config if header contains
  // __metadata__["quantization_config"].
  const std::optional<QuantizationConfig>& GetQuantizationConfig() const {
    return quant_config_;
  }

  // Loads a tensor.
  //
  // BF16 tensors are automatically converted to FP32.
  absl::StatusOr<TensorHandle> LoadTensor(absl::string_view name) const;

  // Loads all tensors into a map.
  absl::StatusOr<absl::flat_hash_map<std::string, TensorHandle>>
  LoadAllTensors() const;

  // Loads weights with name mapping.
  // Converts HuggingFace weight names to model weight names.
  absl::StatusOr<absl::flat_hash_map<std::string, TensorHandle>>
  LoadWeightsWithMapping(
      const absl::flat_hash_map<std::string, std::string>& name_mapping) const;

 private:
  SafetensorLoader() = default;

  // Loads a single safetensor file and appends its tensors.
  absl::Status AddSafetensorFile(const std::string& path);

  // Convert safetensor dtype enum to Type enum.
  static absl::StatusOr<Type> DtypeToType(safetensors::dtype dtype);

  // Map of tensor name to metadata.
  absl::flat_hash_map<std::string, SafetensorTensorInfo> tensor_infos_;

  // Quantization config from header metadata.
  std::optional<QuantizationConfig> quant_config_;
};

// Creates the HuggingFace to model weight name mapping for Gemma4.
absl::flat_hash_map<std::string, std::string> GetGemma4WeightMapping(
    int n_layers);

}  // namespace litert::tensor::examples

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_SAFETENSOR_LOADER_H_

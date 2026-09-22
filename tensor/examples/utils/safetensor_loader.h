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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_UTILS_SAFETENSOR_LOADER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_UTILS_SAFETENSOR_LOADER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <regex>  // NOLINT
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
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

// Quantization configuration.
struct QuantizationConfig {
  enum class Method {
    kUnknown = 0,
    kCompressedTensors,
  };

  // How quantization parameters (scales/zero points) are shared across the
  // elements of a weight.
  enum class Strategy {
    kUnknown = 0,
    kTensor,   // Per tensor scale.
    kChannel,  // Per channel scale.
    kGroup,    // One scale per `group_size` elements along the input dim.
  };

  // Parameters of a single `config_groups` entry, together with the modules it
  // applies to.
  struct Scheme {
    Strategy strategy = Strategy::kUnknown;
    int num_bits = 0;
    int group_size = 0;  // Only applies when `strategy` is `kGroup`.
    bool symmetric = true;

    // Modules this scheme applies to, named exactly as in the checkpoint.
    absl::flat_hash_set<std::string> modules;

    // Compiled "re:"-prefixed target patterns.
    std::vector<std::regex> patterns;

    // True when the scheme names no specific module, either because it lists
    // no targets at all or because its targets are module *classes* such as
    // "Linear". Such a scheme applies to every quantized module.
    bool matches_any_module = false;

    // Returns whether this scheme applies to `module`.
    bool Matches(absl::string_view module) const;
  };

  Method quant_method = Method::kUnknown;

  // Parameters of the first config group, for callers that assume a single
  // model-wide scheme. Prefer `FindScheme()`.
  int num_bits = 0;
  int group_size = 0;
  bool symmetric = true;

  std::vector<Scheme> schemes;

  // Modules excluded from quantization.
  //
  // Elements are matched against full module path or as a prefix or suffix.
  std::vector<std::string> ignore;

  // Modules excluded from quantization by regular expressions.
  std::vector<std::regex> ignore_regexes;

  // Returns the scheme that applies to `module`, or nullptr when the module is
  // ignored or when no scheme claims it.
  const Scheme* FindScheme(absl::string_view module) const;

  // Returns whether `module` appears in the configuration `ignore` list.
  bool IsIgnored(absl::string_view module) const;

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
  friend void AbslStringify(Sink& sink, Strategy strategy) {
    switch (strategy) {
      case Strategy::kTensor:
        sink.Append("tensor");
        break;
      case Strategy::kChannel:
        sink.Append("channel");
        break;
      case Strategy::kGroup:
        sink.Append("group");
        break;
      default:
        sink.Append("unknown");
        break;
    }
  }
};

// SafeTensor file loader using safetensors-cpp library.
// Supports loading tensors from HuggingFace safetensor format, including
// weights compressed with the `compressed-tensors` library.
class SafetensorLoader {
 public:
  // Loads a safetensor file or a directory of safetensor files.
  //
  // The quantization config is read from the safetensors header metadata when
  // present, and otherwise from a `config.json` sitting next to the weights.
  static absl::StatusOr<SafetensorLoader> Load(const std::string& path);

  // Gets list of all tensor names.
  std::vector<std::string> GetTensorNames() const;

  // Gets tensor info by name.
  absl::StatusOr<SafetensorTensorInfo> GetTensorInfo(
      absl::string_view name) const;

  // Gets the quantization config, if the checkpoint declares one.
  const std::optional<QuantizationConfig>& GetQuantizationConfig() const {
    return quant_config_;
  }

  // Loads a tensor.
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

  // Reads `quantization_config` from a HuggingFace `config.json`. Returns
  // `absl::NotFoundError` if the file does not exist.
  absl::Status AddQuantizationConfigFromJsonFile(const std::string& path);

  // Convert safetensor dtype enum to Type enum.
  static absl::StatusOr<Type> DtypeToType(safetensors::dtype dtype);

  // Map of tensor name to metadata.
  absl::flat_hash_map<std::string, SafetensorTensorInfo> tensor_infos_;

  // Quantization config from the header metadata or from `config.json`.
  std::optional<QuantizationConfig> quant_config_;
};

}  // namespace litert::tensor::examples

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_UTILS_SAFETENSOR_LOADER_H_

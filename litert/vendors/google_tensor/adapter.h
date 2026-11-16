// Copyright 2024 Google LLC.
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
#ifndef ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_ADAPTER_H_
#define ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_ADAPTER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"

namespace litert::google_tensor {

// Represents an unsupported operation with its index and rejection reason.
struct UnsupportedOp {
  int32_t op_index;
  std::string reason;
};

// Describes a composite operation, including its name and the element types of
// its operands and results, for support validation.
struct CompositeOpDescriptor {
  std::string name;
  std::vector<LiteRtElementType> operand_types;
  std::vector<LiteRtElementType> result_types;
};

// This class adapts the google tensor compiler API for dynamic loading.
class Adapter {
 public:
  // A smart pointer for managing TensorAdapter objects.
  using Ptr = std::unique_ptr<Adapter>;

  Adapter() = default;
  virtual ~Adapter() = default;

  // Creates a new TensorAdapter and loads the compiler API symbols.
  static litert::Expected<Ptr> Create(
      std::optional<std::string> shared_library_dir);

  virtual Expected<void> Compile(const char* tfl_buffer_data,
                                 size_t tfl_buffer_size, const char* options,
                                 size_t options_size,
                                 char*** compiled_code_data,
                                 size_t** compiled_code_sizes,
                                 size_t* num_bytecodes) = 0;

  // Returns true if the adapter is for AOT compilation, false for ODC.
  virtual bool IsAot() const = 0;

  virtual void FreeCompiledCode(char** compiled_code_data,
                                size_t* compiled_code_sizes,
                                size_t num_bytecodes) = 0;

  // Checks which operations in the TFLite flatbuffer are supported/unsupported.
  // Returns a vector of unsupported operations with indices and reasons.
  virtual Expected<std::vector<UnsupportedOp>> GetUnsupportedOps(
      const char* tfl_buffer_data, size_t tfl_buffer_size, const char* options,
      size_t options_size) = 0;

  // Checks whether each composite op in `composite_ops` (including operand and
  // result element types) is supported under the given serialized `options`
  // (may be nullptr when `options_size == 0`).
  // Returns `kLiteRtStatusErrorUnsupported` by default for Adapter subclasses
  // that do not support standalone composite validation (e.g., `AdapterOdc`).
  virtual Expected<std::vector<bool>> AreCompositesSupported(
      absl::Span<const CompositeOpDescriptor> composite_ops,
      const char* options, size_t options_size) {
    return litert::Unexpected(kLiteRtStatusErrorUnsupported,
                              "AreCompositesSupported is not supported");
  }

  // Checks if multiple composite ops are supported without options.
  Expected<std::vector<bool>> AreCompositesSupported(
      absl::Span<const CompositeOpDescriptor> composite_ops) {
    return AreCompositesSupported(composite_ops, nullptr, 0);
  }

  // Checks whether each composite op in `composite_names` is supported by op
  // name only (without validating operand or result element types). Prefer the
  // `CompositeOpDescriptor` overload above when tensor element types are
  // available.
  virtual Expected<std::vector<bool>> AreCompositesSupported(
      absl::Span<const std::string> composite_names, const char* options,
      size_t options_size) {
    std::vector<CompositeOpDescriptor> descriptors;
    descriptors.reserve(composite_names.size());
    for (const std::string& name : composite_names) {
      descriptors.push_back(CompositeOpDescriptor{.name = name});
    }
    return AreCompositesSupported(descriptors, options, options_size);
  }

  // Checks composite op support by name only, without compiler options.
  Expected<std::vector<bool>> AreCompositesSupported(
      absl::Span<const std::string> composite_names) {
    return AreCompositesSupported(composite_names, nullptr, 0);
  }
};

}  // namespace litert::google_tensor

#endif  // ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_ADAPTER_H_

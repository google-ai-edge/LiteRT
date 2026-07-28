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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_TEST_UTIL_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_TEST_UTIL_H_

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"

namespace litert::internal {

// A highly configurable mock shape inference context that can be used across
// op unit tests to replace duplicated test contexts.
class MockShapeInferenceContext : public ShapeInferenceContext {
 public:
  struct TensorData {
    Dims shape;
    std::vector<uint8_t> data;
    LiteRtElementType element_type = kLiteRtElementTypeNone;
  };

  explicit MockShapeInferenceContext(LiteRtOpCode op_code,
                                     std::vector<TensorData> inputs = {},
                                     TflOptions options = {},
                                     size_t num_outputs = 1)
      : op_code_(op_code),
        inputs_(std::move(inputs)),
        options_(std::move(options)),
        num_outputs_(num_outputs) {}

  size_t GetNumInputs() const override { return inputs_.size(); }
  size_t GetNumOutputs() const override { return num_outputs_; }

  Dims GetInputShape(size_t index) const override {
    if (index >= inputs_.size()) return {};
    return inputs_[index].shape;
  }

  absl::Span<const uint8_t> GetInputData(size_t index) const override {
    if (index >= inputs_.size()) return {};
    return absl::MakeConstSpan(inputs_[index].data);
  }

  LiteRtElementType GetInputElementType(size_t index) const override {
    if (index >= inputs_.size()) return kLiteRtElementTypeNone;
    if (inputs_[index].element_type != kLiteRtElementTypeNone) {
      return inputs_[index].element_type;
    }
    // Heuristic: deduce element type based on buffer size and shape
    if (!inputs_[index].shape.empty() && inputs_[index].shape[0] > 0) {
      size_t num_elems = 1;
      for (auto d : inputs_[index].shape) {
        if (d < 0) {
          num_elems = 0;
          break;
        }
        num_elems *= static_cast<size_t>(d);
      }
      if (num_elems > 0 &&
          inputs_[index].data.size() == num_elems * sizeof(int64_t)) {
        return kLiteRtElementTypeInt64;
      }
    }
    return kLiteRtElementTypeInt32;
  }

  const TflOptions& GetOptions() const override { return options_; }
  LiteRtOpCode GetOpCode() const override { return op_code_; }

 private:
  LiteRtOpCode op_code_;
  std::vector<TensorData> inputs_;
  TflOptions options_;
  size_t num_outputs_;
};

// Helper utility to construct a mock TensorData struct with serialized bytes.
template <typename T>
inline MockShapeInferenceContext::TensorData MakeTensorData(
    Dims shape, const std::vector<T>& data,
    LiteRtElementType element_type = kLiteRtElementTypeNone) {
  size_t size = data.size() * sizeof(T);
  std::vector<uint8_t> byte_data(size);
  std::memcpy(byte_data.data(), data.data(), size);
  return {std::move(shape), std::move(byte_data), element_type};
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_TEST_UTIL_H_

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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TESTING_NUMERICAL_TEST_BRIDGE_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TESTING_NUMERICAL_TEST_BRIDGE_H_

#include <cstddef>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/tensor.h"

namespace litert::tensor {

class TestBackendBridge {
 public:
  virtual ~TestBackendBridge() = default;

  // Initializes the backend environment.
  virtual absl::Status Initialize() = 0;

  // Compiles and lowers the LiteRT Tensor Graph to the backend format.
  virtual absl::Status BuildGraph(absl::Span<const TensorHandle> inputs,
                                  absl::Span<const TensorHandle> outputs) = 0;

  // Writes input data to the backend memory.
  virtual absl::Status SetInput(const TensorHandle& tensor,
                                absl::Span<const std::byte> data) = 0;

  // Runs the backend graph/model.
  virtual absl::Status Execute() = 0;

  // Reads back output data from the backend memory.
  virtual absl::Status GetOutput(const TensorHandle& tensor,
                                 absl::Span<std::byte> data) = 0;
};

// Helper to convert std::vector data to absl::Span of bytes.
template <typename T>
absl::Span<const std::byte> AsBytes(const std::vector<T>& data) {
  return absl::MakeConstSpan(reinterpret_cast<const std::byte*>(data.data()),
                             data.size() * sizeof(T));
}

template <typename T>
absl::Span<std::byte> AsBytes(std::vector<T>& data) {
  return absl::MakeSpan(reinterpret_cast<std::byte*>(data.data()),
                        data.size() * sizeof(T));
}

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TESTING_NUMERICAL_TEST_BRIDGE_H_

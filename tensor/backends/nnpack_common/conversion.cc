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

#include "tensor/backends/nnpack_common/conversion.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

absl::StatusOr<std::vector<float>> DequantizeInt8ConstantTensor(
    const graph::TensorInformation& info,
    absl::Span<const std::byte> raw_data) {
  if (info.quantization == nullptr) {
    return absl::InvalidArgumentError("Missing quantization parameters");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const auto& pcq,
      info.quantization->As<const graph::PerChannelAffineQuantization>());

  if (info.shape.size() != 2) {
    return absl::InvalidArgumentError("Only 2D weights currently supported");
  }
  const size_t out_channels = info.shape[0];
  const size_t in_channels = info.shape[1];
  const size_t num_elements = out_channels * in_channels;

  if (raw_data.size() < num_elements) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Raw data size %zu is smaller than expected %zu",
                        raw_data.size(), num_elements));
  }

  std::vector<float> dequantized(num_elements);
  const int8_t* quantized_data =
      reinterpret_cast<const int8_t*>(raw_data.data());

  for (size_t o = 0; o < out_channels; ++o) {
    const float scale = pcq.scales[o];
    const int32_t zero_point =
        pcq.zero_points.empty() ? 0 : static_cast<int32_t>(pcq.zero_points[o]);
    const size_t row_offset = o * in_channels;
    for (size_t i = 0; i < in_channels; ++i) {
      const size_t idx = row_offset + i;
      dequantized[idx] =
          static_cast<float>(quantized_data[idx] - zero_point) * scale;
    }
  }

  return dequantized;
}

absl::Status TopologicalSort(
    graph::Tensor output,
    const absl::flat_hash_set<graph::Tensor>& inlined_inputs,
    absl::flat_hash_set<graph::Tensor>& visited_tensors,
    absl::flat_hash_set<const graph::Operation*>& visited_ops,
    std::vector<const graph::Operation*>& ordered_ops) {
  using StackEntry = std::variant<graph::Tensor, const graph::Operation*>;
  std::vector<StackEntry> stack;
  stack.push_back(output);

  while (!stack.empty()) {
    StackEntry entry = std::move(stack.back());
    stack.pop_back();

    if (std::holds_alternative<const graph::Operation*>(entry)) {
      ordered_ops.push_back(std::get<const graph::Operation*>(entry));
      continue;
    }

    graph::Tensor tensor = std::get<graph::Tensor>(entry);

    if (inlined_inputs.contains(tensor)) {
      continue;
    }
    if (!visited_tensors.insert(tensor).second) {
      continue;
    }

    LRT_TENSOR_ASSIGN_OR_RETURN(std::shared_ptr<graph::Operation> producer,
                                graph::GetProducer(tensor));
    if (producer == nullptr) {
      continue;
    }

    if (!visited_ops.insert(producer.get()).second) {
      continue;
    }

    stack.push_back(producer.get());

    for (auto it = producer->inputs.rbegin(); it != producer->inputs.rend();
         ++it) {
      stack.push_back(*it);
    }
  }

  return absl::OkStatus();
}

}  // namespace litert::tensor

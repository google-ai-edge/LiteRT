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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_COMMON_NNPACK_GRAPH_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_COMMON_NNPACK_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"

namespace litert::tensor {

struct NnpackValue {
  graph::TensorInformation info;
  uint32_t id = UINT32_MAX;
  uint32_t flags = 0;
  // True when this static constant is consumed exclusively as a packed weight
  // (e.g., a static `FullyConnected` filter) and its raw host buffer is no
  // longer read after runtime creation.
  bool packed_only = false;
  LockedBufferSpan<const std::byte> data =
      LockedBufferSpan<const std::byte>::Empty();
};

class NnpackGraph {
 public:
  NnpackGraph() = default;
  virtual ~NnpackGraph() = default;

  NnpackGraph(NnpackGraph&&) = default;
  NnpackGraph& operator=(NnpackGraph&&) = default;
  NnpackGraph(const NnpackGraph&) = delete;
  NnpackGraph& operator=(const NnpackGraph&) = delete;

  std::vector<NnpackValue>& mutable_values() { return values_; }
  const std::vector<NnpackValue>& values() const { return values_; }

  absl::flat_hash_map<graph::Tensor, size_t>& mutable_tensor_index() {
    return tensor_index_;
  }
  const absl::flat_hash_map<graph::Tensor, size_t>& tensor_index() const {
    return tensor_index_;
  }

  absl::flat_hash_set<graph::Tensor>& mutable_external_outputs() {
    return external_outputs_;
  }
  const absl::flat_hash_set<graph::Tensor>& external_outputs() const {
    return external_outputs_;
  }

  std::vector<std::vector<float>>& dequantized_buffers() {
    return dequantized_buffers_;
  }
  const std::vector<std::vector<float>>& dequantized_buffers() const {
    return dequantized_buffers_;
  }

  std::vector<std::vector<fp16_t>>& fp16_buffers() { return fp16_buffers_; }
  const std::vector<std::vector<fp16_t>>& fp16_buffers() const {
    return fp16_buffers_;
  }

  // Retains shared ownership of cached BF16 blockwise scales across the
  // lifetime of the lowered XNNPACK/YNNPACK subgraph. XNNPACK's blockwise
  // 4-bit quantized dot microkernels (`qb4w`) natively require
  // `xnn_datatype_bf16` scales (via
  // `xnn_define_blockwise_quantized_tensor_value_v2`), which halve scale
  // bandwidth vs. FP32 while preserving FP32's 8-bit exponent dynamic range.
  std::vector<std::shared_ptr<const std::vector<bf16_t>>>& bf16_scale_refs() {
    return bf16_scale_refs_;
  }
  const std::vector<std::shared_ptr<const std::vector<bf16_t>>>&
  bf16_scale_refs() const {
    return bf16_scale_refs_;
  }

  std::vector<
      std::shared_ptr<const BlockwiseQuantization::CpuSymmetricFallback>>&
  cpu_fallback_refs() {
    return cpu_fallback_refs_;
  }
  const std::vector<
      std::shared_ptr<const BlockwiseQuantization::CpuSymmetricFallback>>&
  cpu_fallback_refs() const {
    return cpu_fallback_refs_;
  }

  std::vector<std::vector<char>>& constant_buffers() {
    return constant_buffers_;
  }
  const std::vector<std::vector<char>>& constant_buffers() const {
    return constant_buffers_;
  }

  std::vector<std::shared_ptr<Buffer>>& keep_alive_buffers() {
    return keep_alive_buffers_;
  }
  const std::vector<std::shared_ptr<Buffer>>& keep_alive_buffers() const {
    return keep_alive_buffers_;
  }

  absl::StatusOr<size_t> Lookup(const TensorHandle& tensor) const;

 protected:
  std::vector<NnpackValue> values_;
  absl::flat_hash_map<graph::Tensor, size_t> tensor_index_;
  absl::flat_hash_set<graph::Tensor> external_outputs_;
  std::vector<std::vector<float>> dequantized_buffers_;
  std::vector<std::vector<fp16_t>> fp16_buffers_;
  std::vector<std::shared_ptr<const std::vector<bf16_t>>> bf16_scale_refs_;
  std::vector<
      std::shared_ptr<const BlockwiseQuantization::CpuSymmetricFallback>>
      cpu_fallback_refs_;
  std::vector<std::vector<char>> constant_buffers_;
  std::vector<std::shared_ptr<Buffer>> keep_alive_buffers_;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_COMMON_NNPACK_GRAPH_H_

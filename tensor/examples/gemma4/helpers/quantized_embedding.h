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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_HELPERS_QUANTIZED_EMBEDDING_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_HELPERS_QUANTIZED_EMBEDDING_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::gemma4 {

// Embedding table base interface.
//
// The interface provides batch and single token lookups that are used
// respectively during the prefill and the decode phase.
class GemmaEmbeddingTable {
 public:
  virtual ~GemmaEmbeddingTable() = default;

  // Creates the concrete embedding table depending on the tensor type.
  static absl::StatusOr<std::unique_ptr<GemmaEmbeddingTable>> Create(
      TensorHandle tensor, int expected_emb_dim = 0);

  int VocabSize() const { return vocab_size_; }
  int EmbeddingDim() const { return emb_dim_; }

  // Batch lookup for multiple token IDs (used in prefill pass).
  virtual absl::Status Lookup(absl::Span<const int32_t> token_ids,
                              absl::Span<float> output) const;

  // Single-token lookup (used in decode step).
  virtual absl::StatusOr<LockedBufferSpan<const float>> Lookup(
      int32_t token_id) const = 0;

  // Batch per-layer lookup for multiple token IDs (used in prefill pass).
  virtual absl::Status LookupPerLayer(
      absl::Span<const int32_t> token_ids, int num_layers, int per_layer_dim,
      absl::Span<std::vector<float>> output_per_layer) const;

  // Single-token per-layer lookup (used in decode step).
  virtual absl::StatusOr<std::vector<LockedBufferSpan<const float>>>
  LookupPerLayer(int32_t token_id, int num_layers, int per_layer_dim) const;

 protected:
  GemmaEmbeddingTable(TensorHandle tensor, int vocab_size, int emb_dim,
                      Type type);

  virtual void DecodeRow(int32_t row, int col_start, int num_cols,
                         float* dst) const = 0;

  int32_t ClampTokenId(int32_t token_id) const;

  TensorHandle tensor_;
  int vocab_size_ = 0;
  int emb_dim_ = 0;
  Type type_ = Type::kUnknown;
};

}  // namespace litert::tensor::examples::gemma4

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_HELPERS_QUANTIZED_EMBEDDING_H_

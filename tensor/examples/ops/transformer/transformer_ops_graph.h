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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_OPS_TRANSFORMER_TRANSFORMER_OPS_GRAPH_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_OPS_TRANSFORMER_TRANSFORMER_OPS_GRAPH_H_

#include <optional>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/internal/graph.h"

namespace litert::tensor::graph {

struct RotaryEmbeddingOperationData {
  float min_timescale;
  float max_timescale;
};

struct RotaryEmbeddingOperation : public RotaryEmbeddingOperationData,
                                  Operation {
  absl::string_view GetName() const override { return "RotaryEmbedding"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct RmsNormOperationData {
  float epsilon;
};

struct RmsNormOperation : public RmsNormOperationData, Operation {
  absl::string_view GetName() const override { return "RmsNorm"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct FillSegmentPosOperationData {
  int param_index;
};

struct FillSegmentPosOperation : public FillSegmentPosOperationData, Operation {
  absl::string_view GetName() const override { return "FillSegmentPos"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct FillAttentionMaskOperationData {
  bool is_local;
  int sliding_window_size;
};

struct FillRopeCosSinOperationData {
  float rope_base;
};

struct FillAttentionMaskOperation : public FillAttentionMaskOperationData,
                                    Operation {
  absl::string_view GetName() const override { return "FillAttentionMask"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct FillRopeCosSinOperation : public FillRopeCosSinOperationData, Operation {
  absl::string_view GetName() const override { return "FillRopeCosSin"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct AddValuesToKvCacheOperationData {
  int token_index_offset;
  int active_tokens;
  int num_of_kv_heads;
  int kv_cache_batch_size;
  int cache_size;
  int head_dimension;
};

struct AddValuesToKvCacheOperation : public AddValuesToKvCacheOperationData,
                                     Operation {
  absl::string_view GetName() const override { return "AddValuesToKvCache"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct AddValuesToCacheOperationData {
  int token_index_offset;
  int active_tokens;
  int num_of_kv_heads;
  int kv_cache_batch_size;
  int cache_size;
  int head_dimension;
};

struct AddValuesToCacheOperation : public AddValuesToCacheOperationData,
                                   Operation {
  absl::string_view GetName() const override { return "AddValuesToCache"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct QkNormOperation : Operation {
  absl::string_view GetName() const override { return "QkNorm"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct ActivationSparsityOperationData {
  float stddev_multiplier;
};

struct ActivationSparsityOperation : public ActivationSparsityOperationData,
                                     Operation {
  absl::string_view GetName() const override { return "ActivationSparsity"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct SelectMaskOperation : Operation {
  absl::string_view GetName() const override { return "SelectMask"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct WriteCurrentTokensOperation : Operation {
  absl::string_view GetName() const override { return "WriteCurrentTokens"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct MatMulWithCacheOperationData {
  bool is_v = false;
  bool is_local = false;
  int sliding_window_size = 0;
};

struct MatMulWithCacheOperation : public MatMulWithCacheOperationData,
                                  Operation {
  absl::string_view GetName() const override { return "MatMulWithCache"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

struct SoftmaxWithRuntimeCheckOperationData {
  std::optional<int> start_ch_index;
  std::optional<int> end_ch_index;
};

struct SoftmaxWithRuntimeCheckOperation
    : public SoftmaxWithRuntimeCheckOperationData,
      Operation {
  absl::string_view GetName() const override {
    return "SoftmaxWithRuntimeCheck";
  }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

}  // namespace litert::tensor::graph

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_OPS_TRANSFORMER_TRANSFORMER_OPS_GRAPH_H_

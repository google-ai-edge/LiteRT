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
#include "tensor/internal/mixin.h"

namespace litert::tensor::graph {

struct RotaryEmbeddingOperationData {
  float min_timescale;
  float max_timescale;
};

template <class... Mixins>
struct RotaryEmbeddingOperation
    : public RotaryEmbeddingOperationData,
      virtual Operation,
      virtual OpMixin<struct RotaryEmbeddingOperationTag, Mixins...> {
 public:
  absl::string_view GetName() const override { return "RotaryEmbedding"; }
};

struct RmsNormOperationData {
  float epsilon;
};

template <class... Mixins>
struct RmsNormOperation : public RmsNormOperationData,
                          virtual Operation,
                          virtual OpMixin<struct RmsNormOperationTag, Mixins...>
                          {
 public:
  absl::string_view GetName() const override { return "RmsNorm"; }
};

struct FillSegmentPosOperationData {
  int param_index;
};

template <class... Mixins>
struct FillSegmentPosOperation
    : public FillSegmentPosOperationData,
      virtual Operation,
      virtual OpMixin<struct FillSegmentPosOperationTag, Mixins...> {
 public:
  absl::string_view GetName() const override { return "FillSegmentPos"; }
};

struct FillAttentionMaskOperationData {
  bool is_local;
  int sliding_window_size;
};

struct FillRopeCosSinOperationData {
  float rope_base;
};

template <class... Mixins>
struct FillAttentionMaskOperation
    : public FillAttentionMaskOperationData,
      virtual Operation,
      virtual OpMixin<struct FillAttentionMaskOperationTag, Mixins...> {
 public:
  absl::string_view GetName() const override { return "FillAttentionMask"; }
};

template <class... Mixins>
struct FillRopeCosSinOperation
    : virtual Operation,
      FillRopeCosSinOperationData,
      virtual OpMixin<struct FillRopeCosSinOperationTag, Mixins>... {
 public:
  absl::string_view GetName() const override { return "FillRopeCosSin"; }
};

struct AddValuesToKvCacheOperationData {
  int token_index_offset;
  int active_tokens;
  int num_of_kv_heads;
  int kv_cache_batch_size;
  int cache_size;
  int head_dimension;
};

template <class... Mixins>
struct AddValuesToKvCacheOperation
    : public AddValuesToKvCacheOperationData,
      virtual Operation,
      virtual OpMixin<struct AddValuesToKvCacheOperationTag, Mixins...> {
 public:
  absl::string_view GetName() const override { return "AddValuesToKvCache"; }
};

struct AddValuesToCacheOperationData {
  int token_index_offset;
  int active_tokens;
  int num_of_kv_heads;
  int kv_cache_batch_size;
  int cache_size;
  int head_dimension;
};

template <class... Mixins>
struct AddValuesToCacheOperation
    : public AddValuesToCacheOperationData,
      virtual Operation,
      virtual OpMixin<struct AddValuesToCacheOperationTag, Mixins...> {
 public:
  absl::string_view GetName() const override { return "AddValuesToCache"; }
};

template <class... Mixins>
struct QkNormOperation : virtual Operation,
                         virtual OpMixin<struct QkNormOperationTag, Mixins...> {
 public:
  absl::string_view GetName() const override { return "QkNorm"; }
};

struct ActivationSparsityOperationData {
  float stddev_multiplier;
};

template <class... Mixins>
struct ActivationSparsityOperation
    : public ActivationSparsityOperationData,
      virtual Operation,
      virtual OpMixin<struct ActivationSparsityOperationTag, Mixins...> {
 public:
  absl::string_view GetName() const override { return "ActivationSparsity"; }
};

template <class... Mixins>
struct SelectMaskOperation
    : virtual Operation,
      virtual OpMixin<struct SelectMaskOperationTag, Mixins...> {
 public:
  absl::string_view GetName() const override { return "SelectMask"; }
};

template <class... Mixins>
struct WriteCurrentTokensOperation
    : virtual Operation,
      virtual OpMixin<struct WriteCurrentTokensOperationTag, Mixins...> {
 public:
  absl::string_view GetName() const override {
    return "WriteCurrentTokens";
  }
};

struct MatMulWithCacheOperationData {
  bool is_v = false;
};

template <class... Mixins>
struct MatMulWithCacheOperation
    : public MatMulWithCacheOperationData,
      virtual Operation,
      virtual OpMixin<struct MatMulWithCacheOperationTag, Mixins...> {
 public:
  absl::string_view GetName() const override { return "MatMulWithCache"; }
};

struct SoftmaxWithRuntimeCheckOperationData {
  std::optional<int> start_ch_index;
  std::optional<int> end_ch_index;
};

template <class... Mixins>
struct SoftmaxWithRuntimeCheckOperation
    : public SoftmaxWithRuntimeCheckOperationData,
      virtual Operation,
      virtual OpMixin<struct SoftmaxWithRuntimeCheckOperationTag, Mixins...> {
 public:
  absl::string_view GetName() const override {
    return "SoftmaxWithRuntimeCheck";
  }
};

}  // namespace litert::tensor::graph

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_OPS_TRANSFORMER_TRANSFORMER_OPS_GRAPH_H_

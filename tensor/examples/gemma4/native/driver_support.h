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

#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_DRIVER_SUPPORT_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_DRIVER_SUPPORT_H_

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"      // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/flags/flag.h"               // from @com_google_absl
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"     // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"    // from @com_google_absl
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"
#include "absl/types/span.h"  // from @com_google_absl
#include "xnnpack.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/backends/xnnpack/utils.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/native/model/gemma4_graph.h"
#include "tensor/examples/gemma4/helpers/quantized_embedding.h"
#include "tensor/examples/gemma4/helpers/rope.h"
#include "tensor/examples/ops/transformer/transformer_ops_xnnpack.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"


namespace litert::tensor::examples::gemma4::native {
namespace {

using XnnTensor = Tensor<XnnpackMixinTag>;

struct WeightsCacheDeleter {
  void operator()(xnn_weights_cache_t cache) const {
    if (cache != nullptr) {
      xnn_delete_weights_cache(cache);
    }
  }
};
using WeightsCache =
    std::unique_ptr<xnn_weights_cache_provider, WeightsCacheDeleter>;

absl::StatusOr<std::vector<int32_t>> ParseTokenIds(absl::string_view text) {
  std::vector<int32_t> ids;
  for (absl::string_view field : absl::StrSplit(text, ',')) {
    int32_t token;
    if (!absl::SimpleAtoi(absl::StripAsciiWhitespace(field), &token) ||
        token < 0) {
      return absl::InvalidArgumentError(
          "--token_ids must contain comma-separated nonnegative int32 IDs");
    }
    ids.push_back(token);
  }
  return ids;
}

std::string PassSuffix(int generated_index) {
  return generated_index == 0
             ? ".prefill"
             : absl::StrFormat(".decode_%04d", generated_index);
}

std::string LogitsSuffix(int generated_index) {
  return absl::StrCat(PassSuffix(generated_index), ".f32");
}

absl::Status WriteRawFloats(absl::string_view path,
                            absl::Span<const float> values) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  std::ofstream output(std::string(path), std::ios::binary | std::ios::trunc);
  const uint16_t byte_order = 1;
  if (*reinterpret_cast<const unsigned char*>(&byte_order) == 1) {
    output.write(reinterpret_cast<const char*>(values.data()),
                 values.size() * sizeof(float));
  } else {
    for (float value : values) {
      uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      const char bytes[] = {
          static_cast<char>(bits), static_cast<char>(bits >> 8),
          static_cast<char>(bits >> 16), static_cast<char>(bits >> 24)};
      output.write(bytes, sizeof(bytes));
    }
  }
  output.close();
  if (!output) {
    return absl::InternalError(
        absl::StrCat("Failed to write float32 output: ", path));
  }
  return absl::OkStatus();
}

absl::StatusOr<int32_t> SelectTokenAndDump(absl::Span<const float> logits,
                                           absl::string_view dump_prefix,
                                           int generated_index) {
  if (logits.empty() ||
      std::any_of(logits.begin(), logits.end(),
                  [](float value) { return !std::isfinite(value); })) {
    return absl::InternalError("Model logits must be nonempty and finite");
  }
  if (!dump_prefix.empty()) {
    LRT_TENSOR_RETURN_IF_ERROR(WriteRawFloats(
        absl::StrCat(dump_prefix, LogitsSuffix(generated_index)), logits));
  }
  return static_cast<int32_t>(absl::c_max_element(logits) - logits.begin());
}

absl::Status FillAttentionMask(const Shape& shape, const absl::Span<float> mask,
                               const bool is_local,
                               const int sliding_window_size) {
  if (shape.size() < 2) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "FillAttentionMask output shape must have at least 2 dims, got %zu",
        shape.size()));
  }

  const int64_t seq_q = shape[shape.size() - 2];
  const int64_t seq_k = shape[shape.size() - 1];
  if (seq_q <= 0 || seq_k <= 0 || seq_q != seq_k) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "FillAttentionMask expects output shape [..., S, S] with S>0; got [%s]",
        absl::StrJoin(shape, ", ")));
  }
  if (std::any_of(shape.begin(), shape.end() - 2,
                  [](auto d) { return d <= 0; })) {
    return absl::InvalidArgumentError(
        absl::StrFormat("FillAttentionMask does not support non-positive "
                        "leading dims. Got shape [%s]",
                        absl::StrJoin(shape, ", ")));
  }
  const int64_t leading =
      std::accumulate(shape.begin(), shape.end() - 2, 1, std::multiplies<>());
  const int64_t matrix_size = seq_q * seq_k;
  const int64_t tensor_size = leading * matrix_size;

  if (mask.size() != tensor_size) {
    return absl::InvalidArgumentError(
        absl::StrFormat("FillAttentionMask output mask should hold %" PRIi64
                        " elements but holds %zu",
                        tensor_size, mask.size()));
  }

  const float neg_inf = std::numeric_limits<float>::lowest();

  for (int64_t b = 0; b < leading; ++b) {
    const int64_t base = b * matrix_size;
    for (int64_t i = 0; i < seq_q; ++i) {
      const int64_t I = i * seq_q;
      for (int64_t j = 0; j < seq_q; ++j) {
        const bool is_causal_masked = (j > i);
        const bool is_sliding_masked = is_local && (sliding_window_size > 0) &&
                                       (i - j >= sliding_window_size);
        mask[static_cast<size_t>(base + I + j)] =
            (is_causal_masked || is_sliding_masked) ? neg_inf : 0.0f;
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<XnnTensor> AttentionMask(Shape shape, const bool is_local,
                                        const int sliding_window_size) {
  auto buffer = OwningCpuBuffer::Allocate<Type::kFP32>(shape);
  XnnTensor mask({.name = "attention_mask",
                  .type = Type::kFP32,
                  .shape = std::move(shape),
                  .buffer = buffer});
  LRT_TENSOR_RETURN_IF_ERROR(FillAttentionMask(
      mask.GetShape(), buffer->Span<float>(), is_local, sliding_window_size));
  return mask;
}

struct LoadedTensors {
  absl::flat_hash_map<std::string, TensorHandle> weights_handle;
  std::unique_ptr<GemmaEmbeddingTable> token_embedding;
  std::unique_ptr<GemmaEmbeddingTable> emb_per_layer_table;
};

absl::StatusOr<Gemma4Inputs<XnnpackMixinTag>> CreateGemma4Inputs(
    const Config& config, int input_seq_len, int kv_cache_len,
    const absl::flat_hash_map<std::string, TensorHandle>& weights_handle,
    bool verbose) {
  const int batch_size = 1;
  Gemma4Inputs<XnnpackMixinTag> inputs;

  inputs.embedded_input =
      XnnTensor({.name = "embedded_input",
                 .type = Type::kFP32,
                 .shape = {batch_size, input_seq_len, config.embed_dim}});

  if (kv_cache_len == 0) {
    std::tie(inputs.rope_global_cos, inputs.rope_global_sin) =
        RopeCosSin(input_seq_len, config.global_key_size,
                   config.global_base_frequency, config.global_rope_proportion);
    inputs.rope_global_cos.SetName("rope_global_cos");
    inputs.rope_global_sin.SetName("rope_global_sin");

    std::tie(inputs.rope_local_cos, inputs.rope_local_sin) =
        RopeCosSin(input_seq_len, config.head_dim, config.local_base_frequency,
                   config.local_rope_proportion);
    inputs.rope_local_cos.SetName("rope_local_cos");
    inputs.rope_local_sin.SetName("rope_local_sin");

    LRT_TENSOR_ASSIGN_OR_RETURN(
        inputs.global_attention_mask,
        AttentionMask({1, 1, input_seq_len, input_seq_len},
                      /*is_local=*/false, config.sliding_window_size));
    inputs.global_attention_mask.SetName("global_attention_mask");

    LRT_TENSOR_ASSIGN_OR_RETURN(
        inputs.sliding_attention_mask,
        AttentionMask({1, 1, input_seq_len, input_seq_len},
                      /*is_local=*/true, config.sliding_window_size));
    inputs.sliding_attention_mask.SetName("sliding_attention_mask");
  } else {
    inputs.rope_global_cos.Set({.name = "rope_global_cos",
                                .type = Type::kFP32,
                                .shape = {1, 1, 1, config.global_key_size}});
    inputs.rope_global_sin.Set({.name = "rope_global_sin",
                                .type = Type::kFP32,
                                .shape = {1, 1, 1, config.global_key_size}});
    inputs.rope_local_cos.Set({.name = "rope_local_cos",
                               .type = Type::kFP32,
                               .shape = {1, 1, 1, config.head_dim}});
    inputs.rope_local_sin.Set({.name = "rope_local_sin",
                               .type = Type::kFP32,
                               .shape = {1, 1, 1, config.head_dim}});
    inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                       .type = Type::kFP32,
                                       .shape = {1, 1, 1, kv_cache_len + 1}});
    inputs.global_attention_mask.Set({.name = "global_attention_mask",
                                      .type = Type::kFP32,
                                      .shape = {1, 1, 1, kv_cache_len + 1}});
  }

  inputs.key_caches.reserve(config.num_layers);
  inputs.value_caches.reserve(config.num_layers);
  for (int i = 0; i < config.num_layers; ++i) {
    bool is_global = config.GetLayerType(i) == Config::LayerType::kGlobal;
    int head_dim = is_global ? config.global_key_size : config.head_dim;
    inputs.key_caches.push_back(XnnTensor(
        {.name = absl::StrCat("key_cache_", i),
         .type = Type::kFP32,
         .shape = {batch_size, config.num_kv_heads, kv_cache_len, head_dim}}));
    inputs.value_caches.push_back(XnnTensor(
        {.name = absl::StrCat("value_cache_", i),
         .type = Type::kFP32,
         .shape = {batch_size, config.num_kv_heads, kv_cache_len, head_dim}}));
  }

  inputs.per_layer_token_embeddings.reserve(config.num_layers);
  for (int l = 0; l < config.num_layers; ++l) {
    inputs.per_layer_token_embeddings.push_back(XnnTensor(
        {.name = absl::StrCat("per_layer_token_embedding_", l),
         .type = Type::kFP32,
         .shape = {batch_size, input_seq_len, config.per_layer_input_dim}}));
  }

  for (auto& [name, xnnpack_tensor] : weights_handle) {
    if (xnnpack_tensor.GetBuffer().ok()) {
      inputs.weights.emplace(name, xnnpack_tensor);
      if (verbose) {
        ABSL_LOG(INFO) << "Added weight: " << name << " shape: ["
                       << absl::StrJoin(xnnpack_tensor.GetShape(), ", ") << "]";
      }
    }
  }

  return inputs;
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4::native

#endif  // LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_DRIVER_SUPPORT_H_

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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_RUNNER_H_
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "tensor/buffer.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/litert/bundle_loader.h"
#include "tensor/examples/gemma4/litert/host_buffer.h"
#include "tensor/examples/gemma4/litert/kv_bank.h"
#include "tensor/examples/gemma4/litert/kv_cache.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::examples::gemma4::cpu {
// CPU model client. One CompiledModel owns both signatures and the delegate.
// The model and bank are reused across requests; invocation is synchronous.
class Runner {
 public:
  static absl::StatusOr<std::unique_ptr<Runner>> Create(
      Environment& env, Options& options, absl::string_view model_path,
      absl::string_view bundle_dir, int capacity, int alignment = 32,
      bool compact_int2 = false, bool dynamic_prefill_rows = false,
      bool strict_shapes = false, ModelVariant variant = ModelVariant::kE2B) {
    const Config config = Config::From(variant);
    if (alignment < 1 || alignment > 128 || (alignment & (alignment - 1))) {
      return absl::InvalidArgumentError(
          "Alignment must be a power of two <=128");
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(
        std::vector<KvOwnerSpec> specs,
        LoadPublishedKvSpecs(bundle_dir, config, compact_int2));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        ActiveKvBank bank, ActiveKvBank::Create(std::move(specs), capacity));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        LoadedTensors embeddings,
        LoadPublishedBundle(bundle_dir, config, true, compact_int2));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        LitertDynamicRunner model,
        LitertDynamicRunner::Create(env, std::string(model_path), options));
    std::unique_ptr<Runner> result(new Runner(
        env, std::move(model), std::move(embeddings), std::move(bank),
        alignment, dynamic_prefill_rows, strict_shapes, config));
    for (int i = 0; i < 2; ++i) {
      Signature& sig = result->signatures_[i];
      sig.name = i == 0 ? "prefill" : "decode";
      LRT_TENSOR_ASSIGN_OR_RETURN(
          TensorHandle input,
          result->model_.GetInput(sig.name, "embedded_input"));
      if (input.GetShape().size() != 3 || input.GetShape()[0] != 1 ||
          input.GetShape()[2] != result->config_.embed_dim ||
          input.GetShape()[1] <= 0 || (i == 1 && input.GetShape()[1] != 1)) {
        return absl::InvalidArgumentError(
            "Invalid Gemma4 signature input shape");
      }
      sig.rows = input.GetShape()[1];
      sig.ple.resize(result->config_.num_layers);
      for (int layer = 0; layer < result->config_.num_layers; ++layer) {
        absl::StatusOr<size_t> index = result->model_.GetInputIndex(
            sig.name, absl::StrCat("per_layer_token_embedding_", layer));
        if (index.ok()) {
          sig.ple_indices.push_back({layer, *index});
        } else if (!absl::IsNotFound(index.status())) {
          return index.status();
        }
      }
    }
    result->padding_.Resize(size_t(alignment) *
                            result->config_.global_key_size);
    return result;
  }

  void Reset() { bank_.Reset(); }
  int prefill_rows() const { return signatures_[0].rows; }
  const ActiveKvBank& bank() const { return bank_; }

  // Prefill processes prompt[0..N-2]. The final prompt token goes through
  // Decode to produce logits using all layers. No logits are computed for the
  // prefix.
  absl::Status Prefill(absl::Span<const int32_t> prefix) {
    while (!prefix.empty()) {
      const size_t rows = std::min(prefix.size(), size_t(prefill_rows()));
      LRT_TENSOR_RETURN_IF_ERROR(
          RunChunk(signatures_[0], prefix.subspan(0, rows), false));
      prefix.remove_prefix(rows);
    }
    return absl::OkStatus();
  }
  absl::StatusOr<LockedBufferSpan<const float>> Decode(int32_t token) {
    LRT_TENSOR_RETURN_IF_ERROR(RunChunk(signatures_[1], {&token, 1}, true));
    LRT_TENSOR_ASSIGN_OR_RETURN(TensorHandle output,
                                model_.GetOutput("decode", "logits"));
    return output.GetBufferPtr()->Lock().As<const float>();
  }

 private:
  struct Signature {
    std::string name;
    int rows = 0;
    std::vector<float> embeddings, positions;
    std::vector<std::vector<float>> ple;
    std::vector<std::pair<int, size_t>> ple_indices;
    std::array<HostBuffer, 2> masks;
  };
  Runner(Environment& env, LitertDynamicRunner model, LoadedTensors embeddings,
         ActiveKvBank bank, int alignment, bool dynamic_prefill_rows,
         bool strict_shapes, Config config)
      : env_(env),
        config_(config),
        embeddings_(std::move(embeddings)),
        bank_(std::move(bank)),
        alignment_(alignment),
        dynamic_prefill_rows_(dynamic_prefill_rows),
        resize_mode_(strict_shapes ? ResizeMode::kStrict
                                   : ResizeMode::kNonStrict),
        model_(std::move(model)) {}

  template <class T>
  absl::Status CopyInput(Signature& sig, absl::string_view name,
                         const std::vector<T>& data) {
    return model_.SetInput(
        sig.name, std::string(name),
        absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(data.data()),
                                  data.size() * sizeof(T)));
  }

  absl::Status Bind(Signature& sig, absl::string_view name, ElementType type,
                    const Shape& shape, const void* pointer, size_t bytes) {
    const std::string input_name(name);
    LRT_TENSOR_RETURN_IF_ERROR(
        model_.ResizeInput(sig.name, input_name, shape, resize_mode_));
    LITERT_ASSIGN_OR_RETURN(
        TensorBuffer buffer,
        TensorBuffer::CreateFromHostMemory(
            env_,
            RankedTensorType(type,
                             Layout(Dimensions(shape.begin(), shape.end()))),
            const_cast<void*>(pointer), bytes + 64));
    return model_.SetInputBuffer(sig.name, input_name, std::move(buffer));
  }

  absl::Status RunChunk(Signature& sig, absl::Span<const int32_t> tokens,
                        bool decode) {
    const int count = tokens.size(), start = bank_.length();
    const int rows = dynamic_prefill_rows_ && !decode ? count : sig.rows;
    if (count <= 0 || count > sig.rows) {
      return absl::InvalidArgumentError("Invalid chunk size");
    }
    for (int32_t token : tokens) {
      if (token < 0 || token >= config_.vocab_size) {
        return absl::InvalidArgumentError("Token outside Gemma4 vocabulary");
      }
    }
    LRT_TENSOR_RETURN_IF_ERROR(bank_.BeginAppend(count));
    struct Transaction {
      ActiveKvBank& bank;
      ~Transaction() {
        if (bank.append_in_flight()) {
          bank.Abort();
        }
      }
    } transaction{bank_};
    if (dynamic_prefill_rows_ && !decode) {
      LRT_TENSOR_RETURN_IF_ERROR(
          model_.ResizeInput(sig.name, "embedded_input",
                             {1, rows, config_.embed_dim}, resize_mode_));
      LRT_TENSOR_RETURN_IF_ERROR(model_.ResizeInput(
          sig.name, "positions", {1, 1, rows, 1}, resize_mode_));
      for (auto [layer, index] : sig.ple_indices) {
        LRT_TENSOR_RETURN_IF_ERROR(model_.ResizeInput(
            sig.name, index, {1, rows, config_.per_layer_input_dim},
            resize_mode_));
      }
    }
    std::vector<int32_t> padded(rows, 0);
    absl::c_copy(tokens, padded.begin());
    sig.embeddings.resize(size_t(rows) * config_.embed_dim);
    for (std::vector<float>& layer : sig.ple) {
      layer.resize(size_t(rows) * config_.per_layer_input_dim);
    }
    LRT_TENSOR_RETURN_IF_ERROR(embeddings_.token_embedding->Lookup(
        padded, absl::MakeSpan(sig.embeddings)));
    LRT_TENSOR_RETURN_IF_ERROR(embeddings_.emb_per_layer_table->LookupPerLayer(
        padded, config_.num_layers, config_.per_layer_input_dim,
        absl::MakeSpan(sig.ple)));
    LRT_TENSOR_RETURN_IF_ERROR(
        CopyInput(sig, "embedded_input", sig.embeddings));
    for (auto [layer, index] : sig.ple_indices) {
      const std::vector<float>& data = sig.ple[layer];
      LRT_TENSOR_RETURN_IF_ERROR(
          model_.SetInput(sig.name, index,
                          absl::Span<const uint8_t>(
                              reinterpret_cast<const uint8_t*>(data.data()),
                              data.size() * sizeof(float))));
    }
    sig.positions.resize(rows);
    for (int r = 0; r < rows; ++r) {
      sig.positions[r] = static_cast<float>(start + r);
    }
    LRT_TENSOR_RETURN_IF_ERROR(CopyInput(sig, "positions", sig.positions));
    const int graph_end = start + rows;
    const int padded_end =
        (graph_end + alignment_ - 1) / alignment_ * alignment_;
    // Prefill's final KV producer has no attention or MLP in this graph.
    for (const KvOwnerSpec& spec : bank_.specs()) {
      if (!decode && spec.owner == bank_.specs().back().owner) {
        continue;
      }
      const bool global =
          config_.GetLayerType(spec.owner) == Config::LayerType::kGlobal;
      int begin =
          global ? 0 : std::max(0, start - config_.sliding_window_size + 1);
      begin = begin / alignment_ * alignment_;
      for (int head = 0; head < spec.num_heads; ++head) {
        LRT_TENSOR_ASSIGN_OR_RETURN(absl::Span<const int8_t> keys,
                                    bank_.Keys(spec.owner, begin, start, head));
        LRT_TENSOR_ASSIGN_OR_RETURN(
            absl::Span<const int8_t> values,
            bank_.Values(spec.owner, begin, start, head));
        const std::string prefix = absl::StrCat(
            "model.layers.", spec.owner, ".self_attn",
            spec.num_heads > 1 ? absl::StrCat(".head_", head) : "");
        const Shape past_shape{1, 1, start - begin, spec.head_dim};
        const Shape pad_shape{1, 1, padded_end - graph_end, spec.head_dim};
        LRT_TENSOR_RETURN_IF_ERROR(Bind(sig, absl::StrCat(prefix, ".past_k"),
                                        ElementType::Int8, past_shape,
                                        keys.data(), keys.size()));
        LRT_TENSOR_RETURN_IF_ERROR(Bind(sig, absl::StrCat(prefix, ".past_v"),
                                        ElementType::Int8, past_shape,
                                        values.data(), values.size()));
        for (absl::string_view suffix : {".pad_k", ".pad_v"}) {
          LRT_TENSOR_RETURN_IF_ERROR(Bind(
              sig, absl::StrCat(prefix, suffix), ElementType::Int8, pad_shape,
              padding_.data(), size_t(padded_end - graph_end) * spec.head_dim));
        }
      }
    }
    for (int global = 0; global < 2; ++global) {
      int begin =
          global ? 0 : std::max(0, start - config_.sliding_window_size + 1);
      begin = begin / alignment_ * alignment_;
      const int extent = padded_end - begin;
      const int mask_heads =
          decode ? config_.num_heads / config_.num_kv_heads : 1;
      HostBuffer& buffer = sig.masks[global];
      buffer.Resize(size_t(mask_heads) * rows * extent * sizeof(float));
      float* data = reinterpret_cast<float*>(buffer.data());
      std::fill_n(data, size_t(mask_heads) * rows * extent,
                  std::numeric_limits<float>::lowest());
      for (int h = 0; h < mask_heads; ++h) {
        for (int r = 0; r < count; ++r) {
          int lower =
              global ? 0
                     : std::max(0, start + r - config_.sliding_window_size + 1);
          float* row = data + (size_t(h) * rows + r) * extent;
          std::fill(row + std::max(lower, begin) - begin,
                    row + start + r + 1 - begin, 0.0f);
        }
      }
      LRT_TENSOR_RETURN_IF_ERROR(
          Bind(sig, global ? "joined_global_mask" : "joined_local_mask",
               ElementType::Float32, {1, 1, mask_heads * rows, extent},
               buffer.data(), buffer.size()));
    }
    LRT_TENSOR_RETURN_IF_ERROR(model_.Run(sig.name));
    for (const KvOwnerSpec& spec : bank_.specs()) {
      for (int head = 0; head < spec.num_heads; ++head) {
        const std::string suffix =
            spec.num_heads > 1 ? absl::StrCat("_head_", head) : "";
        LRT_TENSOR_ASSIGN_OR_RETURN(
            TensorHandle keys,
            model_.GetOutput(sig.name,
                             absl::StrCat("new_key_", spec.owner, suffix)));
        LRT_TENSOR_ASSIGN_OR_RETURN(
            TensorHandle values,
            model_.GetOutput(sig.name,
                             absl::StrCat("new_value_", spec.owner, suffix)));
        LockedBufferSpan<const int8_t> k =
            keys.GetBufferPtr()->Lock().As<const int8_t>();
        LockedBufferSpan<const int8_t> v =
            values.GetBufferPtr()->Lock().As<const int8_t>();
        const size_t size = size_t(count) * spec.head_dim;
        if (k.size() < size || v.size() < size) {
          return absl::InternalError("Short KV output");
        }
        LRT_TENSOR_RETURN_IF_ERROR(
            bank_.Append(spec.owner, {k.data(), size}, {v.data(), size}, head));
      }
    }
    return bank_.Commit();
  }

  Environment& env_;
  Config config_ = Config::E2B();
  LoadedTensors embeddings_;
  ActiveKvBank bank_;
  int alignment_;
  bool dynamic_prefill_rows_;
  ResizeMode resize_mode_;
  std::array<Signature, 2> signatures_;
  HostBuffer padding_;
  // Destroy the model before buffers it borrows.
  LitertDynamicRunner model_;
};
}  // namespace litert::tensor::examples::gemma4::cpu
#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_RUNNER_H_

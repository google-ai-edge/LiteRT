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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_OPS_TRANSFORMER_TRANSFORMER_OPS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_OPS_TRANSFORMER_TRANSFORMER_OPS_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "tensor/datatypes.h"
#include "tensor/examples/ops/transformer/transformer_ops_graph.h"
#include "tensor/internal/arithmetic_helpers.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/source_location.h"

namespace litert::tensor {

template <class... Mixins>
Tensor<Mixins...> RotaryEmbedding(const Tensor<Mixins...>& input,
                                  const Tensor<Mixins...>& segment_pos,
                                  float min_timescale, float max_timescale,
                                  float rope_wavelength) {
  auto op = std::make_shared<graph::RotaryEmbeddingOperation>();
  RegisterMixins<Mixins...>(op);
  op->min_timescale = min_timescale;
  op->max_timescale = max_timescale;
  // op->rope_wavelength = rope_wavelength;
  op->inputs = {input.GetRaw(), segment_pos.GetRaw()};
  Tensor<Mixins...> output = AddOutput(op, source_location::current());
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = input_info.shape;
  output_info.type = input_info.type;
  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> RmsNorm(const Tensor<Mixins...>& input,
                          const Tensor<Mixins...>& scale,
                          const Tensor<Mixins...>& epsilon) {
  auto op = std::make_shared<graph::RmsNormOperation>();
  RegisterMixins<Mixins...>(op);
  op->inputs = {input.GetRaw(), scale.GetRaw(), epsilon.GetRaw()};
  Tensor<Mixins...> output = AddOutput(op, source_location::current());
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = input_info.shape;
  output_info.type = input_info.type;
  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> FillSegmentPos(const Tensor<Mixins...>& params,
                                 const std::vector<int>& shape,
                                 int param_index) {
  auto op = std::make_shared<graph::FillSegmentPosOperation>();
  RegisterMixins<Mixins...>(op);
  op->param_index = param_index;
  op->inputs = {params.GetRaw()};
  Tensor<Mixins...> output = AddOutput(op, source_location::current());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = shape;
  output_info.type = Type::kI32;
  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> FillAttentionMask(const Tensor<Mixins...>& params,
                                    const std::vector<int>& shape,
                                    bool is_local, int sliding_window_size) {
  auto op = std::make_shared<graph::FillAttentionMaskOperation>();
  RegisterMixins<Mixins...>(op);
  op->is_local = is_local;
  op->sliding_window_size = sliding_window_size;
  op->inputs = {params.GetRaw()};
  Tensor<Mixins...> output = AddOutput(op, source_location::current());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = shape;
  output_info.type = Type::kFP32;
  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
std::pair<Tensor<Mixins...>, Tensor<Mixins...>> FillRopeCosSin(
    int seq_len, int head_dim, float rope_base) {
  auto op = std::make_shared<graph::FillRopeCosSinOperation>();
  RegisterMixins<Mixins...>(op);
  op->rope_base = rope_base;

  auto out_group = graph::NewTensorGroup(2, source_location::current());
  op->outputs_group = out_group;
  out_group->producer = op;

  std::pair<Tensor<Mixins...>, Tensor<Mixins...>> outputs{
      Tensor<Mixins...>(graph::GetTensor(0, out_group)),
      Tensor<Mixins...>(graph::GetTensor(1, out_group))};

  graph::TensorInformation& cos_info = *GetInfo(outputs.first.GetRaw());
  cos_info.shape = {1, 1, seq_len, head_dim};
  cos_info.type = Type::kFP32;
  graph::TensorInformation& sin_info = *GetInfo(outputs.second.GetRaw());
  sin_info.shape = {1, 1, seq_len, head_dim};
  sin_info.type = Type::kFP32;

  graph::OpDebugger::DebugOp(*op);
  return outputs;
}

template <class... Mixins>
std::vector<Tensor<Mixins...>> AddValuesToKvCache(
    const Tensor<Mixins...>& key_cache, const Tensor<Mixins...>& value_cache,
    const Tensor<Mixins...>& key, const Tensor<Mixins...>& value,
    const Tensor<Mixins...>& params, int num_of_kv_heads,
    int kv_cache_batch_size, int cache_size, int head_dimension,
    int token_index_offset, int active_tokens) {
  auto op = std::make_shared<graph::AddValuesToKvCacheOperation>();
  RegisterMixins<Mixins...>(op);
  op->num_of_kv_heads = num_of_kv_heads;
  op->kv_cache_batch_size = kv_cache_batch_size;
  op->token_index_offset = token_index_offset;
  op->active_tokens = active_tokens;
  op->cache_size = cache_size;
  op->head_dimension = head_dimension;
  op->inputs = {key_cache.GetRaw(), value_cache.GetRaw(), key.GetRaw(),
                value.GetRaw(), params.GetRaw()};

  auto out_group = graph::NewTensorGroup(2, source_location::current());
  op->outputs_group = out_group;
  out_group->producer = op;
  std::vector<Tensor<Mixins...>> outputs;
  outputs.reserve(2);

  const graph::TensorInformation& key_cache_info = *GetInfo(key_cache.GetRaw());
  auto& output_key_cache_info = out_group->tensor_infos[0];
  output_key_cache_info.shape = key_cache_info.shape;
  output_key_cache_info.type = key_cache_info.type;
  outputs.emplace_back(graph::GetTensor(0, out_group));

  const graph::TensorInformation& value_cache_info =
      *GetInfo(value_cache.GetRaw());
  auto& output_value_cache_info = out_group->tensor_infos[1];
  output_value_cache_info.shape = value_cache_info.shape;
  output_value_cache_info.type = value_cache_info.type;
  outputs.emplace_back(graph::GetTensor(1, out_group));

  graph::OpDebugger::DebugOp(*op);
  return outputs;
}

template <class... Mixins>
std::vector<Tensor<Mixins...>> AddValuesToCache(
    const Tensor<Mixins...>& key_cache, const Tensor<Mixins...>& value_cache,
    const Tensor<Mixins...>& key, const Tensor<Mixins...>& value,
    const Tensor<Mixins...>& params, int num_of_kv_heads,
    int kv_cache_batch_size, int cache_size, int head_dimension,
    int token_index_offset, int active_tokens) {
  auto op = std::make_shared<graph::AddValuesToCacheOperation>();
  RegisterMixins<Mixins...>(op);
  op->num_of_kv_heads = num_of_kv_heads;
  op->kv_cache_batch_size = kv_cache_batch_size;
  op->token_index_offset = token_index_offset;
  op->active_tokens = active_tokens;
  op->cache_size = cache_size;
  op->head_dimension = head_dimension;
  op->inputs = {key_cache.GetRaw(), value_cache.GetRaw(), key.GetRaw(),
                value.GetRaw(), params.GetRaw()};

  auto out_group = graph::NewTensorGroup(2, source_location::current());
  op->outputs_group = out_group;
  out_group->producer = op;
  std::vector<Tensor<Mixins...>> outputs;
  outputs.reserve(2);

  const graph::TensorInformation& key_cache_info = *GetInfo(key_cache.GetRaw());
  auto& output_key_cache_info = out_group->tensor_infos[0];
  output_key_cache_info.shape = key_cache_info.shape;
  output_key_cache_info.type = key_cache_info.type;
  outputs.emplace_back(graph::GetTensor(0, out_group));

  const graph::TensorInformation& value_cache_info =
      *GetInfo(value_cache.GetRaw());
  auto& output_value_cache_info = out_group->tensor_infos[1];
  output_value_cache_info.shape = value_cache_info.shape;
  output_value_cache_info.type = value_cache_info.type;
  outputs.emplace_back(graph::GetTensor(1, out_group));

  graph::OpDebugger::DebugOp(*op);
  return outputs;
}

template <class... Mixins>
Tensor<Mixins...> QkNorm(const Tensor<Mixins...>& input,
                         const Tensor<Mixins...>& scale) {
  auto op = std::make_shared<graph::QkNormOperation>();
  RegisterMixins<Mixins...>(op);
  op->inputs = {input.GetRaw(), scale.GetRaw()};
  Tensor<Mixins...> output = AddOutput(op, source_location::current());
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = input_info.shape;
  output_info.type = input_info.type;
  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> ActivationSparsity(const Tensor<Mixins...>& input,
                                     float stddev_multiplier) {
  auto op = std::make_shared<graph::ActivationSparsityOperation>();
  RegisterMixins<Mixins...>(op);
  op->stddev_multiplier = stddev_multiplier;
  op->inputs = {input.GetRaw()};
  Tensor<Mixins...> output = AddOutput(op, source_location::current());
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = input_info.shape;
  output_info.type = input_info.type;
  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> SelectMask(const Tensor<Mixins...>& scores,
                             const Tensor<Mixins...>& mask) {
  auto op = std::make_shared<graph::SelectMaskOperation>();
  RegisterMixins<Mixins...>(op);
  op->inputs = {scores.GetRaw(), mask.GetRaw()};
  Tensor<Mixins...> output = AddOutput(op, source_location::current());
  const graph::TensorInformation& scores_info = *GetInfo(scores.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = scores_info.shape;
  output_info.type = scores_info.type;
  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> WriteCurrentTokens(const Tensor<Mixins...>& input,
                                     const Tensor<Mixins...>& params) {
  auto op = std::make_shared<graph::WriteCurrentTokensOperation>();
  RegisterMixins<Mixins...>(op);
  op->inputs = {input.GetRaw(), params.GetRaw()};
  LRT_TENSOR_ASSIGN_OR_ABORT(auto input_info, graph::GetInfo(input.GetRaw()));
  Tensor<Mixins...> output = AddOutput(op, source_location::current());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = {1};
  output_info.type = input_info.type;
  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> MatMulWithCache(const Tensor<Mixins...>& lhs,
                                  const Tensor<Mixins...>& rhs,
                                  const Tensor<Mixins...>& params,
                                  bool is_v = false, bool is_local = false,
                                  int sliding_window_size = 0) {
  auto op = std::make_shared<graph::MatMulWithCacheOperation>();
  RegisterMixins<Mixins...>(op);
  op->inputs = {lhs.GetRaw(), rhs.GetRaw(), params.GetRaw()};
  op->is_v = is_v;
  op->is_local = is_local;
  op->sliding_window_size = sliding_window_size;
  LRT_TENSOR_ASSIGN_OR_ABORT(auto lhs_info, graph::GetInfo(lhs.GetRaw()));
  LRT_TENSOR_ASSIGN_OR_ABORT(auto rhs_info, graph::GetInfo(rhs.GetRaw()));
  Tensor<Mixins...> output = AddOutput(op, source_location::current());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());

  std::vector<int> out_shape = lhs_info.shape;
  if (!out_shape.empty() && rhs_info.shape.size() >= 2) {
    if (is_v) {
      out_shape.back() = rhs_info.shape.back();
    } else {
      out_shape.back() = rhs_info.shape[rhs_info.shape.size() - 2];
    }
  }

  output_info.shape = out_shape;
  output_info.type = lhs_info.type;
  graph::OpDebugger::DebugOp(*op);
  return output;
}

template <class... Mixins>
Tensor<Mixins...> SoftmaxWithRuntimeCheck(
    const Tensor<Mixins...>& input, const Tensor<Mixins...>& params,
    std::optional<int> start_ch_index = std::nullopt,
    std::optional<int> end_ch_index = std::nullopt) {
  auto op = std::make_shared<graph::SoftmaxWithRuntimeCheckOperation>();
  RegisterMixins<Mixins...>(op);
  op->start_ch_index = start_ch_index;
  op->end_ch_index = end_ch_index;
  op->inputs = {input.GetRaw(), params.GetRaw()};
  Tensor<Mixins...> output = AddOutput(op, source_location::current());
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = input_info.shape;
  output_info.type = input_info.type;
  graph::OpDebugger::DebugOp(*op);
  return output;
}

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_OPS_TRANSFORMER_TRANSFORMER_OPS_H_

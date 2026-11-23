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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_QSA_DECOMPOSITIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_QSA_DECOMPOSITIONS_H_

// The decompositions in TFLite builtin ops of Qwen 4's sparse-attention
// composites (odml.qsa_indexer_update, odml.cache_update over float caches,
// odml.sparse_sdpa_transposed) at the reduced shapes of the artificial
// models (qsa_artifact.cc, qwen4_artifact.cc), written from the composites'
// contracts (ml_drift_delegate/delegate/composite/*_parser.h).

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/vendors/nvidia/tools/tflite_builder.h"
#include "tflite/schema/schema_generated.h"

namespace litert::nvidia::artifact {

// The reduced layer: 4 query heads on 2 KV heads of 64 (the 27B has 24 on 4
// of 256), an indexer of 4 heads of 32 with 16 rotary dims (8 of 128 with
// 64), a 256-token cache (64 blocks of 4) and 8 blocks per row (512).
constexpr int kHidden = 256;
constexpr int kQueryHeads = 4;
constexpr int kKvHeads = 2;
constexpr int kGroup = kQueryHeads / kKvHeads;
constexpr int kDepth = 64;
constexpr int kIndexHeads = 4;
constexpr int kIndexDim = 32;
constexpr int kRotary = 16;
constexpr int kCacheLen = 256;
constexpr int kBlocks = kCacheLen / 4;
constexpr int kTopk = 8;
constexpr float kEps = 1e-6f;

using tflite::BuiltinOperator_EQUAL;
using tflite::BuiltinOperator_GREATER;
using tflite::BuiltinOperator_GREATER_EQUAL;
using tflite::BuiltinOperator_LESS;
using tflite::BuiltinOperator_LESS_EQUAL;
using tflite::TensorType_BOOL;
using tflite::TensorType_FLOAT32;
using tflite::TensorType_INT32;

// odml.qsa_indexer_update over `rows` tokens, from its contract: the
// diagonal of the mask decides the real rows; the ring keeps the incomplete
// block; completed blocks are averaged, RMS-normalized and rotated into the
// pool; complete blocks below (start_pos + rows) / 4 score
// scale * sum_h relu(q_h . pool[b]) (0 when no more than block_topk blocks
// exist); each row's best block_topk blocks (ties to the lower one) are
// listed in ascending order, then -1. The mask is bool and covers exactly
// the pool's tokens.
inline int32_t AddIndexerDecomposition(Builder& b, int rows, float scale) {
  const int32_t graph =
      b.NewSubgraph(absl::StrCat("odml.qsa_indexer_update.impl_", rows));
  const int S = rows, H = kIndexHeads, D = kIndexDim, MB = kBlocks;
  const int L = kCacheLen, K = kTopk, half = kRotary / 2;
  const int NC = S / 4 + 2;  // blocks a step can complete
  const int32_t q = b.Input(b.F32("q_idx", {1, S, H, D}));
  const int32_t k_new = b.Input(b.F32("k_idx_new", {1, S, D}));
  const int32_t ring = b.Input(b.F32("ring_k", {1, 4, D}));
  const int32_t pool = b.Input(b.F32("block_k_pool", {1, MB, D}));
  const int32_t norm = b.Input(b.F32("k_norm_weight", {D}));
  const int32_t cos = b.Input(b.F32("block_cos", {1, MB, kRotary}));
  const int32_t sin = b.Input(b.F32("block_sin", {1, MB, kRotary}));
  const int32_t start = b.Input(b.I32("start_pos", {1}));
  const int32_t mask = b.Input(b.Bool("attention_mask", {1, 1, S, L}));
  const int32_t zero = b.OneI32("zero", 0);
  const int32_t one = b.OneI32("one", 1);
  const int32_t four = b.OneI32("four", 4);

  // pos0, and the real rows: the last row whose own key the mask shows.
  const int32_t pos0 = b.MaxI(start, zero, {1});
  const int32_t iota_s = b.Iota("iota_s", S);
  const int32_t columns = b.AddI(iota_s, pos0, {S});
  const int32_t in_range =
      b.Compare(BuiltinOperator_LESS, columns, b.OneI32("len", L), {S});
  const int32_t clamped = b.MinI(columns, b.OneI32("last", L - 1), {S});
  const int32_t mask_f = b.Reshape(b.Cast(mask, TensorType_FLOAT32), {S, L});
  const int32_t pairs = b.I32("pairs", {S, 2});
  {
    auto& op = b.Op(tflite::BuiltinOperator_PACK, {iota_s, clamped}, {pairs});
    tflite::PackOptionsT options;
    options.values_count = 2;
    options.axis = 1;
    op.builtin_options.Set(options);
  }
  const int32_t diagonal_f = b.F32("diagonal", {S});
  b.Op(tflite::BuiltinOperator_GATHER_ND, {mask_f, pairs}, {diagonal_f});
  const int32_t diagonal =
      b.And(b.Compare(BuiltinOperator_GREATER, diagonal_f,
                      b.ConstF32("half_f", {1}, {0.5f}), {S}),
            in_range, {S});
  const int32_t row_ends = b.Cast(b.AddI(iota_s, one, {S}), TensorType_FLOAT32);
  const int32_t valid_f =
      b.Reduce(tflite::BuiltinOperator_REDUCE_MAX,
               b.Select(diagonal, row_ends, b.ConstF32("zero_f", {1}, {0.0f}),
                        TensorType_FLOAT32, {S}),
               {0}, true, {1});
  const int32_t valid = b.Cast(valid_f, TensorType_INT32);
  const int32_t end_pos = b.AddI(pos0, valid, {1});
  const int32_t end_blk = b.FloorDivI(end_pos, four, {1});
  const int32_t start_blk = b.FloorDivI(pos0, four, {1});

  // The candidate blocks' keys: the ring for tokens before pos0, else the
  // new keys, from source = ring ++ k_new.
  const int32_t source = b.Concat(
      {b.Reshape(ring, {4, D}), b.Reshape(k_new, {S, D})}, 0, {4 + S, D});
  const int32_t blocks = b.AddI(b.Iota("iota_nc", NC), start_blk, {NC});
  const int32_t iota4 = b.Iota("iota4", 4);
  const int32_t tokens =
      b.AddI(b.MulI(b.Reshape(blocks, {NC, 1}), four, {NC, 1}), iota4, {NC, 4});
  const int32_t is_new =
      b.Compare(BuiltinOperator_GREATER_EQUAL, tokens, pos0, {NC, 4});
  const int32_t new_index =
      b.AddI(b.SubI(tokens, pos0, {NC, 4}), four, {NC, 4});
  const int32_t old_index =
      b.AddI(b.MulI(tokens, zero, {NC, 4}), iota4, {NC, 4});
  const int32_t index = b.MinI(
      b.MaxI(b.Select(is_new, new_index, old_index, TensorType_INT32, {NC, 4}),
             zero, {NC, 4}),
      b.OneI32("source_last", 3 + S), {NC, 4});
  const int32_t members =
      b.Gather(source, b.Reshape(index, {NC * 4}), 0, {NC * 4, D});
  const int32_t pooled =
      b.Reduce(tflite::BuiltinOperator_MEAN, b.Reshape(members, {NC, 4, D}),
               {1}, false, {NC, D});
  const int32_t square =
      b.Reduce(tflite::BuiltinOperator_MEAN, b.MulF(pooled, pooled, {NC, D}),
               {1}, true, {NC, 1});
  const int32_t inv_rms =
      b.Unary(tflite::BuiltinOperator_RSQRT,
              b.AddF(square, b.ConstF32("eps", {1}, {kEps}), {NC, 1}),
              TensorType_FLOAT32, {NC, 1});
  const int32_t normed =
      b.MulF(b.MulF(pooled, inv_rms, {NC, D}), norm, {NC, D});
  const int32_t table_index =
      b.MinI(b.MaxI(blocks, zero, {NC}), b.OneI32("blocks_last", MB - 1), {NC});
  const int32_t cos_c =
      b.Gather(b.Reshape(cos, {MB, kRotary}), table_index, 0, {NC, kRotary});
  const int32_t sin_c =
      b.Gather(b.Reshape(sin, {MB, kRotary}), table_index, 0, {NC, kRotary});
  const int32_t x1 = b.Slice(normed, {0, 0}, {NC, half});
  const int32_t x2 = b.Slice(normed, {0, half}, {NC, half});
  const int32_t rotated_half = b.Concat(
      {b.Unary(tflite::BuiltinOperator_NEG, x2, TensorType_FLOAT32, {NC, half}),
       x1},
      1, {NC, kRotary});
  const int32_t x_rot = b.Slice(normed, {0, 0}, {NC, kRotary});
  const int32_t roped =
      b.AddF(b.MulF(x_rot, cos_c, {NC, kRotary}),
             b.MulF(rotated_half, sin_c, {NC, kRotary}), {NC, kRotary});
  const int32_t rest = b.Slice(normed, {0, kRotary}, {NC, D - kRotary});
  const int32_t compressed = b.Concat({roped, rest}, 1, {NC, D});

  // The pool: the completed blocks' rows replaced.
  const int32_t iota_mb = b.Iota("iota_mb", MB);
  const int32_t relative = b.SubI(iota_mb, start_blk, {MB});
  const int32_t rows_index =
      b.MinI(b.MaxI(relative, zero, {MB}), b.OneI32("nc_last", NC - 1), {MB});
  const int32_t replacements = b.Gather(compressed, rows_index, 0, {MB, D});
  const int32_t updated = b.And(
      b.And(b.Compare(BuiltinOperator_GREATER_EQUAL, iota_mb, start_blk, {MB}),
            b.Compare(BuiltinOperator_LESS, iota_mb, end_blk, {MB}), {MB}),
      b.Compare(BuiltinOperator_LESS, relative, b.OneI32("nc", NC), {MB}),
      {MB});
  const int32_t new_pool =
      b.Reshape(b.Select(b.Reshape(updated, {MB, 1}), replacements,
                         b.Reshape(pool, {MB, D}), TensorType_FLOAT32, {MB, D}),
                {1, MB, D});

  // The ring: the incomplete block at end_pos (old keys only while no block
  // completed), zero elsewhere; untouched without real rows.
  const int32_t ring_tokens = b.AddI(b.MulI(end_blk, four, {1}), iota4, {4});
  const int32_t from_new =
      b.And(b.Compare(BuiltinOperator_GREATER_EQUAL, ring_tokens, pos0, {4}),
            b.Compare(BuiltinOperator_LESS, ring_tokens, end_pos, {4}), {4});
  const int32_t from_old =
      b.And(b.Compare(BuiltinOperator_LESS, ring_tokens, pos0, {4}),
            b.Compare(BuiltinOperator_EQUAL, end_blk, start_blk, {1}), {4});
  const int32_t ring_index =
      b.Select(from_new, b.AddI(b.SubI(ring_tokens, pos0, {4}), four, {4}),
               b.Select(from_old, iota4, b.OneI32("zero_row", 4 + S),
                        TensorType_INT32, {4}),
               TensorType_INT32, {4});
  const int32_t source_zero = b.Concat(
      {source, b.ConstF32("zeros_row", {1, D}, std::vector<float>(D, 0.0f))}, 0,
      {5 + S, D});
  const int32_t ring_new =
      b.Gather(source_zero,
               b.MinI(b.MaxI(ring_index, zero, {4}),
                      b.OneI32("zero_row_last", 4 + S), {4}),
               0, {4, D});
  const int32_t new_ring = b.Reshape(
      b.Select(b.Compare(BuiltinOperator_GREATER, valid, zero, {1}), ring_new,
               b.Reshape(ring, {4, D}), TensorType_FLOAT32, {4, D}),
      {1, 4, D});

  // Scores over the blocks below the bound, complete for the row.
  const int32_t bound = b.MinI(
      b.MaxI(b.FloorDivI(b.AddI(pos0, b.OneI32("rows", S), {1}), four, {1}),
             zero, {1}),
      b.OneI32("max_blocks", MB), {1});
  const int32_t dense =
      b.Compare(BuiltinOperator_LESS_EQUAL, bound, b.OneI32("k", K), {1});
  const int32_t dots = b.BatchMatMul(b.Reshape(q, {1, S * H, D}), new_pool,
                                     true, {1, S * H, MB});
  const int32_t relu =
      b.Unary(tflite::BuiltinOperator_RELU, b.Reshape(dots, {S, H, MB}),
              TensorType_FLOAT32, {S, H, MB});
  const int32_t summed =
      b.Reduce(tflite::BuiltinOperator_SUM, relu, {1}, false, {S, MB});
  const int32_t scores =
      b.MulF(summed, b.ConstF32("scale", {1}, {scale}), {S, MB});
  const int32_t complete =
      b.Compare(BuiltinOperator_GREATER,
                b.Reduce(tflite::BuiltinOperator_REDUCE_MIN,
                         b.Reshape(mask_f, {S, MB, 4}), {2}, false, {S, MB}),
                b.ConstF32("half_f2", {1}, {0.5f}), {S, MB});
  const int32_t candidate = b.And(
      complete, b.Compare(BuiltinOperator_LESS, iota_mb, bound, {MB}), {S, MB});
  const int32_t value = b.Select(
      candidate,
      b.Select(dense, b.ConstF32("zero_score", {1}, {0.0f}), scores,
               TensorType_FLOAT32, {S, MB}),
      b.ConstF32("hidden_score", {1}, {-1.0e4f}), TensorType_FLOAT32, {S, MB});

  // The selection, then its blocks in ascending order (the second top-k
  // sorts the negated indices), then -1.
  const int32_t k_scalar = b.ScalarI32("k_scalar", K);
  const int32_t top_values = b.F32("top_values", {S, K});
  const int32_t top_indices = b.I32("top_indices", {S, K});
  b.Op(tflite::BuiltinOperator_TOPK_V2, {value, k_scalar},
       {top_values, top_indices});
  const int32_t taken =
      b.Compare(BuiltinOperator_GREATER, top_values,
                b.ConstF32("taken_below", {1}, {-9999.0f}), {S, K});
  const int32_t marked = b.Select(taken, top_indices, b.OneI32("past", MB),
                                  TensorType_INT32, {S, K});
  const int32_t negated =
      b.Unary(tflite::BuiltinOperator_NEG, marked, TensorType_INT32, {S, K});
  const int32_t sorted_values = b.I32("sorted_values", {S, K});
  const int32_t sorted_indices = b.I32("sorted_indices", {S, K});
  b.Op(tflite::BuiltinOperator_TOPK_V2, {negated, k_scalar},
       {sorted_values, sorted_indices});
  const int32_t ascending = b.Unary(tflite::BuiltinOperator_NEG, sorted_values,
                                    TensorType_INT32, {S, K});
  const int32_t topk = b.Reshape(
      b.Select(b.Compare(BuiltinOperator_GREATER_EQUAL, ascending,
                         b.OneI32("past2", MB), {S, K}),
               b.OneI32("minus_one", -1), ascending, TensorType_INT32, {S, K}),
      {1, S, K});
  b.Output(topk);
  b.Output(new_ring);
  b.Output(new_pool);
  return graph;
}

// odml.cache_update: k [1, H, n, D] into the key cache at k_start, v
// [1, H, n, D] transposed into the value cache [1, H, D, L] at v_start.
inline int32_t AddCacheUpdateDecomposition(Builder& b, int rows) {
  const int32_t graph =
      b.NewSubgraph(absl::StrCat("odml.cache_update.impl_", rows));
  const int32_t k = b.Input(b.F32("k", {1, kKvHeads, rows, kDepth}));
  const int32_t v = b.Input(b.F32("v", {1, kKvHeads, rows, kDepth}));
  b.Input(b.I32("param_tensor", {1, 1, 1, 7}));
  const int32_t k_cache =
      b.Input(b.F32("k_cache", {1, kKvHeads, kCacheLen, kDepth}));
  const int32_t v_cache =
      b.Input(b.F32("v_cache", {1, kKvHeads, kDepth, kCacheLen}));
  const int32_t k_start = b.Input(b.I32("k_start", {4}));
  const int32_t v_start = b.Input(b.I32("v_start", {4}));
  const int32_t k_out = b.F32("k_cache_out", {1, kKvHeads, kCacheLen, kDepth});
  b.Op(tflite::BuiltinOperator_DYNAMIC_UPDATE_SLICE, {k_cache, k, k_start},
       {k_out});
  const int32_t v_columns =
      b.Transpose(v, {0, 1, 3, 2}, {1, kKvHeads, kDepth, rows});
  const int32_t v_out = b.F32("v_cache_out", {1, kKvHeads, kDepth, kCacheLen});
  b.Op(tflite::BuiltinOperator_DYNAMIC_UPDATE_SLICE,
       {v_cache, v_columns, v_start}, {v_out});
  b.Output(k_out);
  b.Output(v_out);
  return graph;
}

// odml.sparse_sdpa_transposed: attention over the keys of the selected blocks
// and of every partly visible block, restricted to visible keys; zero for a
// row that sees none.
inline int32_t AddSparseDecomposition(Builder& b, int rows, float scale) {
  const int32_t graph =
      b.NewSubgraph(absl::StrCat("odml.sparse_sdpa_transposed.impl_", rows));
  const int S = rows, L = kCacheLen, MB = kBlocks, K = kTopk;
  const int32_t q = b.Input(b.F32("query", {1, kQueryHeads, S, kDepth}));
  const int32_t k = b.Input(b.F32("key_cache", {1, kKvHeads, L, kDepth}));
  const int32_t v = b.Input(b.F32("value_cache", {1, kKvHeads, kDepth, L}));
  const int32_t topk = b.Input(b.I32("topk_block_indices", {1, S, K}));
  const int32_t mask = b.Input(b.Bool("attention_mask", {1, 1, S, L}));
  const int32_t selected_hot = b.F32("selected_hot", {1, S, K, MB});
  {
    auto& op = b.Op(tflite::BuiltinOperator_ONE_HOT,
                    {topk, b.ScalarI32("depth", MB), b.ScalarF32("on", 1.0f),
                     b.ScalarF32("off", 0.0f)},
                    {selected_hot});
    tflite::OneHotOptionsT options;
    options.axis = -1;
    op.builtin_options.Set(options);
  }
  const int32_t selected = b.Reduce(tflite::BuiltinOperator_REDUCE_MAX,
                                    selected_hot, {2}, false, {1, S, MB});
  const int32_t mask_f = b.Cast(mask, TensorType_FLOAT32);
  const int32_t seen =
      b.Reduce(tflite::BuiltinOperator_SUM, b.Reshape(mask_f, {1, S, MB, 4}),
               {3}, false, {1, S, MB});
  const int32_t partly =
      b.Cast(b.And(b.Compare(BuiltinOperator_GREATER, seen,
                             b.ConstF32("half", {1}, {0.5f}), {1, S, MB}),
                   b.Compare(BuiltinOperator_LESS, seen,
                             b.ConstF32("three_half", {1}, {3.5f}), {1, S, MB}),
                   {1, S, MB}),
             TensorType_FLOAT32);
  const int32_t kept = b.Binary(tflite::BuiltinOperator_MAXIMUM, selected,
                                partly, TensorType_FLOAT32, {1, S, MB});
  const int32_t kept_keys = b.Reshape(
      b.Tile(b.Reshape(kept, {1, S, MB, 1}), {1, 1, 1, 4}, {1, S, MB, 4}),
      {1, 1, S, L});
  const int32_t attended_f = b.MulF(kept_keys, mask_f, {1, 1, S, L});
  const int32_t attended_rows =
      b.Tile(attended_f, {1, 1, kGroup, 1}, {1, 1, kGroup * S, L});
  const int32_t attended =
      b.Compare(BuiltinOperator_GREATER, attended_rows,
                b.ConstF32("half2", {1}, {0.5f}), {1, 1, kGroup * S, L});
  const int32_t grouped = b.Reshape(q, {1, kKvHeads, kGroup * S, kDepth});
  const int32_t logits =
      b.MulF(b.BatchMatMul(grouped, k, true, {1, kKvHeads, kGroup * S, L}),
             b.ConstF32("scale", {1}, {scale}), {1, kKvHeads, kGroup * S, L});
  const int32_t masked =
      b.Select(attended, logits, b.ConstF32("fill", {1}, {-1.0e30f}),
               TensorType_FLOAT32, {1, kKvHeads, kGroup * S, L});
  const int32_t probs = b.F32("probs", {1, kKvHeads, kGroup * S, L});
  {
    auto& op = b.Op(tflite::BuiltinOperator_SOFTMAX, {masked}, {probs});
    tflite::SoftmaxOptionsT options;
    options.beta = 1.0f;
    op.builtin_options.Set(options);
  }
  const int32_t values =
      b.BatchMatMul(probs, v, true, {1, kKvHeads, kGroup * S, kDepth});
  const int32_t any = b.Reduce(tflite::BuiltinOperator_REDUCE_MAX,
                               attended_rows, {3}, true, {1, 1, kGroup * S, 1});
  const int32_t kept_values =
      b.MulF(values, any, {1, kKvHeads, kGroup * S, kDepth});
  const int32_t out =
      b.Transpose(b.Reshape(kept_values, {1, kQueryHeads, S, kDepth}),
                  {0, 2, 1, 3}, {1, S, kQueryHeads, kDepth});
  b.Output(out);
  return graph;
}

}  // namespace litert::nvidia::artifact

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_QSA_DECOMPOSITIONS_H_

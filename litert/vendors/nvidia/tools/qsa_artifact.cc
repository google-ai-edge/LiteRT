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

// qsa_artifact: a model with the attention layers of Qwen 4 at reduced sizes
// and random weights, for checking the NVIDIA backend's QSA lowering end to
// end without a Qwen 4 export. Each of its two layers is an RMS norm, a fused
// projection to the queries, keys and values of the attention and of its
// sparse-attention indexer, odml.qsa_indexer_update, odml.cache_update,
// odml.sparse_sdpa_transposed reading the updated caches with the indexer's
// selection, an output projection and a residual add. Every composite has a
// decomposition in TFLite builtin ops (written here from the composites'
// contracts, ml_drift_delegate/delegate/composite/*_parser.h); LiteRT's CPU
// backend runs the XNNPACK delegate's own kernels for the QSA composites
// (tflite/delegates/xnnpack/qsa_delegate_kernel.cc) and the NVIDIA plugin
// its kernels, so the comparison is of two independent implementations.
// Signatures "prefill_16" and "decode" share the states, named as LiteRT-LM
// names caches (kv_cache_*), which LiteRT-LM binds in place.
//
// USAGE
//
//   qsa_artifact --write_model=FILE.tflite
//   qsa_artifact --model=FILE.tflite --accelerator=cpu --out_dir=DIR
//   qsa_artifact --model=FILE.tflite --accelerator=npu \
//                --runtime_libs=DIR --out_dir=DIR
//   qsa_artifact --compare --cpu_dir=DIR --npu_dir=DIR
//
// A run feeds a 40-token prompt in prefill chunks (the last one
// right-padded) and --decode_steps decode steps, with pseudo-random
// embeddings from a fixed seed, carrying every state between steps (in place
// on the NPU, as LiteRT-LM does; copied back on the CPU), and writes each
// step's hidden states, selections, indexer rings and pools. --compare
// requires the selections to agree and the states and hidden states to
// agree to --tolerance; the KV caches stay in each backend's own layout.
// qsa_artifact_check.sh runs the whole check.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "tflite/schema/schema_generated.h"

ABSL_FLAG(std::string, write_model, "", "Write the model to this file.");
ABSL_FLAG(std::string, model, "", "The model to run.");
ABSL_FLAG(std::string, accelerator, "cpu", "cpu, or npu (the NVIDIA plugin).");
ABSL_FLAG(std::string, runtime_libs, "",
          "The directory of the NVIDIA compiler plugin and dispatch library.");
ABSL_FLAG(std::string, out_dir, "", "Where a run writes its outputs.");
ABSL_FLAG(int, decode_steps, 12, "Decode steps after the prompt.");
ABSL_FLAG(bool, compare, false, "Compare the outputs of two runs.");
ABSL_FLAG(std::string, cpu_dir, "", "The CPU run's outputs.");
ABSL_FLAG(std::string, npu_dir, "", "The NPU run's outputs.");
ABSL_FLAG(double, tolerance, 2e-2,
          "Largest error allowed, relative to the largest magnitude of an "
          "output.");

namespace {

using litert::Error;
using litert::Expected;

// The reduced layer: 4 query heads on 2 KV heads of 64 (the 27B has 24 on 4
// of 256), an indexer of 4 heads of 32 with 16 rotary dims (8 of 128 with
// 64), a 256-token cache (64 blocks of 4) and 8 blocks per row (512).
constexpr int kHidden = 256;
constexpr int kLayers = 2;
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
constexpr int kPrefill = 16;
constexpr float kEps = 1e-6f;
constexpr int kProjection = kQueryHeads * kDepth + 2 * kKvHeads * kDepth +
                            kIndexHeads * kIndexDim + kIndexDim;
constexpr int kPrompt = 40;

using Shape = std::vector<int32_t>;

class Random {
 public:
  explicit Random(uint32_t seed) : state_(seed * 2654435761u + 7u) {}
  float Next() {  // [-1, 1)
    state_ = state_ * 1664525u + 1013904223u;
    return static_cast<float>(state_ >> 8) / 8388608.0f - 1.0f;
  }
  std::vector<float> Values(size_t count, float scale) {
    std::vector<float> values(count);
    for (auto& value : values) value = Next() * scale;
    return values;
  }

 private:
  uint32_t state_;
};

// A TFLite model under construction, one subgraph at a time.
class Builder {
 public:
  Builder() {
    model_.version = 3;
    model_.buffers.push_back(std::make_unique<tflite::BufferT>());
  }

  int32_t NewSubgraph(const std::string& name) {
    auto subgraph = std::make_unique<tflite::SubGraphT>();
    subgraph->name = name;
    model_.subgraphs.push_back(std::move(subgraph));
    current_ = static_cast<int32_t>(model_.subgraphs.size()) - 1;
    return current_;
  }
  void Select(int32_t subgraph) { current_ = subgraph; }
  tflite::SubGraphT& graph() { return *model_.subgraphs[current_]; }

  int32_t Tensor(const std::string& name, tflite::TensorType type,
                 const Shape& shape) {
    auto tensor = std::make_unique<tflite::TensorT>();
    tensor->name = name;
    tensor->type = type;
    tensor->shape = shape;
    tensor->buffer = 0;
    graph().tensors.push_back(std::move(tensor));
    return static_cast<int32_t>(graph().tensors.size()) - 1;
  }
  int32_t F32(const std::string& name, const Shape& shape) {
    return Tensor(name, tflite::TensorType_FLOAT32, shape);
  }
  int32_t I32(const std::string& name, const Shape& shape) {
    return Tensor(name, tflite::TensorType_INT32, shape);
  }
  int32_t Bool(const std::string& name, const Shape& shape) {
    return Tensor(name, tflite::TensorType_BOOL, shape);
  }
  int32_t Input(int32_t tensor) {
    graph().inputs.push_back(tensor);
    return tensor;
  }
  void Output(int32_t tensor) { graph().outputs.push_back(tensor); }

  template <typename T>
  int32_t Constant(const std::string& name, tflite::TensorType type,
                   const Shape& shape, const std::vector<T>& values) {
    const int32_t tensor = Tensor(name, type, shape);
    auto buffer = std::make_unique<tflite::BufferT>();
    buffer->data.resize(values.size() * sizeof(T));
    std::memcpy(buffer->data.data(), values.data(), buffer->data.size());
    model_.buffers.push_back(std::move(buffer));
    graph().tensors[tensor]->buffer =
        static_cast<uint32_t>(model_.buffers.size()) - 1;
    return tensor;
  }
  int32_t ConstF32(const std::string& name, const Shape& shape,
                   const std::vector<float>& values) {
    return Constant(name, tflite::TensorType_FLOAT32, shape, values);
  }
  int32_t ConstI32(const std::string& name, const Shape& shape,
                   const std::vector<int32_t>& values) {
    return Constant(name, tflite::TensorType_INT32, shape, values);
  }
  int32_t ScalarI32(const std::string& name, int32_t value) {
    return ConstI32(name, {}, {value});
  }
  int32_t OneI32(const std::string& name, int32_t value) {
    return ConstI32(name, {1}, {value});
  }
  int32_t ScalarF32(const std::string& name, float value) {
    return ConstF32(name, {}, {value});
  }
  int32_t Iota(const std::string& name, int32_t count) {
    std::vector<int32_t> values(count);
    for (int32_t i = 0; i < count; ++i) values[i] = i;
    return ConstI32(name, {count}, values);
  }

  tflite::OperatorT& Op(tflite::BuiltinOperator code,
                        std::vector<int32_t> inputs,
                        std::vector<int32_t> outputs) {
    size_t index = 0;
    for (; index < model_.operator_codes.size(); ++index) {
      if (model_.operator_codes[index]->builtin_code == code) break;
    }
    if (index == model_.operator_codes.size()) {
      auto op_code = std::make_unique<tflite::OperatorCodeT>();
      op_code->builtin_code = code;
      op_code->deprecated_builtin_code = static_cast<int8_t>(
          code < tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES
              ? code
              : tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES);
      op_code->version = 1;
      model_.operator_codes.push_back(std::move(op_code));
    }
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = static_cast<uint32_t>(index);
    op->inputs = std::move(inputs);
    op->outputs = std::move(outputs);
    graph().operators.push_back(std::move(op));
    return *graph().operators.back();
  }

  // Ops whose result is a new tensor of the given type and shape.
  int32_t Unary(tflite::BuiltinOperator code, int32_t x,
                tflite::TensorType type, const Shape& shape) {
    const int32_t out = Tensor(Name(), type, shape);
    Op(code, {x}, {out});
    return out;
  }
  int32_t Binary(tflite::BuiltinOperator code, int32_t a, int32_t b,
                 tflite::TensorType type, const Shape& shape) {
    const int32_t out = Tensor(Name(), type, shape);
    auto& op = Op(code, {a, b}, {out});
    switch (code) {
      case tflite::BuiltinOperator_ADD: {
        tflite::AddOptionsT options;
        op.builtin_options.Set(options);
        break;
      }
      case tflite::BuiltinOperator_SUB: {
        tflite::SubOptionsT options;
        op.builtin_options.Set(options);
        break;
      }
      case tflite::BuiltinOperator_MUL: {
        tflite::MulOptionsT options;
        op.builtin_options.Set(options);
        break;
      }
      default:
        break;
    }
    return out;
  }
  int32_t AddF(int32_t a, int32_t b, const Shape& s) {
    return Binary(tflite::BuiltinOperator_ADD, a, b, tflite::TensorType_FLOAT32,
                  s);
  }
  int32_t MulF(int32_t a, int32_t b, const Shape& s) {
    return Binary(tflite::BuiltinOperator_MUL, a, b, tflite::TensorType_FLOAT32,
                  s);
  }
  int32_t AddI(int32_t a, int32_t b, const Shape& s) {
    return Binary(tflite::BuiltinOperator_ADD, a, b, tflite::TensorType_INT32,
                  s);
  }
  int32_t SubI(int32_t a, int32_t b, const Shape& s) {
    return Binary(tflite::BuiltinOperator_SUB, a, b, tflite::TensorType_INT32,
                  s);
  }
  int32_t MulI(int32_t a, int32_t b, const Shape& s) {
    return Binary(tflite::BuiltinOperator_MUL, a, b, tflite::TensorType_INT32,
                  s);
  }
  int32_t MaxI(int32_t a, int32_t b, const Shape& s) {
    return Binary(tflite::BuiltinOperator_MAXIMUM, a, b,
                  tflite::TensorType_INT32, s);
  }
  int32_t MinI(int32_t a, int32_t b, const Shape& s) {
    return Binary(tflite::BuiltinOperator_MINIMUM, a, b,
                  tflite::TensorType_INT32, s);
  }
  int32_t FloorDivI(int32_t a, int32_t b, const Shape& s) {
    return Binary(tflite::BuiltinOperator_FLOOR_DIV, a, b,
                  tflite::TensorType_INT32, s);
  }
  int32_t Compare(tflite::BuiltinOperator code, int32_t a, int32_t b,
                  const Shape& s) {
    return Binary(code, a, b, tflite::TensorType_BOOL, s);
  }
  int32_t And(int32_t a, int32_t b, const Shape& s) {
    return Binary(tflite::BuiltinOperator_LOGICAL_AND, a, b,
                  tflite::TensorType_BOOL, s);
  }
  int32_t Select(int32_t condition, int32_t a, int32_t b,
                 tflite::TensorType type, const Shape& s) {
    const int32_t out = Tensor(Name(), type, s);
    Op(tflite::BuiltinOperator_SELECT_V2, {condition, a, b}, {out});
    return out;
  }
  int32_t Reshape(int32_t x, const Shape& shape) {
    const auto type = graph().tensors[x]->type;
    const int32_t out = Tensor(Name(), type, shape);
    const int32_t shape_tensor =
        ConstI32(Name(), {static_cast<int32_t>(shape.size())},
                 std::vector<int32_t>(shape));
    auto& op = Op(tflite::BuiltinOperator_RESHAPE, {x, shape_tensor}, {out});
    tflite::ReshapeOptionsT options;
    options.new_shape = shape;
    op.builtin_options.Set(options);
    return out;
  }
  int32_t Transpose(int32_t x, const std::vector<int32_t>& perm,
                    const Shape& shape) {
    const auto type = graph().tensors[x]->type;
    const int32_t out = Tensor(Name(), type, shape);
    const int32_t perm_tensor =
        ConstI32(Name(), {static_cast<int32_t>(perm.size())}, perm);
    Op(tflite::BuiltinOperator_TRANSPOSE, {x, perm_tensor}, {out});
    return out;
  }
  int32_t Slice(int32_t x, const std::vector<int32_t>& begin,
                const Shape& size) {
    const auto type = graph().tensors[x]->type;
    const int32_t out = Tensor(Name(), type, size);
    const int32_t b =
        ConstI32(Name(), {static_cast<int32_t>(begin.size())}, begin);
    const int32_t s = ConstI32(Name(), {static_cast<int32_t>(size.size())},
                               std::vector<int32_t>(size));
    Op(tflite::BuiltinOperator_SLICE, {x, b, s}, {out});
    return out;
  }
  int32_t Concat(const std::vector<int32_t>& parts, int32_t axis,
                 const Shape& shape) {
    const auto type = graph().tensors[parts[0]]->type;
    const int32_t out = Tensor(Name(), type, shape);
    auto& op = Op(tflite::BuiltinOperator_CONCATENATION, parts, {out});
    tflite::ConcatenationOptionsT options;
    options.axis = axis;
    op.builtin_options.Set(options);
    return out;
  }
  int32_t Gather(int32_t params, int32_t indices, int32_t axis,
                 const Shape& shape) {
    const auto type = graph().tensors[params]->type;
    const int32_t out = Tensor(Name(), type, shape);
    auto& op = Op(tflite::BuiltinOperator_GATHER, {params, indices}, {out});
    tflite::GatherOptionsT options;
    options.axis = axis;
    op.builtin_options.Set(options);
    return out;
  }
  int32_t Reduce(tflite::BuiltinOperator code, int32_t x,
                 const std::vector<int32_t>& axes, bool keep_dims,
                 const Shape& shape) {
    const auto type = graph().tensors[x]->type;
    const int32_t out = Tensor(Name(), type, shape);
    const int32_t axes_tensor =
        ConstI32(Name(), {static_cast<int32_t>(axes.size())}, axes);
    auto& op = Op(code, {x, axes_tensor}, {out});
    tflite::ReducerOptionsT options;
    options.keep_dims = keep_dims;
    op.builtin_options.Set(options);
    return out;
  }
  int32_t Cast(int32_t x, tflite::TensorType type) {
    const int32_t out = Tensor(Name(), type, graph().tensors[x]->shape);
    auto& op = Op(tflite::BuiltinOperator_CAST, {x}, {out});
    tflite::CastOptionsT options;
    options.in_data_type = graph().tensors[x]->type;
    options.out_data_type = type;
    op.builtin_options.Set(options);
    return out;
  }
  int32_t BatchMatMul(int32_t a, int32_t b, bool adj_y, const Shape& shape) {
    const int32_t out = F32(Name(), shape);
    auto& op = Op(tflite::BuiltinOperator_BATCH_MATMUL, {a, b}, {out});
    tflite::BatchMatMulOptionsT options;
    options.adj_y = adj_y;
    op.builtin_options.Set(options);
    return out;
  }
  int32_t FullyConnected(int32_t x, int32_t weights, const Shape& shape) {
    const int32_t out = F32(Name(), shape);
    auto& op =
        Op(tflite::BuiltinOperator_FULLY_CONNECTED, {x, weights, -1}, {out});
    tflite::FullyConnectedOptionsT options;
    op.builtin_options.Set(options);
    return out;
  }
  int32_t Tile(int32_t x, const std::vector<int32_t>& multiples,
               const Shape& shape) {
    const auto type = graph().tensors[x]->type;
    const int32_t out = Tensor(Name(), type, shape);
    const int32_t m =
        ConstI32(Name(), {static_cast<int32_t>(multiples.size())}, multiples);
    Op(tflite::BuiltinOperator_TILE, {x, m}, {out});
    return out;
  }

  void Composite(const std::string& name, int32_t decomposition,
                 const std::function<void(flexbuffers::Builder*)>& attributes,
                 std::vector<int32_t> inputs, std::vector<int32_t> outputs) {
    auto& op = Op(tflite::BuiltinOperator_STABLEHLO_COMPOSITE,
                  std::move(inputs), std::move(outputs));
    flexbuffers::Builder fbb;
    const auto map = fbb.StartMap();
    attributes(&fbb);
    fbb.EndMap(map);
    fbb.Finish();
    tflite::StableHLOCompositeOptionsT options;
    options.name = name;
    options.decomposition_subgraph_index = decomposition;
    options.composite_attributes = fbb.GetBuffer();
    options.composite_attributes_format =
        tflite::CustomOptionsFormat_FLEXBUFFERS;
    op.builtin_options_2.Set(std::move(options));
  }

  void Signature(const std::string& key, int32_t subgraph,
                 const std::vector<std::pair<std::string, int32_t>>& inputs,
                 const std::vector<std::pair<std::string, int32_t>>& outputs) {
    auto signature = std::make_unique<tflite::SignatureDefT>();
    signature->signature_key = key;
    signature->subgraph_index = static_cast<uint32_t>(subgraph);
    for (const auto& [name, tensor] : inputs) {
      auto map = std::make_unique<tflite::TensorMapT>();
      map->name = name;
      map->tensor_index = static_cast<uint32_t>(tensor);
      signature->inputs.push_back(std::move(map));
    }
    for (const auto& [name, tensor] : outputs) {
      auto map = std::make_unique<tflite::TensorMapT>();
      map->name = name;
      map->tensor_index = static_cast<uint32_t>(tensor);
      signature->outputs.push_back(std::move(map));
    }
    model_.signature_defs.push_back(std::move(signature));
  }

  std::string Pack() {
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model_));
    return std::string(
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        builder.GetSize());
  }

 private:
  std::string Name() { return absl::StrCat("t", next_name_++); }

  tflite::ModelT model_;
  int32_t current_ = -1;
  int next_name_ = 0;
};

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
int32_t AddIndexerDecomposition(Builder& b, int rows, float scale) {
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
int32_t AddCacheUpdateDecomposition(Builder& b, int rows) {
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
int32_t AddSparseDecomposition(Builder& b, int rows, float scale) {
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

struct LayerWeights {
  std::vector<float> norm, projection, output, k_norm, cos, sin;
};

std::vector<LayerWeights> MakeWeights() {
  std::vector<LayerWeights> layers(kLayers);
  for (int l = 0; l < kLayers; ++l) {
    Random random(100 + l);
    auto& w = layers[l];
    w.norm = random.Values(kHidden, 0.2f);
    for (auto& x : w.norm) x += 1.0f;
    w.projection = random.Values(static_cast<size_t>(kProjection) * kHidden,
                                 1.0f / std::sqrt(static_cast<float>(kHidden)));
    w.output = random.Values(
        static_cast<size_t>(kHidden) * kQueryHeads * kDepth,
        0.5f / std::sqrt(static_cast<float>(kQueryHeads * kDepth)));
    w.k_norm = random.Values(kIndexDim, 0.2f);
    for (auto& x : w.k_norm) x += 1.0f;
    w.cos.resize(kBlocks * kRotary);
    w.sin.resize(kBlocks * kRotary);
    for (int b = 0; b < kBlocks; ++b) {
      for (int i = 0; i < kRotary; ++i) {
        const double angle =
            4.0 * b * std::pow(10000.0, -2.0 * (i % (kRotary / 2)) / kRotary);
        w.cos[b * kRotary + i] = static_cast<float>(std::cos(angle));
        w.sin[b * kRotary + i] = static_cast<float>(std::sin(angle));
      }
    }
  }
  return layers;
}

std::string StateName(const char* kind, int layer) {
  return absl::StrCat("kv_cache_", kind, "_", layer);
}

// One signature over `rows` tokens: inputs embeddings, start_pos, mask,
// param_tensor, k_start, v_start, then each layer's states (ring, pool, key
// cache, value cache); outputs hidden, each layer's topk, then the states.
void AddSignature(Builder& b, const std::string& key, int rows,
                  const std::vector<LayerWeights>& weights) {
  const float index_scale = 1.0f / std::sqrt(static_cast<float>(kIndexDim));
  const float attention_scale = 1.0f / std::sqrt(static_cast<float>(kDepth));
  const int32_t indexer = AddIndexerDecomposition(b, rows, index_scale);
  const int32_t cache_update = AddCacheUpdateDecomposition(b, rows);
  const int32_t sparse = AddSparseDecomposition(b, rows, attention_scale);
  const int32_t graph = b.NewSubgraph(key);
  const int n = rows;
  std::vector<std::pair<std::string, int32_t>> inputs, outputs;
  const auto input = [&](const std::string& name, int32_t tensor) {
    b.Input(tensor);
    inputs.push_back({name, tensor});
    return tensor;
  };
  int32_t x = input("embeddings", b.F32("embeddings", {1, n, kHidden}));
  const int32_t start = input("start_pos", b.I32("start_pos", {1}));
  const int32_t mask = input("mask", b.Bool("mask", {1, 1, n, kCacheLen}));
  const int32_t param = input("param_tensor", b.I32("param", {1, 1, 1, 7}));
  const int32_t k_start = input("k_start", b.I32("k_start", {4}));
  const int32_t v_start = input("v_start", b.I32("v_start", {4}));
  std::vector<int32_t> state_in, state_out;
  for (int l = 0; l < kLayers; ++l) {
    for (const auto& [kind, shape] : std::vector<std::pair<const char*, Shape>>{
             {"qsa_ring", {1, 4, kIndexDim}},
             {"qsa_pool", {1, kBlocks, kIndexDim}},
             {"k", {1, kKvHeads, kCacheLen, kDepth}},
             {"v", {1, kKvHeads, kDepth, kCacheLen}}}) {
      const std::string name = StateName(kind, l);
      state_in.push_back(input(name, b.F32(name, shape)));
    }
  }
  std::vector<int32_t> topks;
  for (int l = 0; l < kLayers; ++l) {
    const auto& w = weights[l];
    const int32_t ring = state_in[4 * l + 0];
    const int32_t pool = state_in[4 * l + 1];
    const int32_t k_cache = state_in[4 * l + 2];
    const int32_t v_cache = state_in[4 * l + 3];
    // RMS norm.
    const int32_t square =
        b.Reduce(tflite::BuiltinOperator_MEAN, b.MulF(x, x, {1, n, kHidden}),
                 {2}, true, {1, n, 1});
    const int32_t inv =
        b.Unary(tflite::BuiltinOperator_RSQRT,
                b.AddF(square, b.ConstF32(absl::StrCat("eps", l), {1}, {kEps}),
                       {1, n, 1}),
                TensorType_FLOAT32, {1, n, 1});
    const int32_t normed =
        b.MulF(b.MulF(x, inv, {1, n, kHidden}),
               b.ConstF32(absl::StrCat("norm", l), {kHidden}, w.norm),
               {1, n, kHidden});
    // The fused projection and its parts.
    const int32_t projected =
        b.FullyConnected(b.Reshape(normed, {n, kHidden}),
                         b.ConstF32(absl::StrCat("projection", l),
                                    {kProjection, kHidden}, w.projection),
                         {n, kProjection});
    int offset = 0;
    const auto part = [&](int width) {
      const int32_t slice = b.Slice(projected, {0, offset}, {n, width});
      offset += width;
      return slice;
    };
    const int32_t q = b.Reshape(
        b.Transpose(
            b.Reshape(part(kQueryHeads * kDepth), {n, kQueryHeads, kDepth}),
            {1, 0, 2}, {kQueryHeads, n, kDepth}),
        {1, kQueryHeads, n, kDepth});
    const int32_t k = b.Reshape(
        b.Transpose(b.Reshape(part(kKvHeads * kDepth), {n, kKvHeads, kDepth}),
                    {1, 0, 2}, {kKvHeads, n, kDepth}),
        {1, kKvHeads, n, kDepth});
    const int32_t v = b.Reshape(
        b.Transpose(b.Reshape(part(kKvHeads * kDepth), {n, kKvHeads, kDepth}),
                    {1, 0, 2}, {kKvHeads, n, kDepth}),
        {1, kKvHeads, n, kDepth});
    const int32_t q_idx = b.Reshape(part(kIndexHeads * kIndexDim),
                                    {1, n, kIndexHeads, kIndexDim});
    const int32_t k_idx = b.Reshape(part(kIndexDim), {1, n, kIndexDim});

    const int32_t topk = b.I32(absl::StrCat("topk_", l), {1, n, kTopk});
    const int32_t ring_out =
        b.F32(StateName("qsa_ring", l) + "_out", {1, 4, kIndexDim});
    const int32_t pool_out =
        b.F32(StateName("qsa_pool", l) + "_out", {1, kBlocks, kIndexDim});
    b.Composite(
        "odml.qsa_indexer_update", indexer,
        [&](flexbuffers::Builder* a) {
          a->Int("block_topk", kTopk);
          a->Int("compress_ratio", 4);
          a->Float("rms_norm_eps", kEps);
          a->Float("scale", index_scale);
        },
        {q_idx, k_idx, ring, pool,
         b.ConstF32(absl::StrCat("k_norm", l), {kIndexDim}, w.k_norm),
         b.ConstF32(absl::StrCat("block_cos", l), {1, kBlocks, kRotary}, w.cos),
         b.ConstF32(absl::StrCat("block_sin", l), {1, kBlocks, kRotary}, w.sin),
         start, mask},
        {topk, ring_out, pool_out});
    const int32_t k_out =
        b.F32(StateName("k", l) + "_out", {1, kKvHeads, kCacheLen, kDepth});
    const int32_t v_out =
        b.F32(StateName("v", l) + "_out", {1, kKvHeads, kDepth, kCacheLen});
    b.Composite("odml.cache_update", cache_update,
                [&](flexbuffers::Builder* a) {
                  a->Int("cache_size", kCacheLen);
                  a->Int("head_size", kDepth);
                  a->Int("k_cache_ts_idx", 2);
                  a->Int("k_update_ts_idx", 2);
                  a->Int("kv_cache_batch_size", kKvHeads);
                  a->Int("v_cache_ts_idx", 3);
                  a->Int("v_update_ts_idx", 2);
                },
                {k, v, param, k_cache, v_cache, k_start, v_start},
                {k_out, v_out});
    const int32_t attention =
        b.F32(absl::StrCat("attention_", l), {1, n, kQueryHeads, kDepth});
    b.Composite("odml.sparse_sdpa_transposed", sparse,
                [&](flexbuffers::Builder* a) {
                  a->Int("block_topk", kTopk);
                  a->Int("compress_ratio", 4);
                  a->Int("k_ts_idx", 2);
                  a->Float("scale", attention_scale);
                  a->Int("v_ts_idx", 3);
                },
                {q, k_out, v_out, topk, mask}, {attention});
    const int32_t projected_out =
        b.FullyConnected(b.Reshape(attention, {n, kQueryHeads * kDepth}),
                         b.ConstF32(absl::StrCat("output", l),
                                    {kHidden, kQueryHeads * kDepth}, w.output),
                         {n, kHidden});
    x = b.AddF(x, b.Reshape(projected_out, {1, n, kHidden}), {1, n, kHidden});
    topks.push_back(topk);
    state_out.insert(state_out.end(), {ring_out, pool_out, k_out, v_out});
  }
  const auto output = [&](const std::string& name, int32_t tensor) {
    b.Output(tensor);
    outputs.push_back({name, tensor});
  };
  output("hidden", x);
  for (int l = 0; l < kLayers; ++l) output(absl::StrCat("topk_", l), topks[l]);
  for (int l = 0; l < kLayers; ++l) {
    output(StateName("qsa_ring", l), state_out[4 * l + 0]);
    output(StateName("qsa_pool", l), state_out[4 * l + 1]);
    output(StateName("k", l), state_out[4 * l + 2]);
    output(StateName("v", l), state_out[4 * l + 3]);
  }
  b.Signature(key, graph, inputs, outputs);
}

std::string BuildModel() {
  const auto weights = MakeWeights();
  Builder b;
  AddSignature(b, "prefill_16", kPrefill, weights);
  AddSignature(b, "decode", 1, weights);
  return b.Pack();
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

Expected<void> WriteFile(const std::string& path, const void* data,
                         size_t size) {
  std::ofstream out(path, std::ios::binary);
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
  if (!out) return Error(kLiteRtStatusErrorFileIO, "Cannot write " + path);
  return {};
}

Expected<std::string> ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return Error(kLiteRtStatusErrorFileIO, "Cannot open " + path);
  std::string bytes(static_cast<size_t>(in.tellg()), '\0');
  in.seekg(0);
  in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return bytes;
}

template <typename T>
Expected<void> Fill(litert::TensorBuffer& buffer,
                    const std::vector<T>& values) {
  LITERT_ASSIGN_OR_RETURN(auto lock,
                          litert::TensorBufferScopedLock::Create<T>(
                              buffer, litert::TensorBuffer::LockMode::kWrite));
  LITERT_ASSIGN_OR_RETURN(const size_t size, buffer.PackedSize());
  if (size != values.size() * sizeof(T)) {
    return Error(kLiteRtStatusErrorInvalidArgument, "Input size mismatch");
  }
  std::memcpy(lock.second, values.data(), size);
  return {};
}

Expected<std::string> Bytes(litert::TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(const size_t size, buffer.PackedSize());
  LITERT_ASSIGN_OR_RETURN(auto lock,
                          litert::TensorBufferScopedLock::Create<const uint8_t>(
                              buffer, litert::TensorBuffer::LockMode::kRead));
  return std::string(reinterpret_cast<const char*>(lock.second), size);
}

// One step of the script: `rows` positions from `start`, of which `valid`
// are real.
struct Step {
  bool prefill;
  int start;
  int valid;
};

std::vector<Step> Script(int decode_steps) {
  std::vector<Step> steps;
  for (int start = 0; start < kPrompt; start += kPrefill) {
    steps.push_back({true, start, std::min(kPrefill, kPrompt - start)});
  }
  for (int i = 0; i < decode_steps; ++i) {
    steps.push_back({false, kPrompt + i, 1});
  }
  return steps;
}

Expected<void> Run() {
  const bool npu = absl::GetFlag(FLAGS_accelerator) == "npu";
  std::vector<litert::EnvironmentOptions::Option> env_options;
  const std::string libs = absl::GetFlag(FLAGS_runtime_libs);
  if (npu) {
    env_options.push_back(
        {litert::EnvironmentOptions::Tag::kDispatchLibraryDir, libs.c_str()});
    env_options.push_back(
        {litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
         libs.c_str()});
  }
  LITERT_ASSIGN_OR_RETURN(
      auto env,
      litert::Environment::Create(litert::EnvironmentOptions(env_options)));
  LITERT_ASSIGN_OR_RETURN(auto options, litert::Options::Create());
  LITERT_RETURN_IF_ERROR(options.SetHardwareAccelerators(
      npu ? litert::HwAccelerators::kNpu | litert::HwAccelerators::kCpu
          : litert::HwAcceleratorSet(litert::HwAccelerators::kCpu)));
  LITERT_ASSIGN_OR_RETURN(
      auto model,
      litert::CompiledModel::Create(env, absl::GetFlag(FLAGS_model), options));
  LITERT_ASSIGN_OR_RETURN(const size_t prefill,
                          model.GetSignatureIndex("prefill_16"));
  LITERT_ASSIGN_OR_RETURN(const size_t decode,
                          model.GetSignatureIndex("decode"));
  // The states, one buffer each, shared by both signatures; on the NPU also
  // their outputs (in place, as LiteRT-LM binds caches).
  std::vector<std::string> state_names;
  for (int l = 0; l < kLayers; ++l) {
    for (const char* kind : {"qsa_ring", "qsa_pool", "k", "v"}) {
      state_names.push_back(StateName(kind, l));
    }
  }
  std::vector<litert::TensorBuffer> states;
  for (const auto& name : state_names) {
    LITERT_ASSIGN_OR_RETURN(auto buffer,
                            model.CreateInputBuffer("prefill_16", name));
    LITERT_ASSIGN_OR_RETURN(const size_t size, buffer.PackedSize());
    LITERT_RETURN_IF_ERROR(Fill(buffer, std::vector<uint8_t>(size, 0)));
    states.push_back(std::move(buffer));
  }
  const std::string out_dir = absl::GetFlag(FLAGS_out_dir);
  int index = 0;
  for (const Step& step : Script(absl::GetFlag(FLAGS_decode_steps))) {
    const size_t signature = step.prefill ? prefill : decode;
    const int n = step.prefill ? kPrefill : 1;
    LITERT_ASSIGN_OR_RETURN(auto input_names,
                            model.GetSignatureInputNames(signature));
    LITERT_ASSIGN_OR_RETURN(auto output_names,
                            model.GetSignatureOutputNames(signature));
    LITERT_ASSIGN_OR_RETURN(auto inputs, model.CreateInputBuffers(signature));
    LITERT_ASSIGN_OR_RETURN(auto outputs, model.CreateOutputBuffers(signature));
    // The buffers by name (the signature's order is the runtime's).
    const auto input = [&](const std::string& name) -> litert::TensorBuffer& {
      for (size_t i = 0; i < input_names.size(); ++i) {
        if (input_names[i] == name) return inputs[i];
      }
      ABSL_LOG(FATAL) << "No input " << name;
    };
    const auto output_index = [&](const std::string& name) -> size_t {
      for (size_t i = 0; i < output_names.size(); ++i) {
        if (output_names[i] == name) return i;
      }
      ABSL_LOG(FATAL) << "No output " << name;
    };
    Random random(1000 + index);
    LITERT_RETURN_IF_ERROR(
        Fill(input("embeddings"),
             random.Values(static_cast<size_t>(n) * kHidden, 1.0f)));
    LITERT_RETURN_IF_ERROR(
        Fill(input("start_pos"), std::vector<int32_t>{step.start}));
    std::vector<uint8_t> mask(static_cast<size_t>(n) * kCacheLen, 0);
    for (int t = 0; t < step.valid; ++t) {
      for (int c = 0; c <= step.start + t; ++c) mask[t * kCacheLen + c] = 1;
    }
    LITERT_RETURN_IF_ERROR(Fill(input("mask"), mask));
    LITERT_RETURN_IF_ERROR(
        Fill(input("param_tensor"),
             std::vector<int32_t>{step.start, step.start + step.valid, 0, 0, 0,
                                  0, 0}));
    LITERT_RETURN_IF_ERROR(
        Fill(input("k_start"), std::vector<int32_t>{0, 0, step.start, 0}));
    LITERT_RETURN_IF_ERROR(
        Fill(input("v_start"), std::vector<int32_t>{0, 0, 0, step.start}));
    for (size_t s = 0; s < states.size(); ++s) {
      LITERT_ASSIGN_OR_RETURN(input(state_names[s]), states[s].Duplicate());
      if (npu) {
        LITERT_ASSIGN_OR_RETURN(outputs[output_index(state_names[s])],
                                states[s].Duplicate());
      }
    }
    LITERT_RETURN_IF_ERROR(model.Run(signature, inputs, outputs));
    const std::string prefix =
        absl::StrCat(out_dir, "/step", index < 10 ? "0" : "", index);
    LITERT_ASSIGN_OR_RETURN(const std::string hidden,
                            Bytes(outputs[output_index("hidden")]));
    LITERT_RETURN_IF_ERROR(
        WriteFile(prefix + ".hidden.raw", hidden.data(), hidden.size()));
    for (int l = 0; l < kLayers; ++l) {
      const std::string name = absl::StrCat("topk_", l);
      LITERT_ASSIGN_OR_RETURN(const std::string topk,
                              Bytes(outputs[output_index(name)]));
      LITERT_RETURN_IF_ERROR(WriteFile(absl::StrCat(prefix, ".", name, ".raw"),
                                       topk.data(), topk.size()));
    }
    for (size_t s = 0; s < states.size(); ++s) {
      // The CPU's states come back through its outputs.
      if (!npu) {
        LITERT_ASSIGN_OR_RETURN(const std::string bytes,
                                Bytes(outputs[output_index(state_names[s])]));
        LITERT_RETURN_IF_ERROR(
            Fill(states[s], std::vector<uint8_t>(bytes.begin(), bytes.end())));
      }
      if (state_names[s].find("qsa") != std::string::npos) {
        LITERT_ASSIGN_OR_RETURN(const std::string bytes, Bytes(states[s]));
        LITERT_RETURN_IF_ERROR(
            WriteFile(absl::StrCat(prefix, ".", state_names[s], ".raw"),
                      bytes.data(), bytes.size()));
      }
    }
    ABSL_LOG(INFO) << prefix << (step.prefill ? " prefill" : " decode")
                   << " start " << step.start << " rows " << step.valid;
    ++index;
  }
  return {};
}

// ---------------------------------------------------------------------------
// Comparing
// ---------------------------------------------------------------------------

std::vector<float> Floats(const std::string& bytes) {
  std::vector<float> values(bytes.size() / 4);
  std::memcpy(values.data(), bytes.data(), values.size() * 4);
  return values;
}

// Compares two runs: the hidden states and the pools to the tolerance; each
// row's selection as a set (LiteRT's CPU kernels, in the XNNPACK delegate,
// list a selection by score, the GPU delegate and this backend by block);
// the rings in the slots that hold the incomplete block's keys (the CPU
// keeps a completed block's keys in the others, the GPU delegate and this
// backend clear them; no step reads them).
Expected<bool> Compare() {
  const std::string cpu = absl::GetFlag(FLAGS_cpu_dir);
  const std::string npu = absl::GetFlag(FLAGS_npu_dir);
  const double tolerance = absl::GetFlag(FLAGS_tolerance);
  bool ok = true;
  const auto steps = Script(absl::GetFlag(FLAGS_decode_steps));
  for (size_t index = 0; index < steps.size(); ++index) {
    const Step& step = steps[index];
    const std::string prefix =
        absl::StrCat("/step", index < 10 ? "0" : "", index, ".");
    std::string line =
        absl::StrCat("step ", index, step.prefill ? " prefill" : " decode",
                     " start ", step.start, ":");
    const auto compare = [&](const std::string& name,
                             size_t count) -> Expected<void> {
      LITERT_ASSIGN_OR_RETURN(const std::string a,
                              ReadFile(cpu + prefix + name + ".raw"));
      LITERT_ASSIGN_OR_RETURN(const std::string b,
                              ReadFile(npu + prefix + name + ".raw"));
      const auto x = Floats(a);
      const auto y = Floats(b);
      count = std::min({count, x.size(), y.size()});
      double largest = 0.0, error = 0.0;
      for (size_t i = 0; i < count; ++i) {
        largest = std::max(largest, std::fabs(static_cast<double>(x[i])));
        error = std::max(error, std::fabs(static_cast<double>(x[i]) - y[i]));
      }
      const double relative = largest > 0.0 ? error / largest : error;
      const bool fine = x.size() == y.size() && relative <= tolerance;
      ok &= fine;
      absl::StrAppend(&line, " ", name, " ", relative, fine ? "" : " (!)");
      return {};
    };
    LITERT_RETURN_IF_ERROR(compare("hidden", SIZE_MAX));
    // The incomplete block's keys: slots [0, end % 4).
    const int end = step.start + step.valid;
    for (int l = 0; l < kLayers; ++l) {
      LITERT_RETURN_IF_ERROR(compare(absl::StrCat("kv_cache_qsa_ring_", l),
                                     static_cast<size_t>(end % 4) * kIndexDim));
      LITERT_RETURN_IF_ERROR(
          compare(absl::StrCat("kv_cache_qsa_pool_", l), SIZE_MAX));
    }
    for (int l = 0; l < kLayers; ++l) {
      const std::string name = absl::StrCat("topk_", l);
      LITERT_ASSIGN_OR_RETURN(const std::string a,
                              ReadFile(cpu + prefix + name + ".raw"));
      LITERT_ASSIGN_OR_RETURN(const std::string b,
                              ReadFile(npu + prefix + name + ".raw"));
      const int rows = step.prefill ? kPrefill : 1;
      int differing = 0;
      for (int r = 0; r < rows; ++r) {
        std::vector<int32_t> x(kTopk), y(kTopk);
        std::memcpy(x.data(), a.data() + r * kTopk * 4, kTopk * 4);
        std::memcpy(y.data(), b.data() + r * kTopk * 4, kTopk * 4);
        std::sort(x.begin(), x.end());
        std::sort(y.begin(), y.end());
        differing += x != y;
      }
      ok &= differing == 0;
      absl::StrAppend(&line, " ", name, " rows differing ", differing);
    }
    std::printf("%s\n", line.c_str());
  }
  std::printf("%s\n", ok ? "MATCH" : "MISMATCH");
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  if (!absl::GetFlag(FLAGS_write_model).empty()) {
    const std::string bytes = BuildModel();
    if (!WriteFile(absl::GetFlag(FLAGS_write_model), bytes.data(),
                   bytes.size())) {
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  }
  if (absl::GetFlag(FLAGS_compare)) {
    auto ok = Compare();
    if (!ok) {
      ABSL_LOG(ERROR) << ok.Error().Message();
      return EXIT_FAILURE;
    }
    return *ok ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  if (auto result = Run(); !result) {
    ABSL_LOG(ERROR) << result.Error().Message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

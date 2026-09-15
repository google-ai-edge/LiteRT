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

// Local correctness probe: live-length INT8 K/V views, independent of model
// projection/cache-update authoring. No wall-clock or throughput measurements.
#include "tensor/arithmetic.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/examples/gemma4/native/model/helpers/int8_kv_cache.h"
#include "tensor/runners/xnnpack/runner.h"
#include "xnnpack/operator.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/subgraph.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace litert::tensor;
using namespace litert::tensor::examples::gemma4::native;
using XT = Tensor<XnnpackMixinTag>;
constexpr int kHeads = 8;
constexpr int kWindow = 512;

void Check(absl::Status status) {
  if (!status.ok()) throw std::runtime_error(status.ToString());
}
void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
template <class T> T Take(absl::StatusOr<T> value) {
  Check(value.status());
  return std::move(*value);
}
XT F(const std::string& name, Shape shape) {
  return XT({.name = name, .type = Type::kFP32, .shape = shape});
}
XT I(const std::string& name, Shape shape, float scale) {
  return XT({.name = name, .type = Type::kI8, .shape = shape,
             .quantization = KvQuantization(scale)});
}
uint32_t Hash(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU;
  x ^= x >> 15; x *= 0x846ca68bU;
  return x ^ (x >> 16);
}

struct Error {
  double max_abs = 0, squared = 0;
  uint64_t count = 0, unequal_bits = 0;
  void Add(float actual, double expected) {
    Require(std::isfinite(actual) && std::isfinite(expected), "nonfinite result");
    double delta = std::abs(double(actual) - expected);
    max_abs = std::max(max_abs, delta);
    squared += delta * delta; ++count;
  }
  void Compare(const std::vector<float>& a, const std::vector<float>& b) {
    Require(a.size() == b.size(), "output extent differs");
    for (size_t i = 0; i < a.size(); ++i) {
      Add(a[i], b[i]);
      if (std::memcmp(&a[i], &b[i], sizeof(float))) ++unequal_bits;
    }
  }
  void Print() const {
    std::cout << "{\"count\":" << count << ",\"max_abs\":" << max_abs
              << ",\"rmse\":" << std::sqrt(squared / std::max<uint64_t>(1, count))
              << ",\"unequal_float_bits\":" << unequal_bits << "}";
  }
};
struct Result {
  std::vector<float> qk, probs, context;
};
struct Case {
  int start, rows;
};
struct Inputs {
  int capacity, dim, start, rows, begin, extent;
  std::vector<int8_t> keys, values, transposed_values;
  std::vector<float> q, mask, folded_mask;
  Inputs(int capacity_, int dim_, Case c, bool local, uint32_t salt = 0)
      : capacity(capacity_), dim(dim_), start(c.start), rows(c.rows),
        begin(local ? std::max(0, start - kWindow + 1) : 0),
        extent(start + rows - begin),
        keys(size_t(capacity) * dim + XNN_EXTRA_BYTES),
        values(size_t(capacity) * dim + XNN_EXTRA_BYTES),
        transposed_values(size_t(extent) * dim + XNN_EXTRA_BYTES),
        q(size_t(kHeads) * rows * dim), mask(size_t(rows) * extent),
        folded_mask(size_t(kHeads) * rows * extent) {
    Require(start >= 0 && rows > 0 && start + rows <= capacity,
            "invalid test cache extent");
    for (int t = 0; t < capacity; ++t)
      for (int d = 0; d < dim; ++d) {
        // Valid bytes depend only on absolute position, independent of capacity.
        // Unused tails differ between capacity probes to detect accidental reads.
        uint32_t poison = t >= start + rows ? salt : 0;
        keys[size_t(t) * dim + d] = int8_t(int(Hash(t * 8191U + d * 31U + 77U + poison) & 255U) - 128);
        values[size_t(t) * dim + d] = int8_t(int(Hash(t * 7919U + d * 37U + 31U + poison) & 255U) - 128);
      }
    for (int t = 0; t < extent; ++t)
      for (int d = 0; d < dim; ++d)
        transposed_values[size_t(d) * extent + t] = values[size_t(begin + t) * dim + d];
    for (int h = 0; h < kHeads; ++h)
      for (int r = 0; r < rows; ++r)
        for (int d = 0; d < dim; ++d) {
          int code = int(Hash((start + r) * 8971U + h * 1009U + d * 17U + 43U) & 1023U) - 512;
          q[(size_t(h) * rows + r) * dim + d] = float(code) / (512.0f * std::sqrt(float(dim)));
        }
    for (int r = 0; r < rows; ++r)
      for (int t = 0; t < extent; ++t) {
        int position = start + r, key_position = begin + t;
        bool visible = key_position <= position && (!local || key_position >= position - kWindow + 1);
        float value = visible ? 0 : std::numeric_limits<float>::lowest();
        mask[size_t(r) * extent + t] = value;
        for (int h = 0; h < kHeads; ++h)
          folded_mask[(size_t(h) * rows + r) * extent + t] = value;
      }
  }
};

// This graph has no static reshape operations. The fold is a host-side view of
// already head-major queries: [1,H,R,D] and [1,1,H*R,D] have identical bytes.
struct Attention {
  bool transposed, folded;
  int dim;
  XT q, k, v, mask, qk, probs, context;
  XnnpackRunner runner;
  xnn_runtime_t first_runtime = nullptr;
  size_t inspected_runs = 0;
  std::set<std::string> operator_names;
  Attention(int dim_, float ks, float vs, bool transposed_, bool folded_, uint32_t flags = 0)
      : transposed(transposed_), folded(folded_), dim(dim_),
        q(F("query", folded ? Shape{1, 1, kHeads, dim} : Shape{1, kHeads, 1, dim})),
        k(I("key_view", {1, 1, 1, dim}, ks)),
        v(I("value_view", transposed ? Shape{1, 1, dim, 1} : Shape{1, 1, 1, dim}, vs)),
        mask(F("mask", folded ? Shape{1, 1, kHeads, 1} : Shape{1, 1, 1, 1})),
        qk(BatchMatMul(q, Cast(k, Type::kFP32), false, true)),
        probs(Softmax(Add(qk, mask))),
        context(BatchMatMul(probs, Cast(v, Type::kFP32), false, transposed)),
        runner(Take(XnnpackRunner::Create({context, qk, probs}, flags))) {
    Check(context.GetStatus());
    runner.SetNumThreads(1);
  }
  uint32_t Id(const XT& t) const {
    return runner.graph().values()[Take(runner.graph().Lookup(t))].id;
  }
  void Inspect(const Inputs& in) {
    auto runtime = runner.runtime();
    Require(runtime != nullptr, "runtime missing");
    if (!first_runtime) first_runtime = runtime;
    Require(runtime == first_runtime, "reshape recreated the runtime");
    int fused_bmm = 0, ordinary_bmm = 0, fp32_convert = 0, int8_convert = 0;
    for (size_t i = 0; i < runtime->num_ops; ++i) {
      const auto& op = runtime->opdata[i];
      for (const auto* object : op.operator_objects) {
        if (!object) continue;
        std::string name = xnn_operator_type_to_string(object->type);
        operator_names.insert(name);
        if (object->type == xnn_operator_type_convert_nc_qs8_qc8) {
          ++int8_convert;
          Require(op.num_inputs == 1 && op.num_outputs == 1, "unexpected INT8 adapter arity");
          const auto& src = runtime->values[op.inputs[0]];
          const auto& dst = runtime->values[op.outputs[0]];
          Require(src.datatype == xnn_datatype_qint8 && dst.datatype == xnn_datatype_qcint8,
                  "INT8 adapter materialized FP32");
          // Workspace intermediates may have been reused after their last
          // consumer; checking their bytes after invocation is invalid.
        } else if (name.find("Convert") != std::string::npos) ++fp32_convert;
        if (object->type == xnn_operator_type_batch_matrix_multiply_nc_f32_qc8w) {
          ++fused_bmm;
          Require(op.num_inputs == 2, "unexpected BMM inputs");
          const auto& rhs = runtime->values[op.inputs[1]];
          Require(rhs.datatype == xnn_datatype_qint8 || rhs.datatype == xnn_datatype_qcint8,
                  "BMM RHS materialized as FP32");
          Require(rhs.shape.num_dims == 4, "BMM RHS rank changed");
          size_t elements = 1;
          for (size_t d = 0; d < rhs.shape.num_dims; ++d) elements *= rhs.shape.dim[d];
          Require(elements == size_t(in.extent) * dim, "BMM consumes allocated capacity instead of active view");
        } else if (name.find("Batch Matrix Multiply") != std::string::npos) ++ordinary_bmm;
      }
    }
    if (fused_bmm != 2 || ordinary_bmm != 0 || fp32_convert != 0 || int8_convert != 2)
      for (const auto& name : operator_names) std::cerr << "operator: " << name << '\n';
    Require(fused_bmm == 2 && ordinary_bmm == 0 && fp32_convert == 0 && int8_convert == 2,
            "expected two F32/QC8W BMMs and no dequantization operators: fused=" +
                std::to_string(fused_bmm) + " ordinary=" + std::to_string(ordinary_bmm) +
                " fp32_convert=" + std::to_string(fp32_convert) + " int8_convert=" + std::to_string(int8_convert));
    const auto& rk = runtime->values[Id(k)];
    const auto& rv = runtime->values[Id(v)];
    Require(rk.data == in.keys.data() + size_t(in.begin) * dim, "K live window was copied or misbound");
    Require(rv.data == (transposed ? in.transposed_values.data() : in.values.data() + size_t(in.begin) * dim),
            "V live window was copied or misbound");
    ++inspected_runs;
  }
  Result Run(const Inputs& in) {
    Check(runner.ReshapeInput(q, folded ? Shape{1, 1, kHeads * in.rows, dim} : Shape{1, kHeads, in.rows, dim}));
    Check(runner.ReshapeInput(k, {1, 1, in.extent, dim}));
    Check(runner.ReshapeInput(v, transposed ? Shape{1, 1, dim, in.extent} : Shape{1, 1, in.extent, dim}));
    Check(runner.ReshapeInput(mask, folded ? Shape{1, 1, kHeads * in.rows, in.extent} : Shape{1, 1, in.rows, in.extent}));
    Check(runner.SetInput(q, in.q));
    const absl::Span<const int8_t> key_view(in.keys.data() + size_t(in.begin) * dim, size_t(in.extent) * dim);
    const absl::Span<const int8_t> value_view(transposed ? in.transposed_values.data() : in.values.data() + size_t(in.begin) * dim, size_t(in.extent) * dim);
    Check(runner.SetInput(k, key_view));
    Check(runner.SetInput(v, value_view));
    Check(runner.SetInput(mask, folded ? in.folded_mask : in.mask));
    Check(runner.Run());
    Inspect(in);
    auto copy = [&](const XT& t) {
      auto lock = Take(runner.ReadOutputAs<float>(t));
      const auto& shape = runner.graph().values()[Take(runner.graph().Lookup(t))].info.shape;
      size_t count = 1;
      for (int d : shape) count *= d;
      // ReadOutputAs exposes retained allocation size after a shrink.
      Require(lock.size() >= count, "output storage smaller than logical shape");
      return std::vector<float>(lock.begin(), lock.begin() + count);
    };
    Result out{copy(qk), copy(probs), copy(context)};
    Require(out.qk.size() == size_t(kHeads) * in.rows * in.extent && out.context.size() == in.q.size(),
            "runtime shape propagation failed: T=" + std::to_string(in.extent) +
            " R=" + std::to_string(in.rows) + " folded=" + std::to_string(folded) +
            " transposed=" + std::to_string(transposed) + " qk=" + std::to_string(out.qk.size()) +
            " context=" + std::to_string(out.context.size()));
    return out;
  }
};

void Reference(const Inputs& in, float ks, float vs, const Result& actual,
               Error& qk_error, Error& prob_error, Error& context_error) {
  std::vector<double> p(in.extent);
  for (int h = 0; h < kHeads; ++h)
    for (int r = 0; r < in.rows; ++r) {
      size_t row = size_t(h) * in.rows + r;
      double max = -std::numeric_limits<double>::infinity();
      for (int t = 0; t < in.extent; ++t) {
        double score = 0;
        for (int d = 0; d < in.dim; ++d)
          score += double(in.q[row * in.dim + d]) * in.keys[size_t(in.begin + t) * in.dim + d] * ks;
        qk_error.Add(actual.qk[row * in.extent + t], score);
        p[t] = in.mask[size_t(r) * in.extent + t] == 0 ? score : -std::numeric_limits<double>::infinity();
        max = std::max(max, p[t]);
      }
      double total = 0;
      for (double& value : p) { value = std::exp(value - max); total += value; }
      for (int t = 0; t < in.extent; ++t) {
        p[t] /= total;
        prob_error.Add(actual.probs[row * in.extent + t], p[t]);
      }
      for (int d = 0; d < in.dim; ++d) {
        double expected = 0;
        for (int t = 0; t < in.extent; ++t)
          expected += p[t] * in.values[size_t(in.begin + t) * in.dim + d] * vs;
        context_error.Add(actual.context[row * in.dim + d], expected);
      }
    }
}

int main() {
  try {
    const std::vector<Case> cases = {{0,1},{1,1},{14,1},{126,1},{127,1},{128,1},
        {509,1},{510,1},{511,1},{512,1},{513,1},{1023,1},{2047,1},{1,1},
        {0,5},{126,5},{509,5},{510,17},{0,128},{511,2},{512,1}};
    Error qk_error, prob_error, context_error, layout_error, fold_error, capacity_error;
    size_t tested = 0, inspected = 0, capacity_pairs = 0;
    std::set<std::string> operators;
    for (int dim : {256, 512}) {
      auto found = std::find_if(PublishedE2BKvOwnerSpecs().begin(), PublishedE2BKvOwnerSpecs().end(),
                               [&](const KvOwnerSpec& spec) { return spec.head_dim == dim; });
      Require(found != PublishedE2BKvOwnerSpecs().end(), "published dimension missing");
      float ks = found->key_scale, vs = found->value_scale;
      Attention active(dim, ks, vs, false, true), old(dim, ks, vs, true, true), unfolded(dim, ks, vs, false, false);
      for (bool local : {false, true}) {
        for (const auto c : cases) {
          Inputs in(2048, dim, c, local);
          const auto keys_before = in.keys, values_before = in.values;
          auto a = active.Run(in), b = old.Run(in), u = unfolded.Run(in);
          Require(in.keys == keys_before && in.values == values_before, "attention mutated persistent INT8 cache");
          Reference(in, ks, vs, a, qk_error, prob_error, context_error);
          layout_error.Compare(a.qk, b.qk); layout_error.Compare(a.probs, b.probs); layout_error.Compare(a.context, b.context);
          fold_error.Compare(a.qk, u.qk); fold_error.Compare(a.probs, u.probs); fold_error.Compare(a.context, u.context);
          if ((c.rows == 1 && (c.start == 0 || c.start == 512 || c.start == 2047)) || c.rows == 17) {
            Inputs extra(4096, dim, c, local, 0x2468aceU);
            auto e = active.Run(extra);
            capacity_error.Compare(a.qk, e.qk); capacity_error.Compare(a.probs, e.probs); capacity_error.Compare(a.context, e.context);
            ++capacity_pairs;
          }
          ++tested;
        }
      }
      for (const auto* graph : {&active, &old, &unfolded}) {
        inspected += graph->inspected_runs;
        operators.insert(graph->operator_names.begin(), graph->operator_names.end());
      }
      std::cerr << "PASS dim=" << dim << " published_owner=" << found->owner
                << " global/local, dynamic grow/shrink, folded/head-broadcast, V-layout, capacity\n";
    }
    // Tolerances cover FP32 dot products/softmax, not an activation quantizer.
    Require(qk_error.max_abs < 2e-5, "QK differs from independent FP64 reference");
    Require(prob_error.max_abs < 2e-6, "probability differs from independent FP64 reference");
    Require(context_error.max_abs < 2e-5, "context differs from independent FP64 reference");
    Require(layout_error.max_abs < 2e-5, "V orientation changed attention beyond FP32 tolerance");
    Require(fold_error.max_abs < 2e-5, "folding query heads changed attention beyond FP32 tolerance");
    Require(capacity_error.unequal_bits == 0, "allocation capacity changed arithmetic");
    std::cout << std::setprecision(12) << "{\"status\":\"PASS\",\"cases\":" << tested
              << ",\"runtime_inspections\":" << inspected << ",\"capacity_pairs\":" << capacity_pairs
              << ",\"query_heads\":8,\"kv_heads\":1,\"head_dims\":[256,512],\"capacities\":[2048,4096],"
              << "\"qk_fp64_error\":"; qk_error.Print();
    std::cout << ",\"probability_fp64_error\":"; prob_error.Print();
    std::cout << ",\"context_fp64_error\":"; context_error.Print();
    std::cout << ",\"old_new_layout_error\":"; layout_error.Print();
    std::cout << ",\"folded_unfolded_error\":"; fold_error.Print();
    std::cout << ",\"capacity_error\":"; capacity_error.Print();
    std::cout << ",\"operators\":[";
    bool first = true;
    for (const auto& name : operators) { if (!first) std::cout << ','; first = false; std::cout << '"' << name << '"'; }
    std::cout << "],\"host_performance_measured\":false}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}

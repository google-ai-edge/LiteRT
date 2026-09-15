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

// Correctness-only numerical experiment. Includes the previously frozen probe
// helpers without changing that probe or its binary. No timing measurements.
#define main original_active_attention_probe_main
#include "tensor/examples/gemma4/native/tests/active_attention_test.cc"
#undef main

#include <array>
#include <fstream>
#include <map>

int RoundExtent(int length, int alignment) {
  return ((length + alignment - 1) / alignment) * alignment;
}

void SetExtent(Inputs& in, int extent, bool local) {
  Require(extent > 0 && in.begin + extent <= in.capacity, "padded extent exceeds storage");
  in.extent = extent;
  in.mask.assign(size_t(in.rows) * extent, std::numeric_limits<float>::lowest());
  in.folded_mask.assign(size_t(kHeads) * in.rows * extent, std::numeric_limits<float>::lowest());
  for (int r = 0; r < in.rows; ++r)
    for (int t = 0; t < extent; ++t) {
      const int position = in.start + r, key_position = in.begin + t;
      const bool visible = key_position <= position && (!local || key_position >= position - kWindow + 1);
      if (!visible) continue;
      in.mask[size_t(r) * extent + t] = 0;
      for (int h = 0; h < kHeads; ++h)
        in.folded_mask[(size_t(h) * in.rows + r) * extent + t] = 0;
    }
  in.transposed_values.resize(size_t(extent) * in.dim + XNN_EXTRA_BYTES);
  for (int t = 0; t < extent; ++t)
    for (int d = 0; d < in.dim; ++d)
      in.transposed_values[size_t(d) * extent + t] = in.values[size_t(in.begin + t) * in.dim + d];
}

void Seed(Inputs& in, int seed) {
  for (int t = 0; t < in.capacity; ++t)
    for (int d = 0; d < in.dim; ++d) {
      in.keys[size_t(t) * in.dim + d] = int8_t(int(Hash(t * 8191U + d * 31U + 77U + seed * 13007U) & 255U) - 128);
      in.values[size_t(t) * in.dim + d] = int8_t(int(Hash(t * 7919U + d * 37U + 31U + seed * 11003U) & 255U) - 128);
    }
  // Alternate modest logits and unit-scale queries resembling normalized Q.
  const float denominator = (seed & 1) ? 256.0f : 512.0f * std::sqrt(float(in.dim));
  for (int h = 0; h < kHeads; ++h)
    for (int r = 0; r < in.rows; ++r)
      for (int d = 0; d < in.dim; ++d) {
        int code = int(Hash((in.start + r) * 8971U + h * 1009U + d * 17U + seed * 7001U + 43U) & 1023U) - 512;
        in.q[(size_t(h) * in.rows + r) * in.dim + d] = float(code) / denominator;
      }
}

std::vector<float> PrefixRows(const std::vector<float>& matrix, int rows,
                              int stride, int depth) {
  Require(matrix.size() == size_t(rows) * stride && depth <= stride, "matrix shape mismatch");
  std::vector<float> out(size_t(rows) * depth);
  for (int r = 0; r < rows; ++r)
    std::copy_n(matrix.data() + size_t(r) * stride, depth, out.data() + size_t(r) * depth);
  return out;
}

// Independent PV graph receives the SAME fixed-width reference probabilities,
// cropped/padded with exact zeros. Any difference here comes from PV's extent,
// rather than softmax producing a slightly different denominator.
struct PvOnly {
  int dim;
  XT p, v, out;
  XnnpackRunner runner;
  explicit PvOnly(int dim_, float vs)
      : dim(dim_), p(F("frozen_probabilities", {1,1,kHeads,1})),
        v(I("value_rows", {1,1,1,dim}, vs)),
        out(BatchMatMul(p, Cast(v, Type::kFP32), false, false)),
        runner(Take(XnnpackRunner::Create({out}))) {
    runner.SetNumThreads(1);
  }
  std::vector<float> Run(const Inputs& in, const std::vector<float>& fixed_probabilities,
                         int fixed_extent, int source_offset = 0) {
    const int matrix_rows = kHeads * in.rows;
    std::vector<float> probabilities(size_t(matrix_rows) * in.extent, 0);
    Require(source_offset >= 0 && source_offset + in.extent <= fixed_extent, "PV-only extent exceeds reference");
    for (int r = 0; r < matrix_rows; ++r)
      std::copy_n(fixed_probabilities.data() + size_t(r) * fixed_extent + source_offset,
                  in.extent, probabilities.data() + size_t(r) * in.extent);
    Check(runner.ReshapeInput(p, {1,1,matrix_rows,in.extent}));
    Check(runner.ReshapeInput(v, {1,1,in.extent,dim}));
    const absl::Span<const int8_t> values(in.values.data() + size_t(in.begin) * dim,
                                         size_t(in.extent) * dim);
    Check(runner.SetInput(p, probabilities));
    Check(runner.SetInput(v, values));
    Check(runner.Run());
    int bmm_count = 0;
    for (size_t i = 0; i < runner.runtime()->num_ops; ++i)
      for (const auto* object : runner.runtime()->opdata[i].operator_objects)
        if (object && object->type == xnn_operator_type_batch_matrix_multiply_nc_f32_qc8w)
          ++bmm_count;
    Require(bmm_count == 1, "PV-only graph lost FP32/QC8W fusion");
    auto result = Take(runner.ReadOutputAs<float>(out));
    Require(result.size() >= size_t(matrix_rows) * dim, "PV-only output too short");
    return {result.begin(), result.begin() + size_t(matrix_rows) * dim};
  }
};

struct Summary {
  std::string experiment = "decode";
  bool align_begin = false;
  int dim = 0, alignment = 0;
  bool local = false;
  size_t cases = 0, qk_first = 0, softmax_first = 0, pv_first = 0, exact = 0;
  size_t fixed_probability_pv_differing_cases = 0;
  Error qk, probability, context, fixed_probability_pv;
};

int main(int argc, char** argv) {
  try {
    std::ofstream details;
    if (argc > 1) {
      details.open(argv[1]);
      Require(bool(details), "could not open detail CSV");
      details << "dim,local,seed,absolute_length,live_depth,fixed_extent,alignment,rounded_extent,first_difference,"
                 "qk_unequal,probability_unequal,context_unequal,pv_fixed_probability_unequal,"
                 "qk_max_abs,probability_max_abs,context_max_abs,pv_fixed_probability_max_abs,experiment,rows,view_begin,align_begin\n";
      details << std::setprecision(12);
    }
    const std::vector<int> alignments = {1,4,8,16,32,64,128};
    const std::vector<int> lengths = {1,2,3,4,7,15,16,17,127,128,129,255,256,257,510,511,512,513,529,574,1024,2048};
    std::vector<Summary> summaries;
    size_t total_cases = 0;
    for (int dim : {256,512}) {
      const auto& specs = PublishedE2BKvOwnerSpecs();
      auto spec = std::find_if(specs.begin(), specs.end(), [=](const auto& s) { return s.head_dim == dim; });
      Require(spec != specs.end(), "no published scale for dimension");
      Attention attention(dim, spec->key_scale, spec->value_scale, false, true);
      PvOnly pv(dim, spec->value_scale);
      for (bool local : {false,true}) {
        const int fixed_extent = local ? 512 : 2048;
        std::array<Summary,7> aggregate;
        for (size_t a = 0; a < alignments.size(); ++a) {
          aggregate[a].dim = dim; aggregate[a].local = local;
          aggregate[a].alignment = alignments[a];
        }
        for (int seed = 0; seed < 4; ++seed)
          for (int length : lengths) {
            Inputs inputs(4096, dim, {length - 1,1}, local);
            Seed(inputs, seed);
            const int live_depth = inputs.extent;
            SetExtent(inputs, fixed_extent, local);
            const auto fixed = attention.Run(inputs);
            const auto pv_fixed = pv.Run(inputs, fixed.probs, fixed_extent);
            Error control;
            control.Compare(fixed.context, pv_fixed);
            Require(control.unequal_bits == 0, "PV-only control differs from coupled fixed-extent attention");
            const auto fixed_qk = PrefixRows(fixed.qk, kHeads, fixed_extent, live_depth);
            const auto fixed_probs = PrefixRows(fixed.probs, kHeads, fixed_extent, live_depth);
            for (size_t a = 0; a < alignments.size(); ++a) {
              const int extent = RoundExtent(live_depth, alignments[a]);
              SetExtent(inputs, extent, local);
              const auto rounded = attention.Run(inputs);
              const auto pv_rounded = pv.Run(inputs, fixed.probs, fixed_extent);
              Error qk, probability, context, pv_error;
              qk.Compare(PrefixRows(rounded.qk, kHeads, extent, live_depth), fixed_qk);
              probability.Compare(PrefixRows(rounded.probs, kHeads, extent, live_depth), fixed_probs);
              context.Compare(rounded.context, fixed.context);
              pv_error.Compare(pv_rounded, pv_fixed);
              auto& stats = aggregate[a];
              stats.qk.Compare(PrefixRows(rounded.qk, kHeads, extent, live_depth), fixed_qk);
              stats.probability.Compare(PrefixRows(rounded.probs, kHeads, extent, live_depth), fixed_probs);
              stats.context.Compare(rounded.context, fixed.context);
              stats.fixed_probability_pv.Compare(pv_rounded, pv_fixed);
              ++stats.cases; ++total_cases;
              std::string first = "none";
              if (qk.unequal_bits) { first = "QK"; ++stats.qk_first; }
              else if (probability.unequal_bits) { first = "softmax"; ++stats.softmax_first; }
              else if (context.unequal_bits) { first = "PV"; ++stats.pv_first; }
              else ++stats.exact;
              if (pv_error.unequal_bits) ++stats.fixed_probability_pv_differing_cases;
              if (details)
                details << dim << ',' << local << ',' << seed << ',' << length << ',' << live_depth << ','
                        << fixed_extent << ',' << alignments[a] << ',' << extent << ',' << first << ','
                        << qk.unequal_bits << ',' << probability.unequal_bits << ',' << context.unequal_bits << ','
                        << pv_error.unequal_bits << ',' << qk.max_abs << ',' << probability.max_abs << ','
                        << context.max_abs << ',' << pv_error.max_abs << ",decode,1," << inputs.begin << ",0\n";
              Require(context.max_abs < 2e-4, "unexpectedly large extent rounding discrepancy");
            }
          }
        summaries.insert(summaries.end(), aggregate.begin(), aggregate.end());
        std::cerr << "PASS numerical extent experiment D=" << dim << " local=" << local
                  << " cases=" << lengths.size() * 4 * alignments.size() << '\n';
      }
    }
    // Compare local chunks to the same query rows within the frozen 1024-row
    // prefill's [0,1535) cache view. This isolates attention indexing/reduction
    // from projection authoring: Q/K/V are identical at absolute positions.
    {
      constexpr int dim = 256, fixed_extent = 1535;
      const auto& spec = PublishedE2BKvOwnerSpecs().front();
      Attention attention(dim, spec.key_scale, spec.value_scale, false, true);
      PvOnly pv(dim, spec.value_scale);
      const std::vector<Case> chunks = {{512,128},{640,128},{768,128},{529,17},{510,5},{511,128}};
      const std::vector<int> chunk_alignments = {4,16,32,64,128};
      for (bool align_begin : {false,true}) {
        std::array<Summary,5> aggregate;
        for (size_t a = 0; a < chunk_alignments.size(); ++a) {
          auto& s = aggregate[a];
          s.dim = dim; s.local = true; s.experiment = "local_chunk";
          s.align_begin = align_begin; s.alignment = chunk_alignments[a];
        }
        for (int seed = 0; seed < 2; ++seed)
          for (const auto c : chunks) {
            Inputs inputs(4096, dim, c, true);
            Seed(inputs, seed);
            const int raw_begin = inputs.begin;
            inputs.begin = 0;
            SetExtent(inputs, fixed_extent, true);
            const auto fixed = attention.Run(inputs);
            const auto pv_fixed = pv.Run(inputs, fixed.probs, fixed_extent);
            Error control;
            control.Compare(fixed.context, pv_fixed);
            Require(control.unequal_bits == 0, "chunk PV control differs");
            const int matrix_rows = kHeads * c.rows;
            for (size_t a = 0; a < chunk_alignments.size(); ++a) {
              const int alignment = chunk_alignments[a];
              inputs.begin = align_begin ? (raw_begin / alignment) * alignment : raw_begin;
              const int live_depth = c.start + c.rows - inputs.begin;
              const int extent = RoundExtent(live_depth, alignment);
              SetExtent(inputs, extent, true);
              const auto rounded = attention.Run(inputs);
              const auto pv_rounded = pv.Run(inputs, fixed.probs, fixed_extent, inputs.begin);
              auto slice = [&](const std::vector<float>& values) {
                std::vector<float> selected(size_t(matrix_rows) * extent);
                for (int r = 0; r < matrix_rows; ++r)
                  std::copy_n(values.data() + size_t(r) * fixed_extent + inputs.begin,
                               extent, selected.data() + size_t(r) * extent);
                return selected;
              };
              const auto fixed_qk = slice(fixed.qk), fixed_probabilities = slice(fixed.probs);
              Error qk, probability, context, pv_error;
              qk.Compare(rounded.qk, fixed_qk);
              probability.Compare(rounded.probs, fixed_probabilities);
              context.Compare(rounded.context, fixed.context);
              pv_error.Compare(pv_rounded, pv_fixed);
              auto& stats = aggregate[a];
              stats.qk.Compare(rounded.qk, fixed_qk);
              stats.probability.Compare(rounded.probs, fixed_probabilities);
              stats.context.Compare(rounded.context, fixed.context);
              stats.fixed_probability_pv.Compare(pv_rounded, pv_fixed);
              ++stats.cases; ++total_cases;
              std::string first = "none";
              if (qk.unequal_bits) { first = "QK"; ++stats.qk_first; }
              else if (probability.unequal_bits) { first = "softmax"; ++stats.softmax_first; }
              else if (context.unequal_bits) { first = "PV"; ++stats.pv_first; }
              else ++stats.exact;
              if (pv_error.unequal_bits) ++stats.fixed_probability_pv_differing_cases;
              if (details)
                details << dim << ",1," << seed << ',' << c.start + c.rows << ',' << live_depth << ','
                        << fixed_extent << ',' << alignment << ',' << extent << ',' << first << ','
                        << qk.unequal_bits << ',' << probability.unequal_bits << ',' << context.unequal_bits << ','
                        << pv_error.unequal_bits << ',' << qk.max_abs << ',' << probability.max_abs << ','
                        << context.max_abs << ',' << pv_error.max_abs << ",local_chunk," << c.rows << ','
                        << inputs.begin << ',' << align_begin << '\n';
              Require(context.max_abs < 2e-4, "unexpectedly large chunk attention discrepancy");
            }
          }
        summaries.insert(summaries.end(), aggregate.begin(), aggregate.end());
        std::cerr << "PASS local chunk begin/extent experiment align_begin=" << align_begin
                  << " cases=" << chunks.size() * 2 * chunk_alignments.size() << '\n';
      }
    }
    std::cout << std::setprecision(12) << "{\"status\":\"PASS\",\"meaning\":\"comparisons completed; PASS does not imply bitwise parity\","
              << "\"cases\":" << total_cases << ",\"host_performance_measured\":false,\"summaries\":[";
    bool first = true;
    for (const auto& s : summaries) {
      if (!first) std::cout << ',';
      first = false;
      std::cout << "{\"experiment\":\"" << s.experiment << "\",\"align_begin\":" << (s.align_begin ? "true" : "false")
                << ",\"dim\":" << s.dim << ",\"local\":" << (s.local ? "true" : "false")
                << ",\"alignment\":" << s.alignment << ",\"cases\":" << s.cases
                << ",\"exact_cases\":" << s.exact << ",\"first_qk\":" << s.qk_first
                << ",\"first_softmax\":" << s.softmax_first << ",\"first_pv\":" << s.pv_first
                << ",\"fixed_probability_pv_differing_cases\":" << s.fixed_probability_pv_differing_cases
                << ",\"qk\":"; s.qk.Print();
      std::cout << ",\"probability\":"; s.probability.Print();
      std::cout << ",\"context\":"; s.context.Print();
      std::cout << ",\"pv_fixed_probability\":"; s.fixed_probability_pv.Print();
      std::cout << '}';
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}

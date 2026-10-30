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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/tensorflow_lite/frontend.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/tensor.hpp"
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/common.h"
#include "litert/vendors/intel_openvino/compiler/alias_shared_constants.h"
#include "litert/vendors/intel_openvino/compiler/global_graph.h"
#include "litert/vendors/intel_openvino/compiler/graph_iterator.h"
#include "litert/vendors/intel_openvino/compiler/openvino_compile_context.h"
#include "litert/vendors/intel_openvino/compiler/weight_bank.h"
#include "litert/vendors/intel_openvino/compiler/weightless_caching_attributes.hpp"
#include "tflite/core/subgraph.h"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"

namespace litert {
namespace openvino {
namespace {

using ::litert::intel_openvino::IntelOpenVinoOptions;

constexpr char kMlirTestModelRelPath[] =
    "vendors/intel_openvino/compiler/testdata/npu_transformer_e2e.tflite";

template <typename T>
size_t CountOps(const std::shared_ptr<ov::Model>& model) {
  size_t count = 0;
  for (const auto& node : model->get_ops()) {
    if (std::dynamic_pointer_cast<T>(node)) {
      ++count;
    }
  }
  return count;
}

// Evaluates |model| on |inputs| using OpenVINO's built-in reference kernels
// after decomposing ScaledDotProductAttention into primitive ops that implement
// Node::evaluate.
ov::TensorVector EvaluateOvModel(const std::shared_ptr<ov::Model>& model,
                                 const ov::TensorVector& inputs) {
  auto eval_model = model->clone();
  ov::pass::Manager pm;
  pm.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
  pm.run_passes(eval_model);

  ov::TensorVector outputs;
  for (const auto& out : eval_model->outputs()) {
    outputs.emplace_back(out.get_element_type(), out.get_shape());
  }
  EXPECT_TRUE(eval_model->evaluate(outputs, inputs));
  return outputs;
}

// Fills |data| with deterministic values in [lo, hi) from |seed|.
void FillRange(std::vector<float>& data, uint32_t seed, float lo, float hi) {
  uint64_t state = seed * 2654435761u + 1u;
  for (float& v : data) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    const float unit = static_cast<float>((state >> 40) & 0xFFFF) / 65536.0f;
    v = lo + unit * (hi - lo);
  }
}

struct CompiledPipelineArtifacts {
  WeightBank bank;
  OpenVinoGlobalGraph global_graph;
  std::vector<std::shared_ptr<ov::Model>> ov_models;
  std::map<int32_t, size_t> pool_offset_of;
  std::string weights_bank_name;
  bool any_multi_token_chunk = false;
};

// Runs the two-phase NPU weight-sharing compiler pipeline on |model| (matching
// LiteRtCompilerPluginCompile in openvino_compiler_plugin.cc):
//   Phase 1: WeightBank::AddSubgraph + GraphIteratorDelegate + OpenVinoDecoder
//            + FrontEnd::convert + OptimizeModel(npu_weight_sharing=true) +
//            HarvestSharedConstants + BuildPool(prune_dead=true) + FarmHash64
//            NPUW_WEIGHTS_BANK name.
//   Phase 2: ConfigureForNpuWeightSharing + AliasAndTagSharedConstants over
//            each subgraph's ov::Model.
CompiledPipelineArtifacts RunTwoPhaseNpuPipeline(
    litert::compiler::Model& model, const LiteRtCompilerContext* compiler_ctx,
    OpenVinoCompileContext& compile_ctx) {
  CompiledPipelineArtifacts out;
  const size_t num_subgraphs = model.NumSubgraphs();
  auto frontend = std::make_shared<ov::frontend::tensorflow_lite::FrontEnd>();
  std::unordered_set<std::string> live_constant_names;
  out.ov_models.resize(num_subgraphs);

  for (size_t i = 0; i < num_subgraphs; ++i) {
    auto subgraph = model.Subgraph(i);
    EXPECT_TRUE(subgraph.HasValue());
    out.bank.AddSubgraph(subgraph.Value());

    std::shared_ptr<ov::frontend::tensorflow_lite::GraphIterator> delegate =
        std::make_shared<GraphIteratorDelegate>(compiler_ctx, &subgraph.Value(),
                                                compile_ctx.Device());
    auto ov_model = frontend->convert(frontend->load(delegate));
    EXPECT_NE(ov_model, nullptr);
    const auto opt_result =
        compile_ctx.OptimizeModel(ov_model, /*npu_weight_sharing=*/true);
    if (opt_result.moe_is_multi_token_chunk) {
      out.any_multi_token_chunk = true;
    }
    HarvestSharedConstants(ov_model, out.bank, live_constant_names);
    out.ov_models[i] = std::move(ov_model);
  }

  const PoolLayout layout =
      BuildPool(live_constant_names, out.bank, /*prune_dead=*/true);
  out.pool_offset_of = layout.OffsetMap();
  out.global_graph.buffers.reserve(layout.buffers.size());
  for (const auto& entry : layout.buffers) {
    out.global_graph.buffers.push_back({static_cast<uint32_t>(entry.buffer_id),
                                        entry.pool_offset, entry.bytes});
  }
  out.weights_bank_name = layout.WeightsBankName();

  EXPECT_EQ(compile_ctx.ConfigureForNpuWeightSharing(out.weights_bank_name,
                                                     out.any_multi_token_chunk),
            kLiteRtStatusOk);

  for (size_t i = 0; i < num_subgraphs; ++i) {
    auto aliased = AliasAndTagSharedConstants(
        out.ov_models[i], out.bank, out.pool_offset_of, static_cast<int>(i));
    EXPECT_TRUE(aliased.HasValue()) << aliased.Error().Message();
  }
  return out;
}

TEST(MlirPipelineE2ETest,
     CompilesMultiSubgraphMlirModelWithSharedPoolAndDecomposedBooleans) {
  const std::string model_path = testing::GetLiteRtPath(kMlirTestModelRelPath);
  auto ext_model_or = ExtendedModel::CreateFromFile(model_path);
  ASSERT_TRUE(ext_model_or.HasValue());
  const LiteRtCompilerContext* compiler_ctx = LrtGetCompilerContext();
  litert::compiler::Model model(compiler_ctx, ext_model_or->Get());
  ASSERT_EQ(model.NumSubgraphs(), 2u);

  auto opts = IntelOpenVinoOptions::Create();
  ASSERT_TRUE(opts.HasValue());
  opts->SetConfigsMapOption("fuse_split_attention_to_sdpa", "true");
  // Set caller overrides on NPUW knobs to verify ConfigureForNpuWeightSharing
  // preserves caller entries while still populating the FarmHash64 bank name.
  // NPUW_FOLD=YES keeps weight sharing valid with NPUW_CWAI=NO (NPUW needs one
  // of the two).
  opts->SetConfigsMapOption("NPUW_CWAI", "NO");
  opts->SetConfigsMapOption("NPUW_FOLD", "YES");
  opts->SetConfigsMapOption("NPUW_ONLINE_PIPELINE", "REP");
  auto compile_ctx_or = OpenVinoCompileContext::Create(opts);
  ASSERT_TRUE(compile_ctx_or.HasValue());
  OpenVinoCompileContext& compile_ctx = compile_ctx_or.Value();

  ASSERT_EQ(LiteRtUseSinkLogger(), kLiteRtStatusOk);
  LiteRtLogger logger = LiteRtGetDefaultLogger();
  LiteRtLogSeverity prev_severity = kLiteRtLogSeverityInfo;
  ASSERT_EQ(LiteRtGetMinLoggerSeverity(logger, &prev_severity),
            kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetMinLoggerSeverity(logger, kLiteRtLogSeverityDebug),
            kLiteRtStatusOk);
  ASSERT_EQ(LiteRtClearSinkLogger(logger), kLiteRtStatusOk);

  CompiledPipelineArtifacts artifacts =
      RunTwoPhaseNpuPipeline(model, compiler_ctx, compile_ctx);

  size_t log_count = 0;
  ASSERT_EQ(LiteRtGetSinkLoggerSize(logger, &log_count), kLiteRtStatusOk);
  bool saw_main_sdpa_shapes_log = false;
  bool saw_decode_sdpa_shapes_log = false;
  for (size_t i = 0; i < log_count; ++i) {
    const char* msg = nullptr;
    ASSERT_EQ(LiteRtGetSinkLoggerMessage(logger, i, &msg), kLiteRtStatusOk);
    ASSERT_NE(msg, nullptr);
    const std::string line(msg);
    if (line.find("shapes Q=[1,2,4,8] K_cache=[1,2,8,8] K_new=[1,2,4,8] "
                  "V_cache=[1,2,8,8] V_new=[1,2,8,4] mask=[1,1,4,12]") !=
        std::string::npos) {
      saw_main_sdpa_shapes_log = true;
    }
    if (line.find("shapes Q=[1,2,2,8] K_cache=[1,2,8,8] K_new=[1,2,4,8] "
                  "V_cache=[1,2,8,8] V_new=[1,2,8,4] mask=[1,1,2,12]") !=
        std::string::npos) {
      saw_decode_sdpa_shapes_log = true;
    }
  }
  EXPECT_EQ(LiteRtSetMinLoggerSeverity(logger, prev_severity), kLiteRtStatusOk);
  EXPECT_EQ(LiteRtUseStandardLogger(), kLiteRtStatusOk);
  EXPECT_TRUE(saw_main_sdpa_shapes_log)
      << "expected FuseSplitAttentionToSDPA DEBUG shape log for @main";
  EXPECT_TRUE(saw_decode_sdpa_shapes_log)
      << "expected FuseSplitAttentionToSDPA DEBUG shape log for @decode";

  // 1. Verify NPUW configuration and FarmHash64 bank name.
  ASSERT_EQ(artifacts.weights_bank_name.size(), 28u);
  EXPECT_EQ(artifacts.weights_bank_name.substr(0, 12), "litert_npuw_");
  const auto& cfg = compile_ctx.ConfigsMap();
  EXPECT_EQ(cfg.at("NPU_USE_NPUW").as<std::string>(), "YES");
  EXPECT_EQ(cfg.at("NPUW_WEIGHTS_BANK").as<std::string>(),
            artifacts.weights_bank_name);
  EXPECT_EQ(cfg.at("NPUW_CWAI").as<std::string>(), "NO");
  EXPECT_EQ(cfg.at("NPUW_FOLD").as<std::string>(), "YES");
  EXPECT_EQ(cfg.at("NPUW_ONLINE_PIPELINE").as<std::string>(), "REP");

  // 2. Verify BuildPool(prune_dead=true) pruned the slice-index buffers that
  //    FuseSplitAttentionToSDPA eliminated, while keeping all live buffers.
  EXPECT_LT(artifacts.global_graph.buffers.size(),
            artifacts.bank.Buffers().size());
  EXPECT_GE(artifacts.global_graph.BankBytes(), 512u + 512u + 64u + 16u);

  // 3. Verify IR invariants on both @main (subgraph 0, v8::Slice) and @decode
  //    (subgraph 1, v1::StridedSlice).
  std::vector<std::set<const void*>> aliased_ptrs_per_subgraph(2);
  std::vector<std::set<size_t>> bin_offsets_per_subgraph(2);
  for (size_t s = 0; s < 2u; ++s) {
    const auto& ov_model = artifacts.ov_models[s];
    // Split-cache attention (4 MatMuls + Softmax + Slice/StridedSlice) fused
    // into 1 ScaledDotProductAttention, leaving only the 2 projection MatMuls
    // (%w_q and %w_out), 1 Maximum (-3e4 mask floor), and 3 Pad ops (padding
    // S_kv = 12 up to 16).
    EXPECT_EQ(CountOps<ov::op::v13::ScaledDotProductAttention>(ov_model), 1u)
        << "subgraph " << s;
    EXPECT_EQ(CountOps<ov::op::v8::Softmax>(ov_model), 0u) << "subgraph " << s;
    EXPECT_EQ(CountOps<ov::op::v0::MatMul>(ov_model), 2u) << "subgraph " << s;
    EXPECT_EQ(CountOps<ov::op::v1::Maximum>(ov_model), 1u) << "subgraph " << s;
    EXPECT_EQ(CountOps<ov::op::v1::Pad>(ov_model), 3u) << "subgraph " << s;

    // Both tfl.reduce_any ops decoded to ReduceLogicalOr with keep_dims=true
    // and keep_dims=false.
    bool saw_keep_dims_true = false;
    bool saw_keep_dims_false = false;
    for (const auto& node : ov_model->get_ops()) {
      if (auto r =
              std::dynamic_pointer_cast<ov::op::v1::ReduceLogicalOr>(node)) {
        if (r->get_keep_dims()) {
          saw_keep_dims_true = true;
        } else {
          saw_keep_dims_false = true;
        }
      }
    }
    EXPECT_TRUE(saw_keep_dims_true) << "subgraph " << s;
    EXPECT_TRUE(saw_keep_dims_false) << "subgraph " << s;

    // DecomposeBooleanConstant rewrote the 16-byte %shared_bool_gate boolean
    // Constant into Constant(u8) + Convert(boolean) named "<name>_to_bool",
    // and AliasAndTagSharedConstants aliased all 4 shared constants (3 f32 +
    // 1 u8) to the canonical WeightBank buffers with WeightlessCacheAttribute.
    size_t tagged_constants = 0;
    size_t tagged_u8_constants = 0;
    bool saw_to_bool_convert = false;
    for (const auto& node : ov_model->get_ordered_ops()) {
      if (auto conv = ov::as_type_ptr<ov::op::v0::Convert>(node)) {
        if (conv->get_destination_type() == ov::element::boolean &&
            conv->get_friendly_name().find("_to_bool") != std::string::npos) {
          saw_to_bool_convert = true;
        }
      }
      auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(node);
      if (!cnst) continue;
      if (cnst->get_element_type() == ov::element::boolean) {
        EXPECT_LT(cnst->get_byte_size(), 16u)
            << "sizable boolean constant was not decomposed in subgraph " << s;
      }
      const auto& rt = cnst->get_rt_info();
      const auto it =
          rt.find(ov::WeightlessCacheAttribute::get_type_info_static());
      if (it == rt.end()) continue;

      ++tagged_constants;
      const auto& wlca = it->second.as<ov::WeightlessCacheAttribute>();
      const auto buffer_id =
          artifacts.bank.BufferIdOfName(cnst->get_friendly_name());
      ASSERT_TRUE(buffer_id.has_value()) << cnst->get_friendly_name();
      const auto* entry =
          artifacts.global_graph.FindBuffer(static_cast<uint32_t>(*buffer_id));
      ASSERT_NE(entry, nullptr);
      EXPECT_EQ(cnst->get_data_ptr<uint8_t>(), entry->bytes.data());
      EXPECT_EQ(wlca.bin_offset, entry->pool_offset);
      EXPECT_EQ(wlca.original_size, cnst->get_byte_size());
      aliased_ptrs_per_subgraph[s].insert(cnst->get_data_ptr());
      bin_offsets_per_subgraph[s].insert(wlca.bin_offset);
      if (cnst->get_element_type() == ov::element::u8) {
        ++tagged_u8_constants;
        EXPECT_EQ(wlca.original_size, 16u);
        EXPECT_EQ(wlca.original_dtype, ov::element::u8);
      }
    }
    EXPECT_TRUE(saw_to_bool_convert) << "subgraph " << s;
    EXPECT_EQ(tagged_constants, 4u) << "subgraph " << s;
    EXPECT_EQ(tagged_u8_constants, 1u) << "subgraph " << s;
  }

  // Both subgraphs' aliased Constants must point to the exact same 4 buffer
  // addresses in WeightBank and carry the exact same 4 WLCA bin_offsets.
  EXPECT_EQ(aliased_ptrs_per_subgraph[0].size(), 4u);
  EXPECT_EQ(aliased_ptrs_per_subgraph[0], aliased_ptrs_per_subgraph[1]);
  EXPECT_EQ(bin_offsets_per_subgraph[0], bin_offsets_per_subgraph[1]);

  // 4. Verify OpenVinoGlobalGraph container round-trip with the compiled
  //    shared pool and per-subgraph payloads, and confirm each WLCA bin_offset
  //    resolves to the exact weight bytes inside the contiguous serialized
  //    pool.
  artifacts.global_graph.payload_store.push_back("IR_MAIN");
  artifacts.global_graph.subgraphs["Partition_0"] = {
      "Partition_0",
      0,
      /*kv_tier=*/0,
      {},
      absl::MakeConstSpan(reinterpret_cast<const uint8_t*>(
                              artifacts.global_graph.payload_store[0].data()),
                          artifacts.global_graph.payload_store[0].size())};
  artifacts.global_graph.payload_store.push_back("IR_DECODE");
  artifacts.global_graph.subgraphs["Partition_1"] = {
      "Partition_1",
      0,
      /*kv_tier=*/0,
      {},
      absl::MakeConstSpan(reinterpret_cast<const uint8_t*>(
                              artifacts.global_graph.payload_store[1].data()),
                          artifacts.global_graph.payload_store[1].size())};

  const std::string blob = artifacts.global_graph.Serialize();
  auto parsed = OpenVinoGlobalGraph::Parse(
      reinterpret_cast<const uint8_t*>(blob.data()), blob.size());
  ASSERT_TRUE(parsed.HasValue());
  ASSERT_EQ(parsed->subgraphs.size(), 2u);
  EXPECT_EQ(parsed->pool.size(), artifacts.global_graph.BankBytes());
  for (const auto& entry : artifacts.global_graph.buffers) {
    ASSERT_LE(entry.pool_offset + entry.bytes.size(), parsed->pool.size());
    EXPECT_EQ(std::memcmp(parsed->pool.data() + entry.pool_offset,
                          entry.bytes.data(), entry.bytes.size()),
              0);
  }
}

TEST(MlirPipelineE2ETest,
     PrefillAndDecodeMatchTfliteReferenceInterpreterAndSurviveAllMaskedRow) {
  const std::string model_path = testing::GetLiteRtPath(kMlirTestModelRelPath);
  auto ext_model_or = ExtendedModel::CreateFromFile(model_path);
  ASSERT_TRUE(ext_model_or.HasValue());
  const LiteRtCompilerContext* compiler_ctx = LrtGetCompilerContext();
  litert::compiler::Model model(compiler_ctx, ext_model_or->Get());
  ASSERT_EQ(model.NumSubgraphs(), 2u);

  auto opts = IntelOpenVinoOptions::Create();
  ASSERT_TRUE(opts.HasValue());
  opts->SetConfigsMapOption("fuse_split_attention_to_sdpa", "true");
  auto compile_ctx_or = OpenVinoCompileContext::Create(opts);
  ASSERT_TRUE(compile_ctx_or.HasValue());
  CompiledPipelineArtifacts artifacts =
      RunTwoPhaseNpuPipeline(model, compiler_ctx, compile_ctx_or.Value());

  auto flatbuffer_or =
      internal::FlatbufferWrapper::CreateFromTflFile(model_path);
  ASSERT_TRUE(flatbuffer_or.HasValue());
  auto runtime_or =
      testing::TflRuntime::CreateFromFlatBuffer(std::move(*flatbuffer_or));
  ASSERT_TRUE(runtime_or.HasValue());
  ::tflite::Interpreter& interp = (*runtime_or)->Interpreter();
  ASSERT_EQ(interp.subgraphs_size(), 2u);

  // Test both @main (subgraph 0, S_q = 4, tfl.slice) and @decode (subgraph 1,
  // S_q = 2, tfl.strided_slice).
  const size_t kQueryLengths[2] = {4u, 2u};
  for (size_t s = 0; s < 2u; ++s) {
    const size_t s_q = kQueryLengths[s];
    ::tflite::Subgraph* tfl_sub = interp.subgraph(static_cast<int>(s));
    ASSERT_NE(tfl_sub, nullptr);
    ASSERT_EQ(tfl_sub->AllocateTensors(), kTfLiteOk);
    ASSERT_EQ(tfl_sub->inputs().size(), 6u);
    ASSERT_EQ(tfl_sub->outputs().size(), 3u);

    const auto& ov_model = artifacts.ov_models[s];
    ASSERT_EQ(ov_model->inputs().size(), 6u);
    ASSERT_EQ(ov_model->outputs().size(), 3u);

    // Prepare inputs:
    //   0: hidden    [1, 2, S_q, 8]
    //   1: k_cache   [1, 2, 8, 8]
    //   2: k_new     [1, 2, 4, 8]
    //   3: v_cache   [1, 2, 8, 8]  -- kept negative [-1.0, -0.2)
    //   4: v_new     [1, 2, 8, 4]  -- kept positive [+0.2, +1.0) so any bug
    //                                 in V-branch order recovery flips signs!
    //   5: attn_mask [1, 1, S_q, 12] -- causal/sparse mask with -inf entries.
    ov::TensorVector ov_inputs;
    for (size_t i = 0; i < 6u; ++i) {
      const auto& port = ov_model->input(i);
      ov::Tensor t(port.get_element_type(), port.get_shape());
      std::vector<float> buf(t.get_size());
      if (i == 3) {
        FillRange(buf, static_cast<uint32_t>(100 * s + i), -1.0f, -0.2f);
      } else if (i == 4) {
        FillRange(buf, static_cast<uint32_t>(100 * s + i), 0.2f, 1.0f);
      } else if (i == 5) {
        for (size_t q_idx = 0; q_idx < s_q; ++q_idx) {
          for (size_t k_idx = 0; k_idx < 12u; ++k_idx) {
            // Mask out cached positions 5..7 and new positions > q_idx with
            // -inf so both cached and new KV regions exercise -inf clamping.
            const bool masked =
                (k_idx >= 5u && k_idx < 8u) || (k_idx >= 8u + q_idx + 1u);
            buf[q_idx * 12u + k_idx] =
                masked ? -std::numeric_limits<float>::infinity() : 0.0f;
          }
        }
      } else {
        FillRange(buf, static_cast<uint32_t>(100 * s + i), -0.8f, 0.8f);
      }
      std::memcpy(t.data<float>(), buf.data(), buf.size() * sizeof(float));
      TfLiteTensor* tfl_in = tfl_sub->tensor(tfl_sub->inputs()[i]);
      ASSERT_NE(tfl_in, nullptr);
      ASSERT_EQ(tfl_in->bytes, buf.size() * sizeof(float));
      std::memcpy(tfl_in->data.f, buf.data(), buf.size() * sizeof(float));
      ov_inputs.push_back(std::move(t));
    }

    ASSERT_EQ(tfl_sub->Invoke(), kTfLiteOk);
    const ov::TensorVector ov_outputs = EvaluateOvModel(ov_model, ov_inputs);
    ASSERT_EQ(ov_outputs.size(), 3u);

    // Output 0: %gated_out [1, 2, S_q, 8] (f32)
    const TfLiteTensor* tfl_out0 = tfl_sub->tensor(tfl_sub->outputs()[0]);
    ASSERT_EQ(ov_outputs[0].get_size(), 2u * s_q * 8u);
    const float* ov_f = ov_outputs[0].data<float>();
    float max_diff = 0.0f;
    for (size_t j = 0; j < ov_outputs[0].get_size(); ++j) {
      ASSERT_TRUE(std::isfinite(ov_f[j])) << "subgraph " << s << " index " << j;
      max_diff = std::max(max_diff, std::abs(ov_f[j] - tfl_out0->data.f[j]));
    }
    EXPECT_LT(max_diff, 1e-4f) << "subgraph " << s;

    // Output 1: %any_keep [1, 2, S_q, 1] (bool, keep_dims = true)
    const TfLiteTensor* tfl_out1 = tfl_sub->tensor(tfl_sub->outputs()[1]);
    EXPECT_EQ(ov_outputs[1].get_shape(), (ov::Shape{1, 2, s_q, 1}));
    const uint8_t* ov_b1 = ov_outputs[1].data<uint8_t>();
    for (size_t j = 0; j < ov_outputs[1].get_size(); ++j) {
      EXPECT_EQ(ov_b1[j] != 0, tfl_out1->data.b[j])
          << "subgraph " << s << " any_keep[" << j << "]";
    }

    // Output 2: %any_drop [1, 2, S_q] (bool, keep_dims = false)
    const TfLiteTensor* tfl_out2 = tfl_sub->tensor(tfl_sub->outputs()[2]);
    EXPECT_EQ(ov_outputs[2].get_shape(), (ov::Shape{1, 2, s_q}));
    const uint8_t* ov_b2 = ov_outputs[2].data<uint8_t>();
    for (size_t j = 0; j < ov_outputs[2].get_size(); ++j) {
      EXPECT_EQ(ov_b2[j] != 0, tfl_out2->data.b[j])
          << "subgraph " << s << " any_drop[" << j << "]";
    }

    // Now set the entire first query row of attn_mask to -inf (simulating a
    // flash-attention tile whose KV positions are all masked). Because
    // FuseSplitAttentionToSDPA clamps the mask to -3e4f, every output element
    // must remain finite (no NaN contamination).
    float* mask_ptr = ov_inputs[5].data<float>();
    for (size_t k_idx = 0; k_idx < 12u; ++k_idx) {
      mask_ptr[k_idx] = -std::numeric_limits<float>::infinity();
    }
    const ov::TensorVector all_masked_outputs =
        EvaluateOvModel(ov_model, ov_inputs);
    const float* masked_f = all_masked_outputs[0].data<float>();
    for (size_t j = 0; j < all_masked_outputs[0].get_size(); ++j) {
      EXPECT_TRUE(std::isfinite(masked_f[j]))
          << "all-masked query row produced non-finite output in subgraph " << s
          << " at index " << j;
    }
  }
}

}  // namespace
}  // namespace openvino
}  // namespace litert

// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/vendors/intel_openvino/compiler/openvino_compile_context.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"

namespace litert {
namespace openvino {
namespace {

using ::litert::intel_openvino::IntelOpenVinoOptions;
using Configs = std::vector<std::pair<const char*, const char*>>;

// Context for graph |graph_index| of a model with model-wide configs
// |model_wide| and, for graph |graph_index|, per-graph configs |per_graph|.
OpenVinoCompileContext ContextFor(const Configs& model_wide,
                                  const Configs& per_graph = {},
                                  int graph_index = 0) {
  auto options = IntelOpenVinoOptions::Create();
  EXPECT_TRUE(options.HasValue());
  for (const auto& [key, value] : model_wide) {
    options->SetConfigsMapOption(key, value);
  }
  for (const auto& [key, value] : per_graph) {
    options->SetGraphConfigsMapOption(graph_index, key, value);
  }
  auto context = OpenVinoCompileContext::Create(options, graph_index);
  EXPECT_TRUE(context.HasValue());
  return std::move(*context);
}

std::string Get(const OpenVinoCompileContext& context, const char* key) {
  const auto it = context.ConfigsMap().find(key);
  return it == context.ConfigsMap().end() ? "" : it->second.as<std::string>();
}

// One additive-mask SDPA whose query covers |tokens| tokens.
std::shared_ptr<ov::Model> SingleSdpa(size_t tokens) {
  using ov::op::v0::Parameter;
  const auto f = ov::element::f32;
  auto q = std::make_shared<Parameter>(f, ov::Shape{1, 2, tokens, 4});
  auto k = std::make_shared<Parameter>(f, ov::Shape{1, 2, 16, 4});
  auto v = std::make_shared<Parameter>(f, ov::Shape{1, 2, 16, 4});
  auto mask = std::make_shared<Parameter>(f, ov::Shape{1, 1, tokens, 16});
  auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
      ov::OutputVector{q, k, v, mask}, /*causal=*/false);
  return std::make_shared<ov::Model>(ov::OutputVector{sdpa->output(0)},
                                     ov::ParameterVector{q, k, v, mask},
                                     "sdpa");
}

size_t NumDecomposed(const OpenVinoCompileContext& context, size_t tokens) {
  return context.OptimizeModel(SingleSdpa(tokens)).num_sdpa_decomposed_for_hfa;
}

const OpenVinoCompileContext::OptimizeModelResult kMoePrefill = {
    .moe_is_multi_token_chunk = true, .num_sdpa_decomposed_for_hfa = 29};
const OpenVinoCompileContext::OptimizeModelResult kMoeDecode = {
    .moe_is_multi_token_chunk = false, .num_sdpa_decomposed_for_hfa = 0};

TEST(OpenVinoCompileContextHfaTest, EnableHfaDecomposesOnlyMultiTokenSdpas) {
  const auto context = ContextFor({{"enable_hfa", "true"}});
  EXPECT_EQ(NumDecomposed(context, /*tokens=*/8), 1u);
  EXPECT_EQ(NumDecomposed(context, /*tokens=*/1), 0u);
}

TEST(OpenVinoCompileContextHfaTest, DisabledByDefault) {
  const auto context = ContextFor({});
  EXPECT_EQ(NumDecomposed(context, /*tokens=*/8), 0u);
}

TEST(OpenVinoCompileContextHfaTest, PerGraphEnableHfaFalseOptsOut) {
  EXPECT_EQ(NumDecomposed(
                ContextFor({{"enable_hfa", "true"}}, {{"enable_hfa", "false"}}),
                /*tokens=*/8),
            0u);
}

TEST(OpenVinoCompileContextHfaTest, WeightSharingSetsHfaOnDecomposedGraph) {
  auto prefill = ContextFor({{"enable_hfa", "true"}}, {}, /*graph_index=*/0);
  prefill.ConfigureForNpuWeightSharing(kMoePrefill);
  EXPECT_EQ(Get(prefill, "NPUW_ATTN"), "HFA");
  EXPECT_EQ(Get(prefill, "NPUW_ONLINE_PIPELINE"), "REP");
  EXPECT_EQ(Get(prefill, "NPUW_ONLINE_ISOLATE"), "ATTN,MOE");
  // The HFA sub-options are left at NPUW's defaults.
  EXPECT_EQ(Get(prefill, "NPUW_ATTN_HFA_FUSED"), "");
  EXPECT_EQ(Get(prefill, "NPUW_ATTN_HFA_MASK_SKIPPING"), "");

  // Decode had nothing decomposed, so NPUW keeps its STATIC default.
  auto decode = ContextFor({{"enable_hfa", "true"}}, {}, /*graph_index=*/1);
  decode.ConfigureForNpuWeightSharing(kMoeDecode);
  EXPECT_EQ(Get(decode, "NPUW_ATTN"), "");
  EXPECT_EQ(Get(decode, "NPUW_ONLINE_PIPELINE"), "NONE");
}

TEST(OpenVinoCompileContextHfaTest, WeightSharingKeepsExplicitNpuwAttn) {
  auto context = ContextFor({{"enable_hfa", "true"}, {"NPUW_ATTN", "STATIC"}});
  context.ConfigureForNpuWeightSharing(kMoePrefill);
  EXPECT_EQ(Get(context, "NPUW_ATTN"), "STATIC");
}

const OpenVinoCompileContext::OptimizeModelResult kDensePrefill = {
    .moe_is_multi_token_chunk = false, .num_sdpa_decomposed_for_hfa = 47};
const OpenVinoCompileContext::OptimizeModelResult kDenseDecode = {
    .moe_is_multi_token_chunk = false, .num_sdpa_decomposed_for_hfa = 0};

TEST(OpenVinoCompileContextHfaTest, WeightSharingGivesDenseHfaGraphRep) {
  // Under NONE a dense prefill is one function with no attention block to
  // convert, so HFA needs REP with attention (and no MoE) isolated.
  auto prefill = ContextFor({{"enable_hfa", "true"}}, {}, /*graph_index=*/0);
  prefill.ConfigureForNpuWeightSharing(kDensePrefill);
  EXPECT_EQ(Get(prefill, "NPUW_ATTN"), "HFA");
  EXPECT_EQ(Get(prefill, "NPUW_ONLINE_PIPELINE"), "REP");
  EXPECT_EQ(Get(prefill, "NPUW_ONLINE_ISOLATE"), "ATTN");
  EXPECT_EQ(Get(prefill, "NPUW_FOLD"), "YES");
  EXPECT_EQ(Get(prefill, "NPUW_MOE_TOKEN_CHUNK_SIZE"), "");
  EXPECT_EQ(Get(prefill, "NPUW_CWAI"), "");

  auto decode = ContextFor({{"enable_hfa", "true"}}, {}, /*graph_index=*/1);
  decode.ConfigureForNpuWeightSharing(kDenseDecode);
  EXPECT_EQ(Get(decode, "NPUW_ATTN"), "");
  EXPECT_EQ(Get(decode, "NPUW_ONLINE_PIPELINE"), "NONE");
  EXPECT_EQ(Get(decode, "NPUW_CWAI"), "YES");
}

TEST(OpenVinoCompileContextHfaTest, WeightSharingGivesDenseStaticControlRep) {
  // A STATIC control over the same decomposed graph gets the same pipeline.
  auto context = ContextFor({{"enable_hfa", "true"}}, {{"NPUW_ATTN", "STATIC"}},
                            /*graph_index=*/0);
  context.ConfigureForNpuWeightSharing(kDensePrefill);
  EXPECT_EQ(Get(context, "NPUW_ATTN"), "STATIC");
  EXPECT_EQ(Get(context, "NPUW_ONLINE_PIPELINE"), "REP");
  EXPECT_EQ(Get(context, "NPUW_ONLINE_ISOLATE"), "ATTN");
}

}  // namespace
}  // namespace openvino
}  // namespace litert

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

#include "litert/vendors/intel_openvino/compiler/openvino_compile_context.h"

#include <string>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_intel_openvino_options.h"

namespace litert {
namespace openvino {
namespace {

using ::litert::intel_openvino::IntelOpenVinoOptions;

Expected<OpenVinoCompileContext> CreateDefaultNpuContext() {
  auto opts = IntelOpenVinoOptions::Create();
  return OpenVinoCompileContext::Create(opts);
}

TEST(OpenVinoCompileContextTest,
     ConfigureForNpuWeightSharingSetsSingleTokenCwaiDefaultsAndBankName) {
  auto ctx_or = CreateDefaultNpuContext();
  ASSERT_TRUE(ctx_or.HasValue());
  OpenVinoCompileContext& ctx = ctx_or.Value();
  ASSERT_EQ(ctx.Device(), "NPU");

  ctx.ConfigureForNpuWeightSharing("litert_npuw_0123456789abcdef",
                                   /*moe_multi_token_chunk=*/false);

  const auto& cfg = ctx.ConfigsMap();
  ASSERT_NE(cfg.find("NPU_USE_NPUW"), cfg.end());
  EXPECT_EQ(cfg.at("NPU_USE_NPUW").as<std::string>(), "YES");
  EXPECT_EQ(cfg.at("NPUW_DEVICES").as<std::string>(), "NPU");
  EXPECT_EQ(cfg.at("NPUW_WEIGHTS_BANK").as<std::string>(),
            "litert_npuw_0123456789abcdef");
  EXPECT_EQ(cfg.at("NPUW_FUNCALL_FOR_ALL").as<std::string>(), "YES");
  EXPECT_EQ(cfg.at("NPUW_ONLINE_PIPELINE").as<std::string>(), "NONE");
  EXPECT_EQ(cfg.at("NPUW_CWAI").as<std::string>(), "YES");
  EXPECT_EQ(cfg.find("NPUW_UNFOLD_IREQS"), cfg.end());
  EXPECT_EQ(cfg.find("NPUW_ONLINE_ISOLATE"), cfg.end());
}

TEST(OpenVinoCompileContextTest,
     ConfigureForNpuWeightSharingFallsBackToSharedWhenBankNameEmpty) {
  auto ctx_or = CreateDefaultNpuContext();
  ASSERT_TRUE(ctx_or.HasValue());
  OpenVinoCompileContext& ctx = ctx_or.Value();

  ctx.ConfigureForNpuWeightSharing(/*weights_bank_name=*/"",
                                   /*moe_multi_token_chunk=*/false);

  const auto& cfg = ctx.ConfigsMap();
  ASSERT_NE(cfg.find("NPUW_WEIGHTS_BANK"), cfg.end());
  EXPECT_EQ(cfg.at("NPUW_WEIGHTS_BANK").as<std::string>(), "shared");
}

TEST(OpenVinoCompileContextTest,
     ConfigureForNpuWeightSharingSetsMultiTokenMoePipelineAndBankName) {
  auto ctx_or = CreateDefaultNpuContext();
  ASSERT_TRUE(ctx_or.HasValue());
  OpenVinoCompileContext& ctx = ctx_or.Value();

  ctx.ConfigureForNpuWeightSharing("litert_npuw_fedcba9876543210",
                                   /*moe_multi_token_chunk=*/true);

  const auto& cfg = ctx.ConfigsMap();
  EXPECT_EQ(cfg.at("NPU_USE_NPUW").as<std::string>(), "YES");
  EXPECT_EQ(cfg.at("NPUW_DEVICES").as<std::string>(), "NPU");
  EXPECT_EQ(cfg.at("NPUW_WEIGHTS_BANK").as<std::string>(),
            "litert_npuw_fedcba9876543210");
  EXPECT_EQ(cfg.at("NPUW_FUNCALL_FOR_ALL").as<std::string>(), "YES");
  EXPECT_EQ(cfg.at("NPUW_ONLINE_PIPELINE").as<std::string>(), "REP");
  EXPECT_EQ(cfg.at("NPUW_UNFOLD_IREQS").as<std::string>(), "NO");
  EXPECT_EQ(cfg.at("NPUW_ONLINE_ISOLATE").as<std::string>(), "ATTN,MOE");
  EXPECT_EQ(cfg.at("NPUW_MOE_TOKEN_CHUNK_SIZE").as<std::string>(), "0");
  EXPECT_EQ(cfg.at("NPUW_ONLINE_KEEP_BLOCK_SIZE").as<std::string>(), "4");
  EXPECT_EQ(cfg.at("NPUW_FOLD").as<std::string>(), "YES");
  EXPECT_EQ(cfg.at("NPUW_ONLINE_KEEP_BLOCKS").as<std::string>(), "3");
  EXPECT_EQ(cfg.find("NPUW_CWAI"), cfg.end());
}

TEST(OpenVinoCompileContextTest,
     ConfigureForNpuWeightSharingIsNoOpOnNonNpuDevice) {
  auto opts = IntelOpenVinoOptions::Create();
  ASSERT_TRUE(opts.HasValue());
  opts->SetGraphBackend(/*graph_index=*/0, kLiteRtIntelOpenVinoGraphBackendCPU);
  auto ctx_or = OpenVinoCompileContext::Create(opts, /*graph_index=*/0);
  ASSERT_TRUE(ctx_or.HasValue());
  OpenVinoCompileContext& ctx = ctx_or.Value();
  ASSERT_EQ(ctx.Device(), "CPU");

  ctx.ConfigureForNpuWeightSharing("litert_npuw_0123456789abcdef",
                                   /*moe_multi_token_chunk=*/false);

  const auto& cfg = ctx.ConfigsMap();
  EXPECT_EQ(cfg.find("NPU_USE_NPUW"), cfg.end());
  EXPECT_EQ(cfg.find("NPUW_WEIGHTS_BANK"), cfg.end());
}

}  // namespace
}  // namespace openvino
}  // namespace litert

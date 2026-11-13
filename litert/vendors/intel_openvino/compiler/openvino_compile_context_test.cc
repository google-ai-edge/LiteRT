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

  ASSERT_EQ(ctx.ConfigureForNpuWeightSharing("litert_npuw_0123456789abcdef",
                                             /*moe_multi_token_chunk=*/false),
            kLiteRtStatusOk);

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

  ASSERT_EQ(ctx.ConfigureForNpuWeightSharing(/*weights_bank_name=*/"",
                                             /*moe_multi_token_chunk=*/false),
            kLiteRtStatusOk);

  const auto& cfg = ctx.ConfigsMap();
  ASSERT_NE(cfg.find("NPUW_WEIGHTS_BANK"), cfg.end());
  EXPECT_EQ(cfg.at("NPUW_WEIGHTS_BANK").as<std::string>(), "shared");
}

TEST(OpenVinoCompileContextTest,
     ConfigureForNpuWeightSharingSetsMultiTokenMoePipelineAndBankName) {
  auto ctx_or = CreateDefaultNpuContext();
  ASSERT_TRUE(ctx_or.HasValue());
  OpenVinoCompileContext& ctx = ctx_or.Value();

  ASSERT_EQ(ctx.ConfigureForNpuWeightSharing("litert_npuw_fedcba9876543210",
                                             /*moe_multi_token_chunk=*/true),
            kLiteRtStatusOk);

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

  ASSERT_EQ(ctx.ConfigureForNpuWeightSharing("litert_npuw_0123456789abcdef",
                                             /*moe_multi_token_chunk=*/false),
            kLiteRtStatusOk);

  const auto& cfg = ctx.ConfigsMap();
  EXPECT_EQ(cfg.find("NPU_USE_NPUW"), cfg.end());
  EXPECT_EQ(cfg.find("NPUW_WEIGHTS_BANK"), cfg.end());
}

TEST(OpenVinoCompileContextTest,
     ConfigureForNpuWeightSharingPreservesCallerOverridesSingleToken) {
  auto opts = IntelOpenVinoOptions::Create();
  ASSERT_TRUE(opts.HasValue());
  // Caller overrides single-token defaults (e.g., opting a dense model into
  // NPUW's online REP pipeline, swapping NPUW_CWAI for NPUW_FOLD, or forcing a
  // custom bank name). NPUW_FOLD=YES keeps weight sharing valid with
  // NPUW_CWAI=NO.
  opts->SetConfigsMapOption("NPUW_ONLINE_PIPELINE", "REP");
  opts->SetConfigsMapOption("NPUW_CWAI", "NO");
  opts->SetConfigsMapOption("NPUW_FOLD", "YES");
  opts->SetConfigsMapOption("NPUW_WEIGHTS_BANK", "custom_caller_bank");
  opts->SetConfigsMapOption("NPUW_DEVICES", "NPU,CPU");

  auto ctx_or = OpenVinoCompileContext::Create(opts);
  ASSERT_TRUE(ctx_or.HasValue());
  OpenVinoCompileContext& ctx = ctx_or.Value();

  ASSERT_EQ(ctx.ConfigureForNpuWeightSharing("litert_npuw_0123456789abcdef",
                                             /*moe_multi_token_chunk=*/false),
            kLiteRtStatusOk);

  const auto& cfg = ctx.ConfigsMap();
  // Caller-supplied entries win over defaults.
  EXPECT_EQ(cfg.at("NPUW_ONLINE_PIPELINE").as<std::string>(), "REP");
  EXPECT_EQ(cfg.at("NPUW_CWAI").as<std::string>(), "NO");
  EXPECT_EQ(cfg.at("NPUW_FOLD").as<std::string>(), "YES");
  EXPECT_EQ(cfg.at("NPUW_WEIGHTS_BANK").as<std::string>(),
            "custom_caller_bank");
  EXPECT_EQ(cfg.at("NPUW_DEVICES").as<std::string>(), "NPU,CPU");
  // Unset keys still receive defaults.
  EXPECT_EQ(cfg.at("NPU_USE_NPUW").as<std::string>(), "YES");
  EXPECT_EQ(cfg.at("NPUW_FUNCALL_FOR_ALL").as<std::string>(), "YES");
}

TEST(OpenVinoCompileContextTest,
     ConfigureForNpuWeightSharingPreservesPerGraphOverridesMultiToken) {
  auto opts = IntelOpenVinoOptions::Create();
  ASSERT_TRUE(opts.HasValue());
  opts->SetConfigsMapOption("NPUW_ONLINE_KEEP_BLOCK_SIZE", "8");
  opts->SetGraphConfigsMapOption(/*graph_index=*/1, "NPUW_ONLINE_KEEP_BLOCKS",
                                 "5");
  opts->SetGraphConfigsMapOption(/*graph_index=*/1, "NPUW_UNFOLD_IREQS", "YES");

  auto ctx_or = OpenVinoCompileContext::Create(opts, /*graph_index=*/1);
  ASSERT_TRUE(ctx_or.HasValue());
  OpenVinoCompileContext& ctx = ctx_or.Value();

  ASSERT_EQ(ctx.ConfigureForNpuWeightSharing("litert_npuw_fedcba9876543210",
                                             /*moe_multi_token_chunk=*/true),
            kLiteRtStatusOk);

  const auto& cfg = ctx.ConfigsMap();
  // Both model-wide and per-graph overrides win over multi-token defaults.
  EXPECT_EQ(cfg.at("NPUW_ONLINE_KEEP_BLOCK_SIZE").as<std::string>(), "8");
  EXPECT_EQ(cfg.at("NPUW_ONLINE_KEEP_BLOCKS").as<std::string>(), "5");
  EXPECT_EQ(cfg.at("NPUW_UNFOLD_IREQS").as<std::string>(), "YES");
  // Unset multi-token keys still receive defaults.
  EXPECT_EQ(cfg.at("NPUW_ONLINE_PIPELINE").as<std::string>(), "REP");
  EXPECT_EQ(cfg.at("NPUW_ONLINE_ISOLATE").as<std::string>(), "ATTN,MOE");
  EXPECT_EQ(cfg.at("NPUW_MOE_TOKEN_CHUNK_SIZE").as<std::string>(), "0");
  EXPECT_EQ(cfg.at("NPUW_FOLD").as<std::string>(), "YES");
  EXPECT_EQ(cfg.at("NPUW_WEIGHTS_BANK").as<std::string>(),
            "litert_npuw_fedcba9876543210");
}

TEST(OpenVinoCompileContextTest,
     ConfigureForNpuWeightSharingRejectsCwaiAndFoldBothDisabled) {
  auto opts = IntelOpenVinoOptions::Create();
  ASSERT_TRUE(opts.HasValue());
  // With NPUW on (the default) and both CWAI and FOLD off, NPUW has no way to
  // share weights and would assert during compile.
  opts->SetConfigsMapOption("NPUW_CWAI", "NO");
  opts->SetConfigsMapOption("NPUW_FOLD", "NO");

  auto ctx_or = OpenVinoCompileContext::Create(opts);
  ASSERT_TRUE(ctx_or.HasValue());
  OpenVinoCompileContext& ctx = ctx_or.Value();

  EXPECT_EQ(ctx.ConfigureForNpuWeightSharing("litert_npuw_0123456789abcdef",
                                             /*moe_multi_token_chunk=*/false),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(OpenVinoCompileContextTest,
     ConfigureForNpuWeightSharingRejectsFoldDisabledOnMultiTokenPipeline) {
  auto opts = IntelOpenVinoOptions::Create();
  ASSERT_TRUE(opts.HasValue());
  // The multi-token pipeline defaults NPUW_FOLD=YES and leaves NPUW_CWAI
  // unset, so turning FOLD off leaves neither sharing mechanism enabled.
  opts->SetConfigsMapOption("NPUW_FOLD", "NO");

  auto ctx_or = OpenVinoCompileContext::Create(opts);
  ASSERT_TRUE(ctx_or.HasValue());
  OpenVinoCompileContext& ctx = ctx_or.Value();

  EXPECT_EQ(ctx.ConfigureForNpuWeightSharing("litert_npuw_fedcba9876543210",
                                             /*moe_multi_token_chunk=*/true),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(OpenVinoCompileContextTest,
     ConfigureForNpuWeightSharingAllowsFoldOnlyWhenCwaiAndFoldDisabled) {
  auto opts = IntelOpenVinoOptions::Create();
  ASSERT_TRUE(opts.HasValue());
  opts->SetConfigsMapOption("NPUW_CWAI", "NO");
  opts->SetConfigsMapOption("NPUW_FOLD", "NO");
  opts->SetConfigsMapOption("NPUW_FOLD_ONLY", "ATTN");

  auto ctx_or = OpenVinoCompileContext::Create(opts);
  ASSERT_TRUE(ctx_or.HasValue());
  OpenVinoCompileContext& ctx = ctx_or.Value();

  EXPECT_EQ(ctx.ConfigureForNpuWeightSharing("litert_npuw_0123456789abcdef",
                                             /*moe_multi_token_chunk=*/false),
            kLiteRtStatusOk);
}

TEST(OpenVinoCompileContextTest,
     ConfigureForNpuWeightSharingAllowsCwaiAndFoldDisabledWithoutNpuw) {
  auto opts = IntelOpenVinoOptions::Create();
  ASSERT_TRUE(opts.HasValue());
  // CWAI and FOLD are NPUW options; with NPUW off they have no effect, so the
  // combination is not rejected.
  opts->SetConfigsMapOption("NPU_USE_NPUW", "NO");
  opts->SetConfigsMapOption("NPUW_CWAI", "NO");
  opts->SetConfigsMapOption("NPUW_FOLD", "NO");

  auto ctx_or = OpenVinoCompileContext::Create(opts);
  ASSERT_TRUE(ctx_or.HasValue());
  OpenVinoCompileContext& ctx = ctx_or.Value();

  EXPECT_EQ(ctx.ConfigureForNpuWeightSharing("litert_npuw_0123456789abcdef",
                                             /*moe_multi_token_chunk=*/false),
            kLiteRtStatusOk);
}

}  // namespace
}  // namespace openvino
}  // namespace litert

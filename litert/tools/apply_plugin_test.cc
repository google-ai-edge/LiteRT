// Copyright 2024 Google LLC.
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

#include "litert/tools/apply_plugin.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_model.h"
#include "litert/core/build_stamp.h"
#include "litert/core/dispatch_op_schema.h"
#include "litert/core/model/model.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/tools/outstream.h"

namespace litert::tools {
namespace {

using ::litert::internal::kLiteRtBuildStampKey;
using ::litert::internal::ParseBuildStamp;
using ::litert::testing::GetLiteRtPath;
using ::testing::HasSubstr;
using ::testing::litert::IsError;

static constexpr absl::string_view kSocManufacturer = "ExampleSocManufacturer";

static constexpr absl::string_view kSocModel = "ExampleSocModel";

absl::string_view TestModelPath(absl::string_view filename) {
  static char kModelPath[512] = {};
  const auto model_path = ::litert::testing::GetTestFilePath(filename);
  ABSL_CHECK(model_path.size() < 512);
  model_path.copy(kModelPath, model_path.size(), 0);
  return kModelPath;
}

ApplyPluginRun::Ptr MakeBaseRun(
    ApplyPluginRun::Cmd cmd, absl::string_view model_path = "one_mul.tflite") {
  auto run = std::make_unique<ApplyPluginRun>();
  run->cmd = cmd;
  run->lib_search_paths.push_back(GetLiteRtPath("vendors/examples/"));
  run->model.emplace(TestModelPath(model_path));
  run->soc_manufacturer.emplace(std::string(kSocManufacturer));
  run->soc_models.push_back(std::string(kSocModel));
  run->outs.clear();
  run->dump_out = UserStream(std::cerr);
  return run;
}

TEST(TestApplyPluginTool, TestInfoBadConfig) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::INFO);
  run->lib_search_paths.clear();
  EXPECT_THAT(ApplyPlugin(std::move(run)),
              IsError(kLiteRtStatusErrorInvalidToolConfig));
}

TEST(TestApplyPluginTool, TestInfo) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::INFO);
  std::stringstream out;
  run->outs.push_back(out);
  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));
  EXPECT_THAT(out.str(),
              ::testing::HasSubstr(
                  "< LiteRtCompilerPlugin > \"ExampleSocManufacturer\" | "
                  "\"ExampleSocModel\""));
}

TEST(TestApplyPluginTool, TestNoopBadConfig) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::NOOP);
  run->model.reset();
  EXPECT_THAT(ApplyPlugin(std::move(run)),
              IsError(kLiteRtStatusErrorInvalidToolConfig));
}

TEST(TestApplyPluginTool, TestNoop) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::NOOP);
  std::stringstream out;
  run->outs.push_back(out);
  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));

  std::string out_str = out.str();
  BufferRef<uint8_t> serialized(out_str.data(), out_str.size());
  LITERT_ASSERT_OK_AND_ASSIGN(auto model, Model::CreateFromBuffer(serialized));

  EXPECT_EQ(model.Get()->NumSubgraphs(), 1);
}

TEST(TestApplyPluginTool, TestPartitionBadConfig) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::PARTITION);
  run->model.reset();
  EXPECT_THAT(ApplyPlugin(std::move(run)),
              IsError(kLiteRtStatusErrorInvalidToolConfig));
}

TEST(TestApplyPluginTool, TestPartition) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::PARTITION);
  std::stringstream out;
  run->outs.push_back(out);
  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));
  EXPECT_FALSE(out.str().empty());
}

TEST(TestApplyPluginTool, TestCompileBadConfig) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::COMPILE);
  run->model.reset();
  EXPECT_THAT(ApplyPlugin(std::move(run)),
              IsError(kLiteRtStatusErrorInvalidToolConfig));
}

TEST(TestApplyPluginTool, TestCompile) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::COMPILE);
  std::stringstream out;
  run->outs.push_back(out);
  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));
  EXPECT_FALSE(out.str().empty());
  EXPECT_THAT(out.str(), HasSubstr("inputs:0,1\noutputs:2\nconst_map:\ntensors:"
                                   "[2x2],[2x2],[2x2]\nops:mul(0,1)(2)"));
}

TEST(TestApplyPluginTool, TestApplyBadConfig) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::APPLY);
  run->model.reset();
  EXPECT_THAT(ApplyPlugin(std::move(run)),
              IsError(kLiteRtStatusErrorInvalidToolConfig));
}

TEST(TestApplyPluginTool, TestApply) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::APPLY);
  std::stringstream out;
  run->outs.push_back(out);
  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));

  const auto out_str = out.str();
  BufferRef<uint8_t> serialized(out_str.data(), out_str.size());

  auto model = Model::CreateFromBuffer(serialized);
  EXPECT_EQ(model->Get()->NumSubgraphs(), 1);

  {
    auto stamp_buffer = model->Get()->FindMetadata(kLiteRtBuildStampKey);
    auto stamp = ParseBuildStamp(*stamp_buffer);
    auto [man, soc_model] = *stamp;
    EXPECT_EQ(man, kSocManufacturer);
    EXPECT_EQ(soc_model, kSocModel);
  }

  auto* op = model->Get()->MainSubgraph()->Ops().front();
  ASSERT_EQ(op->OpCode(), kLiteRtOpCodeTflCustom);

  const auto options = internal::GetDispatchOpOptions(op->CustomOptions());
  const auto& [size, offset, name] = options;
  EXPECT_EQ(name, "partition_0");
  ASSERT_LE(offset + size, serialized.Size());

  EXPECT_THAT(serialized.StrView().substr(offset, size),
              HasSubstr("inputs:0,1\noutputs:2\nconst_map:\ntensors:[2x2],[2x2]"
                        ",[2x2]\nops:mul(0,1)(2)"));
}

TEST(TestApplyPluginTool, TestCompileToMultiByteCode) {
  auto run =
      MakeBaseRun(ApplyPluginRun::Cmd::COMPILE, "multi_subgraph_mul.tflite");
  std::stringstream out_0;
  std::stringstream out_1;
  run->outs.push_back(out_0);
  run->outs.push_back(out_1);

  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));
  EXPECT_FALSE(out_0.str().empty());
  EXPECT_FALSE(out_1.str().empty());
  EXPECT_THAT(out_0.str(),
              HasSubstr("inputs:0,1\noutputs:2\nconst_map:\ntensors:[2x2],[2x2]"
                        ",[2x2]\nops:mul(0,1)(2)"));
  EXPECT_THAT(out_1.str(),
              HasSubstr("inputs:0,1\noutputs:2\nconst_map:\ntensors:[4x4],[4x4]"
                        ",[4x4]\nops:mul(0,1)(2)"));
}

}  // namespace
}  // namespace litert::tools

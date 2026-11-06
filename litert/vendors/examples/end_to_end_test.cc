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

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_api_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_runtime_options.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"

namespace litert::example {
namespace {

using ::litert::testing::GetLiteRtPath;
using ::litert::testing::GetTestFilePath;
using ::testing::ElementsAre;

static constexpr absl::string_view kLibsPath = "vendors/examples";
static constexpr absl::string_view kModel = "one_mul.tflite";

TEST(ExampleEndToEndTest, JIT) {
  const auto libs_path = GetLiteRtPath(kLibsPath);
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          libs_path,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(litert::EnvironmentOptions(
                    absl::MakeConstSpan(environment_options))));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto cm, CompiledModel::Create(env, GetTestFilePath(kModel),
                                     HwAccelerators::kNpu));
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              cm.CreateInputBuffers(cm.DefaultSignatureKey()));
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              cm.CreateOutputBuffers(cm.DefaultSignatureKey()));
  LITERT_ASSERT_OK(input_buffers[0].Write<float>({1.0f, 2.0f, 3.0f, 4.0f}));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>({1.0f, 2.0f, 3.0f, 4.0f}));
  LITERT_ASSERT_OK(
      cm.Run(cm.DefaultSignatureKey(), input_buffers, output_buffers));
  std::vector<float> output(4);
  LITERT_ASSERT_OK(output_buffers[0].Read(absl::MakeSpan(output)));
  EXPECT_THAT(output, ElementsAre(1.0f, 4.0f, 9.0f, 16.0f));
}

TEST(ExampleEndToEndTest, JitHandleExecution) {
  const auto libs_path = GetLiteRtPath(kLibsPath);
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          libs_path,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(litert::EnvironmentOptions(
                    absl::MakeConstSpan(environment_options))));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto cm, CompiledModel::Create(env, GetTestFilePath(kModel),
                                     HwAccelerators::kNpu));
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              cm.CreateInputBuffers(cm.DefaultSignatureKey()));
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              cm.CreateOutputBuffers(cm.DefaultSignatureKey()));
  LITERT_ASSERT_OK(input_buffers[0].Write<float>({2.0f, 2.0f, 2.0f, 2.0f}));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>({1.0f, 2.0f, 3.0f, 4.0f}));
  LITERT_ASSERT_OK(
      cm.Run(cm.DefaultSignatureKey(), input_buffers, output_buffers));
  std::vector<float> output(4);
  LITERT_ASSERT_OK(output_buffers[0].Read(absl::MakeSpan(output)));
  EXPECT_THAT(output, ElementsAre(2.0f, 4.0f, 6.0f, 8.0f));
}

TEST(ExampleEndToEndTest, SelectedSignaturesWithMultiSubgraphModel) {
  const auto libs_path = GetLiteRtPath(kLibsPath);
  const std::vector<litert::EnvironmentOptions::Option> environment_options = {
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          libs_path,
      },
      litert::EnvironmentOptions::Option{
          litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          libs_path,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(litert::EnvironmentOptions(
                    absl::MakeConstSpan(environment_options))));

  LiteRtModel raw_model = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateModelFromFile(
      env.Get(), GetTestFilePath("cst_multi_subgraph.tflite").c_str(),
      &raw_model));
  char* signatures[] = {const_cast<char*>("sig_0"), const_cast<char*>("sig_1")};
  OwningBufferRef<uint8_t> serialized_model;
  auto [data, size, offset] = serialized_model.GetWeak();
  const LiteRtModelSerializationOptions serialization_options = {
      /*bytecode_alignment=*/64};
  LITERT_ASSERT_OK(LiteRtSerializeModelWithSignatures(
      raw_model, &data, &size, &offset, /*destroy_model=*/true, signatures,
      /*num_signatures=*/2, serialization_options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());
  options.SetHardwareAccelerators(HwAccelerators::kNpu);
  LITERT_ASSERT_OK_AND_ASSIGN(auto& runtime_options,
                              options.GetRuntimeOptions());
  const StringView selected_signatures[] = {"sig_1"};
  LITERT_ASSERT_OK(runtime_options.SetSelectedSignatures(selected_signatures));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto cm, CompiledModel::Create(env, serialized_model, options));
  EXPECT_EQ(cm.GetNumSignatures(), 1);
  LITERT_ASSERT_OK_AND_ASSIGN(auto signature_keys, cm.GetSignatureKeys());
  EXPECT_THAT(signature_keys, ElementsAre("sig_1"));

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              cm.CreateInputBuffers("sig_1"));
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              cm.CreateOutputBuffers("sig_1"));
  LITERT_ASSERT_OK(input_buffers[0].Write<float>({10.0f, 20.0f, 30.0f, 40.0f}));
  LITERT_ASSERT_OK(cm.Run("sig_1", input_buffers, output_buffers));
  std::vector<float> output(4);
  LITERT_ASSERT_OK(output_buffers[0].Read(absl::MakeSpan(output)));
  EXPECT_THAT(output, ElementsAre(10.0f, 40.0f, 90.0f, 160.0f));
}

}  // namespace
}  // namespace litert::example

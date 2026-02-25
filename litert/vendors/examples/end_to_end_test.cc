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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
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

}  // namespace
}  // namespace litert::example

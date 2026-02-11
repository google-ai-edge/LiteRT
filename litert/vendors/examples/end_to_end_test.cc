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

#include <cstdio>
#include <fstream>
#include <ios>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_compiler_options.h"
#include "litert/compiler/plugin/compiler_plugin.h"
#include "litert/core/model/model_load.h"
#include "litert/core/model/model_serialize.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"

namespace litert::example {
namespace {

using ::litert::testing::GetLiteRtPath;
using ::litert::testing::GetTestFilePath;
using ::testing::ElementsAre;

static constexpr absl::string_view kLibsPath = "vendors/examples";
static constexpr absl::string_view kModel = "one_mul.tflite";

TEST(ExampleEndToEndTest, E2E_CustomOp) {
  std::string compiled_model_path = std::tmpnam(nullptr);
  const auto libs_path = GetLiteRtPath(kLibsPath);

  // Compile the model with custom op info and serialize it to a temporary
  // file.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto model, litert::internal::LoadModelFromFile(
                        GetTestFilePath("simple_npu_model_custom_op.tflite")));

    // Create compiler options with custom op info.
    LITERT_ASSERT_OK_AND_ASSIGN(auto compiler_options,
                                litert::CompilerOptions::Create());
    LITERT_ASSERT_OK(compiler_options.AddCustomOpInfo(
        "litert_cust", GetLiteRtPath("test/testdata/litert_cust.so")));

    // Prepare options for loading plugins.
    LITERT_ASSERT_OK_AND_ASSIGN(auto options, litert::Options::Create());
    options.AddOpaqueOptions(std::move(compiler_options));

    // Load and apply plugin.
    std::vector<absl::string_view> search_paths = {libs_path};
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto plugins, litert::internal::CompilerPlugin::LoadPlugins(
                          search_paths, /*env=*/nullptr, options.Get()));
    ASSERT_EQ(plugins.size(), 1);
    LITERT_ASSERT_OK(litert::internal::ApplyPlugin(plugins[0], *model));

    // Serialize the compiled model.
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto serialized_buf,
        litert::internal::SerializeModel(std::move(*model)));

    // Write serialized model to a temporary file.

    std::ofstream out(compiled_model_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(serialized_buf.Data()),
              serialized_buf.Size());
    out.close();
  }

  // Create environment options for runtime.
  const std::vector<Environment::Option> runtime_environment_options = {
      Environment::Option{
          Environment::OptionTag::DispatchLibraryDir,
          libs_path,
      },
  };

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto runtime_env,
      Environment::Create(absl::MakeConstSpan(runtime_environment_options)));
  // Load and run the compiled model.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto cm, CompiledModel::Create(runtime_env, compiled_model_path,
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

  std::remove(compiled_model_path.c_str());
}

TEST(ExampleEndToEndTest, JIT) {
  const auto libs_path = GetLiteRtPath(kLibsPath);
  const std::vector<Environment::Option> environment_options = {
      Environment::Option{
          Environment::OptionTag::CompilerPluginLibraryDir,
          libs_path,
      },
      Environment::Option{
          Environment::OptionTag::DispatchLibraryDir,
          libs_path,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(absl::MakeConstSpan(environment_options)));
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

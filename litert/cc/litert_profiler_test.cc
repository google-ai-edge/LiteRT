// Copyright 2025 Google LLC.
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

#include <gtest/gtest.h>
#include "absl/strings/match.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/test/common.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

namespace litert {
namespace {

TEST(LiteRtProfilerCcTest, GetProfileSummary) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());

  auto options = Options::Create();
  ASSERT_TRUE(options.HasValue());
  options->SetHardwareAccelerators(HwAccelerators::kCpu);
  auto runtime_options = options->GetRuntimeOptions();
  ASSERT_TRUE(runtime_options.HasValue());
  runtime_options->SetEnableProfiling(true);
  auto compiled_model = CompiledModel::Create(
      *env, testing::GetTestFilePath(kModelFileName), *options);
  ASSERT_TRUE(compiled_model.HasValue());

  auto profiler = compiled_model->GetProfiler();
  ASSERT_TRUE(profiler.HasValue());

  EXPECT_TRUE(profiler->StartProfiling());
  EXPECT_TRUE(profiler->StopProfiling());

  auto summary = profiler->GetProfileSummary(compiled_model->Get());
  ASSERT_TRUE(summary.HasValue());
  EXPECT_TRUE(absl::StrContains(summary.Value(), "nodes observed"));
}

}  // namespace
}  // namespace litert

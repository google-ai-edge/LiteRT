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

#include "litert/vendors/google_tensor/adapter.h"

#include <sys/types.h>

#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/options/litert_google_tensor_options.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"

namespace litert {
namespace google_tensor {

TEST(AdapterTest, CreateSuccess) {
  auto adapter_result = Adapter::Create(/*shared_library_dir=*/
                                        std::nullopt);
  if (!adapter_result.HasValue()) {
    LITERT_LOG(LITERT_ERROR, "Failed to create Adapter: %s",
               adapter_result.Error().Message().c_str());
  }
  ASSERT_TRUE(adapter_result.HasValue());
}

TEST(AdapterTest, CreateFailure) {
  auto kLibDarwinnCompilerNoLib = "libcompiler_api_wrapper_no_lib.so";
  auto adapter_result = Adapter::Create(kLibDarwinnCompilerNoLib);
  ASSERT_FALSE(adapter_result.HasValue());
}

TEST(AdapterTest, CompileSuccess) {
  static constexpr absl::string_view kSocModel = "P25";

  auto adapter_result = Adapter::Create(/*shared_library_dir=*/
                                        std::nullopt);
  if (!adapter_result.HasValue()) {
    LITERT_LOG(LITERT_ERROR, "Failed to create Adapter: %s",
               adapter_result.Error().Message().c_str());
  }

  auto model = litert::testing::LoadTestFileModel("mul_simple.tflite");
  LiteRtModel litert_model = model.Get();

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options, ::litert::google_tensor::GoogleTensorOptions::Create());
  options.SetFloatTruncationType(kLiteRtGoogleTensorFloatTruncationTypeHalf);
  options.SetInt64ToInt32Truncation(true);
  options.SetOutputDir("/tmp/");
  options.SetDumpOpTimings(true);

  LITERT_LOG(LITERT_INFO, "Compling model...");

  std::string compiled_code;
  auto compile_status = adapter_result.Value()->api().compile(
      litert_model, kSocModel, options.Get(), &compiled_code);
  ASSERT_OK(compile_status);
  ASSERT_FALSE(compiled_code.empty());
}

}  // namespace google_tensor
}  // namespace litert

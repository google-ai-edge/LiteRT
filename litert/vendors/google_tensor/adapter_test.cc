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

#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
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
  static constexpr absl::string_view kSocModel = "Tensor_G5";

  auto adapter_result = Adapter::Create(/*shared_library_dir=*/
                                        std::nullopt);
  if (!adapter_result.HasValue()) {
    LITERT_LOG(LITERT_ERROR, "Failed to create Adapter: %s",
               adapter_result.Error().Message().c_str());
  }
  const auto& api = adapter_result.Value()->api();

  // Ensure all necessary API functions are loaded
  ASSERT_NE(api.compile, nullptr);
  ASSERT_NE(api.free_compiled_code, nullptr);
  ASSERT_NE(api.free_error_message, nullptr);

  auto model = litert::testing::LoadTestFileModel("mul_simple.tflite");
  ASSERT_NE(model.Get(), nullptr);
  LiteRtModel litert_model = model.Get();

  LITERT_LOG(LITERT_INFO, "%s", "Serializing model");
  litert::OwningBufferRef buf;

  // Using weak pointer to link the data to the buffer.
  auto [data, size, offset] = buf.GetWeak();

  const auto opts = litert::SerializationOptions::Defaults();
  auto serialize_status =
      LiteRtSerializeModel(litert_model, &data, &size, &offset, false, opts);
  ASSERT_EQ(serialize_status, kLiteRtStatusOk);
  ASSERT_GT(buf.Size(), 0);

  LITERT_ASSERT_OK_AND_ASSIGN(auto options, GoogleTensorOptions::Create());
  options.SetFloatTruncationType(kLiteRtGoogleTensorFloatTruncationTypeHalf);
  options.SetInt64ToInt32Truncation(true);
  options.SetOutputDir("/tmp/");
  options.SetDumpOpTimings(true);

  ASSERT_GT(buf.Size(), 0);
  LITERT_LOG(LITERT_INFO, "buffer_str size: %d", buf.Size());
  LITERT_LOG(LITERT_INFO, "Compling model...");

  char* compiled_code_data = nullptr;
  size_t compiled_code_size = 0;
  char* error_message = nullptr;

  absl::string_view soc_model_view(kSocModel);
  absl::string_view model_buffer_view(buf.StrView());
  // Ensure memory allocated by the C API is freed.
  absl::Cleanup error_cleanup = [&] {
    if (error_message) {
      api.free_error_message(error_message);
    }
  };
  absl::Cleanup code_cleanup = [&] {
    if (compiled_code_data) {
      api.free_compiled_code(compiled_code_data);
    }
  };
  auto compile_status =
      api.compile(model_buffer_view.data(), model_buffer_view.size(),
                  soc_model_view.data(), soc_model_view.size(), options.Get(),
                  &compiled_code_data, &compiled_code_size, &error_message);
  ASSERT_TRUE(compile_status);
  ASSERT_NE(compiled_code_data, nullptr);
  ASSERT_GT(compiled_code_size, 0);
}

}  // namespace google_tensor
}  // namespace litert

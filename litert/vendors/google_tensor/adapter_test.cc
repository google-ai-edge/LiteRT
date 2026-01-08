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

#include <cstddef>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/vendors/google_tensor/compiler/google_tensor_options.pb.h"

namespace litert {
namespace google_tensor {

using ::third_party::odml::litert::litert::vendors::google_tensor::compiler::
    DeviceType;
using ::third_party::odml::litert::litert::vendors::google_tensor::compiler::
    GoogleTensorOptions;
using ::third_party::odml::litert::litert::vendors::google_tensor::compiler::
    GoogleTensorOptionsTruncationType;

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
  auto kLibDarwinnCompilerNoLib = "liblitert_plugin_compiler_no_lib.so";
  auto adapter_result = Adapter::Create(kLibDarwinnCompilerNoLib);
  ASSERT_FALSE(adapter_result.HasValue());
}

TEST(AdapterTest, CompileSuccess) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto adapter,
                              Adapter::Create(/*shared_library_dir=*/
                                              std::nullopt));

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

  GoogleTensorOptions google_tensor_options;
  google_tensor_options.set_float_truncation_type(
      GoogleTensorOptionsTruncationType::FLOAT_TRUNCATION_TYPE_HALF);
  google_tensor_options.set_int64_to_int32_truncation(true);
  google_tensor_options.set_dump_op_timings(true);
  google_tensor_options.mutable_compiler_config()->set_device(
      DeviceType::DEVICE_TYPE_TENSOR_G5);
  google_tensor_options.set_output_dir("/tmp/");

  std::string options_str = google_tensor_options.SerializeAsString();

  ASSERT_GT(buf.Size(), 0);
  LITERT_LOG(LITERT_INFO, "buffer_str size: %d", buf.Size());
  LITERT_LOG(LITERT_INFO, "Compling model...");

  char** compiled_code_data = nullptr;
  size_t* compiled_code_sizes = nullptr;
  size_t num_bytecodes = 0;

  absl::string_view model_buffer_view(buf.StrView());
  // Ensure memory allocated by the C API is freed.
  absl::Cleanup code_cleanup = [&] {
    if (compiled_code_data) {
      adapter->FreeCompiledCode(compiled_code_data, compiled_code_sizes,
                                num_bytecodes);
    }
  };
  LITERT_ASSERT_OK(adapter->Compile(
      model_buffer_view.data(), model_buffer_view.size(), options_str.data(),
      options_str.size(), &compiled_code_data,
      &compiled_code_sizes, &num_bytecodes));
  ASSERT_NE(compiled_code_data, nullptr);
  ASSERT_GT(num_bytecodes, 0);
  for (int i = 0; i < num_bytecodes; ++i) {
    ASSERT_GT(compiled_code_sizes[i], 0);
  }
}

}  // namespace google_tensor
}  // namespace litert

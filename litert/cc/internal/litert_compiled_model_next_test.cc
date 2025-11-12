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

#include "litert/cc/internal/litert_compiled_model_next.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::Pointwise;

namespace litert {
namespace {

TEST(CompiledModelTest, Basic) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModelNext compiled_model,
      CompiledModelNext::Create(env, model, HwAccelerators::kCpu));

  // Check fully accelerated.
  LITERT_ASSERT_OK_AND_ASSIGN(auto fullyAccelerated,
                              compiled_model.IsFullyAccelerated());
  ASSERT_TRUE(fullyAccelerated);

  // Check CompiledModel buffer requirements.
  // input and output expect host memory.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg0,
      compiled_model.GetInputBufferRequirements(/*input_name=*/"arg0"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBufferType> input_buffer_types_arg0,
      input_buffer_requirements_arg0.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(TensorBufferType::kHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg1,
      compiled_model.GetInputBufferRequirements(/*input_name=*/"arg1"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBufferType> input_buffer_types_arg1,
      input_buffer_requirements_arg1.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(TensorBufferType::kHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements output_buffer_requirements,
      compiled_model.GetOutputBufferRequirements(/*output_name=*/"tfl.add"));
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBufferType> output_buffer_types,
                              output_buffer_requirements.SupportedTypes());
  EXPECT_THAT(output_buffer_types, ElementsAre(TensorBufferType::kHostMemory));

  // Create and fill input and output buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> input_buffers,
                              compiled_model.CreateInputBuffers());

  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> output_buffers,
                              compiled_model.CreateOutputBuffers());

  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model with input and output buffers.
  compiled_model.Run(input_buffers, output_buffers);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

TEST(CompiledModelTest, DispatchAnnotations) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModelNext compiled_model,
      CompiledModelNext::Create(env, model, HwAccelerators::kCpu));

  // Test 1: Set and get annotation for signature index 0
  {
    LITERT_ASSERT_OK(
        compiled_model.SetDispatchAnnotation(0, "priority", "high"));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value, compiled_model.GetDispatchAnnotation(0, "priority"));
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), "high");
  }

  // Test 2: Set and get annotation for default signature (overload without
  // index)
  {
    LITERT_ASSERT_OK(
        compiled_model.SetDispatchAnnotation("memory_type", "shared"));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value, compiled_model.GetDispatchAnnotation("memory_type"));
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), "shared");
  }

  // Test 3: Get non-existent annotation
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value, compiled_model.GetDispatchAnnotation(0, "nonexistent"));
    EXPECT_FALSE(value.has_value());
  }

  // Test 4: Remove annotation
  {
    // First set an annotation
    LITERT_ASSERT_OK(
        compiled_model.SetDispatchAnnotation(0, "to_remove", "value"));

    // Verify it exists
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value_before,
        compiled_model.GetDispatchAnnotation(0, "to_remove"));
    ASSERT_TRUE(value_before.has_value());

    // Remove it
    LITERT_ASSERT_OK(compiled_model.RemoveDispatchAnnotation(0, "to_remove"));

    // Verify it's gone
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value_after, compiled_model.GetDispatchAnnotation(0, "to_remove"));
    EXPECT_FALSE(value_after.has_value());
  }

  // Test 5: Remove non-existent annotation (should succeed)
  {
    LITERT_ASSERT_OK(
        compiled_model.RemoveDispatchAnnotation(0, "never_existed"));
  }

  // Test 6: Update existing annotation
  {
    LITERT_ASSERT_OK(
        compiled_model.SetDispatchAnnotation(0, "accelerator", "gpu"));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value1, compiled_model.GetDispatchAnnotation(0, "accelerator"));
    ASSERT_TRUE(value1.has_value());
    EXPECT_EQ(value1.value(), "gpu");

    // Update to new value
    LITERT_ASSERT_OK(
        compiled_model.SetDispatchAnnotation(0, "accelerator", "npu"));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value2, compiled_model.GetDispatchAnnotation(0, "accelerator"));
    ASSERT_TRUE(value2.has_value());
    EXPECT_EQ(value2.value(), "npu");
  }

  // Test 7: Multiple annotations on same signature
  {
    LITERT_ASSERT_OK(compiled_model.SetDispatchAnnotation(0, "key1", "value1"));
    LITERT_ASSERT_OK(compiled_model.SetDispatchAnnotation(0, "key2", "value2"));
    LITERT_ASSERT_OK(compiled_model.SetDispatchAnnotation(0, "key3", "value3"));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto val1, compiled_model.GetDispatchAnnotation(0, "key1"));
    ASSERT_TRUE(val1.has_value());
    EXPECT_EQ(val1.value(), "value1");

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto val2, compiled_model.GetDispatchAnnotation(0, "key2"));
    ASSERT_TRUE(val2.has_value());
    EXPECT_EQ(val2.value(), "value2");

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto val3, compiled_model.GetDispatchAnnotation(0, "key3"));
    ASSERT_TRUE(val3.has_value());
    EXPECT_EQ(val3.value(), "value3");
  }

  // Test 8: Empty string values
  {
    LITERT_ASSERT_OK(
        compiled_model.SetDispatchAnnotation(0, "empty_value", ""));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value, compiled_model.GetDispatchAnnotation(0, "empty_value"));
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), "");
  }

  // Test 9: Special characters in keys and values
  {
    LITERT_ASSERT_OK(compiled_model.SetDispatchAnnotation(
        0, "key-with-dashes", "value/with/slashes"));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value, compiled_model.GetDispatchAnnotation(0, "key-with-dashes"));
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), "value/with/slashes");
  }
}

TEST(CompiledModelTest, DispatchAnnotationsWithSignatureName) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  EXPECT_EQ(model.GetNumSignatures(), 1);

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModelNext compiled_model,
      CompiledModelNext::Create(env, model, HwAccelerators::kCpu));

  // Test 1: Set and get annotation using signature name
  {
    LITERT_ASSERT_OK(compiled_model.SetDispatchAnnotation("precision", "fp16"));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value, compiled_model.GetDispatchAnnotation("precision"));
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), "fp16");
  }

  // Test 2: Remove annotation using signature name
  {
    LITERT_ASSERT_OK(
        compiled_model.SetDispatchAnnotation("to_remove", "value"));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value_before, compiled_model.GetDispatchAnnotation("to_remove"));
    ASSERT_TRUE(value_before.has_value());

    LITERT_ASSERT_OK(compiled_model.RemoveDispatchAnnotation("to_remove"));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value_after, compiled_model.GetDispatchAnnotation("to_remove"));
    EXPECT_FALSE(value_after.has_value());
  }

  // Test 3: Annotations set by name should be retrievable by index
  {
    LITERT_ASSERT_OK(
        compiled_model.SetDispatchAnnotation("cross_check", "test_value"));

    // Since this is signature 0, we should be able to get it by index 0
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value, compiled_model.GetDispatchAnnotation(0, "cross_check"));
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), "test_value");
  }
}

TEST(CompiledModelTest, DispatchAnnotationsInvalidSignature) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModelNext compiled_model,
      CompiledModelNext::Create(env, model, HwAccelerators::kCpu));

  // Test with invalid signature index (model only has 1 signature at index 0)
  {
    auto result = compiled_model.SetDispatchAnnotation(999, "key", "value");
    EXPECT_FALSE(result.HasValue());
  }

  {
    auto result = compiled_model.GetDispatchAnnotation(999, "key");
    EXPECT_FALSE(result.HasValue());
  }

  {
    auto result = compiled_model.RemoveDispatchAnnotation(999, "key");
    EXPECT_FALSE(result.HasValue());
  }

  // Test with invalid signature name
  {
    auto result = compiled_model.SetDispatchAnnotation("nonexistent_signature",
                                                       "key", "value");
    EXPECT_FALSE(result.HasValue());
  }

  {
    auto result =
        compiled_model.GetDispatchAnnotation("nonexistent_signature", "key");
    EXPECT_FALSE(result.HasValue());
  }

  {
    auto result =
        compiled_model.RemoveDispatchAnnotation("nonexistent_signature", "key");
    EXPECT_FALSE(result.HasValue());
  }
}

}  // namespace
}  // namespace litert

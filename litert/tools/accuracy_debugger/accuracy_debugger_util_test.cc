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

#include "litert/tools/accuracy_debugger/accuracy_debugger_util.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/core/filesystem.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"

namespace litert::tools {
namespace {

// --- GetFloats Tests ---

TEST(AccuracyDebuggerUtilTest, GetFloatsFloat32ComplicatedShape) {
  std::vector<float> data = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f};
  Dimensions dims = {1, 2, 3};
  RankedTensorType tensor_type(ElementType::Float32, Layout(dims));
  auto buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 24);
  ASSERT_TRUE(buffer.HasValue());
  ASSERT_TRUE(buffer->Write<float>(absl::MakeConstSpan(data)).HasValue());

  auto floats = internal::GetFloats(buffer.Value(), 6);
  ASSERT_TRUE(floats.HasValue());
  EXPECT_EQ(floats->size(), 6);
  for (int i = 0; i < 6; ++i) EXPECT_NEAR((*floats)[i], data[i], 1e-6f);
}

TEST(AccuracyDebuggerUtilTest, GetFloatsInt64) {
  // Use values that are representable in float.
  std::vector<int64_t> data = {12345678LL, -12345678LL, 0LL};
  Dimensions dims = {3, 1};
  RankedTensorType tensor_type(ElementType::Int64, Layout(dims));
  auto buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 24);
  ASSERT_TRUE(buffer.HasValue());
  ASSERT_TRUE(buffer->Write<int64_t>(absl::MakeConstSpan(data)).HasValue());

  auto floats = internal::GetFloats(buffer.Value(), 3);
  ASSERT_TRUE(floats.HasValue());
  EXPECT_NEAR((*floats)[0], 12345678.0f, 1e-6f);
  EXPECT_NEAR((*floats)[1], -12345678.0f, 1e-6f);
  EXPECT_NEAR((*floats)[2], 0.0f, 1e-6f);
}

TEST(AccuracyDebuggerUtilTest, GetFloatsInt16) {
  std::vector<int16_t> data = {32767, -32768, 0};
  Dimensions dims = {1, 3};
  RankedTensorType tensor_type(ElementType::Int16, Layout(dims));
  auto buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 6);
  ASSERT_TRUE(buffer.HasValue());
  ASSERT_TRUE(buffer->Write<int16_t>(absl::MakeConstSpan(data)).HasValue());

  auto floats = internal::GetFloats(buffer.Value(), 3);
  ASSERT_TRUE(floats.HasValue());
  EXPECT_NEAR((*floats)[0], 32767.0f, 1e-6f);
  EXPECT_NEAR((*floats)[1], -32768.0f, 1e-6f);
  EXPECT_NEAR((*floats)[2], 0.0f, 1e-6f);
}

TEST(AccuracyDebuggerUtilTest, GetFloatsUInt16) {
  std::vector<uint16_t> data = {65535, 0, 12345};
  Dimensions dims = {3};
  RankedTensorType tensor_type(ElementType::UInt16, Layout(dims));
  auto buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 6);
  ASSERT_TRUE(buffer.HasValue());
  ASSERT_TRUE(buffer->Write<uint16_t>(absl::MakeConstSpan(data)).HasValue());

  auto floats = internal::GetFloats(buffer.Value(), 3);
  ASSERT_TRUE(floats.HasValue());
  EXPECT_NEAR((*floats)[0], 65535.0f, 1e-6f);
  EXPECT_NEAR((*floats)[1], 0.0f, 1e-6f);
  EXPECT_NEAR((*floats)[2], 12345.0f, 1e-6f);
}

TEST(AccuracyDebuggerUtilTest, GetFloatsInt8) {
  std::vector<int8_t> data = {127, -128, 0, 50};
  Dimensions dims = {2, 2};
  RankedTensorType tensor_type(ElementType::Int8, Layout(dims));
  auto buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 4);
  ASSERT_TRUE(buffer.HasValue());
  ASSERT_TRUE(buffer->Write<int8_t>(absl::MakeConstSpan(data)).HasValue());

  auto floats = internal::GetFloats(buffer.Value(), 4);
  ASSERT_TRUE(floats.HasValue());
  EXPECT_NEAR((*floats)[0], 127.0f, 1e-6f);
  EXPECT_NEAR((*floats)[1], -128.0f, 1e-6f);
  EXPECT_NEAR((*floats)[2], 0.0f, 1e-6f);
  EXPECT_NEAR((*floats)[3], 50.0f, 1e-6f);
}

TEST(AccuracyDebuggerUtilTest, GetFloatsUInt8) {
  std::vector<uint8_t> data = {255, 0, 128};
  Dimensions dims = {1, 1, 3};
  RankedTensorType tensor_type(ElementType::UInt8, Layout(dims));
  auto buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 3);
  ASSERT_TRUE(buffer.HasValue());
  ASSERT_TRUE(buffer->Write<uint8_t>(absl::MakeConstSpan(data)).HasValue());

  auto floats = internal::GetFloats(buffer.Value(), 3);
  ASSERT_TRUE(floats.HasValue());
  EXPECT_NEAR((*floats)[0], 255.0f, 1e-6f);
  EXPECT_NEAR((*floats)[1], 0.0f, 1e-6f);
  EXPECT_NEAR((*floats)[2], 128.0f, 1e-6f);
}

TEST(AccuracyDebuggerUtilTest, GetFloatsBool) {
  std::vector<uint8_t> data = {1, 0, 1, 1};
  Dimensions dims = {4};
  RankedTensorType tensor_type(ElementType::Bool, Layout(dims));
  auto buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 4);
  ASSERT_TRUE(buffer.HasValue());
  ASSERT_TRUE(buffer->Write<uint8_t>(absl::MakeConstSpan(data)).HasValue());

  auto floats = internal::GetFloats(buffer.Value(), 4);
  ASSERT_TRUE(floats.HasValue());
  EXPECT_NEAR((*floats)[0], 1.0f, 1e-6f);
  EXPECT_NEAR((*floats)[1], 0.0f, 1e-6f);
  EXPECT_NEAR((*floats)[2], 1.0f, 1e-6f);
  EXPECT_NEAR((*floats)[3], 1.0f, 1e-6f);
}

TEST(AccuracyDebuggerUtilTest, GetFloatsFp16SpecialValues) {
  // 0.0: 0x0000, -0.0: 0x8000, Inf: 0x7C00, -Inf: 0xFC00
  std::vector<uint16_t> data = {0x0000, 0x8000, 0x7C00, 0xFC00};
  Dimensions dims = {4};
  RankedTensorType tensor_type(ElementType::Float16, Layout(dims));
  auto buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 8);
  ASSERT_TRUE(buffer.HasValue());
  ASSERT_TRUE(buffer->Write<uint16_t>(absl::MakeConstSpan(data)).HasValue());

  auto floats = internal::GetFloats(buffer.Value(), 4);
  ASSERT_TRUE(floats.HasValue());
  EXPECT_NEAR((*floats)[0], 0.0f, 1e-6f);
  EXPECT_NEAR((*floats)[1], -0.0f, 1e-6f);
  EXPECT_TRUE(std::isinf((*floats)[2]));
  EXPECT_GT((*floats)[2], 0.0f);
  EXPECT_TRUE(std::isinf((*floats)[3]));
  EXPECT_LT((*floats)[3], 0.0f);
}

TEST(AccuracyDebuggerUtilTest, GetFloatsFp16Denormal) {
  // Smallest positive denormal in fp16 is 0x0001 (approx 5.96e-8)
  std::vector<uint16_t> data = {0x0001};
  Dimensions dims = {1};
  RankedTensorType tensor_type(ElementType::Float16, Layout(dims));
  auto buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 2);
  ASSERT_TRUE(buffer.HasValue());
  ASSERT_TRUE(buffer->Write<uint16_t>(absl::MakeConstSpan(data)).HasValue());

  auto floats = internal::GetFloats(buffer.Value(), 1);
  ASSERT_TRUE(floats.HasValue());
  EXPECT_NEAR((*floats)[0], 5.9604644775390625e-8f, 1e-12f);
}

TEST(AccuracyDebuggerUtilTest, GetFloatsEmptyElements) {
  Dimensions dims = {1};
  RankedTensorType tensor_type(ElementType::Float32, Layout(dims));
  auto buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 4);
  ASSERT_TRUE(buffer.HasValue());
  auto floats = internal::GetFloats(buffer.Value(), 0);
  ASSERT_TRUE(floats.HasValue());
  EXPECT_TRUE(floats->empty());
}

TEST(AccuracyDebuggerUtilTest, GetFloatsPartialRead) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  Dimensions dims = {3};
  RankedTensorType tensor_type(ElementType::Float32, Layout(dims));
  auto buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 12);
  ASSERT_TRUE(buffer.HasValue());
  ASSERT_TRUE(buffer->Write<float>(absl::MakeConstSpan(data)).HasValue());

  auto floats = internal::GetFloats(buffer.Value(), 2);
  ASSERT_TRUE(floats.HasValue());
  EXPECT_EQ(floats->size(), 2);
  EXPECT_NEAR((*floats)[0], 1.0f, 1e-6f);
  EXPECT_NEAR((*floats)[1], 2.0f, 1e-6f);
}

TEST(AccuracyDebuggerUtilTest, GetFloatsFloat16Max) {
  // Max finite fp16 is 0x7BFF (65504.0)
  std::vector<uint16_t> data = {0x7BFF};
  Dimensions dims = {1};
  RankedTensorType tensor_type(ElementType::Float16, Layout(dims));
  auto buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 2);
  ASSERT_TRUE(buffer.HasValue());
  ASSERT_TRUE(buffer->Write<uint16_t>(absl::MakeConstSpan(data)).HasValue());

  auto floats = internal::GetFloats(buffer.Value(), 1);
  ASSERT_TRUE(floats.HasValue());
  EXPECT_NEAR((*floats)[0], 65504.0f, 1e-1f);
}

TEST(AccuracyDebuggerUtilTest, GetFloatsInt32MinMax) {
  std::vector<int32_t> data = {16777216, -16777216};
  Dimensions dims = {2};
  RankedTensorType tensor_type(ElementType::Int32, Layout(dims));
  auto buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 8);
  ASSERT_TRUE(buffer.HasValue());
  ASSERT_TRUE(buffer->Write<int32_t>(absl::MakeConstSpan(data)).HasValue());

  auto floats = internal::GetFloats(buffer.Value(), 2);
  ASSERT_TRUE(floats.HasValue());
  EXPECT_NEAR((*floats)[0], 16777216.0f, 1e-6f);
  EXPECT_NEAR((*floats)[1], -16777216.0f, 1e-6f);
}

TEST(AccuracyDebuggerUtilTest, GetFloatsInt16MinMax) {
  std::vector<int16_t> data = {32767, -32768};
  Dimensions dims = {2};
  RankedTensorType tensor_type(ElementType::Int16, Layout(dims));
  auto buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 4);
  ASSERT_TRUE(buffer.HasValue());
  ASSERT_TRUE(buffer->Write<int16_t>(absl::MakeConstSpan(data)).HasValue());

  auto floats = internal::GetFloats(buffer.Value(), 2);
  ASSERT_TRUE(floats.HasValue());
  EXPECT_NEAR((*floats)[0], 32767.0f, 1e-6f);
  EXPECT_NEAR((*floats)[1], -32768.0f, 1e-6f);
}

TEST(AccuracyDebuggerUtilTest, GetFloatsUint8Edge) {
  std::vector<uint8_t> data = {0, 255};
  Dimensions dims = {2};
  RankedTensorType tensor_type(ElementType::UInt8, Layout(dims));
  auto buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 2);
  ASSERT_TRUE(buffer.HasValue());
  ASSERT_TRUE(buffer->Write<uint8_t>(absl::MakeConstSpan(data)).HasValue());

  auto floats = internal::GetFloats(buffer.Value(), 2);
  ASSERT_TRUE(floats.HasValue());
  EXPECT_NEAR((*floats)[0], 0.0f, 1e-6f);
  EXPECT_NEAR((*floats)[1], 255.0f, 1e-6f);
}

// --- CompareBuffers Tests ---

TEST(AccuracyDebuggerUtilTest, CompareBuffersCosSimTinyDiff) {
  std::vector<float> cpu_data(1000, 1.0f);
  std::vector<float> accel_data = cpu_data;
  accel_data[0] = 1.0001f;
  Dimensions dims = {1000};
  RankedTensorType tensor_type(ElementType::Float32, Layout(dims));

  auto cpu_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 4000);
  auto accel_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 4000);
  cpu_buffer->Write<float>(absl::MakeConstSpan(cpu_data));
  accel_buffer->Write<float>(absl::MakeConstSpan(accel_data));

  AccuracyThresholds thresholds;
  auto res = internal::CompareBuffers(cpu_buffer.Value(), accel_buffer.Value(),
                                      thresholds, "test");
  ASSERT_TRUE(res.HasValue());
  EXPECT_LT(res->cosine_similarity, 1.0);
}

TEST(AccuracyDebuggerUtilTest, CompareBuffersSNR) {
  std::vector<float> cpu_data = {1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> accel_data = {1.1f, 0.9f, 1.1f, 0.9f};
  Dimensions dims = {2, 2};
  RankedTensorType tensor_type(ElementType::Float32, Layout(dims));

  auto cpu_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 16);
  auto accel_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 16);
  cpu_buffer->Write<float>(absl::MakeConstSpan(cpu_data));
  accel_buffer->Write<float>(absl::MakeConstSpan(accel_data));

  AccuracyThresholds thresholds;
  auto res = internal::CompareBuffers(cpu_buffer.Value(), accel_buffer.Value(),
                                      thresholds, "test");
  ASSERT_TRUE(res.HasValue());
  EXPECT_NEAR(res->snr, 20.0, 1e-5);
}

TEST(AccuracyDebuggerUtilTest, CompareBuffersPSNR) {
  std::vector<float> cpu_data = {0.0f, 0.5f, 1.0f, 2.0f};
  std::vector<float> accel_data = {0.0f, 0.5f, 1.0f, 1.9f};
  Dimensions dims = {1, 4};
  RankedTensorType tensor_type(ElementType::Float32, Layout(dims));

  auto cpu_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 16);
  auto accel_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 16);
  cpu_buffer->Write<float>(absl::MakeConstSpan(cpu_data));
  accel_buffer->Write<float>(absl::MakeConstSpan(accel_data));

  AccuracyThresholds thresholds;
  auto res = internal::CompareBuffers(cpu_buffer.Value(), accel_buffer.Value(),
                                      thresholds, "test");
  ASSERT_TRUE(res.HasValue());
  EXPECT_NEAR(res->psnr, 32.0412, 1e-3);
}

TEST(AccuracyDebuggerUtilTest, CompareBuffersInt64Precision) {
  std::vector<int64_t> cpu_data = {1000000, 2000000};
  std::vector<int64_t> accel_data = {1000100, 1999900};
  Dimensions dims = {2};
  RankedTensorType tensor_type(ElementType::Int64, Layout(dims));

  auto cpu_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 16);
  auto accel_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 16);
  cpu_buffer->Write<int64_t>(absl::MakeConstSpan(cpu_data));
  accel_buffer->Write<int64_t>(absl::MakeConstSpan(accel_data));

  AccuracyThresholds thresholds;
  auto res = internal::CompareBuffers(cpu_buffer.Value(), accel_buffer.Value(),
                                      thresholds, "test");
  ASSERT_TRUE(res.HasValue());
  EXPECT_NEAR(res->max_diff, 100.0f, 1e-6f);
}

TEST(AccuracyDebuggerUtilTest, CompareBuffersOrthogonal) {
  std::vector<float> cpu_data = {1.0f, 0.0f};
  std::vector<float> accel_data = {0.0f, 1.0f};
  Dimensions dims = {2};
  RankedTensorType tensor_type(ElementType::Float32, Layout(dims));

  auto cpu_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 8);
  auto accel_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 8);
  cpu_buffer->Write<float>(absl::MakeConstSpan(cpu_data));
  accel_buffer->Write<float>(absl::MakeConstSpan(accel_data));

  AccuracyThresholds thresholds;
  auto res = internal::CompareBuffers(cpu_buffer.Value(), accel_buffer.Value(),
                                      thresholds, "test");
  ASSERT_TRUE(res.HasValue());
  EXPECT_NEAR(res->cosine_similarity, 0.0, 1e-9);
}

TEST(AccuracyDebuggerUtilTest, CompareBuffersOpposite) {
  std::vector<float> cpu_data = {1.0f, 2.0f};
  std::vector<float> accel_data = {-1.0f, -2.0f};
  Dimensions dims = {2};
  RankedTensorType tensor_type(ElementType::Float32, Layout(dims));

  auto cpu_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 8);
  auto accel_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 8);
  cpu_buffer->Write<float>(absl::MakeConstSpan(cpu_data));
  accel_buffer->Write<float>(absl::MakeConstSpan(accel_data));

  AccuracyThresholds thresholds;
  auto res = internal::CompareBuffers(cpu_buffer.Value(), accel_buffer.Value(),
                                      thresholds, "test");
  ASSERT_TRUE(res.HasValue());
  EXPECT_NEAR(res->cosine_similarity, -1.0, 1e-9);
}

TEST(AccuracyDebuggerUtilTest, CompareBuffersPearsonCorrelation) {
  std::vector<float> cpu_data = {1.0f, 2.0f, 3.0f};
  std::vector<float> accel_data = {2.0f, 4.0f, 6.0f};
  Dimensions dims = {3};
  RankedTensorType tensor_type(ElementType::Float32, Layout(dims));

  auto cpu_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 12);
  auto accel_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 12);
  cpu_buffer->Write<float>(absl::MakeConstSpan(cpu_data));
  accel_buffer->Write<float>(absl::MakeConstSpan(accel_data));

  AccuracyThresholds thresholds;
  auto res = internal::CompareBuffers(cpu_buffer.Value(), accel_buffer.Value(),
                                      thresholds, "test");
  ASSERT_TRUE(res.HasValue());
  EXPECT_NEAR(res->pearson_correlation, 1.0, 1e-7);
}

TEST(AccuracyDebuggerUtilTest, CompareBuffersPearsonCorrelationNegative) {
  std::vector<float> cpu_data = {1.0f, 2.0f, 3.0f};
  std::vector<float> accel_data = {-1.0f, -2.0f, -3.0f};
  Dimensions dims = {3};
  RankedTensorType tensor_type(ElementType::Float32, Layout(dims));

  auto cpu_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 12);
  auto accel_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 12);
  cpu_buffer->Write<float>(absl::MakeConstSpan(cpu_data));
  accel_buffer->Write<float>(absl::MakeConstSpan(accel_data));

  AccuracyThresholds thresholds;
  auto res = internal::CompareBuffers(cpu_buffer.Value(), accel_buffer.Value(),
                                      thresholds, "test");
  ASSERT_TRUE(res.HasValue());
  EXPECT_NEAR(res->pearson_correlation, -1.0, 1e-7);
}

TEST(AccuracyDebuggerUtilTest, CompareBuffersThresholdsTrigger) {
  std::vector<float> cpu_data = {1.0f, 1.0f};
  std::vector<float> accel_data = {1.5f, 1.5f};
  Dimensions dims = {2};
  RankedTensorType tensor_type(ElementType::Float32, Layout(dims));

  auto cpu_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 8);
  auto accel_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 8);
  cpu_buffer->Write<float>(absl::MakeConstSpan(cpu_data));
  accel_buffer->Write<float>(absl::MakeConstSpan(accel_data));

  // Differences are large: MaxDiff=0.5, MSE=0.25, CosSim=1.0, SNR=6.02dB,
  // PSNR=6.02dB
  AccuracyThresholds thresholds;
  thresholds.max_diff = 0.1f;
  thresholds.mse = 0.1;
  thresholds.snr = 80.0;
  thresholds.psnr = 80.0;
  thresholds.cosine_similarity = 0.9999999;  // 1.0 will not fail.

  auto res = internal::CompareBuffers(cpu_buffer.Value(), accel_buffer.Value(),
                                      thresholds, "test");
  ASSERT_TRUE(res.HasValue());
  EXPECT_TRUE(res->failed);
  // MaxDiff, MSE, SNR, PSNR should fail. CosSim (1.0) should pass.
  EXPECT_EQ(res->failing_metrics.size(), 4);
}

TEST(AccuracyDebuggerUtilTest, CompareBuffersAllZero) {
  std::vector<float> cpu_data = {0.0f, 0.0f};
  std::vector<float> accel_data = {0.0f, 0.0f};
  Dimensions dims = {2};
  RankedTensorType tensor_type(ElementType::Float32, Layout(dims));

  auto cpu_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 8);
  auto accel_buffer = TensorBuffer::CreateManagedHostMemory(tensor_type, 8);
  cpu_buffer->Write<float>(absl::MakeConstSpan(cpu_data));
  accel_buffer->Write<float>(absl::MakeConstSpan(accel_data));

  AccuracyThresholds thresholds;
  auto res = internal::CompareBuffers(cpu_buffer.Value(), accel_buffer.Value(),
                                      thresholds, "test");
  ASSERT_TRUE(res.HasValue());
  EXPECT_FALSE(res->failed);
  EXPECT_NEAR(res->cosine_similarity, 1.0, 1e-9);
  EXPECT_NEAR(res->pearson_correlation, 1.0, 1e-9);
}

TEST(AccuracyDebuggerUtilTest, RunAccuracyDebuggerWithCustomInputs) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());

  // Load simple_add_op model.
  std::string model_path = testing::GetTestFilePath("simple_add_op.tflite");
  auto model_res = litert::internal::LoadModelFromFile(model_path);
  ASSERT_TRUE(model_res.HasValue());
  auto model = std::move(model_res.Value());

  const LiteRtSubgraphT* subgraph = model->MainSubgraph();
  ASSERT_NE(subgraph, nullptr);
  ASSERT_EQ(subgraph->NumInputs(), 2);
  ASSERT_EQ(subgraph->NumOutputs(), 1);

  // Create a unique directory for the inputs/outputs.
  auto test_dir_res = testing::UniqueTestDirectory::Create();
  ASSERT_TRUE(test_dir_res.HasValue());
  std::string test_dir = std::string(test_dir_res->Str());
  std::string input_dir = test_dir + "/inputs";
  std::string output_dir = test_dir + "/outputs";
  LITERT_ASSERT_OK(litert::internal::MkDir(input_dir));
  LITERT_ASSERT_OK(litert::internal::MkDir(output_dir));

  std::vector<size_t> input_sizes;
  std::vector<std::vector<float>> custom_inputs_data;
  for (size_t i = 0; i < subgraph->NumInputs(); ++i) {
    const LiteRtTensorT& input_tensor = subgraph->Input(i);
    std::string name = std::string(input_tensor.Name());
    auto ranked_type_res = input_tensor.Ranked();
    ASSERT_TRUE(ranked_type_res.HasValue());
    ASSERT_EQ(ranked_type_res->element_type, kLiteRtElementTypeFloat32);

    size_t num_elements = input_tensor.NumElements();
    ASSERT_GT(num_elements, 0);
    input_sizes.push_back(num_elements);

    std::vector<float> custom_data(num_elements);
    for (size_t j = 0; j < num_elements; ++j) {
      custom_data[j] = j * static_cast<float>(i + 1);
    }
    custom_inputs_data.push_back(custom_data);

    std::string input_file = input_dir + "/" + name + ".raw";
    std::ofstream f(input_file, std::ios::binary);
    ASSERT_TRUE(f.is_open());
    f.write(reinterpret_cast<const char*>(custom_data.data()),
            custom_data.size() * sizeof(float));
  }

  // Setup debugger options.
  AccuracyDebuggerOptions options;
  options.input_dir = input_dir;
  options.output_dir = output_dir;
  options.accelerator = "cpu";
  options.dump_tensors = true;

  // We use CPU as both the reference and targeted accelerator.
  auto accel_opts_res = litert::Options::Create();
  ASSERT_TRUE(accel_opts_res.HasValue());
  accel_opts_res->SetHardwareAccelerators(litert::HwAccelerators::kCpu);

  // Run the debugger.
  AccuracyDebuggerSummary summary;
  auto status =
      RunAccuracyDebugger(*env, *model, *accel_opts_res, options, &summary);
  ASSERT_TRUE(status.ok()) << status.message();

  // Verify that dumped CPU inputs match our custom inputs we wrote to the file
  // system.
  for (size_t i = 0; i < subgraph->NumInputs(); ++i) {
    const LiteRtTensorT& input_tensor = subgraph->Input(i);
    std::string name = std::string(input_tensor.Name());
    std::string dumped_path = absl::StrFormat(
        "%s/cpu/op_0000_tfl.add_in_%d_%s.raw", output_dir, i, name);
    EXPECT_TRUE(litert::internal::Exists(dumped_path));

    std::string input_path = input_dir + "/" + name + ".raw";
    EXPECT_TRUE(litert::internal::Exists(input_path));

    size_t size = input_sizes[i];
    std::vector<float> dumped_data(size);
    std::ifstream f_dumped(dumped_path, std::ios::binary);
    ASSERT_TRUE(f_dumped.is_open());
    f_dumped.read(reinterpret_cast<char*>(dumped_data.data()),
                  size * sizeof(float));

    std::vector<float> expected_input_data(size);
    std::ifstream f_input(input_path, std::ios::binary);
    ASSERT_TRUE(f_input.is_open());
    f_input.read(reinterpret_cast<char*>(expected_input_data.data()),
                 size * sizeof(float));

    for (size_t j = 0; j < size; ++j) {
      EXPECT_EQ(dumped_data[j], expected_input_data[j]);
    }
  }

  // Run in-memory CPU inference using CompiledModel to obtain expected outputs.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(*env, model_path, HwAccelerators::kCpu));

  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> compiled_input_buffers,
                              compiled_model.CreateInputBuffers());
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> compiled_output_buffers,
                              compiled_model.CreateOutputBuffers());

  ASSERT_EQ(compiled_input_buffers.size(), subgraph->NumInputs());
  for (size_t i = 0; i < subgraph->NumInputs(); ++i) {
    const LiteRtTensorT& input_tensor = subgraph->Input(i);
    std::string name = std::string(input_tensor.Name());
    std::string input_path = input_dir + "/" + name + ".raw";
    EXPECT_TRUE(litert::internal::Exists(input_path));

    size_t size = input_sizes[i];
    std::vector<float> expected_input_data(size);
    std::ifstream f_input(input_path, std::ios::binary);
    ASSERT_TRUE(f_input.is_open());
    f_input.read(reinterpret_cast<char*>(expected_input_data.data()),
                 size * sizeof(float));

    ASSERT_TRUE(compiled_input_buffers[i].Write<float>(
        absl::MakeConstSpan(expected_input_data)));
  }

  LITERT_ASSERT_OK(
      compiled_model.Run(compiled_input_buffers, compiled_output_buffers));

  ASSERT_EQ(compiled_output_buffers.size(), subgraph->NumOutputs());
  ASSERT_EQ(subgraph->NumOutputs(), 1);
  const LiteRtTensorT& output_tensor = subgraph->Output(0);
  std::string out_name = std::string(output_tensor.Name());
  auto out_ranked_type_res = output_tensor.Ranked();
  ASSERT_TRUE(out_ranked_type_res.HasValue());
  ASSERT_EQ(out_ranked_type_res->element_type, kLiteRtElementTypeFloat32);
  size_t out_size = output_tensor.NumElements();

  std::vector<float> expected_output_data(out_size);
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            compiled_output_buffers[0], TensorBuffer::LockMode::kRead));
    std::copy(lock_and_addr.second, lock_and_addr.second + out_size,
              expected_output_data.begin());
  }

  // Verify CPU outputs match expected in-memory inference outputs.
  std::string dumped_out_path = absl::StrFormat(
      "%s/cpu/op_0000_tfl.add_out_0_%s.raw", output_dir, out_name);
  EXPECT_TRUE(litert::internal::Exists(dumped_out_path));

  std::vector<float> out_dumped_data(out_size);
  {
    std::ifstream f_out(dumped_out_path, std::ios::binary);
    ASSERT_TRUE(f_out.is_open());
    f_out.read(reinterpret_cast<char*>(out_dumped_data.data()),
               out_size * sizeof(float));
  }
  for (size_t i = 0; i < out_size; ++i) {
    EXPECT_EQ(out_dumped_data[i], expected_output_data[i]);
  }
}

}  // namespace
}  // namespace litert::tools

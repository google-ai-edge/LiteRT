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

#include "litert/tools/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"

namespace litert {
namespace tensor_utils {
namespace {

using ::testing::ElementsAre;
using ::testing::litert::IsError;

TEST(TensorUtilsTest, QuantizeDataInt8) {
  std::vector<float> input = {0.0f, 0.5f, 1.0f, -0.5f, -1.0f, 100.0f, -100.0f};
  float scale = 0.5f;
  int64_t zero_point = 0;

  auto result =
      QuantizeData<int8_t>(absl::MakeConstSpan(input), scale, zero_point);
  EXPECT_THAT(result, ElementsAre(0, 1, 2, -1, -2, 127, -128));
}

TEST(TensorUtilsTest, QuantizeDataInt8WithZeroPoint) {
  std::vector<float> input = {0.0f, 0.1f, 0.2f, -0.1f, -0.2f};
  float scale = 0.1f;
  int64_t zero_point = 10;

  auto result =
      QuantizeData<int8_t>(absl::MakeConstSpan(input), scale, zero_point);
  EXPECT_THAT(result, ElementsAre(10, 11, 12, 9, 8));
}

TEST(TensorUtilsTest, QuantizeDataRounding) {
  // Test round half away from zero with exact half-integers: 2.5 -> 3, -2.5 ->
  // -3
  std::vector<float> input = {1.25f, -1.25f, 1.75f, -1.75f};
  float scale = 0.5f;
  int64_t zero_point = 0;

  auto result =
      QuantizeData<int8_t>(absl::MakeConstSpan(input), scale, zero_point);
  EXPECT_THAT(result, ElementsAre(3, -3, 4, -4));
}

TEST(TensorUtilsTest, QuantizeDataUInt8) {
  std::vector<float> input = {0.0f, 1.0f, 2.0f, -1.0f, -2.0f, 50.0f};
  float scale = 0.1f;
  int64_t zero_point = 128;

  auto result =
      QuantizeData<uint8_t>(absl::MakeConstSpan(input), scale, zero_point);
  EXPECT_THAT(result, ElementsAre(128, 138, 148, 118, 108, 255));
}

TEST(TensorUtilsTest, QuantizeDataInt16) {
  std::vector<float> input = {0.0f, 10.0f, -10.0f, 40000.0f, -40000.0f};
  float scale = 1.0f;
  int64_t zero_point = 0;

  auto result =
      QuantizeData<int16_t>(absl::MakeConstSpan(input), scale, zero_point);
  EXPECT_THAT(result, ElementsAre(0, 10, -10, 32767, -32768));
}

TEST(TensorUtilsTest, QuantizeDataUInt16) {
  std::vector<float> input = {0.0f, 10.0f, -10.0f, 70000.0f};
  float scale = 1.0f;
  int64_t zero_point = 100;

  auto result =
      QuantizeData<uint16_t>(absl::MakeConstSpan(input), scale, zero_point);
  EXPECT_THAT(result, ElementsAre(100, 110, 90, 65535));
}

TEST(TensorUtilsTest, QuantizeDataInt32) {
  std::vector<float> input = {0.0f, 1000.0f, -1000.0f};
  float scale = 0.01f;
  int64_t zero_point = 0;

  auto result =
      QuantizeData<int32_t>(absl::MakeConstSpan(input), scale, zero_point);
  EXPECT_THAT(result, ElementsAre(0, 100000, -100000));
}

TEST(TensorUtilsTest, QuantizeDataInvalidScale) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  int64_t zero_point = 42;

  // Zero scale
  auto result_zero =
      QuantizeData<int8_t>(absl::MakeConstSpan(input), 0.0f, zero_point);
  EXPECT_THAT(result_zero, ElementsAre(42, 42, 42));

  // Negative scale
  auto result_neg =
      QuantizeData<int8_t>(absl::MakeConstSpan(input), -0.5f, zero_point);
  EXPECT_THAT(result_neg, ElementsAre(42, 42, 42));

  // NaN scale
  auto result_nan =
      QuantizeData<int8_t>(absl::MakeConstSpan(input),
                           std::numeric_limits<float>::quiet_NaN(), zero_point);
  EXPECT_THAT(result_nan, ElementsAre(42, 42, 42));

  // Inf scale
  auto result_inf =
      QuantizeData<int8_t>(absl::MakeConstSpan(input),
                           std::numeric_limits<float>::infinity(), zero_point);
  EXPECT_THAT(result_inf, ElementsAre(42, 42, 42));
}

TEST(TensorUtilsTest, QuantizeDataSpecialFloatValues) {
  std::vector<float> input = {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      1e35f,
      -1e35f,
  };
  float scale = 0.5f;
  int64_t zero_point = 10;

  auto result =
      QuantizeData<int8_t>(absl::MakeConstSpan(input), scale, zero_point);
  EXPECT_THAT(result, ElementsAre(10, 127, -128, 127, -128));
}

TEST(TensorUtilsTest, QuantizeDataOutOfRangeZeroPoint) {
  std::vector<float> input = {0.0f, 10.0f};
  float scale = 1.0f;

  // zero_point exceeds uint8_t max (255)
  auto result_uint8 =
      QuantizeData<uint8_t>(absl::MakeConstSpan(input), scale, 500);
  EXPECT_THAT(result_uint8, ElementsAre(255, 255));

  // zero_point below int8_t min (-128)
  auto result_int8 =
      QuantizeData<int8_t>(absl::MakeConstSpan(input), scale, -500);
  EXPECT_THAT(result_int8, ElementsAre(-128, -128));
}

TEST(TensorUtilsTest, QuantizeDataEmptySpan) {
  std::vector<float> empty_input;
  auto result = QuantizeData<int8_t>(absl::MakeConstSpan(empty_input), 1.0f, 0);
  EXPECT_TRUE(result.empty());
}

TEST(TensorUtilsTest, ReadTensorDataFromRawFileValid) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto temp_dir,
                              testing::UniqueTestDirectory::Create());
  std::string file_path =
      (std::filesystem::path(std::string(temp_dir.Str())) / "test_tensor.raw")
          .string();

  std::vector<float> expected_data = {1.0f, 2.0f, 3.0f, 4.0f};
  {
    std::ofstream ofs(file_path, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(expected_data.data()),
              expected_data.size() * sizeof(float));
  }

  LITERT_ASSERT_OK_AND_ASSIGN(auto read_bytes,
                              ReadTensorDataFromRawFile(file_path));
  ASSERT_EQ(read_bytes.size(), expected_data.size() * sizeof(float));

  const float* read_floats = reinterpret_cast<const float*>(read_bytes.data());
  for (size_t i = 0; i < expected_data.size(); ++i) {
    EXPECT_EQ(read_floats[i], expected_data[i]);
  }
}

TEST(TensorUtilsTest, ReadTensorDataFromRawFileNonExistent) {
  auto result = ReadTensorDataFromRawFile("/path/that/does/not/exist/foo.raw");
  EXPECT_THAT(result, IsError(kLiteRtStatusErrorNotFound));
}

TEST(TensorUtilsTest, ReadTensorDataFromRawFileDirectoryPath) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto temp_dir,
                              testing::UniqueTestDirectory::Create());
  auto result = ReadTensorDataFromRawFile(temp_dir.Str());
  EXPECT_THAT(result, IsError(kLiteRtStatusErrorNotFound));
}

TEST(TensorUtilsTest, FillBufferWithCustomDataMismatchedSize) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  RankedTensorType tensor_type(ElementType::Float32, Layout(Dimensions{2, 2}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  tensor_type, 4 * sizeof(float)));

  std::vector<char> small_data(
      2);  // 2 bytes instead of 16 bytes (2*2*sizeof(float))
  EXPECT_THAT(FillBufferWithCustomData(buffer, small_data),
              IsError(kLiteRtStatusErrorRuntimeFailure));
}

TEST(TensorUtilsTest, FillBufferWithCustomDataSuccess) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  RankedTensorType tensor_type(ElementType::Float32, Layout(Dimensions{2, 2}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  tensor_type, 4 * sizeof(float)));

  std::vector<float> expected_floats = {1.5f, 2.5f, 3.5f, 4.5f};
  std::vector<char> data(expected_floats.size() * sizeof(float));
  std::memcpy(data.data(), expected_floats.data(), data.size());

  LITERT_EXPECT_OK(FillBufferWithCustomData(buffer, data));

  std::vector<float> read_floats(4);
  LITERT_EXPECT_OK(buffer.Read<float>(absl::MakeSpan(read_floats)));
  EXPECT_THAT(read_floats, ElementsAre(1.5f, 2.5f, 3.5f, 4.5f));
}

TEST(TensorUtilsTest, FillInputBuffersWithCustomDataQuantizeInputs) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(
          env, testing::GetTestFilePath("simple_quantized_ops.tflite"),
          HwAccelerators::kCpu));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_names,
      compiled_model.GetSignatureInputNames(/*signature_index=*/0));
  ASSERT_FALSE(input_names.empty());

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers,
      compiled_model.CreateInputBuffers(/*signature_index=*/0));
  ASSERT_EQ(input_buffers.size(), input_names.size());

  LITERT_ASSERT_OK_AND_ASSIGN(auto temp_dir,
                              testing::UniqueTestDirectory::Create());

  // Create FP32 raw input files for all inputs
  for (size_t i = 0; i < input_names.size(); ++i) {
    const auto& name = input_names[i];
    LITERT_ASSERT_OK_AND_ASSIGN(auto cur_type, input_buffers[i].TensorType());
    size_t cur_elements = std::accumulate(
        cur_type.Layout().Dimensions().begin(),
        cur_type.Layout().Dimensions().end(), 1, std::multiplies<size_t>());
    std::string file_path =
        (std::filesystem::path(std::string(temp_dir.Str())) /
         (std::string(name.data(), name.size()) + ".raw"))
            .string();
    std::vector<float> fp32_data(cur_elements, 1.0f);
    std::ofstream ofs(file_path, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(fp32_data.data()),
              fp32_data.size() * sizeof(float));
  }

  // Auto-quantize inputs
  LITERT_EXPECT_OK(FillInputBuffersWithCustomData(
      compiled_model, /*signature_index=*/0, input_buffers, temp_dir.Str(),
      /*quantize_inputs=*/true));

  // Verify that input buffer 0 contains correctly quantized data
  LITERT_ASSERT_OK_AND_ASSIGN(auto t0_type, input_buffers[0].TensorType());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto q_params, compiled_model.GetInputTensorPerTensorQuantization(
                         /*signature_index=*/0, input_names[0]));
  EXPECT_GT(q_params.scale, 0.0f);

  size_t t0_elements = std::accumulate(t0_type.Layout().Dimensions().begin(),
                                       t0_type.Layout().Dimensions().end(), 1,
                                       std::multiplies<size_t>());

  if (t0_type.ElementType() == ElementType::Int8) {
    std::vector<int8_t> quantized_result(t0_elements);
    LITERT_EXPECT_OK(
        input_buffers[0].Read<int8_t>(absl::MakeSpan(quantized_result)));
    int8_t expected_quantized_val = static_cast<int8_t>(
        std::clamp(static_cast<int64_t>(std::round(1.0f / q_params.scale)) +
                       q_params.zero_point,
                   static_cast<int64_t>(-128), static_cast<int64_t>(127)));
    EXPECT_THAT(quantized_result, ::testing::Each(expected_quantized_val));
  } else if (t0_type.ElementType() == ElementType::Int16) {
    std::vector<int16_t> quantized_result(t0_elements);
    LITERT_EXPECT_OK(
        input_buffers[0].Read<int16_t>(absl::MakeSpan(quantized_result)));
    int16_t expected_quantized_val = static_cast<int16_t>(
        std::clamp(static_cast<int64_t>(std::round(1.0f / q_params.scale)) +
                       q_params.zero_point,
                   static_cast<int64_t>(-32768), static_cast<int64_t>(32767)));
    EXPECT_THAT(quantized_result, ::testing::Each(expected_quantized_val));
  } else if (t0_type.ElementType() == ElementType::UInt8) {
    std::vector<uint8_t> quantized_result(t0_elements);
    LITERT_EXPECT_OK(
        input_buffers[0].Read<uint8_t>(absl::MakeSpan(quantized_result)));
    uint8_t expected_quantized_val = static_cast<uint8_t>(
        std::clamp(static_cast<int64_t>(std::round(1.0f / q_params.scale)) +
                       q_params.zero_point,
                   static_cast<int64_t>(0), static_cast<int64_t>(255)));
    EXPECT_THAT(quantized_result, ::testing::Each(expected_quantized_val));
  }
}

TEST(TensorUtilsTest, FillInputBuffersWithCustomDataRawQuantizedBytes) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(
          env, testing::GetTestFilePath("simple_quantized_ops.tflite"),
          HwAccelerators::kCpu));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_names,
      compiled_model.GetSignatureInputNames(/*signature_index=*/0));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers,
      compiled_model.CreateInputBuffers(/*signature_index=*/0));

  LITERT_ASSERT_OK_AND_ASSIGN(auto temp_dir,
                              testing::UniqueTestDirectory::Create());

  // Create raw input files matching buffer sizes (already quantized)
  for (size_t i = 0; i < input_names.size(); ++i) {
    const auto& name = input_names[i];
    LITERT_ASSERT_OK_AND_ASSIGN(auto cur_size, input_buffers[i].Size());
    std::string file_path =
        (std::filesystem::path(std::string(temp_dir.Str())) /
         (std::string(name.data(), name.size()) + ".raw"))
            .string();
    std::vector<char> raw_data(cur_size, 42);
    std::ofstream ofs(file_path, std::ios::binary);
    ofs.write(raw_data.data(), raw_data.size());
  }

  // Fill with quantize_inputs=false
  LITERT_EXPECT_OK(FillInputBuffersWithCustomData(
      compiled_model, /*signature_index=*/0, input_buffers, temp_dir.Str(),
      /*quantize_inputs=*/false));

  LITERT_ASSERT_OK_AND_ASSIGN(auto t0_type, input_buffers[0].TensorType());
  LITERT_ASSERT_OK_AND_ASSIGN(auto t0_size, input_buffers[0].Size());
  if (t0_type.ElementType() == ElementType::Int8 ||
      t0_type.ElementType() == ElementType::UInt8) {
    std::vector<uint8_t> read_result(t0_size);
    LITERT_EXPECT_OK(
        input_buffers[0].Read<uint8_t>(absl::MakeSpan(read_result)));
    EXPECT_THAT(read_result, ::testing::Each(42));
  } else if (t0_type.ElementType() == ElementType::Int16 ||
             t0_type.ElementType() == ElementType::UInt16) {
    std::vector<uint16_t> read_result(t0_size / sizeof(uint16_t));
    LITERT_EXPECT_OK(
        input_buffers[0].Read<uint16_t>(absl::MakeSpan(read_result)));
    uint16_t expected_val = static_cast<uint16_t>((42 << 8) | 42);
    EXPECT_THAT(read_result, ::testing::Each(expected_val));
  }

  // Also test with quantize_inputs=true (since size != FP32 size, it falls back
  // to raw fill)
  LITERT_EXPECT_OK(FillInputBuffersWithCustomData(
      compiled_model, /*signature_index=*/0, input_buffers, temp_dir.Str(),
      /*quantize_inputs=*/true));

  if (t0_type.ElementType() == ElementType::Int8 ||
      t0_type.ElementType() == ElementType::UInt8) {
    std::vector<uint8_t> read_result(t0_size);
    LITERT_EXPECT_OK(
        input_buffers[0].Read<uint8_t>(absl::MakeSpan(read_result)));
    EXPECT_THAT(read_result, ::testing::Each(42));
  } else if (t0_type.ElementType() == ElementType::Int16 ||
             t0_type.ElementType() == ElementType::UInt16) {
    std::vector<uint16_t> read_result(t0_size / sizeof(uint16_t));
    LITERT_EXPECT_OK(
        input_buffers[0].Read<uint16_t>(absl::MakeSpan(read_result)));
    uint16_t expected_val = static_cast<uint16_t>((42 << 8) | 42);
    EXPECT_THAT(read_result, ::testing::Each(expected_val));
  }
}

TEST(TensorUtilsTest, FillInputBuffersWithCustomDataMismatchedSize) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(
          env, testing::GetTestFilePath("simple_quantized_ops.tflite"),
          HwAccelerators::kCpu));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_names,
      compiled_model.GetSignatureInputNames(/*signature_index=*/0));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers,
      compiled_model.CreateInputBuffers(/*signature_index=*/0));

  LITERT_ASSERT_OK_AND_ASSIGN(auto temp_dir,
                              testing::UniqueTestDirectory::Create());

  for (const auto& name : input_names) {
    std::string file_path =
        (std::filesystem::path(std::string(temp_dir.Str())) /
         (std::string(name.data(), name.size()) + ".raw"))
            .string();
    std::vector<char> corrupt_data(
        5, 0);  // Neither float32 size nor int8 buffer size
    std::ofstream ofs(file_path, std::ios::binary);
    ofs.write(corrupt_data.data(), corrupt_data.size());
  }

  EXPECT_THAT(FillInputBuffersWithCustomData(
                  compiled_model, /*signature_index=*/0, input_buffers,
                  temp_dir.Str(), /*quantize_inputs=*/true),
              IsError(kLiteRtStatusErrorRuntimeFailure));
}

TEST(TensorUtilsTest, FillInputBuffersWithCustomDataMissingFile) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(
          env, testing::GetTestFilePath("simple_quantized_ops.tflite"),
          HwAccelerators::kCpu));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers,
      compiled_model.CreateInputBuffers(/*signature_index=*/0));

  LITERT_ASSERT_OK_AND_ASSIGN(auto temp_dir,
                              testing::UniqueTestDirectory::Create());

  EXPECT_THAT(FillInputBuffersWithCustomData(
                  compiled_model, /*signature_index=*/0, input_buffers,
                  temp_dir.Str(), /*quantize_inputs=*/true),
              IsError(kLiteRtStatusErrorNotFound));
}

TEST(TensorUtilsTest, FillInputBuffersWithCustomDataMismatchedBufferCount) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(
          env, testing::GetTestFilePath("simple_quantized_ops.tflite"),
          HwAccelerators::kCpu));

  std::vector<TensorBuffer> empty_buffers;
  LITERT_ASSERT_OK_AND_ASSIGN(auto temp_dir,
                              testing::UniqueTestDirectory::Create());

  EXPECT_THAT(FillInputBuffersWithCustomData(
                  compiled_model, /*signature_index=*/0, empty_buffers,
                  temp_dir.Str(), /*quantize_inputs=*/true),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(TensorUtilsTest, WriteOutputBuffersToFilesSuccess) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(
          env, testing::GetTestFilePath("simple_quantized_ops.tflite"),
          HwAccelerators::kCpu));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_names,
      compiled_model.GetSignatureOutputNames(/*signature_index=*/0));
  ASSERT_FALSE(output_names.empty());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers,
      compiled_model.CreateOutputBuffers(/*signature_index=*/0));
  ASSERT_EQ(output_buffers.size(), output_names.size());

  // Fill each output buffer with a distinct known byte pattern.
  std::vector<std::vector<char>> expected_bytes(output_buffers.size());
  for (size_t i = 0; i < output_buffers.size(); ++i) {
    LITERT_ASSERT_OK_AND_ASSIGN(size_t size, output_buffers[i].Size());
    expected_bytes[i].assign(size, static_cast<char>(i + 1));
    LITERT_EXPECT_OK(
        FillBufferWithCustomData(output_buffers[i], expected_bytes[i]));
  }

  LITERT_ASSERT_OK_AND_ASSIGN(auto temp_dir,
                              testing::UniqueTestDirectory::Create());
  LITERT_EXPECT_OK(WriteOutputBuffersToFiles(
      compiled_model, /*signature_index=*/0, output_buffers, temp_dir.Str()));

  // Every output must ba a "<name>.raw" file with the exact bytes.
  for (size_t i = 0; i < output_names.size(); ++i) {
    const auto& name = output_names[i];
    std::string file_path =
        (std::filesystem::path(std::string(temp_dir.Str())) /
         (std::string(name) + ".raw"))
            .string();
    LITERT_ASSERT_OK_AND_ASSIGN(auto read_bytes,
                                ReadTensorDataFromRawFile(file_path));
    EXPECT_EQ(read_bytes, expected_bytes[i]);
  }
}

TEST(TensorUtilsTest, WriteOutputBuffersToFilesUnwritableDir) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(
          env, testing::GetTestFilePath("simple_quantized_ops.tflite"),
          HwAccelerators::kCpu));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers,
      compiled_model.CreateOutputBuffers(/*signature_index=*/0));

  // A non-existent directory makes the output file fail to open.
  EXPECT_THAT(
      WriteOutputBuffersToFiles(compiled_model, /*signature_index=*/0,
                                output_buffers, "/path/that/does/not/exist"),
      IsError(kLiteRtStatusErrorRuntimeFailure));
}

}  // namespace
}  // namespace tensor_utils
}  // namespace litert

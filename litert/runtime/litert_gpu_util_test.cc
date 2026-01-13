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

#include "litert/runtime/litert_gpu_util.h"

#include <gtest/gtest.h>
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/test/matchers.h"
#include "tflite/delegates/gpu/common/data_type.h"

namespace litert::internal {
namespace {

using tflite::gpu::DataType;

TEST(ConvertLiteRtDataTypeToGpuDataTypeTest, Float32) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.element_type = kLiteRtElementTypeFloat32;
  DataType data_type;

  LITERT_ASSERT_OK(ConvertLiteRtDataTypeToGpuDataType(
      &tensor_type, &data_type, kLiteRtTensorBufferTypeAhwb));
  EXPECT_EQ(data_type, DataType::FLOAT32);
}

class Float32AsFloat16Test
    : public ::testing::TestWithParam<LiteRtTensorBufferType> {};

TEST_P(Float32AsFloat16Test, ConvertsToFloat16) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.element_type = kLiteRtElementTypeFloat32;
  DataType data_type;

  LITERT_ASSERT_OK(ConvertLiteRtDataTypeToGpuDataType(
      &tensor_type, &data_type, GetParam()));
  EXPECT_EQ(data_type, DataType::FLOAT16);
}

INSTANTIATE_TEST_SUITE_P(
    Float32AsFloat16Tests, Float32AsFloat16Test,
    ::testing::Values(kLiteRtTensorBufferTypeMetalBufferFp16,
                      kLiteRtTensorBufferTypeMetalTextureFp16,
                      kLiteRtTensorBufferTypeOpenClBufferFp16,
                      kLiteRtTensorBufferTypeOpenClTextureFp16,
                      kLiteRtTensorBufferTypeOpenClImageBufferFp16));

TEST(ConvertLiteRtDataTypeToGpuDataTypeTest, Float16) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.element_type = kLiteRtElementTypeFloat16;
  DataType data_type;

  LITERT_ASSERT_OK(ConvertLiteRtDataTypeToGpuDataType(
      &tensor_type, &data_type, kLiteRtTensorBufferTypeAhwb));
  EXPECT_EQ(data_type, DataType::FLOAT16);
}

TEST(ConvertLiteRtDataTypeToGpuDataTypeTest, Int32) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.element_type = kLiteRtElementTypeInt32;
  DataType data_type;

  LITERT_ASSERT_OK(ConvertLiteRtDataTypeToGpuDataType(
      &tensor_type, &data_type, kLiteRtTensorBufferTypeAhwb));
  EXPECT_EQ(data_type, DataType::INT32);
}

TEST(ConvertLiteRtDataTypeToGpuDataTypeTest, Bool) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.element_type = kLiteRtElementTypeBool;
  DataType data_type;

  LITERT_ASSERT_OK(ConvertLiteRtDataTypeToGpuDataType(
      &tensor_type, &data_type, kLiteRtTensorBufferTypeAhwb));
  EXPECT_EQ(data_type, DataType::BOOL);
}

TEST(ConvertLiteRtDataTypeToGpuDataTypeTest, UnsupportedType) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.element_type = kLiteRtElementTypeInt64;
  DataType data_type;

  EXPECT_FALSE(ConvertLiteRtDataTypeToGpuDataType(
      &tensor_type, &data_type, kLiteRtTensorBufferTypeAhwb).ok());
}

}  // namespace
}  // namespace litert::internal

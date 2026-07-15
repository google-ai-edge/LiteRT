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

#include "ml_drift_delegate/delegate/buffer_handler_utils.h"

#include <cstdint>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"

namespace litert::ml_drift {
namespace {

using ::testing::Eq;

TEST(BufferHandlerUtilsTest, CreateTensorDescriptorRank2Float32) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 2;
  tensor_type.layout.dimensions[0] = 3;
  tensor_type.layout.dimensions[1] = 4;
  tensor_type.element_type = kLiteRtElementTypeFloat32;

  auto desc = CreateTensorDescriptor(
      tensor_type, kLiteRtTensorBufferTypeOpenClImageBuffer);
  ASSERT_TRUE(desc.ok());
  EXPECT_THAT(desc->GetDataType(), Eq(::ml_drift::DataType::FLOAT32));
  EXPECT_THAT(desc->GetStorageType(),
              Eq(::ml_drift::TensorStorageType::IMAGE_BUFFER));
  EXPECT_THAT(desc->GetBHWDCShape().b, Eq(3));
  EXPECT_THAT(desc->GetBHWDCShape().h, Eq(1));
  EXPECT_THAT(desc->GetBHWDCShape().w, Eq(1));
  EXPECT_THAT(desc->GetBHWDCShape().c, Eq(4));
}

TEST(BufferHandlerUtilsTest, CreateTensorDescriptorRank4Float16) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 4;
  tensor_type.layout.dimensions[0] = 1;
  tensor_type.layout.dimensions[1] = 2;
  tensor_type.layout.dimensions[2] = 3;
  tensor_type.layout.dimensions[3] = 4;
  tensor_type.element_type = kLiteRtElementTypeFloat32;

  auto desc = CreateTensorDescriptor(
      tensor_type, kLiteRtTensorBufferTypeMetalBufferFp16);
  ASSERT_TRUE(desc.ok());
  EXPECT_THAT(desc->GetDataType(), Eq(::ml_drift::DataType::FLOAT16));
  EXPECT_THAT(desc->GetStorageType(),
              Eq(::ml_drift::TensorStorageType::BUFFER));
}

TEST(BufferHandlerUtilsTest, ConvertDataToDescriptorFloat) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 2;
  tensor_type.layout.dimensions[0] = 2;
  tensor_type.layout.dimensions[1] = 4;
  tensor_type.element_type = kLiteRtElementTypeFloat32;

  auto desc = CreateTensorDescriptor(
      tensor_type, kLiteRtTensorBufferTypeMetalBuffer);
  ASSERT_TRUE(desc.ok());

  std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                  5.0f, 6.0f, 7.0f, 8.0f};
  ConvertDataToDescriptor(host_data.data(), *desc, kLiteRtElementTypeFloat32);

  auto result_data = desc->GetData();
  ASSERT_EQ(result_data.size(), 8 * sizeof(float));
  const float* float_data = reinterpret_cast<const float*>(result_data.data());
  EXPECT_THAT(float_data[0], Eq(1.0f));
  EXPECT_THAT(float_data[7], Eq(8.0f));
}

TEST(BufferHandlerUtilsTest, ConvertDataFromDescriptorFloat) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 2;
  tensor_type.layout.dimensions[0] = 2;
  tensor_type.layout.dimensions[1] = 4;
  tensor_type.element_type = kLiteRtElementTypeFloat32;

  auto desc = CreateTensorDescriptor(
      tensor_type, kLiteRtTensorBufferTypeMetalBuffer);
  ASSERT_TRUE(desc.ok());

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                   5.0f, 6.0f, 7.0f, 8.0f};
  desc->UploadData(input_data.data());

  std::vector<float> host_data(8);
  ConvertDataFromDescriptor(*desc, host_data.data(), kLiteRtElementTypeFloat32);

  EXPECT_THAT(host_data[0], Eq(1.0f));
  EXPECT_THAT(host_data[7], Eq(8.0f));
}

TEST(BufferHandlerUtilsTest, ConvertDataToDescriptorInt8) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 2;
  tensor_type.layout.dimensions[0] = 2;
  tensor_type.layout.dimensions[1] = 4;
  tensor_type.element_type = kLiteRtElementTypeInt8;

  auto desc = CreateTensorDescriptor(
      tensor_type, kLiteRtTensorBufferTypeMetalBuffer);
  ASSERT_TRUE(desc.ok());

  std::vector<int8_t> host_data = {1, 2, 3, 4, 5, 6, 7, 8};
  ConvertDataToDescriptor(host_data.data(), *desc, kLiteRtElementTypeInt8);

  auto result_data = desc->GetData();
  ASSERT_EQ(result_data.size(), 8 * sizeof(int8_t));
  const int8_t* int8_data = reinterpret_cast<const int8_t*>(result_data.data());
  EXPECT_THAT(int8_data[0], Eq(1));
  EXPECT_THAT(int8_data[7], Eq(8));
}

TEST(BufferHandlerUtilsTest, ConvertDataFromDescriptorInt8) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 2;
  tensor_type.layout.dimensions[0] = 2;
  tensor_type.layout.dimensions[1] = 4;
  tensor_type.element_type = kLiteRtElementTypeInt8;

  auto desc = CreateTensorDescriptor(
      tensor_type, kLiteRtTensorBufferTypeMetalBuffer);
  ASSERT_TRUE(desc.ok());

  std::vector<int8_t> input_data = {1, 2, 3, 4, 5, 6, 7, 8};
  desc->UploadData(input_data.data());

  std::vector<int8_t> host_data(8);
  ConvertDataFromDescriptor(*desc, host_data.data(), kLiteRtElementTypeInt8);

  EXPECT_THAT(host_data[0], Eq(1));
  EXPECT_THAT(host_data[7], Eq(8));
}

TEST(BufferHandlerUtilsTest, ConvertDataToDescriptorUInt8) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 2;
  tensor_type.layout.dimensions[0] = 2;
  tensor_type.layout.dimensions[1] = 4;
  tensor_type.element_type = kLiteRtElementTypeUInt8;

  auto desc =
      CreateTensorDescriptor(tensor_type, kLiteRtTensorBufferTypeMetalBuffer);
  ASSERT_TRUE(desc.ok());

  std::vector<uint8_t> host_data = {1, 2, 3, 4, 5, 6, 7, 8};
  ConvertDataToDescriptor(host_data.data(), *desc, kLiteRtElementTypeUInt8);

  auto result_data = desc->GetData();
  ASSERT_EQ(result_data.size(), 8 * sizeof(uint8_t));
  const uint8_t* uint8_data =
      reinterpret_cast<const uint8_t*>(result_data.data());
  EXPECT_THAT(uint8_data[0], Eq(1));
  EXPECT_THAT(uint8_data[7], Eq(8));
}

TEST(BufferHandlerUtilsTest, ConvertDataFromDescriptorUInt8) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 2;
  tensor_type.layout.dimensions[0] = 2;
  tensor_type.layout.dimensions[1] = 4;
  tensor_type.element_type = kLiteRtElementTypeUInt8;

  auto desc =
      CreateTensorDescriptor(tensor_type, kLiteRtTensorBufferTypeMetalBuffer);
  ASSERT_TRUE(desc.ok());

  std::vector<uint8_t> input_data = {1, 2, 3, 4, 5, 6, 7, 8};
  desc->UploadData(input_data.data());

  std::vector<uint8_t> host_data(8);
  ConvertDataFromDescriptor(*desc, host_data.data(), kLiteRtElementTypeUInt8);

  EXPECT_THAT(host_data[0], Eq(1));
  EXPECT_THAT(host_data[7], Eq(8));
}

TEST(BufferHandlerUtilsTest, ConvertDataToDescriptorUInt32) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 2;
  tensor_type.layout.dimensions[0] = 2;
  tensor_type.layout.dimensions[1] = 4;
  tensor_type.element_type = kLiteRtElementTypeUInt32;

  auto desc =
      CreateTensorDescriptor(tensor_type, kLiteRtTensorBufferTypeMetalBuffer);
  ASSERT_TRUE(desc.ok());

  std::vector<uint32_t> host_data = {1, 2, 3, 4, 5, 6, 7, 8};
  ConvertDataToDescriptor(host_data.data(), *desc, kLiteRtElementTypeUInt32);

  auto result_data = desc->GetData();
  ASSERT_EQ(result_data.size(), 8 * sizeof(uint32_t));
  const uint32_t* uint32_data =
      reinterpret_cast<const uint32_t*>(result_data.data());
  EXPECT_THAT(uint32_data[0], Eq(1));
  EXPECT_THAT(uint32_data[7], Eq(8));
}

TEST(BufferHandlerUtilsTest, ConvertDataFromDescriptorUInt32) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 2;
  tensor_type.layout.dimensions[0] = 2;
  tensor_type.layout.dimensions[1] = 4;
  tensor_type.element_type = kLiteRtElementTypeUInt32;

  auto desc =
      CreateTensorDescriptor(tensor_type, kLiteRtTensorBufferTypeMetalBuffer);
  ASSERT_TRUE(desc.ok());

  std::vector<uint32_t> input_data = {1, 2, 3, 4, 5, 6, 7, 8};
  desc->UploadData(input_data.data());

  std::vector<uint32_t> host_data(8);
  ConvertDataFromDescriptor(*desc, host_data.data(), kLiteRtElementTypeUInt32);

  EXPECT_THAT(host_data[0], Eq(1));
  EXPECT_THAT(host_data[7], Eq(8));
}

TEST(BufferHandlerUtilsTest, ConvertDataToDescriptorInt64) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 2;
  tensor_type.layout.dimensions[0] = 2;
  tensor_type.layout.dimensions[1] = 4;
  tensor_type.element_type = kLiteRtElementTypeInt64;

  auto desc =
      CreateTensorDescriptor(tensor_type, kLiteRtTensorBufferTypeMetalBuffer);
  ASSERT_TRUE(desc.ok());

  std::vector<int64_t> host_data = {1, 2, 3, 4, 5, 6, 7, 8};
  ConvertDataToDescriptor(host_data.data(), *desc, kLiteRtElementTypeInt64);

  auto result_data = desc->GetData();
  ASSERT_EQ(result_data.size(), 8 * sizeof(int64_t));
  const int64_t* int64_data =
      reinterpret_cast<const int64_t*>(result_data.data());
  EXPECT_THAT(int64_data[0], Eq(1));
  EXPECT_THAT(int64_data[7], Eq(8));
}

TEST(BufferHandlerUtilsTest, ConvertDataFromDescriptorInt64) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 2;
  tensor_type.layout.dimensions[0] = 2;
  tensor_type.layout.dimensions[1] = 4;
  tensor_type.element_type = kLiteRtElementTypeInt64;

  auto desc =
      CreateTensorDescriptor(tensor_type, kLiteRtTensorBufferTypeMetalBuffer);
  ASSERT_TRUE(desc.ok());

  std::vector<int64_t> input_data = {1, 2, 3, 4, 5, 6, 7, 8};
  desc->UploadData(input_data.data());

  std::vector<int64_t> host_data(8);
  ConvertDataFromDescriptor(*desc, host_data.data(), kLiteRtElementTypeInt64);

  EXPECT_THAT(host_data[0], Eq(1));
  EXPECT_THAT(host_data[7], Eq(8));
}

TEST(BufferHandlerUtilsTest, ConvertDataToDescriptorUInt64) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 2;
  tensor_type.layout.dimensions[0] = 2;
  tensor_type.layout.dimensions[1] = 4;
  tensor_type.element_type = kLiteRtElementTypeUInt64;

  auto desc =
      CreateTensorDescriptor(tensor_type, kLiteRtTensorBufferTypeMetalBuffer);
  ASSERT_TRUE(desc.ok());

  std::vector<uint64_t> host_data = {1, 2, 3, 4, 5, 6, 7, 8};
  ConvertDataToDescriptor(host_data.data(), *desc, kLiteRtElementTypeUInt64);

  auto result_data = desc->GetData();
  ASSERT_EQ(result_data.size(), 8 * sizeof(uint64_t));
  const uint64_t* uint64_data =
      reinterpret_cast<const uint64_t*>(result_data.data());
  EXPECT_THAT(uint64_data[0], Eq(1));
  EXPECT_THAT(uint64_data[7], Eq(8));
}

TEST(BufferHandlerUtilsTest, ConvertDataFromDescriptorUInt64) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 2;
  tensor_type.layout.dimensions[0] = 2;
  tensor_type.layout.dimensions[1] = 4;
  tensor_type.element_type = kLiteRtElementTypeUInt64;

  auto desc =
      CreateTensorDescriptor(tensor_type, kLiteRtTensorBufferTypeMetalBuffer);
  ASSERT_TRUE(desc.ok());

  std::vector<uint64_t> input_data = {1, 2, 3, 4, 5, 6, 7, 8};
  desc->UploadData(input_data.data());

  std::vector<uint64_t> host_data(8);
  ConvertDataFromDescriptor(*desc, host_data.data(), kLiteRtElementTypeUInt64);

  EXPECT_THAT(host_data[0], Eq(1));
  EXPECT_THAT(host_data[7], Eq(8));
}

TEST(BufferHandlerUtilsTest, ConvertDataToDescriptorInt16) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 2;
  tensor_type.layout.dimensions[0] = 2;
  tensor_type.layout.dimensions[1] = 4;
  tensor_type.element_type = kLiteRtElementTypeInt16;

  auto desc =
      CreateTensorDescriptor(tensor_type, kLiteRtTensorBufferTypeMetalBuffer);
  ASSERT_TRUE(desc.ok());

  std::vector<int16_t> host_data = {1, 2, 3, 4, 5, 6, 7, 8};
  ConvertDataToDescriptor(host_data.data(), *desc, kLiteRtElementTypeInt16);

  auto result_data = desc->GetData();
  ASSERT_EQ(result_data.size(), 8 * sizeof(int16_t));
  const int16_t* int16_data =
      reinterpret_cast<const int16_t*>(result_data.data());
  EXPECT_THAT(int16_data[0], Eq(1));
  EXPECT_THAT(int16_data[7], Eq(8));
}

TEST(BufferHandlerUtilsTest, ConvertDataFromDescriptorInt16) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 2;
  tensor_type.layout.dimensions[0] = 2;
  tensor_type.layout.dimensions[1] = 4;
  tensor_type.element_type = kLiteRtElementTypeInt16;

  auto desc =
      CreateTensorDescriptor(tensor_type, kLiteRtTensorBufferTypeMetalBuffer);
  ASSERT_TRUE(desc.ok());

  std::vector<int16_t> input_data = {1, 2, 3, 4, 5, 6, 7, 8};
  desc->UploadData(input_data.data());

  std::vector<int16_t> host_data(8);
  ConvertDataFromDescriptor(*desc, host_data.data(), kLiteRtElementTypeInt16);

  EXPECT_THAT(host_data[0], Eq(1));
  EXPECT_THAT(host_data[7], Eq(8));
}

TEST(BufferHandlerUtilsTest, ConvertDataToDescriptorUInt16) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 2;
  tensor_type.layout.dimensions[0] = 2;
  tensor_type.layout.dimensions[1] = 4;
  tensor_type.element_type = kLiteRtElementTypeUInt16;

  auto desc =
      CreateTensorDescriptor(tensor_type, kLiteRtTensorBufferTypeMetalBuffer);
  ASSERT_TRUE(desc.ok());

  std::vector<uint16_t> host_data = {1, 2, 3, 4, 5, 6, 7, 8};
  ConvertDataToDescriptor(host_data.data(), *desc, kLiteRtElementTypeUInt16);

  auto result_data = desc->GetData();
  ASSERT_EQ(result_data.size(), 8 * sizeof(uint16_t));
  const uint16_t* uint16_data =
      reinterpret_cast<const uint16_t*>(result_data.data());
  EXPECT_THAT(uint16_data[0], Eq(1));
  EXPECT_THAT(uint16_data[7], Eq(8));
}

TEST(BufferHandlerUtilsTest, ConvertDataFromDescriptorUInt16) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 2;
  tensor_type.layout.dimensions[0] = 2;
  tensor_type.layout.dimensions[1] = 4;
  tensor_type.element_type = kLiteRtElementTypeUInt16;

  auto desc =
      CreateTensorDescriptor(tensor_type, kLiteRtTensorBufferTypeMetalBuffer);
  ASSERT_TRUE(desc.ok());

  std::vector<uint16_t> input_data = {1, 2, 3, 4, 5, 6, 7, 8};
  desc->UploadData(input_data.data());

  std::vector<uint16_t> host_data(8);
  ConvertDataFromDescriptor(*desc, host_data.data(), kLiteRtElementTypeUInt16);

  EXPECT_THAT(host_data[0], Eq(1));
  EXPECT_THAT(host_data[7], Eq(8));
}
}  // namespace
}  // namespace litert::ml_drift

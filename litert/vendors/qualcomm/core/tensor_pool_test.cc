// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/tensor_pool.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {

namespace {

// TODO(Alen): The current test coverage is not exhaustive.
// Some corner cases may not be tested. Narrowed types may lead to unexpected
// behavior.

TEST(TensorPoolConvertStaticTensorTest, ConvertNonStaticTensor) {
  TensorPool tensor_pool;

  auto& tensor_wrapper = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_FLOAT_32, QuantizeParamsWrapperVariant{}, {1, 2, 3});

  auto* res = tensor_pool.ConvertStaticTensorFrom<float>(tensor_wrapper);
  ASSERT_EQ(res, nullptr);
}

TEST(TensorPoolConvertStaticTensorTest, ExceedRangeAndFailToConvert) {
  TensorPool tensor_pool;

  std::vector<std::int32_t> tensor_data{
      std::numeric_limits<std::int32_t>::min(),
      std::numeric_limits<std::int32_t>::max()};
  auto& tensor_wrapper = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, QuantizeParamsWrapperVariant{}, {2},
      sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data());

  auto* res = tensor_pool.ConvertStaticTensorFrom<std::int16_t>(tensor_wrapper);
  ASSERT_EQ(res, nullptr);
}

TEST(TensorPoolConvertStaticTensorTest, SameTypeConversionFloat32) {
  TensorPool tensor_pool;

  std::vector<float> tensor_data{0, 1, 2, 3, 4, 5};
  auto& tensor_wrapper = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, QuantizeParamsWrapperVariant{}, {1, 2, 3},
      sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data());

  auto* res = tensor_pool.ConvertStaticTensorFrom<float>(tensor_wrapper);
  ASSERT_NE(res, nullptr);

  auto converted_data = res->GetTensorData<float>();
  ASSERT_TRUE(converted_data.has_value());

  ASSERT_EQ(tensor_data.size(), converted_data->size());
  for (size_t i = 0; i < tensor_data.size(); ++i) {
    ASSERT_FLOAT_EQ(tensor_data[i], (*converted_data)[i]);
  }
}

TEST(TensorPoolConvertStaticTensorTest, SameTypeConversionInt32) {
  TensorPool tensor_pool;

  std::vector<std::int32_t> tensor_data{0, 1, 2, 3, 4, 5};
  auto& tensor_wrapper = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, QuantizeParamsWrapperVariant{}, {1, 2, 3},
      sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data());

  auto* res = tensor_pool.ConvertStaticTensorFrom<std::int32_t>(tensor_wrapper);
  ASSERT_NE(res, nullptr);

  auto converted_data = res->GetTensorData<std::int32_t>();
  ASSERT_TRUE(converted_data.has_value());

  ASSERT_EQ(tensor_data.size(), converted_data->size());
  for (size_t i = 0; i < tensor_data.size(); ++i) {
    ASSERT_EQ(tensor_data[i], (*converted_data)[i]);
  }
}

TEST(TensorPoolConvertStaticTensorTest, ExpandTypeConversionFloat32) {
  TensorPool tensor_pool;

  std::vector<float> tensor_data{0, 1, 2, 3, 4, 5};
  auto& tensor_wrapper = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, QuantizeParamsWrapperVariant{}, {1, 2, 3},
      sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data());

  auto* res = tensor_pool.ConvertStaticTensorFrom<double>(tensor_wrapper);
  ASSERT_NE(res, nullptr);

  auto converted_data = res->GetTensorData<double>();
  ASSERT_TRUE(converted_data.has_value());

  ASSERT_EQ(tensor_data.size(), converted_data->size());
  for (size_t i = 0; i < tensor_data.size(); ++i) {
    ASSERT_DOUBLE_EQ(tensor_data[i], (*converted_data)[i]);
  }
}

TEST(TensorPoolConvertStaticTensorTest, ExpandTypeConversionInt32) {
  TensorPool tensor_pool;

  std::vector<std::int32_t> tensor_data{0, 1, 2, 3, 4, 5};
  auto& tensor_wrapper = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, QuantizeParamsWrapperVariant{}, {1, 2, 3},
      sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data());

  auto* res = tensor_pool.ConvertStaticTensorFrom<std::int64_t>(tensor_wrapper);
  ASSERT_NE(res, nullptr);

  auto converted_data = res->GetTensorData<std::int64_t>();
  ASSERT_TRUE(converted_data.has_value());

  ASSERT_EQ(tensor_data.size(), converted_data->size());
  for (size_t i = 0; i < tensor_data.size(); ++i) {
    ASSERT_EQ(tensor_data[i], (*converted_data)[i]);
  }
}

TEST(TensorPoolConvertStaticTensorTest, NarrowTypeConversionFloat32) {
  TensorPool tensor_pool;

  std::vector<double> tensor_data{0, 1, 2, 3, 4, 5};
  auto& tensor_wrapper = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_64, QuantizeParamsWrapperVariant{}, {1, 2, 3},
      sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data());

  auto* res = tensor_pool.ConvertStaticTensorFrom<float>(tensor_wrapper);
  ASSERT_NE(res, nullptr);

  auto converted_data = res->GetTensorData<float>();
  ASSERT_TRUE(converted_data.has_value());

  ASSERT_EQ(tensor_data.size(), converted_data->size());
  for (size_t i = 0; i < tensor_data.size(); ++i) {
    ASSERT_DOUBLE_EQ(tensor_data[i], (*converted_data)[i]);
  }
}

TEST(TensorPoolConvertStaticTensorTest, NarrowTypeConversionInt32) {
  TensorPool tensor_pool;

  std::vector<std::int64_t> tensor_data{0, 1, 2, 3, 4, 5};
  auto& tensor_wrapper = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_64, QuantizeParamsWrapperVariant{}, {1, 2, 3},
      sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data());

  auto* res = tensor_pool.ConvertStaticTensorFrom<std::int32_t>(tensor_wrapper);
  ASSERT_NE(res, nullptr);

  auto converted_data = res->GetTensorData<std::int32_t>();
  ASSERT_TRUE(converted_data.has_value());

  ASSERT_EQ(tensor_data.size(), converted_data->size());
  for (size_t i = 0; i < tensor_data.size(); ++i) {
    ASSERT_EQ(tensor_data[i], (*converted_data)[i]);
  }
}

TEST(TensorPoolConvertStaticTensorTest, CreateStatictensorByValueFloat) {
  TensorPool tensor_pool;

  std::vector<float> golden_data = {6.0f, 6.0f, 6.0f};

  TensorWrapper* tensor_wrapper = tensor_pool.CreateStaticTensorWithValue(
      QNN_DATATYPE_FLOAT_32, {}, {1, 1, 3}, 6);
  ASSERT_NE(tensor_wrapper, nullptr);
  const auto tensor_data = tensor_wrapper->GetTensorData<float>();

  EXPECT_TRUE(tensor_data.has_value());
  EXPECT_EQ(tensor_data, golden_data);
}

TEST(TensorPoolConvertStaticTensorTest, CreateStatictensorByValueInt8) {
  TensorPool tensor_pool;

  ScaleOffsetQuantizeParamsWrapper q_param(2, -5);  // offset = 5

  std::vector<float> golden_data = {2.0, 2.0, 2.0};

  TensorWrapper* tensor_wrapper = tensor_pool.CreateStaticTensorWithValue(
      QNN_DATATYPE_SFIXED_POINT_8, q_param, {1, 1, 3}, 2);
  ASSERT_NE(tensor_wrapper, nullptr);

  const auto& q_param_ref = tensor_wrapper->GetQuantParams();
  const float scale =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetScale();
  const std::int32_t zero_point =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetZeroPoint();

  const auto tensor_data = tensor_wrapper->GetTensorData<std::int8_t>();

  EXPECT_TRUE(tensor_data.has_value());

  // Dequantize each element from the tensor data.
  for (int i = 0; i < golden_data.size(); i++) {
    EXPECT_NEAR(Dequantize((*tensor_data)[i], scale, zero_point),
                golden_data[i], 1e-7);
  }
}

TEST(TensorPoolConvertStaticTensorTest, CreateStatictensorByValueUInt8) {
  TensorPool tensor_pool;

  ScaleOffsetQuantizeParamsWrapper q_param(2, 5);  // offset = -5

  std::vector<float> golden_data = {2.0, 2.0, 2.0};

  TensorWrapper* tensor_wrapper = tensor_pool.CreateStaticTensorWithValue(
      QNN_DATATYPE_UFIXED_POINT_8, q_param, {1, 1, 3}, 2);
  ASSERT_NE(tensor_wrapper, nullptr);

  const auto& q_param_ref = tensor_wrapper->GetQuantParams();
  const float scale =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetScale();
  const std::int32_t zero_point =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetZeroPoint();

  const auto tensor_data = tensor_wrapper->GetTensorData<std::uint8_t>();

  EXPECT_TRUE(tensor_data.has_value());

  // Dequantize each element from the tensor data.
  for (int i = 0; i < golden_data.size(); i++) {
    EXPECT_NEAR(Dequantize((*tensor_data)[i], scale, zero_point),
                golden_data[i], 1e-7);
  }
}

TEST(TensorPoolConvertStaticTensorTest, CreateStatictensorByValueInt16) {
  TensorPool tensor_pool;

  ScaleOffsetQuantizeParamsWrapper q_param(2, -5);  // offset = 5

  std::vector<float> golden_data = {2.0, 2.0, 2.0};

  TensorWrapper* tensor_wrapper = tensor_pool.CreateStaticTensorWithValue(
      QNN_DATATYPE_SFIXED_POINT_16, q_param, {1, 1, 3}, 2);
  ASSERT_NE(tensor_wrapper, nullptr);

  const auto& q_param_ref = tensor_wrapper->GetQuantParams();
  const float scale =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetScale();
  const std::int32_t zero_point =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetZeroPoint();

  const auto tensor_data = tensor_wrapper->GetTensorData<std::int16_t>();

  EXPECT_TRUE(tensor_data.has_value());

  // Dequantize each element from the tensor data.
  for (int i = 0; i < golden_data.size(); i++) {
    EXPECT_NEAR(Dequantize((*tensor_data)[i], scale, zero_point),
                golden_data[i], 1e-7);
  }
}

TEST(TensorPoolConvertStaticTensorTest, CreateStatictensorByValueUInt16) {
  TensorPool tensor_pool;

  ScaleOffsetQuantizeParamsWrapper q_param(2, 5);  // offset = -5

  std::vector<float> golden_data = {2.0, 2.0, 2.0};

  TensorWrapper* tensor_wrapper = tensor_pool.CreateStaticTensorWithValue(
      QNN_DATATYPE_UFIXED_POINT_16, q_param, {1, 1, 3}, 2);
  ASSERT_NE(tensor_wrapper, nullptr);

  const auto& q_param_ref = tensor_wrapper->GetQuantParams();
  const float scale =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetScale();
  const std::int32_t zero_point =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetZeroPoint();

  const auto tensor_data = tensor_wrapper->GetTensorData<std::uint16_t>();

  EXPECT_TRUE(tensor_data.has_value());

  // Dequantize each element from the tensor data.
  for (int i = 0; i < golden_data.size(); i++) {
    EXPECT_NEAR(Dequantize((*tensor_data)[i], scale, zero_point),
                golden_data[i], 1e-7);
  }
}

TEST(TensorPoolConvertStaticTensorTest, CreateStatictensorByValueSFixInt32) {
  TensorPool tensor_pool;

  ScaleOffsetQuantizeParamsWrapper q_param(2, -5);  // offset = 5

  std::vector<float> golden_data = {2.0, 2.0, 2.0};

  TensorWrapper* tensor_wrapper = tensor_pool.CreateStaticTensorWithValue(
      QNN_DATATYPE_SFIXED_POINT_32, q_param, {1, 1, 3}, 2.0);
  ASSERT_NE(tensor_wrapper, nullptr);

  const auto& q_param_ref = tensor_wrapper->GetQuantParams();
  const float scale =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetScale();
  const std::int32_t zero_point =
      std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetZeroPoint();

  const auto tensor_data = tensor_wrapper->GetTensorData<std::int32_t>();

  EXPECT_TRUE(tensor_data.has_value());

  // Dequantize each element from the tensor data.
  for (int i = 0; i < golden_data.size(); i++) {
    EXPECT_NEAR(Dequantize((*tensor_data)[i], scale, zero_point),
                golden_data[i], 1e-7);
  }
}

// TODO(@chengwl-qti): Re-enable this test when it passes in dbg mode.
// TEST(TensorPoolConvertStaticTensorTest, CreateStatictensorByValueUFixInt32) {
//   TensorPool tensor_pool;

//   ScaleOffsetQuantizeParamsWrapper q_param(2, -5);  // offset = 5

//   std::vector<float> golden_data = {2, 2, 2};

//   TensorWrapper* tensor_wrapper = tensor_pool.CreateStaticTensorWithValue(
//       QNN_DATATYPE_UFIXED_POINT_32, q_param, {1, 1, 3}, 2.0);
//   ASSERT_NE(tensor_wrapper, nullptr);

//   const auto& q_param_ref = tensor_wrapper->GetQuantParams();
//   const float scale =
//       std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetScale();
//   const std::int32_t zero_point =
//       std::get<ScaleOffsetQuantizeParamsWrapper>(q_param_ref).GetZeroPoint();

//   const auto tensor_data = tensor_wrapper->GetTensorData<std::uint32_t>();

//   EXPECT_TRUE(tensor_data.has_value());

//   // Dequantize each element from the tensor data.
//   for (int i = 0; i < golden_data.size(); i++) {
//     EXPECT_NEAR(Dequantize((*tensor_data)[i], scale, zero_point),
//                 golden_data[i], 1e-7);
//   }
// }

TEST(TensorPoolConvertStaticTensorTest, CreateStatictensorByValueInt32) {
  TensorPool tensor_pool;

  std::vector<std::int32_t> golden_data = {2, 2, 2};

  TensorWrapper* tensor_wrapper = tensor_pool.CreateStaticTensorWithValue(
      QNN_DATATYPE_INT_32, {}, {1, 1, 3}, 2.0);
  ASSERT_NE(tensor_wrapper, nullptr);

  const auto tensor_data = tensor_wrapper->GetTensorData<std::int32_t>();

  EXPECT_TRUE(tensor_data.has_value());

  EXPECT_EQ(tensor_data, golden_data);
}

TEST(TensorPoolConvertStaticTensorTest, CreateStatictensorByValueUInt32) {
  TensorPool tensor_pool;

  std::vector<std::uint32_t> golden_data = {2, 2, 2};

  TensorWrapper* tensor_wrapper = tensor_pool.CreateStaticTensorWithValue(
      QNN_DATATYPE_UINT_32, {}, {1, 1, 3}, 2.0);
  ASSERT_NE(tensor_wrapper, nullptr);

  const auto tensor_data = tensor_wrapper->GetTensorData<std::uint32_t>();

  EXPECT_TRUE(tensor_data.has_value());

  EXPECT_EQ(tensor_data, golden_data);
}

}  // namespace

}  // namespace qnn

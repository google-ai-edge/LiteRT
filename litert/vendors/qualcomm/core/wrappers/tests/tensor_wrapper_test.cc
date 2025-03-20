// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"

namespace qnn {
namespace {

TEST(TensorWrapperTest, SanityTest) {
  TensorWrapper tensor_wrapper{};

  EXPECT_EQ(tensor_wrapper.GetRank(), 0);
  EXPECT_TRUE(tensor_wrapper.GetDims().empty());
  EXPECT_TRUE(std::holds_alternative<UndefinedQuantizeParamsWrapper>(
      tensor_wrapper.GetQuantParams()));
  EXPECT_FALSE(tensor_wrapper.IsPerTensorQuantWithOffsetDiff(tensor_wrapper));
  EXPECT_FALSE(tensor_wrapper.IsQuant8());
  EXPECT_FALSE(tensor_wrapper.IsQuant16());
  EXPECT_EQ(tensor_wrapper.GetDataType(), QNN_DATATYPE_UNDEFINED);
  EXPECT_FALSE(tensor_wrapper.IsSubgraphInput());
  EXPECT_FALSE(tensor_wrapper.IsSubgraphOutput());
  EXPECT_FALSE(tensor_wrapper.IsTensorStatic());
  EXPECT_EQ(tensor_wrapper.GetStaticTensorData<std::uint8_t>(), std::nullopt);
  std::vector<std::uint8_t> data = {1, 2, 3};
  // expect no use, since tensor type not correct
  tensor_wrapper.SetTensorData<std::uint8_t>(
      absl::MakeSpan(data.data(), data.size()));
  EXPECT_EQ(tensor_wrapper.GetStaticTensorData<std::uint8_t>(), std::nullopt);
}

TEST(TensorWrapperTest, CopyTensorTest) {
  std::vector<std::uint32_t> dummy_dims = {1, 1, 3};
  ScaleOffsetQuantizeParamsWrapper q_param(1, 0);
  TensorWrapper tensor_wrapper{0, QNN_TENSOR_TYPE_STATIC,
                               QNN_DATATYPE_UFIXED_POINT_8, q_param,
                               dummy_dims};
  TensorWrapper copied{tensor_wrapper};

  EXPECT_EQ(copied.GetRank(), 3);
  EXPECT_EQ(copied.GetDims(), dummy_dims);
  EXPECT_TRUE(std::holds_alternative<ScaleOffsetQuantizeParamsWrapper>(
      copied.GetQuantParams()));
  EXPECT_FALSE(copied.IsPerTensorQuantWithOffsetDiff(copied));
  EXPECT_TRUE(copied.IsQuant8());
  EXPECT_FALSE(copied.IsQuant16());
  EXPECT_EQ(copied.GetDataType(), QNN_DATATYPE_UFIXED_POINT_8);
  EXPECT_FALSE(copied.IsSubgraphInput());
  EXPECT_FALSE(copied.IsSubgraphOutput());
  EXPECT_TRUE(copied.IsTensorStatic());
  EXPECT_EQ(copied.GetStaticTensorData<std::uint8_t>(), std::nullopt);
  std::vector<std::uint8_t> data = {1, 2, 3};
  copied.SetTensorData<std::uint8_t>(absl::MakeSpan(data.data(), data.size()));
  const auto tensor_data = copied.GetStaticTensorData<std::uint8_t>();
  EXPECT_TRUE(tensor_data.has_value());
  for (size_t i = 0; i < data.size(); i++) {
    EXPECT_EQ((*tensor_data)[i], data[i]);
  }
}

TEST(TensorWrapperTest, MoveTensorTest) {
  std::vector<std::uint32_t> dummy_dims = {1, 1, 3};
  ScaleOffsetQuantizeParamsWrapper q_param(1, 0);
  std::vector<std::uint8_t> data = {1, 2, 3};
  void* data_ptr = reinterpret_cast<void*>(data.data());
  TensorWrapper tensor_wrapper{0,
                               QNN_TENSOR_TYPE_STATIC,
                               QNN_DATATYPE_UFIXED_POINT_8,
                               q_param,
                               dummy_dims,
                               static_cast<uint32_t>(data.size()),
                               data_ptr};
  TensorWrapper moved{tensor_wrapper};

  EXPECT_EQ(moved.GetRank(), 3);
  EXPECT_EQ(moved.GetDims(), dummy_dims);
  EXPECT_TRUE(std::holds_alternative<ScaleOffsetQuantizeParamsWrapper>(
      moved.GetQuantParams()));
  EXPECT_FALSE(moved.IsPerTensorQuantWithOffsetDiff(moved));
  EXPECT_TRUE(moved.IsQuant8());
  EXPECT_FALSE(moved.IsQuant16());
  EXPECT_EQ(moved.GetDataType(), QNN_DATATYPE_UFIXED_POINT_8);
  EXPECT_FALSE(moved.IsSubgraphInput());
  EXPECT_FALSE(moved.IsSubgraphOutput());
  EXPECT_TRUE(moved.IsTensorStatic());
  const auto tensor_data = moved.GetStaticTensorData<std::uint8_t>();
  EXPECT_TRUE(tensor_data.has_value());
  for (size_t i = 0; i < data.size(); i++) {
    EXPECT_EQ(tensor_data.value()[i], data[i]);
  }
}

TEST(TensorWrapperTest, QnnTensorTest) {
  std::vector<std::uint32_t> dummy_dims = {1, 1, 3};
  std::vector<std::uint8_t> data = {1, 2, 3};
  void* data_ptr = reinterpret_cast<void*>(data.data());
  const auto data_size =
      std::accumulate(dummy_dims.begin(), dummy_dims.end(),
                      sizeof(decltype(data)::value_type), std::multiplies<>());

  TensorWrapper tensor_wrapper{0,
                               QNN_TENSOR_TYPE_APP_WRITE,
                               QNN_DATATYPE_UFIXED_POINT_8,
                               QuantizeParamsWrapperVariant(),
                               dummy_dims,
                               static_cast<uint32_t>(data_size),
                               data_ptr};

  Qnn_Tensor_t cloned;
  tensor_wrapper.CloneTo(cloned);
  EXPECT_EQ(cloned.version, QNN_TENSOR_VERSION_2);
  EXPECT_EQ(cloned.v2.id, 0);
  EXPECT_EQ(cloned.v2.type, QNN_TENSOR_TYPE_APP_WRITE);
  EXPECT_EQ(cloned.v2.dataFormat, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
  EXPECT_EQ(cloned.v2.dataType, QNN_DATATYPE_UFIXED_POINT_8);
  EXPECT_EQ(cloned.v2.quantizeParams.encodingDefinition,
            QNN_DEFINITION_UNDEFINED);
  EXPECT_EQ(cloned.v2.rank, dummy_dims.size());
  for (size_t i = 0; i < cloned.v2.rank; i++) {
    EXPECT_EQ(cloned.v2.dimensions[i], dummy_dims[i]);
  }
  EXPECT_EQ(cloned.v2.memType, QNN_TENSORMEMTYPE_RAW);
  EXPECT_EQ(cloned.v2.clientBuf.dataSize, data_size);
  const auto* cloned_data =
      reinterpret_cast<const std::uint8_t*>(cloned.v2.clientBuf.data);
  for (size_t i = 0; i < data.size(); i++) {
    EXPECT_EQ(cloned_data[i], data[i]);
  }

  Qnn_Tensor_t& ref = tensor_wrapper.GetQnnTensor();
  EXPECT_EQ(ref.version, QNN_TENSOR_VERSION_2);
  EXPECT_EQ(ref.v2.id, 0);
  EXPECT_EQ(ref.v2.type, QNN_TENSOR_TYPE_APP_WRITE);
  EXPECT_EQ(ref.v2.dataFormat, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
  EXPECT_EQ(ref.v2.dataType, QNN_DATATYPE_UFIXED_POINT_8);
  EXPECT_EQ(ref.v2.quantizeParams.encodingDefinition, QNN_DEFINITION_UNDEFINED);
  EXPECT_EQ(ref.v2.rank, dummy_dims.size());
  for (size_t i = 0; i < ref.v2.rank; i++) {
    EXPECT_EQ(ref.v2.dimensions[i], dummy_dims[i]);
  }
  EXPECT_EQ(ref.v2.memType, QNN_TENSORMEMTYPE_RAW);
  EXPECT_EQ(ref.v2.clientBuf.dataSize, data_size);
  const auto* ref_data =
      reinterpret_cast<const std::uint8_t*>(ref.v2.clientBuf.data);
  for (size_t i = 0; i < data.size(); i++) {
    EXPECT_EQ(ref_data[i], data[i]);
  }
}

TEST(TensorWrapperTest, DataTypeTest) {
  TensorWrapper tensor_wrapper{};
  tensor_wrapper.SetDataType(QNN_DATATYPE_UFIXED_POINT_8);
  EXPECT_EQ(tensor_wrapper.GetDataType(), QNN_DATATYPE_UFIXED_POINT_8);
  EXPECT_TRUE(tensor_wrapper.IsQuant8());

  tensor_wrapper.SetDataType(QNN_DATATYPE_SFIXED_POINT_8);
  EXPECT_EQ(tensor_wrapper.GetDataType(), QNN_DATATYPE_SFIXED_POINT_8);
  EXPECT_TRUE(tensor_wrapper.IsQuant8());

  tensor_wrapper.SetDataType(QNN_DATATYPE_UFIXED_POINT_16);
  EXPECT_EQ(tensor_wrapper.GetDataType(), QNN_DATATYPE_UFIXED_POINT_16);
  EXPECT_TRUE(tensor_wrapper.IsQuant16());

  tensor_wrapper.SetDataType(QNN_DATATYPE_SFIXED_POINT_16);
  EXPECT_EQ(tensor_wrapper.GetDataType(), QNN_DATATYPE_SFIXED_POINT_16);
  EXPECT_TRUE(tensor_wrapper.IsQuant16());
}

TEST(TensorWrapperTest, IsPerTensorQuantWithOffsetDiff8BitTest) {
  constexpr int kSUFixed8OffsetDiff = 128;
  ScaleOffsetQuantizeParamsWrapper wrapper1(1, 0);
  ScaleOffsetQuantizeParamsWrapper wrapper2(1, kSUFixed8OffsetDiff);
  TensorWrapper tensor_wrapper0{0,
                                QNN_TENSOR_TYPE_STATIC,
                                QNN_DATATYPE_UFIXED_POINT_8,
                                QuantizeParamsWrapperVariant(wrapper1),
                                {}};
  TensorWrapper tensor_wrapper1{0,
                                QNN_TENSOR_TYPE_STATIC,
                                QNN_DATATYPE_UFIXED_POINT_8,
                                QuantizeParamsWrapperVariant(wrapper2),
                                {}};
  EXPECT_FALSE(tensor_wrapper0.IsPerTensorQuantWithOffsetDiff(tensor_wrapper1));
  tensor_wrapper1.SetDataType(QNN_DATATYPE_SFIXED_POINT_8);
  EXPECT_TRUE(tensor_wrapper0.IsPerTensorQuantWithOffsetDiff(tensor_wrapper1));
}

TEST(TensorWrapperTest, IsPerTensorQuantWithOffsetDiff16BitTest) {
  constexpr int kSUFixed16OffsetDiff = 32768;
  ScaleOffsetQuantizeParamsWrapper wrapper1(1, 0);
  ScaleOffsetQuantizeParamsWrapper wrapper2(1, kSUFixed16OffsetDiff);
  TensorWrapper tensor_wrapper0{0,
                                QNN_TENSOR_TYPE_STATIC,
                                QNN_DATATYPE_UFIXED_POINT_16,
                                QuantizeParamsWrapperVariant(wrapper1),
                                {}};
  TensorWrapper tensor_wrapper1{0,
                                QNN_TENSOR_TYPE_STATIC,
                                QNN_DATATYPE_UFIXED_POINT_16,
                                QuantizeParamsWrapperVariant(wrapper2),
                                {}};
  EXPECT_FALSE(tensor_wrapper0.IsPerTensorQuantWithOffsetDiff(tensor_wrapper1));
  tensor_wrapper1.SetDataType(QNN_DATATYPE_SFIXED_POINT_16);
  EXPECT_TRUE(tensor_wrapper0.IsPerTensorQuantWithOffsetDiff(tensor_wrapper1));
}

TEST(TensorWrapperTest, StaticTensorTest) {
  TensorWrapper tensor_wrapper{0,
                               QNN_TENSOR_TYPE_STATIC,
                               QNN_DATATYPE_UNDEFINED,
                               QuantizeParamsWrapperVariant(),
                               {}};

  EXPECT_TRUE(tensor_wrapper.IsTensorStatic());
  EXPECT_FALSE(tensor_wrapper.IsSubgraphInput());
  EXPECT_FALSE(tensor_wrapper.IsSubgraphOutput());
}

TEST(TensorWrapperTest, SubgraphInputTensorTest) {
  TensorWrapper tensor_wrapper{0,
                               QNN_TENSOR_TYPE_APP_WRITE,
                               QNN_DATATYPE_UNDEFINED,
                               QuantizeParamsWrapperVariant(),
                               {}};

  EXPECT_FALSE(tensor_wrapper.IsTensorStatic());
  EXPECT_TRUE(tensor_wrapper.IsSubgraphInput());
  EXPECT_FALSE(tensor_wrapper.IsSubgraphOutput());
}

TEST(TensorWrapperTest, SubgraphOutputTensorTest) {
  TensorWrapper tensor_wrapper{0,
                               QNN_TENSOR_TYPE_APP_READ,
                               QNN_DATATYPE_UNDEFINED,
                               QuantizeParamsWrapperVariant(),
                               {}};

  EXPECT_FALSE(tensor_wrapper.IsTensorStatic());
  EXPECT_FALSE(tensor_wrapper.IsSubgraphInput());
  EXPECT_TRUE(tensor_wrapper.IsSubgraphOutput());
}

TEST(TensorWrapperTest, GetStaticTensorDataNonStaticTest) {
  std::vector<std::uint32_t> dummy_dims = {1, 1, 3};
  ScaleOffsetQuantizeParamsWrapper q_param(1, 0);
  TensorWrapper tensor_wrapper{0, QNN_TENSOR_TYPE_APP_WRITE,
                               QNN_DATATYPE_UFIXED_POINT_8, q_param,
                               dummy_dims};
  EXPECT_FALSE(tensor_wrapper.GetStaticTensorData<std::uint8_t>().has_value());
}

TEST(TensorWrapperTest, GetStaticTensorDataTest) {
  std::vector<std::uint32_t> dummy_dims = {1, 1, 3};
  ScaleOffsetQuantizeParamsWrapper q_param(1, 0);
  TensorWrapper tensor_wrapper{0, QNN_TENSOR_TYPE_STATIC,
                               QNN_DATATYPE_UFIXED_POINT_8, q_param,
                               dummy_dims};

  EXPECT_FALSE(tensor_wrapper.GetStaticTensorData<float>().has_value());
  EXPECT_FALSE(tensor_wrapper.GetStaticTensorData<std::int8_t>().has_value());
  EXPECT_FALSE(tensor_wrapper.GetStaticTensorData<std::uint8_t>().has_value());
  std::vector<std::uint8_t> data = {1, 2, 3};
  tensor_wrapper.SetTensorData<std::uint8_t>(
      absl::MakeSpan(data.data(), data.size()));
  const auto tensor_data =
      *(tensor_wrapper.GetStaticTensorData<std::uint8_t>());
  for (size_t i = 0; i < data.size(); i++) {
    EXPECT_EQ(tensor_data[i], data[i]);
  }
}
}  // namespace
}  // namespace qnn

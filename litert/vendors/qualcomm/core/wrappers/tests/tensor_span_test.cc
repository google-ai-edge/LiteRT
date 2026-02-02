// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/wrappers/tensor_span.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <gtest/gtest.h>

namespace qnn {
namespace {

TEST(TensorSpanTest, GetAndSet) {
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  std::array<std::uint32_t, 3> dims = {1, 2, 3};
  static constexpr std::uint32_t kNumElements = 6;
  static constexpr std::uint32_t kBytes = kNumElements * sizeof(std::int16_t);
  tensor.v1.rank = dims.size();
  tensor.v1.dimensions = dims.data();
  tensor.v1.dataType = QNN_DATATYPE_UFIXED_POINT_16;
  tensor.v1.name = "test_tensor";
  tensor.v1.quantizeParams = {QNN_DEFINITION_DEFINED,
                              QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                              {1.0f, 0}};

  TensorSpan tensor_span(&tensor);

  EXPECT_EQ(tensor_span.Get(), &tensor);

  EXPECT_TRUE(tensor_span.IsQUInt16());
  EXPECT_EQ(tensor_span.GetName(), "test_tensor");

  const auto scale_offset = tensor_span.GetScaleOffset();
  EXPECT_EQ(scale_offset.scale, 1.0f);
  EXPECT_EQ(scale_offset.offset, 0);
  EXPECT_EQ(tensor_span.GetNumElements(), kNumElements);
  EXPECT_EQ(tensor_span.GetBytes(), kBytes);

  std::byte data[kBytes]{};
  tensor_span.SetClientBuf(data, kBytes);
  EXPECT_EQ(tensor.v1.clientBuf.data, data);
  EXPECT_EQ(tensor.v1.clientBuf.dataSize, kBytes);
}

TEST(TensorSpanTest, IsMarkedDump) {
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  tensor.v1.name = "test_tensor_dump";
  tensor.v1.type = QNN_TENSOR_TYPE_APP_READ;

  TensorSpan tensor_span(&tensor);
  EXPECT_TRUE(tensor_span.IsMarkedDump());
}

}  // namespace
}  // namespace qnn

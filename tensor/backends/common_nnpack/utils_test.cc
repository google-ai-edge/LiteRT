/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/backends/common_nnpack/utils.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "tensor/arithmetic_graph.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Ge;
using ::testing::Le;

MATCHER(IsInf, "") { return std::isinf(arg); }
MATCHER(IsPosInf, "") {
  return ExplainMatchResult(AllOf(IsInf(), Ge(0)), arg, result_listener);
}
MATCHER(IsNegInf, "") {
  return ExplainMatchResult(AllOf(IsInf(), Le(0)), arg, result_listener);
}

TEST(CommonNnpackUtilsTest, GetActivationBounds) {
  NnpackActivationBounds none_bounds = GetActivationBounds(kActNone);
  EXPECT_THAT(none_bounds.min, IsNegInf());
  EXPECT_THAT(none_bounds.max, IsPosInf());

  NnpackActivationBounds relu_bounds = GetActivationBounds(kActRelu);
  EXPECT_EQ(relu_bounds.min, 0.0f);
  EXPECT_THAT(relu_bounds.max, IsPosInf());

  NnpackActivationBounds relu6_bounds = GetActivationBounds(kActRelu6);
  EXPECT_EQ(relu6_bounds.min, 0.0f);
  EXPECT_EQ(relu6_bounds.max, 6.0f);

  NnpackActivationBounds relu1_bounds = GetActivationBounds(kActReluN1To1);
  EXPECT_EQ(relu1_bounds.min, -1.0f);
  EXPECT_EQ(relu1_bounds.max, 1.0f);
}

TEST(CommonNnpackUtilsTest, ComputePadding) {
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      NnpackPadding valid_pad,
      ComputePadding(kPaddingValid, /*input_height=*/5, /*input_width=*/5,
                     /*kernel_height=*/3, /*kernel_width=*/3, /*stride_h=*/1,
                     /*stride_w=*/1, /*dilation_h=*/1, /*dilation_w=*/1));
  EXPECT_EQ(valid_pad.top, 0);
  EXPECT_EQ(valid_pad.bottom, 0);
  EXPECT_EQ(valid_pad.left, 0);
  EXPECT_EQ(valid_pad.right, 0);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      NnpackPadding same_pad,
      ComputePadding(kPaddingSame, /*input_height=*/5, /*input_width=*/5,
                     /*kernel_height=*/3, /*kernel_width=*/3, /*stride_h=*/1,
                     /*stride_w=*/1, /*dilation_h=*/1, /*dilation_w=*/1));
  EXPECT_EQ(same_pad.top, 1);
  EXPECT_EQ(same_pad.bottom, 1);
  EXPECT_EQ(same_pad.left, 1);
  EXPECT_EQ(same_pad.right, 1);
}

TEST(CommonNnpackUtilsTest,
     ComputeTransposeConvPaddingRejectsNonPositiveDimensions) {
  EXPECT_THAT(ComputeTransposeConvPadding(kPaddingSame, 0, 3, 3, 3, 1, 1, 5, 5),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ComputeTransposeConvPadding(kPaddingSame, 3, 0, 3, 3, 1, 1, 5, 5),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ComputeTransposeConvPadding(kPaddingSame, 3, 3, 3, 3, 0, 1, 5, 5),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CommonNnpackUtilsTest,
     ComputeTransposeConvPaddingRejectsInvalidPaddingForValid) {
  // base = (input - 1) * stride + filter = (3 - 1) * 2 + 4 = 8.
  // output = 5 -> requested < expected -> error
  EXPECT_THAT(ComputeTransposeConvPadding(
                  kPaddingValid, /*input_height=*/3, /*input_width=*/3,
                  /*kernel_height=*/4, /*kernel_width=*/4,
                  /*stride_h=*/2, /*stride_w=*/2,
                  /*output_height=*/5, /*output_width=*/5),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CommonNnpackUtilsTest, ComputeTransposeConvPaddingRejectsLargeAdjustment) {
  // base = (3 - 1) * 1 + 2 = 3.
  // output = 5 -> adj = 5 - 3 = 2 >= stride (1).
  EXPECT_THAT(ComputeTransposeConvPadding(
                  kPaddingSame, /*input_height=*/3, /*input_width=*/3,
                  /*kernel_height=*/2, /*kernel_width=*/2,
                  /*stride_h=*/1, /*stride_w=*/1,
                  /*output_height=*/5, /*output_width=*/5),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CommonNnpackUtilsTest,
     ComputeTransposeConvPaddingCalculatesAsymmetricPaddingAndAdjustment) {
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      NnpackTransposeConvPadding res,
      ComputeTransposeConvPadding(kPaddingSame, /*input_height=*/3,
                                  /*input_width=*/3,
                                  /*kernel_height=*/4, /*kernel_width=*/3,
                                  /*stride_h=*/2, /*stride_w=*/2,
                                  /*output_height=*/6, /*output_width=*/7));
  EXPECT_EQ(res.top, 1);
  EXPECT_EQ(res.bottom, 1);
  EXPECT_EQ(res.left, 0);
  EXPECT_EQ(res.right, 1);
  EXPECT_EQ(res.adj_h, 0);
  EXPECT_EQ(res.adj_w, 1);
}

TEST(CommonNnpackUtilsTest, ToNnpackDims) {
  const std::vector<int32_t> valid_shape = {1, 3, 224, 224};
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<size_t> dims,
                                  ToNnpackDims(valid_shape));
  EXPECT_THAT(dims, ElementsAre(1, 3, 224, 224));

  const std::vector<int32_t> invalid_shape = {1, -1, 224};
  EXPECT_THAT(ToNnpackDims(invalid_shape),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CommonNnpackUtilsTest, ValidateTensorType) {
  graph::TensorInformation fp32_info{.type = Type::kFP32, .shape = {2}};
  EXPECT_THAT(
      ValidateTensorType("TestOp", fp32_info, {Type::kFP32, Type::kFP16}),
      IsOk());
  EXPECT_THAT(ValidateTensorType("TestOp", fp32_info, {Type::kI32}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CommonNnpackUtilsTest, ValidateFp32OrQuantizedConstantWeights) {
  graph::TensorInformation fp32_info{
      .type = Type::kFP32,
      .shape = {2},
      .buffer = OwningCpuBuffer::Copy<Type::kFP32>({1.0f, 2.0f}),
  };
  EXPECT_THAT(ValidateFp32OrQuantizedConstantWeights("TestOp", fp32_info,
                                                     /*is_external=*/false),
              IsOk());

  EXPECT_THAT(ValidateFp32OrQuantizedConstantWeights("TestOp", fp32_info,
                                                     /*is_external=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument));

  graph::TensorInformation i32_info{
      .type = Type::kI32,
      .shape = {2},
      .buffer = OwningCpuBuffer::Copy<Type::kI32>({1, 2}),
  };
  EXPECT_THAT(ValidateFp32OrQuantizedConstantWeights("TestOp", i32_info,
                                                     /*is_external=*/false),
              StatusIs(absl::StatusCode::kInvalidArgument));

  graph::TensorInformation i8_no_buffer{
      .type = Type::kI8,
      .shape = {2},
      .buffer = nullptr,
  };
  EXPECT_THAT(ValidateFp32OrQuantizedConstantWeights("TestOp", i8_no_buffer,
                                                     /*is_external=*/false),
              StatusIs(absl::StatusCode::kInvalidArgument));

  graph::TensorInformation i8_valid{
      .type = Type::kI8,
      .shape = {2},
      .buffer = OwningCpuBuffer::Copy<Type::kI8>({1, 2}),
  };
  EXPECT_THAT(ValidateFp32OrQuantizedConstantWeights("TestOp", i8_valid,
                                                     /*is_external=*/false),
              IsOk());
}

}  // namespace
}  // namespace litert::tensor

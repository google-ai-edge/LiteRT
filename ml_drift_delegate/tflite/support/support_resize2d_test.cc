// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <string>
#include <tuple>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"

// These tests indirectly verify IsNodeSupported through GetOpsToReplace,
// which in turn uses GetSupportedNodes to leverage existing matchers.
//
// Note that the functionality of tflite::delegates::GraphPartitionHelper is
// intentionally NOT tested, as that's an implementation detail and that should
// be covered by its own unit tests.

namespace litert::ml_drift::ir {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::NotNull;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;

// GetSupportedNodes is module-private (support.cc) and not public (support.h),
// prioritizing encapsulation over test convenience.
extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

inline std::string TfLiteBuiltinCodeGetResize2dName(
    TfLiteBuiltinOperator code) {
  switch (code) {
    case kTfLiteBuiltinResizeBilinear:
      return "ResizeBilinear";
    case kTfLiteBuiltinResizeNearestNeighbor:
      return "ResizeNearestNeighbor";
    default:
      return "Unknown";
  }
}

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;
constexpr std::array<int, 4> kDefaultInputDims = {4, 1, 1, 4};
constexpr std::array<int, 4> kDefaultOutputDims = {2, 2, 2, 2};
constexpr TfLiteResizeBilinearParams kDefaultResizeBilinearParams = {
    .align_corners = false, .half_pixel_centers = false};
constexpr TfLiteResizeNearestNeighborParams
    kDefaultResizeNearestNeighborParams = {.align_corners = false,
                                           .half_pixel_centers = false};

// Test suite for resize 2d x supported version.
using SupportedVersionTest =
    TestWithParam<std::tuple<int, TfLiteBuiltinOperator>>;

TEST_P(SupportedVersionTest, Supports) {
  StubContextBuilder context_builder;
  const int version = std::get<0>(GetParam());
  const TfLiteBuiltinOperator builtin_code = std::get<1>(GetParam());
  const int src = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int dst = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  if (builtin_code == kTfLiteBuiltinResizeBilinear) {
    context_builder.SetOp(builtin_code, version,
                          /*params=*/&kDefaultResizeBilinearParams,
                          /*inputs=*/{src}, /*outputs=*/{dst});
  } else {
    context_builder.SetOp(builtin_code, version,
                          /*params=*/&kDefaultResizeNearestNeighborParams,
                          /*inputs=*/{src}, /*outputs=*/{dst});
  }
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    resize2dOps, SupportedVersionTest,
    ::testing::Combine(::testing::ValuesIn<int>({1, 2, 3}),
                       ::testing::ValuesIn<TfLiteBuiltinOperator>(
                           {kTfLiteBuiltinResizeBilinear,
                            kTfLiteBuiltinResizeNearestNeighbor})),
    [](const TestParamInfo<SupportedVersionTest::ParamType>& info) {
      return absl::StrCat(
          "V_", std::get<0>(info.param), "_",
          TfLiteBuiltinCodeGetResize2dName(std::get<1>(info.param)));
    });

// Test suite for resize 2d x unsupported version.
using UnsupportedVersionTest =
    TestWithParam<std::tuple<int, TfLiteBuiltinOperator>>;

TEST_P(UnsupportedVersionTest, Rejects) {
  StubContextBuilder context_builder;
  const int src = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int dst = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  if (std::get<1>(GetParam()) == kTfLiteBuiltinResizeBilinear) {
    context_builder.SetOp(std::get<1>(GetParam()), std::get<0>(GetParam()),
                          /*params=*/&kDefaultResizeBilinearParams,
                          /*inputs=*/{src}, /*outputs=*/{dst});
  } else {
    context_builder.SetOp(std::get<1>(GetParam()), std::get<0>(GetParam()),
                          /*params=*/&kDefaultResizeNearestNeighborParams,
                          /*inputs=*/{src}, /*outputs=*/{dst});
  }
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    resize2dOps, UnsupportedVersionTest,
    ::testing::Combine(::testing::ValuesIn<int>({0, 4}),
                       ::testing::ValuesIn<TfLiteBuiltinOperator>(
                           {kTfLiteBuiltinResizeBilinear,
                            kTfLiteBuiltinResizeNearestNeighbor})),
    [](const TestParamInfo<UnsupportedVersionTest::ParamType>& info) {
      return absl::StrCat(
          "V_", std::get<0>(info.param), "_",
          TfLiteBuiltinCodeGetResize2dName(std::get<1>(info.param)));
    });

using NumInputOutputTest = TestWithParam<TfLiteBuiltinOperator>;

// Tests for resize 2d ops for different number of I/O tensors.
TEST_P(NumInputOutputTest, Supports1Input1Output) {
  StubContextBuilder context_builder;
  const int src = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int dst = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  if (GetParam() == kTfLiteBuiltinResizeBilinear) {
    context_builder.SetOp(GetParam(), /*version=*/1,
                          /*params=*/&kDefaultResizeBilinearParams,
                          /*inputs=*/{src}, /*outputs=*/{dst});
  } else {
    context_builder.SetOp(GetParam(), /*version=*/1,
                          /*params=*/&kDefaultResizeNearestNeighborParams,
                          /*inputs=*/{src}, /*outputs=*/{dst});
  }
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(NumInputOutputTest, Rejects0Inputs) {
  StubContextBuilder context_builder;
  const int src = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  if (GetParam() == kTfLiteBuiltinResizeBilinear) {
    context_builder.SetOp(GetParam(), /*version=*/1,
                          /*params=*/&kDefaultResizeBilinearParams,
                          /*inputs=*/{}, /*outputs=*/{src});
  } else {
    context_builder.SetOp(GetParam(), /*version=*/1,
                          /*params=*/&kDefaultResizeNearestNeighborParams,
                          /*inputs=*/{}, /*outputs=*/{src});
  }
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects3Inputs) {
  StubContextBuilder context_builder;
  const int src = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int extra = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int dst = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  if (GetParam() == kTfLiteBuiltinResizeBilinear) {
    context_builder.SetOp(GetParam(), /*version=*/1,
                          /*params=*/&kDefaultResizeBilinearParams,
                          /*inputs=*/{src, extra, extra}, /*outputs=*/{dst});
  } else {
    context_builder.SetOp(GetParam(), /*version=*/1,
                          /*params=*/&kDefaultResizeNearestNeighborParams,
                          /*inputs=*/{src, extra, extra}, /*outputs=*/{dst});
  }
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects0Outputs) {
  StubContextBuilder context_builder;
  const int src = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  if (GetParam() == kTfLiteBuiltinResizeBilinear) {
    context_builder.SetOp(GetParam(), /*version=*/1,
                          /*params=*/&kDefaultResizeBilinearParams,
                          /*inputs=*/{}, /*outputs=*/{src});
  } else {
    context_builder.SetOp(GetParam(), /*version=*/1,
                          /*params=*/&kDefaultResizeNearestNeighborParams,
                          /*inputs=*/{}, /*outputs=*/{src});
  }
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects2Outputs) {
  StubContextBuilder context_builder;
  const int src = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int dst = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int extra =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  if (GetParam() == kTfLiteBuiltinResizeBilinear) {
    context_builder.SetOp(GetParam(), /*version=*/1,
                          /*params=*/&kDefaultResizeBilinearParams,
                          /*inputs=*/{src}, /*outputs=*/{dst, extra});
  } else {
    context_builder.SetOp(GetParam(), /*version=*/1,
                          /*params=*/&kDefaultResizeNearestNeighborParams,
                          /*inputs=*/{src}, /*outputs=*/{dst, extra});
  }
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    resize2dOps, NumInputOutputTest,
    ::testing::ValuesIn<TfLiteBuiltinOperator>(
        {kTfLiteBuiltinResizeBilinear, kTfLiteBuiltinResizeNearestNeighbor}),
    [](const TestParamInfo<NumInputOutputTest::ParamType>& info) {
      return absl::StrCat(TfLiteBuiltinCodeGetResize2dName(info.param));
    });

// Test suite for resize 2d ops x supported subject dtypes.
using SupportedDtypeTest =
    TestWithParam<std::tuple<TfLiteType, TfLiteBuiltinOperator>>;

TEST_P(SupportedDtypeTest, SupportsSupportedDtypes) {
  StubContextBuilder context_builder;
  const TfLiteType dtype = std::get<0>(GetParam());
  const TfLiteBuiltinOperator builtin_code = std::get<1>(GetParam());
  const int src = context_builder.AddTensor(dtype, kDefaultInputDims);
  const int dst = context_builder.AddTensor(dtype, kDefaultOutputDims);
  if (builtin_code == kTfLiteBuiltinResizeBilinear) {
    context_builder.SetOp(builtin_code, /*version=*/1,
                          /*params=*/&kDefaultResizeBilinearParams,
                          /*inputs=*/{src}, /*outputs=*/{dst});
  } else {
    context_builder.SetOp(builtin_code, /*version=*/1,
                          /*params=*/&kDefaultResizeNearestNeighborParams,
                          /*inputs=*/{src}, /*outputs=*/{dst});
  }
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedInput) {
  StubContextBuilder context_builder;
  const TfLiteType dtype = std::get<0>(GetParam());
  const TfLiteBuiltinOperator builtin_code = std::get<1>(GetParam());
  const int src = context_builder.AddTensor(kTfLiteNoType, kDefaultInputDims);
  const int dst = context_builder.AddTensor(dtype, kDefaultOutputDims);
  if (builtin_code == kTfLiteBuiltinResizeBilinear) {
    context_builder.SetOp(builtin_code, /*version=*/1,
                          /*params=*/&kDefaultResizeBilinearParams,
                          /*inputs=*/{src}, /*outputs=*/{dst});
  } else {
    context_builder.SetOp(builtin_code, /*version=*/1,
                          /*params=*/&kDefaultResizeNearestNeighborParams,
                          /*inputs=*/{src}, /*outputs=*/{dst});
  }
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedOutput) {
  StubContextBuilder context_builder;
  const TfLiteType dtype = std::get<0>(GetParam());
  const TfLiteBuiltinOperator builtin_code = std::get<1>(GetParam());
  const int src = context_builder.AddTensor(dtype, kDefaultInputDims);
  const int dst = context_builder.AddTensor(kTfLiteNoType, kDefaultOutputDims);
  if (builtin_code == kTfLiteBuiltinResizeBilinear) {
    context_builder.SetOp(builtin_code, /*version=*/1,
                          /*params=*/&kDefaultResizeBilinearParams,
                          /*inputs=*/{src}, /*outputs=*/{dst});
  } else {
    context_builder.SetOp(builtin_code, /*version=*/1,
                          /*params=*/&kDefaultResizeNearestNeighborParams,
                          /*inputs=*/{src}, /*outputs=*/{dst});
  }
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    resize2dOps, SupportedDtypeTest,
    ::testing::Combine(::testing::ValuesIn<TfLiteType>({
                           kTfLiteFloat32,
                           kTfLiteFloat16,
                           kTfLiteBFloat16,
                       }),
                       ::testing::ValuesIn<TfLiteBuiltinOperator>(
                           {kTfLiteBuiltinResizeBilinear,
                            kTfLiteBuiltinResizeNearestNeighbor})),
    [](const TestParamInfo<SupportedDtypeTest::ParamType>& info) {
      return absl::StrCat(
          TfLiteTypeGetName(std::get<0>(info.param)), "_",
          TfLiteBuiltinCodeGetResize2dName(std::get<1>(info.param)));
    });

// Test that we can reject all constant inputs
using ConstantTestSuite = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(ConstantTestSuite, RejectsConstantInput) {
  StubContextBuilder context_builder;
  const int src =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultInputDims);
  const int dst = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  if (GetParam() == kTfLiteBuiltinResizeBilinear) {
    context_builder.SetOp(GetParam(), /*version=*/1,
                          /*params=*/&kDefaultResizeBilinearParams,
                          /*inputs=*/{src}, /*outputs=*/{dst});
  } else {
    context_builder.SetOp(GetParam(), /*version=*/1,
                          /*params=*/&kDefaultResizeNearestNeighborParams,
                          /*inputs=*/{src}, /*outputs=*/{dst});
  }
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    resize2dOps, ConstantTestSuite,
    ::testing::ValuesIn<TfLiteBuiltinOperator>(
        {kTfLiteBuiltinResizeBilinear, kTfLiteBuiltinResizeNearestNeighbor}),
    [](const TestParamInfo<ConstantTestSuite::ParamType>& info) {
      return absl::StrCat(TfLiteBuiltinCodeGetResize2dName(info.param),
                          "_ConstantInput");
    });

// Test suite for checking the number of dimensions.
using DimsTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(DimsTest, Rejects5dInput) {
  StubContextBuilder context_builder;
  const int src = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4, 5});
  const int dst = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(GetParam(), /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{src}, /*outputs=*/{dst});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(DimsTest, Rejects5dOutput) {
  StubContextBuilder context_builder;
  const int src = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int dst = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4, 5});
  context_builder.SetOp(GetParam(), /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{src}, /*outputs=*/{dst});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(resize2dOps, DimsTest,
                         ::testing::ValuesIn<TfLiteBuiltinOperator>(
                             {kTfLiteBuiltinResizeBilinear,
                              kTfLiteBuiltinResizeNearestNeighbor}),
                         [](const TestParamInfo<DimsTest::ParamType>& info) {
                           return absl::StrCat(
                               TfLiteBuiltinCodeGetResize2dName(info.param),
                               "_DimsTest");
                         });

// Test suite for resize 2d ops x params
class ParamsTest : public testing::Test {};
TEST_F(ParamsTest, RejectsNullBilinearParams) {
  StubContextBuilder context_builder;
  const int src = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int dst = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinResizeBilinear, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{src}, /*outputs=*/{dst});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ParamsTest, RejectsBilinearBadParams) {
  TfLiteResizeBilinearParams bad_params = {.align_corners = true,
                                           .half_pixel_centers = true};
  StubContextBuilder context_builder;
  const int src = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int dst = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinResizeBilinear, /*version=*/1,
                        /*params=*/&bad_params,
                        /*inputs=*/{src}, /*outputs=*/{dst});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ParamsTest, RejectsNullNNParams) {
  StubContextBuilder context_builder;
  const int src = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int dst = context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinResizeNearestNeighbor, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{src}, /*outputs=*/{dst});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir

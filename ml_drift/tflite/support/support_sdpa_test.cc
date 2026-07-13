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
#include <string_view>
#include <tuple>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/c_api_types.h"
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
using ::testing::ValuesIn;

extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;
constexpr std::array<int, 4> kDefaultQDims = {1, 2, 3, 4};
constexpr std::array<int, 4> kDefaultKDims = {1, 2, 3, 4};
constexpr std::array<int, 4> kDefaultVDims = {1, 2, 3, 4};
constexpr std::array<int, 4> kDefaultMaskDims = {1, 1, 1, 2};
constexpr std::array<int, 4> kDefaultOutputDims = {1, 2, 3, 4};
constexpr TfLiteStablehloCompositeParams kDefaultSdpaParams = {
    .name = "odml.scaled_dot_product_attention",
};

struct VersionTestCase {
  int version = 0;
};

// Test suite for sdpa logical ops x supported version.
using SupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(SupportedVersionTest, SupportedVersions) {
  StubContextBuilder context_builder;
  const int q = context_builder.AddTensor(kDefaultDtype, kDefaultQDims);
  const int k = context_builder.AddTensor(kDefaultDtype, kDefaultKDims);
  const int v = context_builder.AddTensor(kDefaultDtype, kDefaultVDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, GetParam().version,
                        /*params=*/&kDefaultSdpaParams,
                        /*inputs=*/{q, k, v},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    SdpaOps, SupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {1},  // only supported version
    }),
    [](const TestParamInfo<SupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param.version);
    });

// Test suite for sdpa ops x unsupported version.
using UnsupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(UnsupportedVersionTest, RejectedVersions) {
  StubContextBuilder context_builder;
  const int q = context_builder.AddTensor(kDefaultDtype, kDefaultQDims);
  const int k = context_builder.AddTensor(kDefaultDtype, kDefaultKDims);
  const int v = context_builder.AddTensor(kDefaultDtype, kDefaultVDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, GetParam().version,
                        /*params=*/&kDefaultSdpaParams,
                        /*inputs=*/{q, k, v},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    SdpaOps, UnsupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {0},  // min-1
        {2},  // max+1
    }),
    [](const TestParamInfo<UnsupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param.version);
    });

using NumInputOutputTests = testing::Test;
// Tests for sdpa ops for different number of I/O tensors.
TEST_F(NumInputOutputTests, Supports3Inputs1Output) {
  StubContextBuilder context_builder;
  const int q = context_builder.AddTensor(kDefaultDtype, kDefaultQDims);
  const int k = context_builder.AddTensor(kDefaultDtype, kDefaultKDims);
  const int v = context_builder.AddTensor(kDefaultDtype, kDefaultVDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite,
                        /*version=*/1,
                        /*params=*/&kDefaultSdpaParams,
                        /*inputs=*/{q, k, v},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(NumInputOutputTests, Supports4Inputs1Output) {
  StubContextBuilder context_builder;
  const int q = context_builder.AddTensor(kDefaultDtype, kDefaultQDims);
  const int k = context_builder.AddTensor(kDefaultDtype, kDefaultKDims);
  const int v = context_builder.AddTensor(kDefaultDtype, kDefaultVDims);
  const int mask = context_builder.AddTensor(kDefaultDtype, kDefaultMaskDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&kDefaultSdpaParams,
                        /*inputs=*/{q, k, v, mask},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(NumInputOutputTests, Rejects2Inputs) {
  StubContextBuilder context_builder;
  const int q = context_builder.AddTensor(kDefaultDtype, kDefaultQDims);
  const int k = context_builder.AddTensor(kDefaultDtype, kDefaultKDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&kDefaultSdpaParams, /*inputs=*/{q, k},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTests, Rejects5Inputs) {
  StubContextBuilder context_builder;
  const int q = context_builder.AddTensor(kDefaultDtype, kDefaultQDims);
  const int k = context_builder.AddTensor(kDefaultDtype, kDefaultKDims);
  const int v = context_builder.AddTensor(kDefaultDtype, kDefaultVDims);
  const int mask = context_builder.AddTensor(kDefaultDtype, kDefaultMaskDims);
  const int extra = context_builder.AddTensor(kDefaultDtype, kDefaultQDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&kDefaultSdpaParams,
                        /*inputs=*/{q, k, v, mask, extra},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTests, Rejects0Outputs) {
  StubContextBuilder context_builder;
  const int q = context_builder.AddTensor(kDefaultDtype, kDefaultQDims);
  const int k = context_builder.AddTensor(kDefaultDtype, kDefaultKDims);
  const int v = context_builder.AddTensor(kDefaultDtype, kDefaultVDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&kDefaultSdpaParams,
                        /*inputs=*/{q, k, v}, /*outputs=*/{});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTests, Rejects2Outputs) {
  StubContextBuilder context_builder;
  const int q = context_builder.AddTensor(kDefaultDtype, kDefaultQDims);
  const int k = context_builder.AddTensor(kDefaultDtype, kDefaultKDims);
  const int v = context_builder.AddTensor(kDefaultDtype, kDefaultVDims);
  const int output1 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output2 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&kDefaultSdpaParams,
                        /*inputs=*/{q, k, v}, /*outputs=*/{output1, output2});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for sdpa op x supported subject dtypes.
using SupportedDtypeTest = TestWithParam<TfLiteType>;

TEST_P(SupportedDtypeTest, SupportedDtypes) {
  StubContextBuilder context_builder;
  const int q = context_builder.AddTensor(GetParam(), kDefaultQDims);
  const int k = context_builder.AddTensor(GetParam(), kDefaultKDims);
  const int v = context_builder.AddTensor(GetParam(), kDefaultVDims);
  const int mask = context_builder.AddTensor(GetParam(), kDefaultMaskDims);
  const int output = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&kDefaultSdpaParams,
                        /*inputs=*/{q, k, v, mask},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    SdpaOps, SupportedDtypeTest,
    ValuesIn<TfLiteType>({
        kTfLiteBFloat16,
        kTfLiteFloat16,
        kTfLiteFloat32,
    }),
    [](const TestParamInfo<SupportedDtypeTest::ParamType>& info) {
      return TfLiteTypeGetName(info.param);
    });

// Define a test fixture that is parameterized by a tuple containing:
// 1. The valid TfLiteType for the tensors.
// 2. An integer index representing the tensor to be invalidated.
using SdpaUnsupportedDtypeTest =
    ::testing::TestWithParam<std::tuple<TfLiteType, int>>;
TEST_P(SdpaUnsupportedDtypeTest, RejectsUnsupportedTensor) {
  // Get the valid dtype and the index of the tensor to make unsupported.
  const auto& [supported_type, unsupported_idx] = GetParam();

  // Vector of dtypes for each tensor, defaulted to the supported type.
  std::vector<TfLiteType> dtypes(5, supported_type);
  // Set the specified tensor's dtype to an unsupported type based on the index.
  dtypes[unsupported_idx] = kTfLiteNoType;

  StubContextBuilder context_builder;
  const int q = context_builder.AddTensor(dtypes[0], kDefaultQDims);
  const int k = context_builder.AddTensor(dtypes[1], kDefaultKDims);
  const int v = context_builder.AddTensor(dtypes[2], kDefaultVDims);
  const int mask = context_builder.AddTensor(dtypes[3], kDefaultMaskDims);
  const int output = context_builder.AddTensor(dtypes[4], kDefaultOutputDims);

  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&kDefaultSdpaParams,
                        /*inputs=*/{q, k, v, mask},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

constexpr std::array<std::string_view, 5> kTensorNames = {"Q", "K", "V", "Mask",
                                                          "Output"};
INSTANTIATE_TEST_SUITE_P(
    SdpaOps, SdpaUnsupportedDtypeTest,
    // Combine the supported dtypes with a range of indices (0-4) for each
    // tensor.
    ::testing::Combine(::testing::ValuesIn<TfLiteType>({
                           kTfLiteBFloat16,
                           kTfLiteFloat16,
                           kTfLiteFloat32,
                       }),
                       ::testing::Range(0, 5)),
    // Custom name generator for clear test output.
    [](const testing::TestParamInfo<SdpaUnsupportedDtypeTest::ParamType>&
           info) {
      const TfLiteType dtype = std::get<0>(info.param);
      const int index = std::get<1>(info.param);
      const std::string_view tensor_name =
          index < kTensorNames.size() ? kTensorNames[index] : "Unknown";
      return absl::StrCat(TfLiteTypeGetName(dtype), "_RejectsUnsupported",
                          tensor_name);
    });

// Test that we can reject constant input
using ConstantTestSuite = testing::Test;
TEST_F(ConstantTestSuite, Rejects3ConstInputs) {
  StubContextBuilder context_builder;
  const int q = context_builder.AddConstTensor(kDefaultDtype, kDefaultQDims);
  const int k = context_builder.AddConstTensor(kDefaultDtype, kDefaultKDims);
  const int v = context_builder.AddConstTensor(kDefaultDtype, kDefaultVDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&kDefaultSdpaParams,
                        /*inputs=*/{q, k, v},
                        /*outputs=*/{output});

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ConstantTestSuite, Rejects4ConstInputs) {
  StubContextBuilder context_builder;
  const int q = context_builder.AddConstTensor(kDefaultDtype, kDefaultQDims);
  const int k = context_builder.AddConstTensor(kDefaultDtype, kDefaultKDims);
  const int v = context_builder.AddConstTensor(kDefaultDtype, kDefaultVDims);
  const int mask =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultMaskDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&kDefaultSdpaParams,
                        /*inputs=*/{q, k, v, mask},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for checking the number of dimensions.
// Define a test fixture that is parameterized by a tuple containing:
// 1. An integer index representing the tensor to be invalidated (0-4).
// 2. A vector of ints representing the invalid dimensions.
using DimsTest = ::testing::TestWithParam<std::tuple<int, std::vector<int>>>;
TEST_P(DimsTest, RejectsInvalidDims) {
  // Get the index of the tensor to invalidate and the invalid dimensions.
  const int invalid_tensor_idx = std::get<0>(GetParam());
  const std::vector<int>& invalid_dims = std::get<1>(GetParam());

  // Create a list of default dimensions for all tensors.
  std::vector<std::vector<int>> all_dims = {
      {kDefaultQDims.begin(), kDefaultQDims.end()},
      {kDefaultKDims.begin(), kDefaultKDims.end()},
      {kDefaultVDims.begin(), kDefaultVDims.end()},
      {kDefaultMaskDims.begin(), kDefaultMaskDims.end()},
      {kDefaultOutputDims.begin(), kDefaultOutputDims.end()}};
  // Overwrite the dimensions for the specific tensor being tested.
  all_dims[invalid_tensor_idx] = invalid_dims;

  StubContextBuilder context_builder;
  const int q = context_builder.AddTensor(kDefaultDtype, all_dims[0]);
  const int k = context_builder.AddTensor(kDefaultDtype, all_dims[1]);
  const int v = context_builder.AddTensor(kDefaultDtype, all_dims[2]);
  const int mask = context_builder.AddTensor(kDefaultDtype, all_dims[3]);
  const int output = context_builder.AddTensor(kDefaultDtype, all_dims[4]);

  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&kDefaultSdpaParams,
                        /*inputs=*/{q, k, v, mask},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    SdpaOps, DimsTest,
    // Create a test for every combination of tensor index (0-4) and invalid
    // dimension (0-D and 5-D).
    ::testing::Combine(::testing::Range(0, 5),
                       ::testing::ValuesIn({std::vector<int>{},
                                            {1, 2, 3, 4, 1}})),
    // Custom name generator for clear test output.
    [](const testing::TestParamInfo<DimsTest::ParamType>& info) {
      const int index = std::get<0>(info.param);
      const std::vector<int>& dims = std::get<1>(info.param);
      const std::string_view tensor_name =
          index < kTensorNames.size() ? kTensorNames[index] : "Unknown";
      return absl::StrCat("Rejects", dims.empty() ? "0d" : "5d", tensor_name);
    });

using ShapeTest = testing::Test;
TEST_F(ShapeTest, RejectsQAndKChannelMismatch) {
  StubContextBuilder context_builder;
  const int q = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int k = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 5});
  const int v = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int mask = context_builder.AddTensor(kDefaultDtype, {1, 1, 1, 2});
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&kDefaultSdpaParams,
                        /*inputs=*/{q, k, v, mask},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ShapeTest, RejectsKAndVChannelMismatch) {
  StubContextBuilder context_builder;
  const int q = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int k = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int v = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 5});
  const int mask = context_builder.AddTensor(kDefaultDtype, {1, 1, 1, 2});
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&kDefaultSdpaParams,
                        /*inputs=*/{q, k, v, mask},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ShapeTest, RejectsKAndVShapeMismatch) {
  StubContextBuilder context_builder;
  const int q = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int k = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int v = context_builder.AddTensor(kDefaultDtype, {1, 3, 3, 4});
  const int mask = context_builder.AddTensor(kDefaultDtype, {1, 1, 1, 2});
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&kDefaultSdpaParams,
                        /*inputs=*/{q, k, v, mask},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ShapeTest, RejectsMaskChAndKHeightMismatch) {
  StubContextBuilder context_builder;
  const int q = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int k = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int v = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  const int mask = context_builder.AddTensor(kDefaultDtype, {1, 1, 1, 3});
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4});
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&kDefaultSdpaParams,
                        /*inputs=*/{q, k, v, mask},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir

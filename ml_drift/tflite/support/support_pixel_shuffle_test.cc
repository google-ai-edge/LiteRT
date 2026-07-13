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
#include <cstdint>
#include <string_view>
#include <tuple>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/support/stub_context.h"
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

// GetSupportedNodes is module-private (support.cc) and not public (support.h),
// prioritizing encapsulation over test convenience.
extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

std::vector<uint8_t> CreatePixelShuffleBuffer() {
  flexbuffers::Builder fbb;

  // Start building the root map. The lambda function populates the map.
  fbb.Map([&]() { fbb.Int("num_groups", 3); });

  fbb.Finish();
  return fbb.GetBuffer();
}

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;
constexpr std::array<int, 4> kDefaultInputDims = {1, 9, 4, 4};
constexpr std::array<int, 4> kDefaultOutputDims = {1, 1, 12, 12};

// Test suite for pixel shuffle ops x supported version.
TEST(SupportedVersionTest, SupportedVersions) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreatePixelShuffleBuffer();
  context_builder.SetOpCustom("custom_call.pixel_shuffle", /*version=*/1,
                              /*params=*/nullptr, {input}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

// Test suite for pixel shuffle ops x unsupported version.
using UnsupportedVersionTest = TestWithParam<int>;
TEST_P(UnsupportedVersionTest, RejectedVersions) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreatePixelShuffleBuffer();
  context_builder.SetOpCustom("custom_call.pixel_shuffle", GetParam(),
                              /*params=*/nullptr, {input}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    PixelShuffle, UnsupportedVersionTest,
    ValuesIn<int>({
        0,  // min-1
        2,  // max+1
    }),
    [](const TestParamInfo<UnsupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param);
    });

// Tests for pixel shuffle ops for different number of I/O tensors.
TEST(NumInputOutputTests, Supports1Input1Output) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreatePixelShuffleBuffer();
  context_builder.SetOpCustom("custom_call.pixel_shuffle", /*version=*/1,
                              /*params=*/nullptr, {input}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST(NumInputOutputTests, Rejects0Inputs) {
  StubContextBuilder context_builder;
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreatePixelShuffleBuffer();
  context_builder.SetOpCustom("custom_call.pixel_shuffle", /*version=*/1,
                              /*params=*/nullptr, {}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(NumInputOutputTests, Rejects2Inputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int extra = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreatePixelShuffleBuffer();
  context_builder.SetOpCustom("custom_call.pixel_shuffle", /*version=*/1,
                              /*params=*/nullptr, {input, extra}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(NumInputOutputTests, Rejects0Outputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);

  const std::vector<uint8_t> buffer = CreatePixelShuffleBuffer();
  context_builder.SetOpCustom("custom_call.pixel_shuffle", /*version=*/1,
                              /*params=*/nullptr, {input}, {});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(NumInputOutputTests, Rejects2Outputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);

  const int output1 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output2 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreatePixelShuffleBuffer();
  context_builder.SetOpCustom("custom_call.pixel_shuffle", /*version=*/1,
                              /*params=*/nullptr, {input}, {output1, output2});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for pixel shuffle op x supported subject dtypes.
using SupportedDtypeTest = TestWithParam<TfLiteType>;

TEST_P(SupportedDtypeTest, SupportedDtypes) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int output = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreatePixelShuffleBuffer();
  context_builder.SetOpCustom("custom_call.pixel_shuffle", /*version=*/1,
                              /*params=*/nullptr, {input}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    PixelShuffle, SupportedDtypeTest,
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
using PixelShuffleUnsupportedDtypeTest =
    ::testing::TestWithParam<std::tuple<TfLiteType, int>>;
TEST_P(PixelShuffleUnsupportedDtypeTest, RejectsUnsupportedTensor) {
  // Get the valid dtype and the index of the tensor to make unsupported.
  const auto& [supported_type, unsupported_idx] = GetParam();

  // Vector of dtypes for each tensor, defaulted to the supported type.
  std::vector<TfLiteType> dtypes(2, supported_type);
  // Set the specified tensor's dtype to an unsupported type based on the index
  dtypes[unsupported_idx] = kTfLiteNoType;

  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(dtypes[0], kDefaultInputDims);
  const int output = context_builder.AddTensor(dtypes[1], kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreatePixelShuffleBuffer();
  context_builder.SetOpCustom("custom_call.pixel_shuffle", /*version=*/1,
                              /*params=*/nullptr, {input}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

constexpr std::array<std::string_view, 4> kTensorNames = {"Input", "Output"};
INSTANTIATE_TEST_SUITE_P(
    PixelShuffle, PixelShuffleUnsupportedDtypeTest,
    // Combine the supported dtypes with a range of indices (0-4) for each
    // tensor.
    ::testing::Combine(::testing::ValuesIn<TfLiteType>({
                           kTfLiteBFloat16,
                           kTfLiteFloat16,
                           kTfLiteFloat32,
                       }),
                       ::testing::Range(0, 2)),
    // Custom name generator for clear test output.
    [](const testing::TestParamInfo<
        PixelShuffleUnsupportedDtypeTest::ParamType>& info) {
      const TfLiteType dtype = std::get<0>(info.param);
      const int index = std::get<1>(info.param);
      const std::string_view tensor_name =
          index < kTensorNames.size() ? kTensorNames[index] : "Unknown";
      return absl::StrCat(TfLiteTypeGetName(dtype), "_RejectsUnsupported",
                          tensor_name);
    });

// Test that we can reject constant input
TEST(ConstantTestSuite, RejectsConstInput) {
  StubContextBuilder context_builder;
  const int input =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreatePixelShuffleBuffer();
  context_builder.SetOpCustom("custom_call.pixel_shuffle", /*version=*/1,
                              /*params=*/nullptr, {input}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for checking the number of dimensions.
TEST(DimsTest, Rejects6dInput) {
  StubContextBuilder context_builder;
  const int input =
      context_builder.AddTensor(kDefaultDtype, {1, 1, 1, 9, 4, 4});
  const int output =
      context_builder.AddTensor(kDefaultDtype, {1, 1, 1, 1, 12, 12});

  const std::vector<uint8_t> buffer = CreatePixelShuffleBuffer();
  context_builder.SetOpCustom("custom_call.pixel_shuffle", /*version=*/1,
                              /*params=*/nullptr, {input}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, Rejects2dInput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {4, 4});
  const int output = context_builder.AddTensor(kDefaultDtype, {12, 12});

  const std::vector<uint8_t> buffer = CreatePixelShuffleBuffer();
  context_builder.SetOpCustom("custom_call.pixel_shuffle", /*version=*/1,
                              /*params=*/nullptr, {input}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(DimsTest, RejectsDiffDims) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {9, 4, 4});
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 12, 12, 1});

  const std::vector<uint8_t> buffer = CreatePixelShuffleBuffer();
  context_builder.SetOpCustom("custom_call.pixel_shuffle", /*version=*/1,
                              /*params=*/nullptr, {input}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for checking the params.
TEST(ParamsTest, RejectsNullParams) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  context_builder.SetOpCustom("custom_call.pixel_shuffle", /*version=*/1,
                              /*params=*/nullptr, {input}, {output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(ParamsTest, RejectsNoNumGroups) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  // Start building the root map. Note no num_groups.
  flexbuffers::Builder fbb;
  fbb.Map([&]() {});
  fbb.Finish();
  const std::vector<uint8_t> buffer = fbb.GetBuffer();

  context_builder.SetOpCustom("custom_call.pixel_shuffle", /*version=*/1,
                              /*params=*/nullptr, {input}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(ParamsTest, RejectsNegNumGroups) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  // Build root map with negative num_groups.
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Int("num_groups", -1); });
  fbb.Finish();
  const std::vector<uint8_t> buffer = fbb.GetBuffer();

  context_builder.SetOpCustom("custom_call.pixel_shuffle", /*version=*/1,
                              /*params=*/nullptr, {input}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(ParamsTest, RejectsBadHeight) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1, 10, 4, 4});
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 1, 12, 12});
  // (bad 2nd dim): 1 * num_groups^2 != 10

  const std::vector<uint8_t> buffer = CreatePixelShuffleBuffer();
  context_builder.SetOpCustom("custom_call.pixel_shuffle", /*version=*/1,
                              /*params=*/nullptr, {input}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(ParamsTest, RejectsBadWidth) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1, 9, 4, 4});
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 1, 13, 12});
  // (bad 3rd dim): 4 * num_groups != 13

  const std::vector<uint8_t> buffer = CreatePixelShuffleBuffer();
  context_builder.SetOpCustom("custom_call.pixel_shuffle", /*version=*/1,
                              /*params=*/nullptr, {input}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST(ParamsTest, RejectsBadChannels) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1, 9, 4, 4});
  const int output = context_builder.AddTensor(kDefaultDtype, {1, 1, 12, 13});
  // (bad last dim): 4 * num_groups != 13

  const std::vector<uint8_t> buffer = CreatePixelShuffleBuffer();
  context_builder.SetOpCustom("custom_call.pixel_shuffle", /*version=*/1,
                              /*params=*/nullptr, {input}, {output});
  context_builder.SetOpCustomInitialData(buffer.data(), buffer.size());
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir

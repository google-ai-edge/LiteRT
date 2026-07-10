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

// GetSupportedNodes is module-private (support.cc) and not public (support.h),
// prioritizing encapsulation over test convenience.
extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

std::vector<uint8_t> CreateGroupNormBuffer() {
  flexbuffers::Builder fbb;

  // Start building the root map. The lambda function populates the map.
  fbb.Map([&]() {
    fbb.Int("num_groups", 32);
    fbb.Float("epsilon", 1.0e-5f);
    fbb.Int("channel_axis", 3);  // last axis

    fbb.Map("_TENSOR_V1_reduction_axes", [&]() {
      fbb.Vector("TENSOR_DATA", [&]() {
        fbb.Add(3);  // last axis
      });
    });
  });

  fbb.Finish();
  return fbb.GetBuffer();
}

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;
constexpr std::array<int, 4> kDefaultInputDims = {1, 2, 3, 4};
constexpr std::array<int, 1> kDefaultGreekDims = {4};
constexpr std::array<int, 4> kDefaultOutputDims = {1, 2, 3, 4};

struct VersionTestCase {
  int version = 0;
};

// Test suite for group norm ops x supported version.
using SupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(SupportedVersionTest, SupportedVersions) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateGroupNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, GetParam().version,
                        /*params=*/&params,
                        /*inputs=*/{input},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    GroupNorm, SupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {1},  // only supported version
    }),
    [](const TestParamInfo<SupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param.version);
    });

// Test suite for group norm ops x unsupported version.
using UnsupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(UnsupportedVersionTest, RejectedVersions) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateGroupNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, GetParam().version,
                        /*params=*/&params,
                        /*inputs=*/{input},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    GroupNorm, UnsupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {0},  // min-1
        {2},  // max+1
    }),
    [](const TestParamInfo<UnsupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param.version);
    });

using NumInputOutputTests = testing::Test;
// Tests for group norm ops for different number of I/O tensors.
TEST_F(NumInputOutputTests, Supports2Inputs1Output) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int gamma = context_builder.AddTensor(kDefaultDtype, kDefaultGreekDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateGroupNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite,
                        /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, gamma},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(NumInputOutputTests, Supports3Inputs1Output) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int gamma = context_builder.AddTensor(kDefaultDtype, kDefaultGreekDims);
  const int beta = context_builder.AddTensor(kDefaultDtype, kDefaultGreekDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateGroupNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, gamma, beta},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(NumInputOutputTests, Rejects0Inputs) {
  StubContextBuilder context_builder;
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateGroupNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTests, Rejects4Inputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int gamma = context_builder.AddTensor(kDefaultDtype, kDefaultGreekDims);
  const int beta = context_builder.AddTensor(kDefaultDtype, kDefaultGreekDims);
  const int extra = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateGroupNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, gamma, beta, extra},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTests, Rejects0Outputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);

  const std::vector<uint8_t> buffer = CreateGroupNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input}, /*outputs=*/{});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTests, Rejects2Outputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);

  const int output1 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const int output2 =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateGroupNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input}, /*outputs=*/{output1, output2});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for group norm op x supported subject dtypes.
using SupportedDtypeTest = TestWithParam<TfLiteType>;

TEST_P(SupportedDtypeTest, SupportedDtypes) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int gamma = context_builder.AddTensor(GetParam(), kDefaultGreekDims);
  const int beta = context_builder.AddTensor(GetParam(), kDefaultGreekDims);
  const int output = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateGroupNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, gamma, beta},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    GroupNorm, SupportedDtypeTest,
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
using GroupNormUnsupportedDtypeTest =
    ::testing::TestWithParam<std::tuple<TfLiteType, int>>;
TEST_P(GroupNormUnsupportedDtypeTest, RejectsUnsupportedTensor) {
  // Get the valid dtype and the index of the tensor to make unsupported.
  const auto& [supported_type, unsupported_idx] = GetParam();

  // Vector of dtypes for each tensor, defaulted to the supported type.
  std::vector<TfLiteType> dtypes(4, supported_type);
  // Set the specified tensor's dtype to an unsupported type based on the index
  dtypes[unsupported_idx] = kTfLiteNoType;

  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(dtypes[0], kDefaultInputDims);
  const int gamma = context_builder.AddTensor(dtypes[1], kDefaultGreekDims);
  const int beta = context_builder.AddTensor(dtypes[2], kDefaultGreekDims);
  const int output = context_builder.AddTensor(dtypes[3], kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateGroupNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, gamma, beta},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

constexpr std::array<std::string_view, 4> kTensorNames = {"Input", "Gamma",
                                                          "Beta", "Output"};
INSTANTIATE_TEST_SUITE_P(
    GroupNorm, GroupNormUnsupportedDtypeTest,
    // Combine the supported dtypes with a range of indices (0-4) for each
    // tensor.
    ::testing::Combine(::testing::ValuesIn<TfLiteType>({
                           kTfLiteBFloat16,
                           kTfLiteFloat16,
                           kTfLiteFloat32,
                       }),
                       ::testing::Range(0, 4)),
    // Custom name generator for clear test output.
    [](const testing::TestParamInfo<GroupNormUnsupportedDtypeTest::ParamType>&
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
TEST_F(ConstantTestSuite, RejectsConstInput) {
  StubContextBuilder context_builder;
  const int input =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateGroupNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input},
                        /*outputs=*/{output});

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for checking the number of dimensions.
class DimsTest : public testing::Test {};

TEST_F(DimsTest, Rejects3dInput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1, 2, 3});
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateGroupNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, Rejects0dGamma) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int gamma = context_builder.AddTensor(kDefaultDtype, {4, 1});
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateGroupNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, gamma},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, RejectsGammaShapeMismatch) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int gamma = context_builder.AddTensor(kDefaultDtype, {3});
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateGroupNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, gamma},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, Rejects0dBeta) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int gamma = context_builder.AddTensor(kDefaultDtype, kDefaultGreekDims);
  const int beta = context_builder.AddTensor(kDefaultDtype, {4, 1});
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateGroupNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, gamma, beta},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, RejectsBetaShapeMismatch) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int gamma = context_builder.AddTensor(kDefaultDtype, kDefaultGreekDims);
  const int beta = context_builder.AddTensor(kDefaultDtype, {3});
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateGroupNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, gamma, beta},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// Test suite for checking the params.
class ParamsTest : public testing::Test {};
TEST_F(ParamsTest, RejectsNullParams) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/nullptr,
                        /*inputs=*/{input},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ParamsTest, Rejects2ElemVector) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  flexbuffers::Builder fbb;

  // Start building the root map. The lambda function populates the map.
  fbb.Map([&]() {
    fbb.Int("num_groups", 32);
    fbb.Float("epsilon", 1.0e-5f);
    fbb.Int("channel_axis", 3);  // last axis

    fbb.Map("_TENSOR_V1_reduction_axes", [&]() {
      fbb.Vector("TENSOR_DATA", [&]() {
        fbb.Add(3);  // last axis
        fbb.Add(3);  // bad extra value
      });
    });
  });
  fbb.Finish();
  const std::vector<uint8_t> buffer = fbb.GetBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};

  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ParamsTest, RejectsBadTensorDataAxis) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  flexbuffers::Builder fbb;

  // Start building the root map. The lambda function populates the map.
  fbb.Map([&]() {
    fbb.Int("num_groups", 32);
    fbb.Float("epsilon", 1.0e-5f);
    fbb.Int("channel_axis", 3);  // last axis

    fbb.Map("_TENSOR_V1_reduction_axes", [&]() {
      fbb.Vector("TENSOR_DATA", [&]() {
        fbb.Add(2);  // should be 3
      });
    });
  });
  fbb.Finish();
  const std::vector<uint8_t> buffer = fbb.GetBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};

  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ParamsTest, RejectsBadChannelAxis) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  flexbuffers::Builder fbb;

  // Start building the root map. The lambda function populates the map.
  fbb.Map([&]() {
    fbb.Int("num_groups", 32);
    fbb.Float("epsilon", 1.0e-5f);
    fbb.Int("channel_axis", 2);  // NOT last axis

    fbb.Map("_TENSOR_V1_reduction_axes", [&]() {
      fbb.Vector("TENSOR_DATA", [&]() {
        fbb.Add(3);  // last axis
      });
    });
  });
  fbb.Finish();
  const std::vector<uint8_t> buffer = fbb.GetBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};

  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ParamsTest, RejectsNoNumGroups) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  flexbuffers::Builder fbb;

  // Start building the root map. The lambda function populates the map.
  fbb.Map([&]() {
    fbb.Float("epsilon", 1.0e-5f);
    fbb.Int("channel_axis", 3);  // last axis

    fbb.Map("_TENSOR_V1_reduction_axes", [&]() {
      fbb.Vector("TENSOR_DATA", [&]() {
        fbb.Add(3);  // last axis
      });
    });
  });
  fbb.Finish();
  const std::vector<uint8_t> buffer = fbb.GetBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};

  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(ParamsTest, RejectsNoEpsilon) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  flexbuffers::Builder fbb;

  // Start building the root map. The lambda function populates the map.
  fbb.Map([&]() {
    fbb.Int("num_groups", 32);
    fbb.Int("channel_axis", 3);  // last axis

    fbb.Map("_TENSOR_V1_reduction_axes", [&]() {
      fbb.Vector("TENSOR_DATA", [&]() {
        fbb.Add(3);  // last axis
      });
    });
  });
  fbb.Finish();
  const std::vector<uint8_t> buffer = fbb.GetBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.group_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};

  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir

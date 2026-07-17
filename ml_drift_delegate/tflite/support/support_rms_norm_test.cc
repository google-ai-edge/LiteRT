// Copyright 2026 Google LLC.
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
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

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
constexpr std::array<int, 4> kDefaultInputDims = {1, 2, 3, 4};
constexpr std::array<int, 4> kDefaultOutputDims = {1, 2, 3, 4};
constexpr std::array<int, 1> kDefaultScaleDims = {4};

std::vector<uint8_t> CreateRmsNormBuffer(bool missing_epsilon = false) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    if (!missing_epsilon) {
      fbb.Float("epsilon", 1e-5f);
    }
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

struct VersionTestCase {
  int version = 0;
};

using SupportedVersionTest = TestWithParam<VersionTestCase>;
TEST_P(SupportedVersionTest, SupportedVersions) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int scale =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultScaleDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateRmsNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.rms_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, GetParam().version,
                        /*params=*/&params,
                        /*inputs=*/{input, scale},
                        /*outputs=*/{output});

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    RmsNorm, SupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {1},
    }),
    [](const TestParamInfo<SupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param.version);
    });

using UnsupportedVersionTest = TestWithParam<VersionTestCase>;
TEST_P(UnsupportedVersionTest, RejectedVersions) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int scale =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultScaleDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateRmsNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.rms_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, GetParam().version,
                        /*params=*/&params,
                        /*inputs=*/{input, scale},
                        /*outputs=*/{output});

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    RmsNorm, UnsupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {0},
        {2},
    }),
    [](const TestParamInfo<UnsupportedVersionTest::ParamType>& info) {
      return absl::StrCat("V_", info.param.version);
    });

using NumInputOutputTests = testing::Test;
TEST_F(NumInputOutputTests, Supports1Input) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateRmsNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.rms_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite,
                        /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(NumInputOutputTests, Supports2Inputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int scale =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultScaleDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateRmsNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.rms_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, scale},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_F(NumInputOutputTests, Rejects0Inputs) {
  StubContextBuilder context_builder;
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateRmsNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.rms_norm",
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

TEST_F(NumInputOutputTests, Rejects3Inputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int scale =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultScaleDims);
  const int extra = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateRmsNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.rms_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, scale, extra},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(NumInputOutputTests, Rejects0Outputs) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);

  const std::vector<uint8_t> buffer = CreateRmsNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.rms_norm",
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
  const std::vector<uint8_t> buffer = CreateRmsNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.rms_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input}, /*outputs=*/{output1, output2});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

using SupportedDtypeTest = TestWithParam<TfLiteType>;
TEST_P(SupportedDtypeTest, SupportedDtypes) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(GetParam(), kDefaultInputDims);
  const int scale =
      context_builder.AddConstTensor(GetParam(), kDefaultScaleDims);
  const int output = context_builder.AddTensor(GetParam(), kDefaultOutputDims);
  const std::vector<uint8_t> buffer = CreateRmsNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.rms_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, scale},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

INSTANTIATE_TEST_SUITE_P(
    RmsNorm, SupportedDtypeTest,
    ValuesIn<TfLiteType>({
        kTfLiteBFloat16,
        kTfLiteFloat16,
        kTfLiteFloat32,
    }),
    [](const TestParamInfo<SupportedDtypeTest::ParamType>& info) {
      return TfLiteTypeGetName(info.param);
    });

using RmsNormUnsupportedDtypeTest =
    ::testing::TestWithParam<std::tuple<TfLiteType, int>>;
TEST_P(RmsNormUnsupportedDtypeTest, RejectsUnsupportedTensor) {
  const auto& [supported_type, unsupported_idx] = GetParam();

  std::vector<TfLiteType> dtypes(3, supported_type);
  dtypes[unsupported_idx] = kTfLiteNoType;

  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(dtypes[0], kDefaultInputDims);
  const int scale =
      context_builder.AddConstTensor(dtypes[1], kDefaultScaleDims);
  const int output = context_builder.AddTensor(dtypes[2], kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateRmsNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.rms_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, scale},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

constexpr std::array<std::string_view, 3> kTensorNames = {"Input", "Scale",
                                                          "Output"};
INSTANTIATE_TEST_SUITE_P(
    RmsNorm, RmsNormUnsupportedDtypeTest,
    ::testing::Combine(::testing::ValuesIn<TfLiteType>({
                           kTfLiteBFloat16,
                           kTfLiteFloat16,
                           kTfLiteFloat32,
                       }),
                       ::testing::Range(0, 3)),
    [](const testing::TestParamInfo<RmsNormUnsupportedDtypeTest::ParamType>&
           info) {
      const TfLiteType dtype = std::get<0>(info.param);
      const int index = std::get<1>(info.param);
      const std::string_view tensor_name =
          index < kTensorNames.size() ? kTensorNames[index] : "Unknown";
      return absl::StrCat(TfLiteTypeGetName(dtype), "_RejectsUnsupported",
                          tensor_name);
    });

using ConstantTestSuite = testing::Test;
TEST_F(ConstantTestSuite, RejectsConstInput) {
  StubContextBuilder context_builder;
  const int input =
      context_builder.AddConstTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateRmsNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.rms_norm",
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

TEST_F(ConstantTestSuite, RejectsDynamicScale) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int scale = context_builder.AddTensor(kDefaultDtype, kDefaultScaleDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateRmsNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.rms_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, scale},
                        /*outputs=*/{output});

  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

class DimsTest : public testing::Test {};

TEST_F(DimsTest, Rejects1dInput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1});
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateRmsNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.rms_norm",
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

TEST_F(DimsTest, Rejects5dInput) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, {1, 2, 3, 4, 5});
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateRmsNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.rms_norm",
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

TEST_F(DimsTest, Rejects2dScale) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int scale = context_builder.AddConstTensor(kDefaultDtype, {4, 1});
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateRmsNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.rms_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, scale},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_F(DimsTest, RejectsScaleShapeMismatch) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int scale = context_builder.AddConstTensor(kDefaultDtype, {3});
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer = CreateRmsNormBuffer();
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.rms_norm",
      .attributes = buffer.data(),
      .attributes_size = buffer.size()};
  context_builder.SetOp(kTfLiteBuiltinStablehloComposite, /*version=*/1,
                        /*params=*/&params,
                        /*inputs=*/{input, scale},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  ASSERT_THAT(context, NotNull());
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

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

TEST_F(ParamsTest, RejectsNoEpsilon) {
  StubContextBuilder context_builder;
  const int input = context_builder.AddTensor(kDefaultDtype, kDefaultInputDims);
  const int output =
      context_builder.AddTensor(kDefaultDtype, kDefaultOutputDims);

  const std::vector<uint8_t> buffer =
      CreateRmsNormBuffer(/*missing_epsilon=*/true);
  const TfLiteStablehloCompositeParams params = {
      .name = "odml.rms_norm",
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

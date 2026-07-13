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
#include <tuple>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/schema/schema_generated.h"

// These tests indirectly verify IsNodeSupported through GetOpsToReplace,
// which in turn uses GetSupportedNodes to leverage existing matchers.
//
// Note that the functionality of tflite::delegates::GraphPartitionHelper is
// intentionally NOT tested, as that's an implementation detail and that should
// be covered by its own unit tests.

using ::testing::Combine;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

namespace litert::ml_drift::ir {

extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

constexpr IrModelBuilderOptions kDefaultOptions = {};
constexpr TfLiteType kDefaultDtype = kTfLiteFloat32;
constexpr std::array<int, 4> kDefaultDims = {1, 2, 3, 4};

struct VersionTestCase {
  TfLiteBuiltinOperator op = kTfLiteBuiltinAdd;
  int version = 0;
};

// Test suite for unary arithmetic ops x supported version.
using SupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(SupportedVersionTest, Supports) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(GetParam().op, GetParam().version, /*params=*/nullptr,
                        /*inputs=*/{a}, /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    UnaryArithmeticOps, SupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {kTfLiteBuiltinAbs,      1},  // min
        {kTfLiteBuiltinAbs,      5},  // max
        {kTfLiteBuiltinCast,     1},  // only version
        {kTfLiteBuiltinCeil,     1},  // min
        {kTfLiteBuiltinCeil,     2},  // max
        {kTfLiteBuiltinCos,      1},  // min
        {kTfLiteBuiltinCos,      2},  // max
        {kTfLiteBuiltinElu,      1},  // min
        {kTfLiteBuiltinElu,      2},  // max
        {kTfLiteBuiltinExp,      1},  // min
        {kTfLiteBuiltinExp,      2},  // max
        {kTfLiteBuiltinFloor,    1},  // min
        {kTfLiteBuiltinFloor,    2},  // max
        {kTfLiteBuiltinGelu,     1},  // min
        {kTfLiteBuiltinGelu,     2},
        {kTfLiteBuiltinGelu,     3},  // max
        {kTfLiteBuiltinHardSwish, 1},  // only version
        {kTfLiteBuiltinLog,      1},  // min
        {kTfLiteBuiltinLog,      2},  // max
        {kTfLiteBuiltinLogistic, 1},  // min
        {kTfLiteBuiltinLogistic, 2},  // max
        {kTfLiteBuiltinNeg,      1},  // min
        {kTfLiteBuiltinNeg,      2},  // max
        {kTfLiteBuiltinRound,    1},  // min
        {kTfLiteBuiltinRound,    2},  // max
        {kTfLiteBuiltinRsqrt,    1},  // min
        {kTfLiteBuiltinRsqrt,    2},  // max
        {kTfLiteBuiltinSign,     1},  // min
        {kTfLiteBuiltinSign,     2},  // max
        {kTfLiteBuiltinSin,      1},  // min
        {kTfLiteBuiltinSin,      2},  // max
        {kTfLiteBuiltinSqrt,     1},  // min
        {kTfLiteBuiltinSqrt,     2},  // max
        {kTfLiteBuiltinSquare,   1},  // min
        {kTfLiteBuiltinSquare,   2},  // max
        {kTfLiteBuiltinStablehloCbrt,     1},  // only version
        {kTfLiteBuiltinTanh,     1},  // min
        {kTfLiteBuiltinTanh,     2},  // max
    }),
    [](const TestParamInfo<SupportedVersionTest::ParamType>& info) {
      return absl::StrCat(::tflite::EnumNamesBuiltinOperator()[info.param.op],
                          "_V", info.param.version);
    });
// clang-format on

// Test suite for unary arithmetic ops x unsupported version.
using UnsupportedVersionTest = TestWithParam<VersionTestCase>;

TEST_P(UnsupportedVersionTest, Rejects) {
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(GetParam().op, GetParam().version, /*params=*/nullptr,
                        /*inputs=*/{a}, /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    UnaryArithmeticOps, UnsupportedVersionTest,
    ValuesIn<VersionTestCase>({
        {kTfLiteBuiltinAbs,      0},           // min-1
        {kTfLiteBuiltinAbs,      6},           // max+1
        {kTfLiteBuiltinCast,     0},           // min-1
        {kTfLiteBuiltinCast,     2},           // max+1
        {kTfLiteBuiltinCeil,     0},           // min-1
        {kTfLiteBuiltinCeil,     3},           // max+1
        {kTfLiteBuiltinCos,      0},           // min-1
        {kTfLiteBuiltinCos,      3},           // max+1
        {kTfLiteBuiltinElu,      0},           // min-1
        {kTfLiteBuiltinElu,      3},           // max+1
        {kTfLiteBuiltinExp,      0},           // min-1
        {kTfLiteBuiltinExp,      3},           // max+1
        {kTfLiteBuiltinFloor,    0},           // min-1
        {kTfLiteBuiltinFloor,    3},           // max+1
        {kTfLiteBuiltinGelu,     0},           // min-1
        {kTfLiteBuiltinGelu,     4},           // max+1
        {kTfLiteBuiltinHardSwish, 0},          // min-1
        {kTfLiteBuiltinHardSwish, 2},          // max+1
        {kTfLiteBuiltinLog,      0},           // min-1
        {kTfLiteBuiltinLog,      3},           // max+1
        {kTfLiteBuiltinLogistic, 0},           // min-1
        {kTfLiteBuiltinLogistic, 3},           // max+1
        {kTfLiteBuiltinNeg,      0},           // min-1
        {kTfLiteBuiltinNeg,      3},           // max+1
        {kTfLiteBuiltinRound,    0},           // min-1
        {kTfLiteBuiltinRound,    3},           // max+1
        {kTfLiteBuiltinRsqrt,    0},           // min-1
        {kTfLiteBuiltinRsqrt,    3},           // max+1
        {kTfLiteBuiltinSign,     0},           // min-1
        {kTfLiteBuiltinSign,     3},           // max+1
        {kTfLiteBuiltinSin,      0},           // min-1
        {kTfLiteBuiltinSin,      3},           // max+1
        {kTfLiteBuiltinSqrt,     0},           // min-1
        {kTfLiteBuiltinSqrt,     3},           // max+1
        {kTfLiteBuiltinSquare,   0},           // min-1
        {kTfLiteBuiltinSquare,   3},           // max+1
        {kTfLiteBuiltinStablehloCbrt,     0},  // min-1
        {kTfLiteBuiltinStablehloCbrt,     2},  // max+1
        {kTfLiteBuiltinTanh,     0},           // min-1
        {kTfLiteBuiltinTanh,     3},           // max+1
    }),
    [](const TestParamInfo<UnsupportedVersionTest::ParamType>& info) {
      return absl::StrCat(::tflite::EnumNamesBuiltinOperator()[info.param.op],
                          "_V", info.param.version);
    });
// clang-format on

// Test suite for unary arithmetic ops for different number of I/O tensors.
using NumInputOutputTest = TestWithParam<TfLiteBuiltinOperator>;

TEST_P(NumInputOutputTest, Supports1Input1Output) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr, /*inputs=*/{a},
                        /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(NumInputOutputTest, Rejects0Inputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr, /*inputs=*/{},
                        /*outputs=*/{a});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects2Inputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr,
                        /*inputs=*/{a, b}, /*outputs=*/{c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects0Outputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr, /*inputs=*/{a},
                        /*outputs=*/{});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(NumInputOutputTest, Rejects2Outputs) {
  const TfLiteBuiltinOperator op = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int b = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  const int c = context_builder.AddTensor(kDefaultDtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr, /*inputs=*/{a},
                        /*outputs=*/{b, c});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    UnaryArithmeticOps, NumInputOutputTest,
    ValuesIn<TfLiteBuiltinOperator>({
        kTfLiteBuiltinAbs,           kTfLiteBuiltinCast,
        kTfLiteBuiltinCeil,          kTfLiteBuiltinCos,
        kTfLiteBuiltinElu,           kTfLiteBuiltinExp,
        kTfLiteBuiltinFloor,         kTfLiteBuiltinGelu,
        kTfLiteBuiltinHardSwish,     kTfLiteBuiltinLog,
        kTfLiteBuiltinLogistic,      kTfLiteBuiltinNeg,
        kTfLiteBuiltinRound,         kTfLiteBuiltinRsqrt,
        kTfLiteBuiltinSign,          kTfLiteBuiltinSin,
        kTfLiteBuiltinSqrt,          kTfLiteBuiltinSquare,
        kTfLiteBuiltinStablehloCbrt, kTfLiteBuiltinTanh,
    }),
    [](const TestParamInfo<NumInputOutputTest::ParamType>& info) {
      return ::tflite::EnumNamesBuiltinOperator()[info.param];
    });

// Test suite for unary arithmetic ops x supported subject dtypes.
using SupportedDtypeTest =
    TestWithParam<std::tuple<TfLiteBuiltinOperator, TfLiteType>>;

TEST_P(SupportedDtypeTest, SupportsSupportedDtypes) {
  const auto [op, dtype] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(dtype, kDefaultDims);
  const int b = context_builder.AddTensor(dtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr, /*inputs=*/{a},
                        /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), ElementsAre(0));
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedInput) {
  const auto [op, dtype] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(kTfLiteNoType, kDefaultDims);
  const int b = context_builder.AddTensor(dtype, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr, /*inputs=*/{a},
                        /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

TEST_P(SupportedDtypeTest, RejectsUnsupportedOutput) {
  const auto [op, dtype] = GetParam();
  StubContextBuilder context_builder;
  const int a = context_builder.AddTensor(dtype, kDefaultDims);
  const int b = context_builder.AddTensor(kTfLiteNoType, kDefaultDims);
  context_builder.SetOp(op, /*version=*/1, /*params=*/nullptr, /*inputs=*/{a},
                        /*outputs=*/{b});
  TfLiteContext* context = context_builder.Build();
  ASSERT_TRUE(context != nullptr);
  EXPECT_THAT(GetSupportedNodes(context, kDefaultOptions), IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(
    UnaryArithmeticOps, SupportedDtypeTest,
    Combine(ValuesIn<TfLiteBuiltinOperator>({
                kTfLiteBuiltinAbs,           kTfLiteBuiltinCast,
                kTfLiteBuiltinCeil,          kTfLiteBuiltinCos,
                kTfLiteBuiltinElu,           kTfLiteBuiltinExp,
                kTfLiteBuiltinFloor,         kTfLiteBuiltinGelu,
                kTfLiteBuiltinHardSwish,     kTfLiteBuiltinLog,
                kTfLiteBuiltinLogistic,      kTfLiteBuiltinNeg,
                kTfLiteBuiltinRound,         kTfLiteBuiltinRsqrt,
                kTfLiteBuiltinSign,          kTfLiteBuiltinSin,
                kTfLiteBuiltinSqrt,          kTfLiteBuiltinSquare,
                kTfLiteBuiltinStablehloCbrt, kTfLiteBuiltinTanh,
            }),
            ValuesIn<TfLiteType>({
                kTfLiteFloat32,
                kTfLiteInt32,
                kTfLiteUInt8,
                kTfLiteInt16,
                kTfLiteInt8,
                kTfLiteFloat16,
                kTfLiteUInt32,
                kTfLiteUInt16,
                kTfLiteBFloat16,
            })),
    [](const TestParamInfo<SupportedDtypeTest::ParamType>& info) {
      return absl::StrCat(
          ::tflite::EnumNamesBuiltinOperator()[std::get<0>(info.param)], "_",
          TfLiteTypeGetName(std::get<1>(info.param)));
    });

}  // namespace
}  // namespace litert::ml_drift::ir

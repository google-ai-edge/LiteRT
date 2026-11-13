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
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/stub_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"

// These tests verify IsScatterNdSupported through GetSupportedNodes.

namespace litert::ml_drift::ir {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::NotNull;

extern std::vector<int> GetSupportedNodes(TfLiteContext*,
                                          const IrModelBuilderOptions&);

namespace {

constexpr IrModelBuilderOptions kDefaultOptions = {};

struct ScatterNdCase {
  int version = 1;
  TfLiteType indices_dtype = kTfLiteInt32;
  std::vector<int> indices_dims = {3, 1};
  std::vector<int> updates_dims = {3, 2, 3, 5};
  std::vector<int> output_dims = {4, 2, 3, 5};
  bool constant_shape = true;
};

std::vector<int> SupportedNodes(const ScatterNdCase& c) {
  StubContextBuilder context_builder;
  const int indices =
      context_builder.AddTensor(c.indices_dtype, c.indices_dims);
  const int updates = context_builder.AddTensor(kTfLiteFloat32, c.updates_dims);
  const std::array<int, 1> shape_dims = {4};
  const int shape =
      c.constant_shape
          ? context_builder.AddConst1dTensor<int>(kTfLiteInt32, c.output_dims)
          : context_builder.AddTensor(kTfLiteInt32, shape_dims);
  const int output = context_builder.AddTensor(kTfLiteFloat32, c.output_dims);
  context_builder.SetOp(kTfLiteBuiltinScatterNd, c.version, /*params=*/nullptr,
                        /*inputs=*/{indices, updates, shape},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  EXPECT_THAT(context, NotNull());
  return GetSupportedNodes(context, kDefaultOptions);
}

TEST(SupportScatterNdTest, SupportsIndicesAlongFirstAxis) {
  EXPECT_THAT(SupportedNodes({}), ElementsAre(0));
}

TEST(SupportScatterNdTest, RejectsUnknownVersion) {
  EXPECT_THAT(SupportedNodes({.version = 2}), IsEmpty());
}

TEST(SupportScatterNdTest, RejectsNonInt32Indices) {
  EXPECT_THAT(SupportedNodes({.indices_dtype = kTfLiteInt64}), IsEmpty());
}

TEST(SupportScatterNdTest, RejectsMultiAxisIndices) {
  EXPECT_THAT(SupportedNodes({.indices_dims = {3, 2}}), IsEmpty());
}

TEST(SupportScatterNdTest, RejectsNon4dUpdates) {
  EXPECT_THAT(SupportedNodes({.updates_dims = {3, 5}, .output_dims = {4, 5}}),
              IsEmpty());
}

TEST(SupportScatterNdTest, RejectsMismatchedUpdateCount) {
  EXPECT_THAT(SupportedNodes({.updates_dims = {2, 2, 3, 5}}), IsEmpty());
}

TEST(SupportScatterNdTest, RejectsMismatchedInnerShape) {
  EXPECT_THAT(SupportedNodes({.updates_dims = {3, 2, 3, 4}}), IsEmpty());
}

TEST(SupportScatterNdTest, RejectsRuntimeShape) {
  EXPECT_THAT(SupportedNodes({.constant_shape = false}), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir

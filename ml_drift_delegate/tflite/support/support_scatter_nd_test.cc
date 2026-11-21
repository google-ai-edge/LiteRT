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

#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/stub_context.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
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
  TfLiteType updates_dtype = kTfLiteFloat32;
  TfLiteType output_dtype = kTfLiteFloat32;
  std::vector<int> indices_dims = {3, 3};
  std::vector<int> updates_dims = {3, 5};
  std::vector<int> output_dims = {4, 2, 3, 5};
  bool constant_shape = true;
  bool constant_indices = false;
  bool constant_updates = false;
};

std::vector<int> SupportedNodes(const ScatterNdCase& c) {
  StubContextBuilder context_builder;
  const int indices =
      c.constant_indices
          ? context_builder.AddConstTensor(c.indices_dtype, c.indices_dims)
          : context_builder.AddTensor(c.indices_dtype, c.indices_dims);
  const int updates =
      c.constant_updates
          ? context_builder.AddConstTensor(c.updates_dtype, c.updates_dims)
          : context_builder.AddTensor(c.updates_dtype, c.updates_dims);
  const std::array<int, 1> shape_dims = {4};
  const int shape =
      c.constant_shape
          ? context_builder.AddConst1dTensor<int>(kTfLiteInt32, c.output_dims)
          : context_builder.AddTensor(kTfLiteInt32, shape_dims);
  const int output = context_builder.AddTensor(c.output_dtype, c.output_dims);
  context_builder.SetOp(kTfLiteBuiltinScatterNd, c.version, /*params=*/nullptr,
                        /*inputs=*/{indices, updates, shape},
                        /*outputs=*/{output});
  TfLiteContext* context = context_builder.Build();
  EXPECT_THAT(context, NotNull());
  return GetSupportedNodes(context, kDefaultOptions);
}

TEST(SupportScatterNdTest, SupportsIndexDepth3) {
  EXPECT_THAT(SupportedNodes({}), ElementsAre(0));
}

TEST(SupportScatterNdTest, RejectsUnknownVersion) {
  EXPECT_THAT(SupportedNodes({.version = 2}), IsEmpty());
}

TEST(SupportScatterNdTest, RejectsNonInt32Indices) {
  EXPECT_THAT(SupportedNodes({.indices_dtype = kTfLiteInt64}), IsEmpty());
}

TEST(SupportScatterNdTest, RejectsOtherIndexDepth) {
  EXPECT_THAT(SupportedNodes({.indices_dims = {3, 1}}), IsEmpty());
  EXPECT_THAT(SupportedNodes({.indices_dims = {3, 2}}), IsEmpty());
  EXPECT_THAT(SupportedNodes({.indices_dims = {3, 4}}), IsEmpty());
}

TEST(SupportScatterNdTest, Supports4dIndices) {
  EXPECT_THAT(SupportedNodes({.indices_dims = {1, 8, 1, 3},
                              .updates_dims = {1, 8, 1, 64},
                              .output_dims = {1, 8, 127, 64}}),
              ElementsAre(0));
}

TEST(SupportScatterNdTest, SupportsBoolUpdates) {
  EXPECT_THAT(SupportedNodes({.updates_dtype = kTfLiteBool,
                              .output_dtype = kTfLiteBool,
                              .indices_dims = {1, 8, 1, 3},
                              .updates_dims = {1, 8, 1, 64},
                              .output_dims = {1, 8, 127, 64}}),
              ElementsAre(0));
}

TEST(SupportScatterNdTest, SupportsConstantUpdates) {
  EXPECT_THAT(SupportedNodes({.constant_updates = true}), ElementsAre(0));
  EXPECT_THAT(SupportedNodes({.updates_dtype = kTfLiteBool,
                              .output_dtype = kTfLiteBool,
                              .constant_updates = true}),
              ElementsAre(0));
}

TEST(SupportScatterNdTest, RejectsConstantIndices) {
  EXPECT_THAT(SupportedNodes({.constant_indices = true}), IsEmpty());
}

TEST(SupportScatterNdTest, RejectsInt32Updates) {
  EXPECT_THAT(SupportedNodes({.updates_dtype = kTfLiteInt32,
                              .output_dtype = kTfLiteInt32}),
              IsEmpty());
}

TEST(SupportScatterNdTest, RejectsMismatchedOutputType) {
  EXPECT_THAT(SupportedNodes({.updates_dtype = kTfLiteBool}), IsEmpty());
}

TEST(SupportScatterNdTest, RejectsUnsupportedUpdatesType) {
  EXPECT_THAT(SupportedNodes({.updates_dtype = kTfLiteInt64,
                              .output_dtype = kTfLiteInt64}),
              IsEmpty());
}

TEST(SupportScatterNdTest, RejectsUnsupportedIndicesRank) {
  EXPECT_THAT(SupportedNodes({.indices_dims = {1, 3, 3}}), IsEmpty());
}

TEST(SupportScatterNdTest, RejectsMismatchedUpdatesRank) {
  EXPECT_THAT(SupportedNodes({.updates_dims = {3, 1, 1, 5}}), IsEmpty());
}

TEST(SupportScatterNdTest, RejectsMismatchedLeadingDims) {
  EXPECT_THAT(SupportedNodes({.indices_dims = {1, 8, 1, 3},
                              .updates_dims = {1, 8, 2, 64},
                              .output_dims = {1, 8, 127, 64}}),
              IsEmpty());
}

TEST(SupportScatterNdTest, RejectsNon4dOutput) {
  EXPECT_THAT(SupportedNodes({.output_dims = {4, 2, 5}}), IsEmpty());
}

TEST(SupportScatterNdTest, RejectsMismatchedUpdateCount) {
  EXPECT_THAT(SupportedNodes({.updates_dims = {2, 5}}), IsEmpty());
}

TEST(SupportScatterNdTest, RejectsMismatchedChannels) {
  EXPECT_THAT(SupportedNodes({.updates_dims = {3, 4}}), IsEmpty());
}

TEST(SupportScatterNdTest, RejectsRuntimeShape) {
  EXPECT_THAT(SupportedNodes({.constant_shape = false}), IsEmpty());
}

}  // namespace
}  // namespace litert::ml_drift::ir

// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/cc/litert_op_options.h"

#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_op_code.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/schema/schema_generated.h"

namespace litert {
namespace {

TEST(OpOptionsTest, GetCompositeOptions) {
  static constexpr auto kOptsType =
      ::tflite::BuiltinOptions2_StableHLOCompositeOptions;
  static constexpr absl::string_view kName = "test.composite";
  static constexpr int kSubgraph = 1;

  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeShloComposite);

  tflite::StableHLOCompositeOptionsT options;
  options.name = kName;
  options.decomposition_subgraph_index = kSubgraph;

  internal::TflOptions2 tfl_options;
  tfl_options.type = kOptsType;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions2(op, std::move(tfl_options));

  auto res = GetOptionsAs<CompositeOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->name, kName);
  EXPECT_EQ(res->subgraph, kSubgraph);
}

TEST(OpOptionsTest, GetUnsupportedOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeShloAdd);
  ASSERT_FALSE(GetOptionsAs<CompositeOptions>(&op));
}

TEST(OpOptionsTest, GetAddOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  tflite::AddOptionsT options;
  options.fused_activation_function = tflite::ActivationFunctionType_NONE;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_AddOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<AddOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->fused_activation_function, kActivationFunctionTypeNone);
  EXPECT_NE(res->fused_activation_function, kActivationFunctionTypeRelu);
  EXPECT_EQ(&op, res->op);
}

TEST(OpOptionsTest, TestGetOptionsAsInvalidOpOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeShloComposite);
  ASSERT_FALSE(GetOptionsAs<AddOptions>(&op));
  ASSERT_FALSE(GetOptionsAs<BatchMatmulOptions>(&op));
}

TEST(OpOptionsTest, GetBatchMatmulOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  tflite::BatchMatMulOptionsT options;
  options.adj_x = false;
  options.adj_y = false;
  options.asymmetric_quantize_inputs = true;
  internal::TflOptions tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions_BatchMatMulOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions(op, std::move(tfl_options));

  auto res = GetOptionsAs<BatchMatmulOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->adj_x, false);
  EXPECT_EQ(res->adj_y, false);
  EXPECT_EQ(res->asymmetric_quantize_input, true);
  EXPECT_EQ(&op, res->op);
}
}  // namespace
}  // namespace litert

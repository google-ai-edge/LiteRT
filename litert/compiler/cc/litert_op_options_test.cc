// Copyright 2026 Google LLC.
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

#include "litert/compiler/cc/litert_op_options.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_tfl_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/test/load_test_model.h"

namespace litert::compiler {
namespace {

using ::testing::ElementsAre;

// Instead of manually constructing ops which might rely on internal helpers
// that are hard to access, we can load a test model and inspect its ops. This
// is much cleaner and doesn't rely on internal mutable model APIs.

TEST(CompilerOpOptionsTest, AddOptionsFromModel) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  ASSERT_NE(ctx, nullptr);

  auto cc_model = litert::testing::LoadTestFileModel("simple_multi_op.tflite");
  litert::compiler::Model model(ctx, cc_model.Get());

  auto subgraph_or = model.MainSubgraph();
  ASSERT_TRUE(subgraph_or.HasValue());
  auto subgraph = subgraph_or.Value();

  LiteRtOp add_op = nullptr;
  for (const auto& op : subgraph.Ops()) {
    if (op.Code() == kLiteRtOpCodeTflAdd) {
      add_op = op.Get();
      break;
    }
  }
  ASSERT_NE(add_op, nullptr);

  auto res = GetOptionsAs<AddOptions>(ctx, add_op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->fused_activation_function, kActivationFunctionTypeNone);
  EXPECT_EQ(add_op, res->op);
}

TEST(CompilerOpOptionsTest, CompositeOptionsFromModel) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  ASSERT_NE(ctx, nullptr);

  auto cc_model =
      litert::testing::LoadTestFileModel("simple_shlo_composite.tflite");
  litert::compiler::Model model(ctx, cc_model.Get());

  auto subgraph_or = model.MainSubgraph();
  ASSERT_TRUE(subgraph_or.HasValue());
  auto subgraph = subgraph_or.Value();

  LiteRtOp composite_op = nullptr;
  for (const auto& op : subgraph.Ops()) {
    if (op.Code() == kLiteRtOpCodeShloComposite) {
      composite_op = op.Get();
      break;
    }
  }
  ASSERT_NE(composite_op, nullptr);

  auto res = GetOptionsAs<CompositeOptions>(ctx, composite_op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->name, "stablehlo.add_n");
  EXPECT_EQ(res->version, 3);
  EXPECT_TRUE(res->attributes_map.has_value());
  EXPECT_STREQ(res->attributes_map.value()["an_attribute"].AsString().c_str(),
               "foo");
  EXPECT_EQ(res->attributes_map.value()["meaning_of_life"].AsInt32(), 42);
}

TEST(CompilerOpOptionsTest, ReshapeOptionsFromModel) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  ASSERT_NE(ctx, nullptr);

  auto cc_model = litert::testing::LoadTestFileModel("rms_norm.tflite");
  litert::compiler::Model model(ctx, cc_model.Get());

  auto subgraph_or = model.MainSubgraph();
  ASSERT_TRUE(subgraph_or.HasValue());
  auto subgraph = subgraph_or.Value();

  LiteRtOp reshape_op = nullptr;
  for (const auto& op : subgraph.Ops()) {
    if (op.Code() == kLiteRtOpCodeTflReshape) {
      reshape_op = op.Get();
      break;
    }
  }
  ASSERT_NE(reshape_op, nullptr);

  auto res = GetOptionsAs<ReshapeOptions>(ctx, reshape_op);
  ASSERT_TRUE(res);
  EXPECT_THAT(res->new_shape, ElementsAre(8, 128, 1));
  EXPECT_EQ(reshape_op, res->op);
}

}  // namespace
}  // namespace litert::compiler

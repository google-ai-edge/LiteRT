// Copyright 2025 Google LLC.
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

#include "litert/test/generators/graph_helpers.h"

#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/test/matchers.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {
namespace {

using ::litert::internal::GetTflOptions;

TEST(GraphHelpersTest, TestSingleOpModel) {
  TensorDetails lhs = {{1, 2, 3}, kLiteRtElementTypeInt32, "lhs"};

  TensorDetails rhs = {
      {}, kLiteRtElementTypeInt32, "cst", MakeBufferRef<int32_t>({1})};

  TensorDetails output = {{1, 2, 3}, kLiteRtElementTypeInt32, "output"};

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model, SingleOpModel<kLiteRtOpCodeTflAdd>(
                      {std::move(lhs), std::move(rhs)}, {std::move(output)},
                      tflite::ActivationFunctionType_NONE, false));

  auto& sg = model->Subgraph(0);
  {
    EXPECT_EQ(sg.Inputs().size(), 1);
    EXPECT_EQ(sg.Inputs()[0]->Name(), "lhs");

    EXPECT_EQ(sg.Outputs().size(), 1);
    EXPECT_EQ(sg.Outputs()[0]->Name(), "output");

    ASSERT_EQ(sg.Tensors().size(), 3);
  }

  {
    ASSERT_EQ(sg.Ops().size(), 1);
    const auto& op = sg.Op(0);
    EXPECT_EQ(op.OpCode(), kLiteRtOpCodeTflAdd);
    const auto& tfl_opts = GetTflOptions(op);
    ASSERT_EQ(tfl_opts.type, tflite::BuiltinOptions_AddOptions);
    const auto* add_opts = tfl_opts.AsAddOptions();
    ASSERT_NE(add_opts, nullptr);
    EXPECT_EQ(add_opts->fused_activation_function,
              tflite::ActivationFunctionType_NONE);
    EXPECT_EQ(add_opts->pot_scale_int16, false);
    EXPECT_EQ(op.NumInputs(), 2);
    EXPECT_EQ(op.NumOutputs(), 1);
  }

  {
    const auto& lhs_tensor = sg.Tensor(0);
    EXPECT_EQ(lhs_tensor.Name(), "lhs");
  }

  {
    const auto& rhs_tensor = sg.Tensor(1);
    EXPECT_EQ(rhs_tensor.Name(), "cst");
    EXPECT_EQ(rhs_tensor.Weights().Buffer().Size(), sizeof(int32_t));
  }

  {
    const auto& output_tensor = sg.Tensor(2);
    EXPECT_EQ(output_tensor.Name(), "output");
  }
}

}  // namespace
}  // namespace litert::testing

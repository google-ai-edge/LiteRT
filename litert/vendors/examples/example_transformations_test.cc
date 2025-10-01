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

#include "litert/vendors/examples/example_transformations.h"

#include <gtest/gtest.h>
#include "litert/c/litert_op_code.h"
#include "litert/core/model/model.h"

namespace litert {
namespace {

TEST(ExampleTransformationTest, SimpleAddOpToMulOpTransformation) {
  // Create a subgraph with an add op and a mul op.
  LiteRtSubgraphT subgraph;
  LiteRtRewriterT rewriter;
  LiteRtOpT& add_op = subgraph.EmplaceOp();
  auto& add_op_input_tensor = subgraph.EmplaceTensor();
  auto& mul_op_output_tensor = subgraph.EmplaceTensor();
  add_op.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&add_op_input_tensor, add_op);
  internal::AttachOutput(&mul_op_output_tensor, add_op);

  // Call the transformation.
  SimpleAddOpToMulOpTransformation(&add_op, &rewriter);

  // Apply the changes.
  rewriter.ApplyChanges(&subgraph);

  // Verify the changes.
  EXPECT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(subgraph.Ops().front()->OpCode(), kLiteRtOpCodeTflMul);
}

}  // namespace
}  // namespace litert

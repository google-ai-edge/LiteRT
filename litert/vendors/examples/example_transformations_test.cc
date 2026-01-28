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
  LiteRtBuilderT builder;
  LiteRtOpT& add_op = subgraph.EmplaceOp();
  auto& add_op_input_tensor = subgraph.EmplaceTensor();
  auto& mul_op_output_tensor = subgraph.EmplaceTensor();
  add_op.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&add_op_input_tensor, add_op);
  internal::AttachOutput(&mul_op_output_tensor, add_op);

  // Call the transformation.
  SimpleAddOpToMulOpTransformation(&builder, &add_op);

  // Apply the changes.
  builder.ApplyChanges(&subgraph);

  // Verify the changes.
  EXPECT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(subgraph.Ops().front()->OpCode(), kLiteRtOpCodeTflMul);
}

TEST(ExampleTransformationTest, SqrtMeanSquareTransformation) {
  // Create a subgraph with Sqrt(Mean(Square(x))).
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder;

  auto& x = subgraph.EmplaceTensor();
  auto& axis = subgraph.EmplaceTensor();
  auto& square_out = subgraph.EmplaceTensor();
  auto& mean_out = subgraph.EmplaceTensor();
  auto& sqrt_out = subgraph.EmplaceTensor();

  auto& mul_op = subgraph.EmplaceOp();
  mul_op.SetOpCode(kLiteRtOpCodeTflMul);
  internal::AttachInput(&x, mul_op);
  internal::AttachInput(&x, mul_op);
  internal::AttachOutput(&square_out, mul_op);

  auto& mean_op = subgraph.EmplaceOp();
  mean_op.SetOpCode(kLiteRtOpCodeTflMean);
  internal::AttachInput(&square_out, mean_op);
  internal::AttachInput(&axis, mean_op);
  internal::AttachOutput(&mean_out, mean_op);

  auto& sqrt_op = subgraph.EmplaceOp();
  sqrt_op.SetOpCode(kLiteRtOpCodeTflSqrt);
  internal::AttachInput(&mean_out, sqrt_op);
  internal::AttachOutput(&sqrt_out, sqrt_op);

  // Call the transformation.
  SqrtMeanSquareTransformation(&builder, &sqrt_op);

  // Apply the changes.
  builder.ApplyChanges(&subgraph);

  // Verify the changes.
  // Mul and Mean should be removed. Abs should be added. Sqrt remains.
  // But Builder::ApplyChanges might just mark them as removed in its internal
  // state? No, ApplyChanges applies to subgraph.

  // Note: ApplyChanges might not remove ops from the vector immediately or
  // might replace them? It effectively removes them.

  int abs_count = 0;
  int sqrt_count = 0;
  for (const auto& op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflAbs) abs_count++;
    if (op->OpCode() == kLiteRtOpCodeTflSqrt) sqrt_count++;
  }

  EXPECT_EQ(abs_count, 1);
  EXPECT_EQ(sqrt_count, 1);
  EXPECT_EQ(subgraph.Ops().size(), 2);
}

}  // namespace
}  // namespace litert

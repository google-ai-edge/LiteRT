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

#include "litert/cc/internal/litert_matchers.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"

namespace litert {
namespace {

TEST(MatchersTest, SimpleMatch) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  Op cc_op(&op);
  EXPECT_TRUE(Match(cc_op, m_Op(kLiteRtOpCodeTflAdd)));
  EXPECT_FALSE(Match(cc_op, m_Op(kLiteRtOpCodeTflMul)));
}

TEST(MatchersTest, OpCodeMatch) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  // Add inputs to make m_Op(Code) fail (as it expects 0 inputs)
  auto& input = subgraph.EmplaceTensor();
  internal::AttachInput(&input, op);

  Op cc_op(&op);
  EXPECT_FALSE(Match(cc_op, m_Op(kLiteRtOpCodeTflAdd)));
  EXPECT_TRUE(Match(cc_op, m_OpCode(kLiteRtOpCodeTflAdd)));
}

TEST(MatchersTest, MatchInput) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  auto& input = subgraph.EmplaceTensor();
  internal::AttachInput(&input, op);

  auto& def_op = subgraph.EmplaceOp();
  def_op.SetOpCode(kLiteRtOpCodeTflMul);
  internal::AttachOutput(&input, def_op);

  Op cc_op(&op);

  // Match Add(Mul)
  EXPECT_TRUE(
      Match(cc_op, m_Op(kLiteRtOpCodeTflAdd, m_Op(kLiteRtOpCodeTflMul))));

  // Mismatch
  EXPECT_FALSE(
      Match(cc_op, m_Op(kLiteRtOpCodeTflAdd, m_Op(kLiteRtOpCodeTflSub))));
}

TEST(MatchersTest, Capture) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  Op cc_op(&op);
  Op captured(nullptr);

  EXPECT_TRUE(Match(cc_op, m_Capture(&captured, m_Op(kLiteRtOpCodeTflAdd))));
  EXPECT_EQ(captured.Code(), kLiteRtOpCodeTflAdd);
  EXPECT_EQ(captured.Get(), &op);
}

TEST(MatchersTest, AnyMatchers) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  auto& tensor = subgraph.EmplaceTensor();
  internal::AttachOutput(&tensor, op);

  Op cc_op(&op);
  Tensor cc_tensor(&tensor);

  EXPECT_TRUE(Match(cc_op, m_AnyOp()));
  EXPECT_TRUE(Match(cc_tensor, m_Any()));
  EXPECT_TRUE(Match(cc_tensor, m_AnyOp()));  // Tensor matches its defining op
}

TEST(MatchersTest, ConstantAndSubgraphInput) {
  LiteRtSubgraphT subgraph;
  auto& cst = subgraph.EmplaceTensor();
  SetWeightsFromOwnedBuffer(cst.Weights(), OwningBufferRef<uint8_t>("dummy"));

  auto& input = subgraph.EmplaceTensor();
  // No weights, no defining op -> subgraph input

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&cst, op);
  internal::AttachInput(&input, op);

  Tensor cc_cst(&cst);
  Tensor cc_input(&input);

  EXPECT_TRUE(Match(cc_cst, m_IsConstant()));
  EXPECT_FALSE(Match(cc_cst, m_IsSubgraphInput()));

  EXPECT_TRUE(Match(cc_input, m_IsSubgraphInput()));
  EXPECT_FALSE(Match(cc_input, m_IsConstant()));

  Op cc_op(&op);
  EXPECT_TRUE(Match(
      cc_op, m_Op(kLiteRtOpCodeTflAdd, m_IsConstant(), m_IsSubgraphInput())));
}

TEST(MatchersTest, Predicate) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  Op cc_op(&op);
  EXPECT_TRUE(Match(cc_op, m_Predicate<Op>([](const Op& o) {
                      return o.Code() == kLiteRtOpCodeTflAdd;
                    })));
  EXPECT_FALSE(Match(cc_op, m_Predicate<Op>([](const Op& o) {
                       return o.Code() == kLiteRtOpCodeTflMul;
                     })));
}

/*
Topology:
    T0 -> Op1 -> T1 -> Op2 -> T2 -> Op3 -> T3 -> Op4 -> T4
*/
TEST(MatchersTest, DeepChain) {
  LiteRtSubgraphT subgraph;
  auto* current_tensor = &subgraph.EmplaceTensor();

  for (int i = 0; i < 4; ++i) {
    auto& op = subgraph.EmplaceOp();
    op.SetOpCode(static_cast<LiteRtOpCode>(i + 1));
    internal::AttachInput(current_tensor, op);
    auto& next_tensor = subgraph.EmplaceTensor();
    internal::AttachOutput(&next_tensor, op);
    current_tensor = &next_tensor;
  }

  Tensor last_tensor(current_tensor);
  EXPECT_TRUE(
      Match(last_tensor,
            m_Op(static_cast<LiteRtOpCode>(4),
                 m_Op(static_cast<LiteRtOpCode>(3),
                      m_Op(static_cast<LiteRtOpCode>(2),
                           m_Op(static_cast<LiteRtOpCode>(1), m_Any()))))));

  // Fail due to wrong opcode in the middle
  EXPECT_FALSE(
      Match(last_tensor,
            m_Op(static_cast<LiteRtOpCode>(4),
                 m_Op(static_cast<LiteRtOpCode>(3),
                      m_Op(kLiteRtOpCodeTflAdd,  // Wrong
                           m_Op(static_cast<LiteRtOpCode>(1), m_Any()))))));
}

/*
Topology:
    T1 \
    T2 -> Op -> T_out
    T3 /
*/
TEST(MatchersTest, FanIn) {
  LiteRtSubgraphT subgraph;
  auto& in1 = subgraph.EmplaceTensor();
  auto& in2 = subgraph.EmplaceTensor();
  auto& in3 = subgraph.EmplaceTensor();

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&in1, op);
  internal::AttachInput(&in2, op);
  internal::AttachInput(&in3, op);

  Op cc_op(&op);
  EXPECT_TRUE(
      Match(cc_op, m_Op(kLiteRtOpCodeTflAdd, m_Any(), m_Any(), m_Any())));
  EXPECT_FALSE(Match(cc_op, m_Op(kLiteRtOpCodeTflAdd, m_Any(), m_Any())));
  EXPECT_FALSE(Match(
      cc_op, m_Op(kLiteRtOpCodeTflAdd, m_Any(), m_Any(), m_Any(), m_Any())));
}

/*
Topology:
    T1 --+--> Op -> T_out
         |
         +--/
*/
TEST(MatchersTest, SameInput) {
  LiteRtSubgraphT subgraph;
  auto& in = subgraph.EmplaceTensor();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflMul);
  internal::AttachInput(&in, op);
  internal::AttachInput(&in, op);

  Op cc_op(&op);
  Op captured1(nullptr);
  Op captured2(nullptr);

  // Both matchers see the same tensor, but they don't know it's shared
  // unless we check the captured pointers.
  EXPECT_TRUE(
      Match(cc_op, m_Op(kLiteRtOpCodeTflMul, m_Capture(&captured1, m_Any()),
                        m_Capture(&captured2, m_Any()))));
}

/*
Topology:
          +-> Op_left  -> T_left  -\
    T_in -+                         +-> Op_final -> T_out
          +-> Op_right -> T_right -/
*/
TEST(MatchersTest, DiamondPattern) {
  LiteRtSubgraphT subgraph;
  auto& in = subgraph.EmplaceTensor();

  auto& op_left = subgraph.EmplaceOp();
  op_left.SetOpCode(kLiteRtOpCodeTflAbs);
  internal::AttachInput(&in, op_left);
  auto& out_left = subgraph.EmplaceTensor();
  internal::AttachOutput(&out_left, op_left);

  auto& op_right = subgraph.EmplaceOp();
  op_right.SetOpCode(kLiteRtOpCodeTflNeg);
  internal::AttachInput(&in, op_right);
  auto& out_right = subgraph.EmplaceTensor();
  internal::AttachOutput(&out_right, op_right);

  auto& op_final = subgraph.EmplaceOp();
  op_final.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&out_left, op_final);
  internal::AttachInput(&out_right, op_final);

  Op cc_op(&op_final);
  EXPECT_TRUE(
      Match(cc_op, m_Op(kLiteRtOpCodeTflAdd, m_OpCode(kLiteRtOpCodeTflAbs),
                        m_OpCode(kLiteRtOpCodeTflNeg))));
}

/*
Topology:
    T_in --+--> Op1 -> T_out1 -\
           |                    +-> Op2 -> T_out2
           +--/           T_in2 -/
*/
TEST(MatchersTest, NestedCaptures) {
  LiteRtSubgraphT subgraph;
  auto& in = subgraph.EmplaceTensor();
  auto& op1 = subgraph.EmplaceOp();
  op1.SetOpCode(kLiteRtOpCodeTflMul);
  internal::AttachInput(&in, op1);
  internal::AttachInput(&in, op1);
  auto& out1 = subgraph.EmplaceTensor();
  internal::AttachOutput(&out1, op1);

  auto& op2 = subgraph.EmplaceOp();
  op2.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&out1, op2);
  auto& in2 = subgraph.EmplaceTensor();
  internal::AttachInput(&in2, op2);

  Op root(&op2);
  Op cap_mul(nullptr);
  Op cap_add(nullptr);

  EXPECT_TRUE(Match(
      root, m_Capture(&cap_add,
                      m_Op(kLiteRtOpCodeTflAdd,
                           m_Capture(&cap_mul, m_OpCode(kLiteRtOpCodeTflMul)),
                           m_Any()))));
  EXPECT_EQ(cap_add.Get(), &op2);
  EXPECT_EQ(cap_mul.Get(), &op1);
}

TEST(MatchersTest, ManyTestCasesTopology) {
  // We'll generate a bunch of cases in a loop or just list many variations.
  for (int num_inputs = 0; num_inputs < 10; ++num_inputs) {
    LiteRtSubgraphT subgraph;
    auto& op = subgraph.EmplaceOp();
    op.SetOpCode(kLiteRtOpCodeTflCustom);
    for (int i = 0; i < num_inputs; ++i) {
      auto& in = subgraph.EmplaceTensor();
      internal::AttachInput(&in, op);
    }

    Op cc_op(&op);
    // Exact match for num_inputs
    std::vector<AnyTensorMatcher> matchers(num_inputs);
    // Unfortunately we can't easily build m_Op dynamically with variadic args.
    // But we can check a few specific ones.
    if (num_inputs == 0)
      EXPECT_TRUE(Match(cc_op, m_Op(kLiteRtOpCodeTflCustom)));
    if (num_inputs == 1)
      EXPECT_TRUE(Match(cc_op, m_Op(kLiteRtOpCodeTflCustom, m_Any())));
    if (num_inputs == 2)
      EXPECT_TRUE(Match(cc_op, m_Op(kLiteRtOpCodeTflCustom, m_Any(), m_Any())));
    if (num_inputs == 5)
      EXPECT_TRUE(Match(cc_op, m_Op(kLiteRtOpCodeTflCustom, m_Any(), m_Any(),
                                    m_Any(), m_Any(), m_Any())));
  }
}

TEST(MatchersTest, ChainVariation) {
  // Testing chains of different lengths.
  for (int len = 1; len < 10; ++len) {
    LiteRtSubgraphT subgraph;
    auto* cur = &subgraph.EmplaceTensor();
    for (int i = 0; i < len; ++i) {
      auto& op = subgraph.EmplaceOp();
      op.SetOpCode(kLiteRtOpCodeTflAbs);
      internal::AttachInput(cur, op);
      auto& out = subgraph.EmplaceTensor();
      internal::AttachOutput(&out, op);
      cur = &out;
    }
    Tensor last(cur);
    // We can't easily generate the recursive matcher dynamically for 50 cases,
    // but we can test a few.
    if (len == 1) EXPECT_TRUE(Match(last, m_Op(kLiteRtOpCodeTflAbs, m_Any())));
    if (len == 2)
      EXPECT_TRUE(Match(
          last, m_Op(kLiteRtOpCodeTflAbs, m_Op(kLiteRtOpCodeTflAbs, m_Any()))));
    if (len == 3)
      EXPECT_TRUE(Match(last, m_Op(kLiteRtOpCodeTflAbs,
                                   m_Op(kLiteRtOpCodeTflAbs,
                                        m_Op(kLiteRtOpCodeTflAbs, m_Any())))));
  }
}

/*
Topology:
          +-> Op1 -> T_out1
    T_in -+
          +-> Op2 -> T_out2
*/
TEST(MatchersTest, FanOutMismatch) {
  LiteRtSubgraphT subgraph;
  auto& in = subgraph.EmplaceTensor();

  auto& op1 = subgraph.EmplaceOp();
  op1.SetOpCode(kLiteRtOpCodeTflAbs);
  internal::AttachInput(&in, op1);

  auto& op2 = subgraph.EmplaceOp();
  op2.SetOpCode(kLiteRtOpCodeTflNeg);
  internal::AttachInput(&in, op2);

  // Matcher for op1 shouldn't care about op2.
  EXPECT_TRUE(Match(Op(&op1), m_Op(kLiteRtOpCodeTflAbs, m_Any())));
}

/*
Topology:
    T_in -> Op --+-> T_out1
                 |
                 +-> T_out2
*/
TEST(MatchersTest, MultipleOutputs) {
  LiteRtSubgraphT subgraph;
  auto& in = subgraph.EmplaceTensor();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflSplit);
  internal::AttachInput(&in, op);

  auto& out1 = subgraph.EmplaceTensor();
  internal::AttachOutput(&out1, op);
  auto& out2 = subgraph.EmplaceTensor();
  internal::AttachOutput(&out2, op);

  // OpMatcher only checks inputs.
  EXPECT_TRUE(Match(Op(&op), m_Op(kLiteRtOpCodeTflSplit, m_Any())));

  // Tensor matching from different outputs.
  EXPECT_TRUE(Match(Tensor(&out1), m_Op(kLiteRtOpCodeTflSplit, m_Any())));
  EXPECT_TRUE(Match(Tensor(&out2), m_Op(kLiteRtOpCodeTflSplit, m_Any())));
}

/*
Topology:
    T1 -\
    T2 --\
    ...   -> Op -> T_out
    T8 --/
*/
TEST(MatchersTest, WideFanIn) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflConcatenation);
  std::vector<LiteRtTensor*> inputs;
  for (int i = 0; i < 8; ++i) {
    auto& in = subgraph.EmplaceTensor();
    internal::AttachInput(&in, op);
  }

  EXPECT_TRUE(Match(
      Op(&op), m_Op(kLiteRtOpCodeTflConcatenation, m_Any(), m_Any(), m_Any(),
                    m_Any(), m_Any(), m_Any(), m_Any(), m_Any())));
}

/*
Topology:
    T1 -\
         +-> Op1 -> T_out1 -\
    T2 -/                    \
                              +-> Op3 -> T_out3
    T3 -\                    /
         +-> Op2 -> T_out2 -/
    T4 -/
*/
TEST(MatchersTest, ComplexTree) {
  LiteRtSubgraphT subgraph;
  auto& in1 = subgraph.EmplaceTensor();
  auto& in2 = subgraph.EmplaceTensor();
  auto& op1 = subgraph.EmplaceOp();
  op1.SetOpCode(kLiteRtOpCodeTflMul);
  internal::AttachInput(&in1, op1);
  internal::AttachInput(&in2, op1);
  auto& out1 = subgraph.EmplaceTensor();
  internal::AttachOutput(&out1, op1);

  auto& in3 = subgraph.EmplaceTensor();
  auto& in4 = subgraph.EmplaceTensor();
  auto& op2 = subgraph.EmplaceOp();
  op2.SetOpCode(kLiteRtOpCodeTflSub);
  internal::AttachInput(&in3, op2);
  internal::AttachInput(&in4, op2);
  auto& out2 = subgraph.EmplaceTensor();
  internal::AttachOutput(&out2, op2);

  auto& op3 = subgraph.EmplaceOp();
  op3.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&out1, op3);
  internal::AttachInput(&out2, op3);

  EXPECT_TRUE(
      Match(Op(&op3), m_Op(kLiteRtOpCodeTflAdd, m_OpCode(kLiteRtOpCodeTflMul),
                           m_OpCode(kLiteRtOpCodeTflSub))));
}

TEST(MatchersTest, CaptureTensor) {
  LiteRtSubgraphT subgraph;
  auto& in = subgraph.EmplaceTensor();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAbs);
  internal::AttachInput(&in, op);

  Tensor captured(nullptr);
  EXPECT_TRUE(
      Match(Op(&op), m_Op(kLiteRtOpCodeTflAbs, m_Capture(&captured, m_Any()))));
  EXPECT_EQ(captured.Get(), &in);
}

TEST(MatchersTest, PredicateOnTensor) {
  LiteRtSubgraphT subgraph;
  auto& in = subgraph.EmplaceTensor();
  in.SetName("my_tensor");

  EXPECT_TRUE(Match(Tensor(&in), m_Predicate<Tensor>([](const Tensor& t) {
                      return t.Name() == "my_tensor";
                    })));
}

TEST(MatchersTest, MatchOpCodeWithTensor) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflMul);
  auto& out = subgraph.EmplaceTensor();
  internal::AttachOutput(&out, op);

  EXPECT_TRUE(Match(Tensor(&out), m_OpCode(kLiteRtOpCodeTflMul)));
}

}  // namespace
}  // namespace litert

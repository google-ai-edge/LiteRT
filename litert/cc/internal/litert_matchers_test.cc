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
#include <cstring>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert {
namespace {

TEST(MatchersTest, SimpleMatch) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  Op cc_op(&op);
  EXPECT_TRUE(Match(cc_op, m_Op<kLiteRtOpCodeTflAdd>()));
  EXPECT_FALSE(Match(cc_op, m_Op<kLiteRtOpCodeTflMul>()));
}

TEST(MatchersTest, OpCodeMatch) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  // Add inputs to make m_Op(Code) fail (as it expects 0 inputs)
  auto& input = subgraph.EmplaceTensor();
  internal::AttachInput(&input, op);

  Op cc_op(&op);
  EXPECT_FALSE(Match(cc_op, m_Op<kLiteRtOpCodeTflAdd>()));
  EXPECT_TRUE(Match(cc_op, m_OpCode<kLiteRtOpCodeTflAdd>()));
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
      Match(cc_op, m_Op<kLiteRtOpCodeTflAdd>(m_Op<kLiteRtOpCodeTflMul>())));

  // Mismatch
  EXPECT_FALSE(
      Match(cc_op, m_Op<kLiteRtOpCodeTflAdd>(m_Op<kLiteRtOpCodeTflSub>())));
}

TEST(MatchersTest, Capture) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  Op cc_op(&op);
  Op captured(nullptr);

  EXPECT_TRUE(Match(cc_op, m_Capture(&captured, m_Op<kLiteRtOpCodeTflAdd>())));
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
      cc_op, m_Op<kLiteRtOpCodeTflAdd>(m_IsConstant(), m_IsSubgraphInput())));
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
  EXPECT_TRUE(Match(
      last_tensor,
      m_Op<static_cast<LiteRtOpCode>(4)>(
          m_Op<static_cast<LiteRtOpCode>(3)>(m_Op<static_cast<LiteRtOpCode>(2)>(
              m_Op<static_cast<LiteRtOpCode>(1)>(m_Any()))))));

  // Fail due to wrong opcode in the middle
  EXPECT_FALSE(Match(
      last_tensor,
      m_Op<static_cast<LiteRtOpCode>(4)>(m_Op<static_cast<LiteRtOpCode>(3)>(
          m_Op<kLiteRtOpCodeTflAdd>(  // Wrong
              m_Op<static_cast<LiteRtOpCode>(1)>(m_Any()))))));
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
      Match(cc_op, m_Op<kLiteRtOpCodeTflAdd>(m_Any(), m_Any(), m_Any())));
  EXPECT_FALSE(Match(cc_op, m_Op<kLiteRtOpCodeTflAdd>(m_Any(), m_Any())));
  EXPECT_FALSE(Match(
      cc_op, m_Op<kLiteRtOpCodeTflAdd>(m_Any(), m_Any(), m_Any(), m_Any())));
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
  Tensor captured1(nullptr);
  Tensor captured2(nullptr);

  // Both matchers see the same tensor, but they don't know it's shared
  // unless we check the captured pointers.
  EXPECT_TRUE(
      Match(cc_op, m_Op<kLiteRtOpCodeTflMul>(m_Capture(&captured1, m_Any()),
                                             m_Capture(&captured2, m_Any()))));
  EXPECT_EQ(captured1.Get(), &in);
  EXPECT_EQ(captured2.Get(), &in);
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
      Match(cc_op, m_Op<kLiteRtOpCodeTflAdd>(m_OpCode<kLiteRtOpCodeTflAbs>(),
                                             m_OpCode<kLiteRtOpCodeTflNeg>())));
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
                      m_Op<kLiteRtOpCodeTflAdd>(
                          m_Capture(&cap_mul, m_OpCode<kLiteRtOpCodeTflMul>()),
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
      EXPECT_TRUE(Match(cc_op, m_Op<kLiteRtOpCodeTflCustom>()));
    if (num_inputs == 1)
      EXPECT_TRUE(Match(cc_op, m_Op<kLiteRtOpCodeTflCustom>(m_Any())));
    if (num_inputs == 2)
      EXPECT_TRUE(Match(cc_op, m_Op<kLiteRtOpCodeTflCustom>(m_Any(), m_Any())));
    if (num_inputs == 5)
      EXPECT_TRUE(
          Match(cc_op, m_Op<kLiteRtOpCodeTflCustom>(m_Any(), m_Any(), m_Any(),
                                                    m_Any(), m_Any())));
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
    if (len == 1) EXPECT_TRUE(Match(last, m_Op<kLiteRtOpCodeTflAbs>(m_Any())));
    if (len == 2)
      EXPECT_TRUE(Match(
          last, m_Op<kLiteRtOpCodeTflAbs>(m_Op<kLiteRtOpCodeTflAbs>(m_Any()))));
    if (len == 3)
      EXPECT_TRUE(
          Match(last, m_Op<kLiteRtOpCodeTflAbs>(m_Op<kLiteRtOpCodeTflAbs>(
                          m_Op<kLiteRtOpCodeTflAbs>(m_Any())))));
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
  EXPECT_TRUE(Match(Op(&op1), m_Op<kLiteRtOpCodeTflAbs>(m_Any())));
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
  EXPECT_TRUE(Match(Op(&op), m_Op<kLiteRtOpCodeTflSplit>(m_Any())));

  // Tensor matching from different outputs.
  EXPECT_TRUE(Match(Tensor(&out1), m_Op<kLiteRtOpCodeTflSplit>(m_Any())));
  EXPECT_TRUE(Match(Tensor(&out2), m_Op<kLiteRtOpCodeTflSplit>(m_Any())));
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

  EXPECT_TRUE(Match(Op(&op), m_Op<kLiteRtOpCodeTflConcatenation>(
                                 m_Any(), m_Any(), m_Any(), m_Any(), m_Any(),
                                 m_Any(), m_Any(), m_Any())));
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

  EXPECT_TRUE(Match(
      Op(&op3), m_Op<kLiteRtOpCodeTflAdd>(m_OpCode<kLiteRtOpCodeTflMul>(),
                                          m_OpCode<kLiteRtOpCodeTflSub>())));
}

TEST(MatchersTest, CaptureTensor) {
  LiteRtSubgraphT subgraph;
  auto& in = subgraph.EmplaceTensor();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAbs);
  internal::AttachInput(&in, op);

  Tensor captured(nullptr);
  EXPECT_TRUE(
      Match(Op(&op), m_Op<kLiteRtOpCodeTflAbs>(m_Capture(&captured, m_Any()))));
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

  EXPECT_TRUE(Match(Tensor(&out), m_OpCode<kLiteRtOpCodeTflMul>()));
}

TEST(MatchersTest, Combinators) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  Op cc_op(&op);
  EXPECT_TRUE(
      Match(cc_op, m_AllOf(m_OpCode<kLiteRtOpCodeTflAdd>(), m_AnyOp())));
  EXPECT_FALSE(Match(cc_op, m_AllOf(m_OpCode<kLiteRtOpCodeTflAdd>(),
                                    m_OpCode<kLiteRtOpCodeTflMul>())));

  EXPECT_TRUE(Match(cc_op, m_AnyOf(m_OpCode<kLiteRtOpCodeTflAdd>(),
                                   m_OpCode<kLiteRtOpCodeTflMul>())));
  EXPECT_FALSE(Match(cc_op, m_AnyOf(m_OpCode<kLiteRtOpCodeTflSub>(),
                                    m_OpCode<kLiteRtOpCodeTflMul>())));

  EXPECT_TRUE(Match(cc_op, m_Not(m_OpCode<kLiteRtOpCodeTflMul>())));
  EXPECT_FALSE(Match(cc_op, m_Not(m_OpCode<kLiteRtOpCodeTflAdd>())));
}

TEST(MatchersTest, CaptureFail) {
  LiteRtSubgraphT subgraph;
  auto& input = subgraph.EmplaceTensor();
  // No defining op (subgraph input)

  Tensor cc_input(&input);
  Op captured(nullptr);

  // Match succeeds on tensor level (m_Any), but capture Op fails because no
  // defining op. With fix, this should return false.
  EXPECT_FALSE(Match(cc_input, m_Capture(&captured, m_Any())));
}

TEST(MatchersTest, CustomOpMatching) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflCustom);
  op.SetCustomCode("MyCustomOp");

  auto& in = subgraph.EmplaceTensor();
  internal::AttachInput(&in, op);

  Op cc_op(&op);

  // m_CustomOpCode matches code string
  EXPECT_TRUE(Match(cc_op, m_CustomOpCode("MyCustomOp")));
  EXPECT_FALSE(Match(cc_op, m_CustomOpCode("OtherOp")));

  // m_CustomOp matches code + inputs
  EXPECT_TRUE(Match(cc_op, m_CustomOp("MyCustomOp", m_Any())));
  EXPECT_FALSE(Match(cc_op, m_CustomOp("MyCustomOp")));  // Wrong input count
  EXPECT_FALSE(Match(cc_op, m_CustomOp("OtherOp", m_Any())));
}

TEST(MatchersTest, NameMatching) {
  LiteRtSubgraphT subgraph;
  auto& t = subgraph.EmplaceTensor();
  t.SetName("MyTensor");

  Tensor cc_t(&t);
  EXPECT_TRUE(Match(cc_t, m_Name("MyTensor")));
  EXPECT_FALSE(Match(cc_t, m_Name("Other")));
}

TEST(MatchersTest, CustomLambdaMatching) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  Op cc_op(&op);

  // Custom lambda checking op code
  auto is_add = [](const Op& o) { return o.Code() == kLiteRtOpCodeTflAdd; };
  EXPECT_TRUE(Match(cc_op, m_Custom(is_add)));

  auto is_mul = [](const Op& o) { return o.Code() == kLiteRtOpCodeTflMul; };
  EXPECT_FALSE(Match(cc_op, m_Custom(is_mul)));

  // Custom lambda with state capture
  int match_count = 0;
  auto increment_and_match = [&](const Op& o) {
    match_count++;
    return true;
  };
  EXPECT_TRUE(Match(cc_op, m_Custom(increment_and_match)));
  EXPECT_EQ(match_count, 1);
}

TEST(MatchersTest, OptionsMatching) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflConv2d);

  tflite::Conv2DOptionsT conv_opts;
  conv_opts.stride_w = 1;
  conv_opts.stride_h = 2;
  conv_opts.padding = tflite::Padding_SAME;
  litert::internal::TflOptions tfl_opts;
  tfl_opts.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_opts.Set(std::move(conv_opts));
  litert::internal::SetTflOptions(op, std::move(tfl_opts));

  Op cc_op(&op);

  auto match_stride = [](const Conv2dOptions& opts) {
    return opts.stride_w == 1 && opts.stride_h == 2;
  };

  EXPECT_TRUE(Match(cc_op, m_Options<Conv2dOptions>(match_stride)));

  auto match_wrong = [](const Conv2dOptions& opts) {
    return opts.stride_w == 2;
  };
  EXPECT_FALSE(Match(cc_op, m_Options<Conv2dOptions>(match_wrong)));
}

TEST(MatchersTest, OpVariadicMatching) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflConcatenation);

  auto& in1 = subgraph.EmplaceTensor();
  auto& in2 = subgraph.EmplaceTensor();
  auto& in3 = subgraph.EmplaceTensor();
  internal::AttachInput(&in1, op);
  internal::AttachInput(&in2, op);
  internal::AttachInput(&in3, op);

  Op cc_op(&op);

  // Match prefix 2 inputs
  EXPECT_TRUE(Match(
      cc_op, m_OpVariadic<kLiteRtOpCodeTflConcatenation>(m_Any(), m_Any())));

  // Match prefix 1 input
  EXPECT_TRUE(
      Match(cc_op, m_OpVariadic<kLiteRtOpCodeTflConcatenation>(m_Any())));

  // Match exact 3 inputs (still valid as variadic)
  EXPECT_TRUE(Match(cc_op, m_OpVariadic<kLiteRtOpCodeTflConcatenation>(
                               m_Any(), m_Any(), m_Any())));

  // Match 4 inputs (fail)
  EXPECT_FALSE(Match(cc_op, m_OpVariadic<kLiteRtOpCodeTflConcatenation>(
                                m_Any(), m_Any(), m_Any(), m_Any())));
}

TEST(MatchersTest, CommutativeMatching) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  auto& in1 = subgraph.EmplaceTensor();
  in1.SetName("A");
  auto& in2 = subgraph.EmplaceTensor();
  in2.SetName("B");
  internal::AttachInput(&in1, op);
  internal::AttachInput(&in2, op);

  Op cc_op(&op);

  auto match_a =
      m_Predicate<Tensor>([](const Tensor& t) { return t.Name() == "A"; });
  auto match_b =
      m_Predicate<Tensor>([](const Tensor& t) { return t.Name() == "B"; });

  // Match A, B
  EXPECT_TRUE(
      Match(cc_op, m_CommutativeOp<kLiteRtOpCodeTflAdd>(match_a, match_b)));

  // Match B, A
  EXPECT_TRUE(
      Match(cc_op, m_CommutativeOp<kLiteRtOpCodeTflAdd>(match_b, match_a)));

  // Mismatch
  auto match_c =
      m_Predicate<Tensor>([](const Tensor& t) { return t.Name() == "C"; });
  EXPECT_FALSE(
      Match(cc_op, m_CommutativeOp<kLiteRtOpCodeTflAdd>(match_a, match_c)));
}

TEST(MatchersTest, MixedMatchers) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  // Set options
  tflite::AddOptionsT add_opts;
  add_opts.fused_activation_function = tflite::ActivationFunctionType_RELU;
  litert::internal::TflOptions tfl_opts;
  tfl_opts.type = tflite::BuiltinOptions_AddOptions;
  tfl_opts.Set(std::move(add_opts));
  litert::internal::SetTflOptions(op, std::move(tfl_opts));

  auto& in1 = subgraph.EmplaceTensor();
  auto& in2 = subgraph.EmplaceTensor();
  internal::AttachInput(&in1, op);
  internal::AttachInput(&in2, op);

  Op cc_op(&op);

  auto match_opts = m_Options<AddOptions>([](const AddOptions& o) {
    return o.fused_activation_function == kActivationFunctionTypeRelu;
  });

  // Match OpCode, Options, and Variadic Inputs
  EXPECT_TRUE(Match(
      cc_op, m_AllOf(m_OpCode<kLiteRtOpCodeTflAdd>(), match_opts,
                     m_OpVariadic<kLiteRtOpCodeTflAdd>(m_Any(), m_Any()))));
}

TEST(MatchersTest, CommutativeOpFailCount) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  auto& in1 = subgraph.EmplaceTensor();
  internal::AttachInput(&in1, op);
  // Only 1 input

  Op cc_op(&op);
  EXPECT_FALSE(
      Match(cc_op, m_CommutativeOp<kLiteRtOpCodeTflAdd>(m_Any(), m_Any())));
}

TEST(MatchersTest, ShapeMatching) {
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  tensor.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 224, 224, 3}));

  Tensor cc_tensor(&tensor);
  EXPECT_TRUE(Match(cc_tensor, m_Shape({1, 224, 224, 3})));
  // Rank mismatch
  EXPECT_FALSE(Match(cc_tensor, m_Shape({1, 224, 224})));
  // Dimension mismatch
  EXPECT_FALSE(Match(cc_tensor, m_Shape({1, 224, 224, 1})));
  // Empty shape check (scalar) - fail
  EXPECT_FALSE(Match(cc_tensor, m_Shape({})));
}

TEST(MatchersTest, ShapeWildcardMatching) {
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  tensor.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 224, 224, 3}));

  Tensor cc_tensor(&tensor);
  // Match exact
  EXPECT_TRUE(Match(cc_tensor, m_Shape({1, 224, 224, 3})));
  // Match with wildcards
  EXPECT_TRUE(Match(cc_tensor, m_Shape({1, -1, -1, 3})));
  EXPECT_TRUE(Match(cc_tensor, m_Shape({-1, -1, -1, -1})));

  // Mismatch
  EXPECT_FALSE(Match(cc_tensor, m_Shape({1, -1, 100, 3})));
}

TEST(MatchersTest, RankMatching) {
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  tensor.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 224, 224, 3}));

  Tensor cc_tensor(&tensor);
  EXPECT_TRUE(Match(cc_tensor, m_Rank(4)));
  EXPECT_FALSE(Match(cc_tensor, m_Rank(3)));
  EXPECT_FALSE(Match(cc_tensor, m_Rank(5)));
}

TEST(MatchersTest, ElementTypeMatching) {
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1}));

  Tensor cc_tensor(&tensor);
  EXPECT_TRUE(Match(cc_tensor, m_ElementType(kLiteRtElementTypeFloat32)));
  EXPECT_FALSE(Match(cc_tensor, m_ElementType(kLiteRtElementTypeInt32)));
  EXPECT_FALSE(Match(cc_tensor, m_ElementType(kLiteRtElementTypeBool)));
}

TEST(MatchersTest, OneUseMatching) {
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  Tensor cc_tensor(&tensor);

  // 0 uses
  EXPECT_FALSE(Match(cc_tensor, m_HasOneUse()));

  // 1 use
  auto& op1 = subgraph.EmplaceOp();
  op1.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&tensor, op1);
  EXPECT_TRUE(Match(cc_tensor, m_HasOneUse()));

  // 2 uses
  auto& op2 = subgraph.EmplaceOp();
  op2.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&tensor, op2);
  EXPECT_FALSE(Match(cc_tensor, m_HasOneUse()));
}

TEST(MatchersTest, ConstantValueMatching) {
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1}));

  // Set value 1.0
  float data = 1.0f;
  std::vector<uint8_t> bytes(sizeof(float));
  std::memcpy(bytes.data(), &data, sizeof(float));
  SetWeightsFromOwnedBuffer(tensor.Weights(),
                            OwningBufferRef<uint8_t>(std::move(bytes)));

  Tensor cc_tensor(&tensor);
  EXPECT_TRUE(Match(cc_tensor, m_IsConstant()));
  EXPECT_TRUE(Match(cc_tensor, m_ConstantValue<float>(1.0f)));
  EXPECT_FALSE(Match(cc_tensor, m_ConstantValue<float>(0.0f)));

  // Wrong Type
  EXPECT_FALSE(Match(cc_tensor, m_ConstantValue<int32_t>(1)));
}

TEST(MatchersTest, QuantizationMatching) {
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {1}));

  Tensor cc_tensor(&tensor);
  // Initially no quantization
  EXPECT_FALSE(Match(cc_tensor, m_IsQuantized()));
  EXPECT_TRUE(Match(cc_tensor, m_QType(kLiteRtQuantizationNone)));

  // Add Per-Tensor quantization
  tensor.SetQTypeId(kLiteRtQuantizationPerTensor);
  EXPECT_TRUE(Match(cc_tensor, m_IsQuantized()));
  EXPECT_TRUE(Match(cc_tensor, m_QType(kLiteRtQuantizationPerTensor)));
  EXPECT_FALSE(Match(cc_tensor, m_QType(kLiteRtQuantizationPerChannel)));
}

TEST(MatchersTest, UserCountMatching) {
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  Tensor cc_tensor(&tensor);

  EXPECT_TRUE(Match(cc_tensor, m_HasUsers(0)));
  EXPECT_FALSE(Match(cc_tensor, m_HasUsers(1)));

  auto& op1 = subgraph.EmplaceOp();
  internal::AttachInput(&tensor, op1);
  EXPECT_TRUE(Match(cc_tensor, m_HasUsers(1)));
  EXPECT_TRUE(Match(cc_tensor, m_HasOneUse()));

  auto& op2 = subgraph.EmplaceOp();
  internal::AttachInput(&tensor, op2);
  EXPECT_TRUE(Match(cc_tensor, m_HasUsers(2)));
}

TEST(MatchersTest, SameAsMatching) {
  LiteRtSubgraphT subgraph;
  auto& t1 = subgraph.EmplaceTensor();
  auto& t2 = subgraph.EmplaceTensor();

  Tensor cc_t1(&t1);
  Tensor cc_t2(&t2);

  Tensor captured(nullptr);

  // Capture t1, match t1 against captured (true)
  EXPECT_TRUE(Match(
      cc_t1, m_AllOf(m_Capture(&captured, m_Any()), m_SameAs(&captured))));

  // Capture t1, match t2 against captured (false)
  EXPECT_FALSE(
      Match(cc_t2, m_AllOf(m_Capture(&captured, m_Any()),  // capture t2
                           m_SameAs(&cc_t1))));  // compare t2 vs t1? No wait.

  // Test scenario: Matcher logic
  // m_SameAs(&captured) checks if current value == captured value.

  // Reset captured
  captured = Tensor(nullptr);
  // Match t1, capture it. Then check if t2 is same as t1 (false).
  // But we need to structure it as separate matches if not in same expression.

  Match(cc_t1, m_Capture(&captured, m_Any()));
  EXPECT_TRUE(Match(cc_t1, m_SameAs(&captured)));
  EXPECT_FALSE(Match(cc_t2, m_SameAs(&captured)));
}

TEST(MatchersTest, OutputIndexMatching) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflSplit);
  auto& out0 = subgraph.EmplaceTensor();
  internal::AttachOutput(&out0, op);
  auto& out1 = subgraph.EmplaceTensor();
  internal::AttachOutput(&out1, op);

  Tensor cc_out0(&out0);
  Tensor cc_out1(&out1);

  // Both tensors are produced by the same Split op
  EXPECT_TRUE(Match(cc_out0, m_OpCode<kLiteRtOpCodeTflSplit>()));
  EXPECT_TRUE(Match(cc_out1, m_OpCode<kLiteRtOpCodeTflSplit>()));

  // Distinguish by output index
  EXPECT_TRUE(
      Match(cc_out0, m_OutputIndex(0, m_OpCode<kLiteRtOpCodeTflSplit>())));
  EXPECT_FALSE(
      Match(cc_out0, m_OutputIndex(1, m_OpCode<kLiteRtOpCodeTflSplit>())));

  EXPECT_FALSE(
      Match(cc_out1, m_OutputIndex(0, m_OpCode<kLiteRtOpCodeTflSplit>())));
  EXPECT_TRUE(
      Match(cc_out1, m_OutputIndex(1, m_OpCode<kLiteRtOpCodeTflSplit>())));

  // Fail if defining op doesn't match
  EXPECT_FALSE(
      Match(cc_out0, m_OutputIndex(0, m_OpCode<kLiteRtOpCodeTflAdd>())));
}

TEST(MatchersTest, ComplexResnetBlock) {
  // Topology:
  // Input -> Split (2 outputs)
  // Out0 -> Conv2D (stride=1) -> Add
  // Out1 ----------------------> Add
  // Match the final Add op.

  LiteRtSubgraphT subgraph;
  auto& input = subgraph.EmplaceTensor();

  auto& split_op = subgraph.EmplaceOp();
  split_op.SetOpCode(kLiteRtOpCodeTflSplit);
  internal::AttachInput(&input, split_op);
  auto& split_out0 = subgraph.EmplaceTensor();
  internal::AttachOutput(&split_out0, split_op);
  auto& split_out1 = subgraph.EmplaceTensor();
  internal::AttachOutput(&split_out1, split_op);

  auto& conv_op = subgraph.EmplaceOp();
  conv_op.SetOpCode(kLiteRtOpCodeTflConv2d);
  internal::AttachInput(&split_out0, conv_op);
  // Set Conv2D options (stride=1)
  tflite::Conv2DOptionsT conv_opts;
  conv_opts.stride_w = 1;
  conv_opts.stride_h = 1;
  conv_opts.padding = tflite::Padding_SAME;
  litert::internal::TflOptions tfl_conv_opts;
  tfl_conv_opts.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_conv_opts.Set(std::move(conv_opts));
  litert::internal::SetTflOptions(conv_op, std::move(tfl_conv_opts));

  auto& conv_out = subgraph.EmplaceTensor();
  internal::AttachOutput(&conv_out, conv_op);

  auto& add_op = subgraph.EmplaceOp();
  add_op.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&conv_out, add_op);
  internal::AttachInput(&split_out1, add_op);

  Op root(&add_op);

  // Define sub-matchers for clarity
  auto match_split = m_OpCode<kLiteRtOpCodeTflSplit>();

  auto match_conv_opts = m_Options<Conv2dOptions>([](const Conv2dOptions& o) {
    return o.stride_w == 1 && o.stride_h == 1;
  });

  // Match Conv2D that takes the 0-th output of a Split, and has stride=1.
  auto match_conv =
      m_AllOf(m_Op<kLiteRtOpCodeTflConv2d>(m_OutputIndex(0, match_split)),
              match_conv_opts);

  // Match Add where one input is the Conv path, and other is the skip path
  // (1-st output of the same Split). Use CommutativeOp to handle input order.
  auto match_resnet_add = m_CommutativeOp<kLiteRtOpCodeTflAdd>(
      match_conv,                    // Matches Conv output path
      m_OutputIndex(1, match_split)  // Matches skip path
  );

  EXPECT_TRUE(Match(root, match_resnet_add));
}

TEST(MatchersTest, VariadicTypedConcat) {
  // Topology: Concat(Float32, Float32, Float32)
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflConcatenation);

  for (int i = 0; i < 3; ++i) {
    auto& t = subgraph.EmplaceTensor();
    t.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 10}));
    internal::AttachInput(&t, op);
  }

  Op root(&op);

  auto float_tensor = m_ElementType(kLiteRtElementTypeFloat32);

  // Match Concat taking at least 3 float tensors.
  EXPECT_TRUE(Match(root, m_OpVariadic<kLiteRtOpCodeTflConcatenation>(
                              float_tensor, float_tensor, float_tensor)));

  // Modify last input to Int32 -> Match should fail.
  subgraph.Tensors().back()->SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt32, {1, 10}));
  EXPECT_FALSE(Match(root, m_OpVariadic<kLiteRtOpCodeTflConcatenation>(
                               float_tensor, float_tensor, float_tensor)));
}

TEST(MatchersTest, AllInOneIntegration) {
  // A complex graph pattern that exercises every single matcher type.
  //
  // Graph Topology:
  //   Const(Float32) -\
  //                    Mul (Commutative) -> T_mul
  //   Input(Float32) -/
  //
  //   T_mul -\
  //           Add (Options: FusedActivation=Relu) -> T_add
  //   T_mul -/
  //
  //   Split(T_add, num_splits=2) -> T_split_0, T_split_1
  //
  //   Concat(T_split_0, T_split_1, T_mul) -> Output

  LiteRtSubgraphT subgraph;

  // 1. Inputs for Mul
  auto& cst = subgraph.EmplaceTensor();
  cst.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 2}));
  SetWeightsFromOwnedBuffer(cst.Weights(), OwningBufferRef<uint8_t>("dummy"));

  auto& input = subgraph.EmplaceTensor();
  input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 2}));

  // 2. Mul Op (Commutative)
  auto& mul_op = subgraph.EmplaceOp();
  mul_op.SetOpCode(kLiteRtOpCodeTflMul);
  internal::AttachInput(&cst, mul_op);
  internal::AttachInput(&input, mul_op);
  auto& t_mul = subgraph.EmplaceTensor();
  t_mul.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 2}));
  internal::AttachOutput(&t_mul, mul_op);

  // 3. Add Op (Options)
  auto& add_op = subgraph.EmplaceOp();
  add_op.SetOpCode(kLiteRtOpCodeTflAdd);
  tflite::AddOptionsT add_opts;
  add_opts.fused_activation_function = tflite::ActivationFunctionType_RELU;
  litert::internal::TflOptions tfl_add_opts;
  tfl_add_opts.type = tflite::BuiltinOptions_AddOptions;
  tfl_add_opts.Set(std::move(add_opts));
  litert::internal::SetTflOptions(add_op, std::move(tfl_add_opts));

  internal::AttachInput(&t_mul, add_op);
  internal::AttachInput(&t_mul, add_op);
  auto& t_add = subgraph.EmplaceTensor();
  t_add.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 2}));
  internal::AttachOutput(&t_add, add_op);

  // 4. Split Op (Output Index)
  auto& split_op = subgraph.EmplaceOp();
  split_op.SetOpCode(kLiteRtOpCodeTflSplit);
  internal::AttachInput(&t_add, split_op);
  auto& t_split_0 = subgraph.EmplaceTensor();
  t_split_0.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 1}));
  internal::AttachOutput(&t_split_0, split_op);
  auto& t_split_1 = subgraph.EmplaceTensor();
  t_split_1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 1}));
  internal::AttachOutput(&t_split_1, split_op);

  // 5. Concat Op (Variadic)
  auto& concat_op = subgraph.EmplaceOp();
  concat_op.SetOpCode(kLiteRtOpCodeTflConcatenation);
  internal::AttachInput(&t_split_0, concat_op);
  internal::AttachInput(&t_split_1, concat_op);
  internal::AttachInput(&t_mul, concat_op);

  // --- Matching Logic ---

  Op root(&concat_op);
  Op captured_add(nullptr);

  // Matcher for the commutative Mul inputs: one Const, one Subgraph Input
  auto match_mul_inputs = m_CommutativeOp<kLiteRtOpCodeTflMul>(
      m_AllOf(m_IsConstant(), m_ElementType(kLiteRtElementTypeFloat32)),
      m_AllOf(m_IsSubgraphInput(), m_Shape({1, 2})));

  // Matcher for the Add op with options, capturing it
  auto match_add = m_Capture(
      &captured_add,
      m_AllOf(m_OpCode<kLiteRtOpCodeTflAdd>(),
              m_Options<AddOptions>([](const AddOptions& o) {
                return o.fused_activation_function ==
                       kActivationFunctionTypeRelu;
              }),
              // Verify inputs come from the Mul we matched
              m_Op<kLiteRtOpCodeTflAdd>(match_mul_inputs, match_mul_inputs)));

  // Matcher for Split outputs
  auto match_split_out0 =
      m_OutputIndex(0, m_Op<kLiteRtOpCodeTflSplit>(match_add));
  auto match_split_out1 =
      m_OutputIndex(1, m_OpCode<kLiteRtOpCodeTflSplit>());  // Simple check

  // Root Matcher: Variadic Concat
  // Also using m_Not and m_AnyOf just to show them off.
  auto match_root =
      m_AllOf(m_OpVariadic<kLiteRtOpCodeTflConcatenation>(
                  match_split_out0, match_split_out1,
                  m_AnyOf(match_mul_inputs, m_IsConstant())  // Mul matches
                  ),
              m_Not(m_OpCode<kLiteRtOpCodeTflAdd>())  // Root is NOT Add
      );

  EXPECT_TRUE(Match(root, match_root));
  EXPECT_EQ(captured_add.Get(), &add_op);
}

}  // namespace
}  // namespace litert

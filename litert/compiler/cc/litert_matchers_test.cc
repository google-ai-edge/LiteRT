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

#include "litert/compiler/cc/litert_matchers.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_tfl_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/litert_op_options.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::compiler {
namespace {

using ::SetWeightsFromOwnedBuffer;
using ::litert::OwningBufferRef;
using ::litert::internal::AttachInput;
using ::litert::internal::AttachOutput;

TEST(MatchersTest, SimpleMatch) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  Op cc_op(ctx, &op);
  EXPECT_TRUE(Match(cc_op, m_Op<kLiteRtOpCodeTflAdd>()));
  EXPECT_FALSE(Match(cc_op, m_Op<kLiteRtOpCodeTflMul>()));
}

TEST(MatchersTest, OpCodeMatch) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  // Add inputs to make m_Op(Code) fail (as it expects 0 inputs)
  auto& input = subgraph.EmplaceTensor();
  AttachInput(&input, op);

  Op cc_op(ctx, &op);
  EXPECT_FALSE(Match(cc_op, m_Op<kLiteRtOpCodeTflAdd>()));
  EXPECT_TRUE(Match(cc_op, m_OpCode<kLiteRtOpCodeTflAdd>()));
}

TEST(MatchersTest, MatchInput) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  auto& input = subgraph.EmplaceTensor();
  AttachInput(&input, op);

  auto& def_op = subgraph.EmplaceOp();
  def_op.SetOpCode(kLiteRtOpCodeTflMul);
  AttachOutput(&input, def_op);

  Op cc_op(ctx, &op);

  // Match Add(Mul)
  EXPECT_TRUE(
      Match(cc_op, m_Op<kLiteRtOpCodeTflAdd>(m_Op<kLiteRtOpCodeTflMul>())));

  // Mismatch
  EXPECT_FALSE(
      Match(cc_op, m_Op<kLiteRtOpCodeTflAdd>(m_Op<kLiteRtOpCodeTflSub>())));
}

TEST(MatchersTest, CaptureOrSameAs_Capture) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  Op cc_op(ctx, &op);
  Op captured(ctx, nullptr);

  EXPECT_TRUE(
      Match(cc_op, m_CaptureOrSameAs(&captured, m_Op<kLiteRtOpCodeTflAdd>())));
  EXPECT_EQ(captured.Code(), kLiteRtOpCodeTflAdd);
  EXPECT_EQ(captured.Get(), &op);
}

TEST(MatchersTest, AnyMatchers) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  auto& tensor = subgraph.EmplaceTensor();
  AttachOutput(&tensor, op);

  Op cc_op(ctx, &op);
  Tensor cc_tensor(ctx, &tensor);

  EXPECT_TRUE(Match(cc_op, m_AnyOp()));
  EXPECT_TRUE(Match(cc_tensor, m_Any()));
  EXPECT_TRUE(Match(cc_tensor, m_AnyOp()));  // Tensor matches its defining op
}

TEST(MatchersTest, ConstantAndSubgraphInput) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& cst = subgraph.EmplaceTensor();
  SetWeightsFromOwnedBuffer(cst.Weights(), OwningBufferRef<uint8_t>("dummy"));

  auto& input = subgraph.EmplaceTensor();
  // No weights, no defining op -> subgraph input

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&cst, op);
  AttachInput(&input, op);

  Tensor cc_cst(ctx, &cst);
  Tensor cc_input(ctx, &input);

  EXPECT_TRUE(Match(cc_cst, m_IsConstant()));
  EXPECT_FALSE(Match(cc_cst, m_IsSubgraphInput()));

  EXPECT_TRUE(Match(cc_input, m_IsSubgraphInput()));
  EXPECT_FALSE(Match(cc_input, m_IsConstant()));

  Op cc_op(ctx, &op);
  EXPECT_TRUE(Match(
      cc_op, m_Op<kLiteRtOpCodeTflAdd>(m_IsConstant(), m_IsSubgraphInput())));
}

TEST(MatchersTest, Predicate) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  Op cc_op(ctx, &op);
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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto* current_tensor = &subgraph.EmplaceTensor();

  for (int i = 0; i < 4; ++i) {
    auto& op = subgraph.EmplaceOp();
    op.SetOpCode(static_cast<LiteRtOpCode>(i + 1));
    AttachInput(current_tensor, op);
    auto& next_tensor = subgraph.EmplaceTensor();
    AttachOutput(&next_tensor, op);
    current_tensor = &next_tensor;
  }

  Tensor last_tensor(ctx, current_tensor);
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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& in1 = subgraph.EmplaceTensor();
  auto& in2 = subgraph.EmplaceTensor();
  auto& in3 = subgraph.EmplaceTensor();

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&in1, op);
  AttachInput(&in2, op);
  AttachInput(&in3, op);

  Op cc_op(ctx, &op);
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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& in = subgraph.EmplaceTensor();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&in, op);
  AttachInput(&in, op);

  Op cc_op(ctx, &op);
  Tensor captured1(ctx, nullptr);
  Tensor captured2(ctx, nullptr);

  // Both matchers see the same tensor, but they don't know it's shared
  // unless we check the captured pointers.
  EXPECT_TRUE(Match(cc_op, m_Op<kLiteRtOpCodeTflMul>(
                               m_CaptureOrSameAs(&captured1, m_Any()),
                               m_CaptureOrSameAs(&captured2, m_Any()))));
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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& in = subgraph.EmplaceTensor();

  auto& op_left = subgraph.EmplaceOp();
  op_left.SetOpCode(kLiteRtOpCodeTflAbs);
  AttachInput(&in, op_left);
  auto& out_left = subgraph.EmplaceTensor();
  AttachOutput(&out_left, op_left);

  auto& op_right = subgraph.EmplaceOp();
  op_right.SetOpCode(kLiteRtOpCodeTflNeg);
  AttachInput(&in, op_right);
  auto& out_right = subgraph.EmplaceTensor();
  AttachOutput(&out_right, op_right);

  auto& op_final = subgraph.EmplaceOp();
  op_final.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&out_left, op_final);
  AttachInput(&out_right, op_final);

  Op cc_op(ctx, &op_final);
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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& in = subgraph.EmplaceTensor();
  auto& op1 = subgraph.EmplaceOp();
  op1.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&in, op1);
  AttachInput(&in, op1);
  auto& out1 = subgraph.EmplaceTensor();
  AttachOutput(&out1, op1);

  auto& op2 = subgraph.EmplaceOp();
  op2.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&out1, op2);
  auto& in2 = subgraph.EmplaceTensor();
  AttachInput(&in2, op2);

  Op root(ctx, &op2);
  Op cap_mul(ctx, nullptr);
  Op cap_add(ctx, nullptr);

  EXPECT_TRUE(Match(
      root, m_CaptureOrSameAs(
                &cap_add, m_Op<kLiteRtOpCodeTflAdd>(
                              m_CaptureOrSameAs(
                                  &cap_mul, m_OpCode<kLiteRtOpCodeTflMul>()),
                              m_Any()))));
  EXPECT_EQ(cap_add.Get(), &op2);
  EXPECT_EQ(cap_mul.Get(), &op1);
}

TEST(MatchersTest, ManyTestCasesTopology) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  for (int num_inputs = 0; num_inputs < 10; ++num_inputs) {
    LiteRtSubgraphT subgraph;
    auto& op = subgraph.EmplaceOp();
    op.SetOpCode(kLiteRtOpCodeTflCustom);
    for (int i = 0; i < num_inputs; ++i) {
      auto& in = subgraph.EmplaceTensor();
      AttachInput(&in, op);
    }

    Op cc_op(ctx, &op);
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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  for (int len = 1; len < 10; ++len) {
    LiteRtSubgraphT subgraph;
    auto* cur = &subgraph.EmplaceTensor();
    for (int i = 0; i < len; ++i) {
      auto& op = subgraph.EmplaceOp();
      op.SetOpCode(kLiteRtOpCodeTflAbs);
      AttachInput(cur, op);
      auto& out = subgraph.EmplaceTensor();
      AttachOutput(&out, op);
      cur = &out;
    }
    Tensor last(ctx, cur);
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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& in = subgraph.EmplaceTensor();

  auto& op1 = subgraph.EmplaceOp();
  op1.SetOpCode(kLiteRtOpCodeTflAbs);
  AttachInput(&in, op1);

  auto& op2 = subgraph.EmplaceOp();
  op2.SetOpCode(kLiteRtOpCodeTflNeg);
  AttachInput(&in, op2);

  // Matcher for op1 shouldn't care about op2.
  EXPECT_TRUE(Match(Op(ctx, &op1), m_Op<kLiteRtOpCodeTflAbs>(m_Any())));
}

/*
Topology:
    T_in -> Op --+-> T_out1
                 |
                 +-> T_out2
*/
TEST(MatchersTest, MultipleOutputs) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& in = subgraph.EmplaceTensor();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflSplit);
  AttachInput(&in, op);

  auto& out1 = subgraph.EmplaceTensor();
  AttachOutput(&out1, op);
  auto& out2 = subgraph.EmplaceTensor();
  AttachOutput(&out2, op);

  // OpMatcher only checks inputs.
  EXPECT_TRUE(Match(Op(ctx, &op), m_Op<kLiteRtOpCodeTflSplit>(m_Any())));

  // Tensor matching from different outputs.
  EXPECT_TRUE(Match(Tensor(ctx, &out1), m_Op<kLiteRtOpCodeTflSplit>(m_Any())));
  EXPECT_TRUE(Match(Tensor(ctx, &out2), m_Op<kLiteRtOpCodeTflSplit>(m_Any())));
}

/*
Topology:
    T1 -\
    T2 --\
    ...   -> Op -> T_out
    T8 --/
*/
TEST(MatchersTest, WideFanIn) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflConcatenation);
  for (int i = 0; i < 8; ++i) {
    auto& in = subgraph.EmplaceTensor();
    AttachInput(&in, op);
  }

  EXPECT_TRUE(Match(Op(ctx, &op), m_Op<kLiteRtOpCodeTflConcatenation>(
                                      m_Any(), m_Any(), m_Any(), m_Any(),
                                      m_Any(), m_Any(), m_Any(), m_Any())));
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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& in1 = subgraph.EmplaceTensor();
  auto& in2 = subgraph.EmplaceTensor();
  auto& op1 = subgraph.EmplaceOp();
  op1.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&in1, op1);
  AttachInput(&in2, op1);
  auto& out1 = subgraph.EmplaceTensor();
  AttachOutput(&out1, op1);

  auto& in3 = subgraph.EmplaceTensor();
  auto& in4 = subgraph.EmplaceTensor();
  auto& op2 = subgraph.EmplaceOp();
  op2.SetOpCode(kLiteRtOpCodeTflSub);
  AttachInput(&in3, op2);
  AttachInput(&in4, op2);
  auto& out2 = subgraph.EmplaceTensor();
  AttachOutput(&out2, op2);

  auto& op3 = subgraph.EmplaceOp();
  op3.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&out1, op3);
  AttachInput(&out2, op3);

  EXPECT_TRUE(Match(Op(ctx, &op3), m_Op<kLiteRtOpCodeTflAdd>(
                                       m_OpCode<kLiteRtOpCodeTflMul>(),
                                       m_OpCode<kLiteRtOpCodeTflSub>())));
}

TEST(MatchersTest, CaptureTensor) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& in = subgraph.EmplaceTensor();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAbs);
  AttachInput(&in, op);

  Tensor captured(ctx, nullptr);
  EXPECT_TRUE(Match(Op(ctx, &op), m_Op<kLiteRtOpCodeTflAbs>(
                                      m_CaptureOrSameAs(&captured, m_Any()))));
  EXPECT_EQ(captured.Get(), &in);
}

TEST(MatchersTest, PredicateOnTensor) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& in = subgraph.EmplaceTensor();
  in.SetName("my_tensor");

  EXPECT_TRUE(Match(Tensor(ctx, &in), m_Predicate<Tensor>([](const Tensor& t) {
                      return t.Name() == "my_tensor";
                    })));
}

TEST(MatchersTest, MatchOpCodeWithTensor) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflMul);
  auto& out = subgraph.EmplaceTensor();
  AttachOutput(&out, op);

  EXPECT_TRUE(Match(Tensor(ctx, &out), m_OpCode<kLiteRtOpCodeTflMul>()));
}

TEST(MatchersTest, Combinators) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  Op cc_op(ctx, &op);
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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& input = subgraph.EmplaceTensor();
  // No defining op (subgraph input)

  Tensor cc_input(ctx, &input);
  Op captured(ctx, nullptr);

  // Match succeeds on tensor level (m_Any), but capture Op fails because no
  // defining op.
  EXPECT_FALSE(Match(cc_input, m_CaptureOrSameAs(&captured, m_Any())));
}

TEST(MatchersTest, CustomOpMatching) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflCustom);
  op.SetCustomCode("MyCustomOp");

  auto& in = subgraph.EmplaceTensor();
  AttachInput(&in, op);

  Op cc_op(ctx, &op);

  // m_CustomOpCode matches code string
  EXPECT_TRUE(Match(cc_op, m_CustomOpCode("MyCustomOp")));
  EXPECT_FALSE(Match(cc_op, m_CustomOpCode("OtherOp")));

  // m_CustomOp matches code + inputs
  EXPECT_TRUE(Match(cc_op, m_CustomOp("MyCustomOp", m_Any())));
  EXPECT_FALSE(Match(cc_op, m_CustomOp("MyCustomOp")));  // Wrong input count
  EXPECT_FALSE(Match(cc_op, m_CustomOp("OtherOp", m_Any())));
}

TEST(MatchersTest, NameMatching) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& t = subgraph.EmplaceTensor();
  t.SetName("MyTensor");

  Tensor cc_t(ctx, &t);
  EXPECT_TRUE(Match(cc_t, m_Name("MyTensor")));
  EXPECT_FALSE(Match(cc_t, m_Name("Other")));
}

TEST(MatchersTest, CustomLambdaMatching) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  Op cc_op(ctx, &op);

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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
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

  Op cc_op(ctx, &op);

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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflConcatenation);

  auto& in1 = subgraph.EmplaceTensor();
  auto& in2 = subgraph.EmplaceTensor();
  auto& in3 = subgraph.EmplaceTensor();
  AttachInput(&in1, op);
  AttachInput(&in2, op);
  AttachInput(&in3, op);

  Op cc_op(ctx, &op);

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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  auto& in1 = subgraph.EmplaceTensor();
  in1.SetName("A");
  auto& in2 = subgraph.EmplaceTensor();
  in2.SetName("B");
  AttachInput(&in1, op);
  AttachInput(&in2, op);

  Op cc_op(ctx, &op);

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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
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
  AttachInput(&in1, op);
  AttachInput(&in2, op);

  Op cc_op(ctx, &op);

  auto match_opts = m_Options<AddOptions>([](const AddOptions& o) {
    return o.fused_activation_function == kActivationFunctionTypeRelu;
  });

  // Match OpCode, Options, and Variadic Inputs
  EXPECT_TRUE(Match(
      cc_op, m_AllOf(m_OpCode<kLiteRtOpCodeTflAdd>(), match_opts,
                     m_OpVariadic<kLiteRtOpCodeTflAdd>(m_Any(), m_Any()))));
}

TEST(MatchersTest, CommutativeOpFailCount) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  auto& in1 = subgraph.EmplaceTensor();
  AttachInput(&in1, op);
  // Only 1 input

  Op cc_op(ctx, &op);
  EXPECT_FALSE(
      Match(cc_op, m_CommutativeOp<kLiteRtOpCodeTflAdd>(m_Any(), m_Any())));
}

TEST(MatchersTest, ShapeMatching) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  tensor.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 224, 224, 3}));

  Tensor cc_tensor(ctx, &tensor);
  EXPECT_TRUE(Match(cc_tensor, m_Shape({1, 224, 224, 3})));
  // Rank mismatch
  EXPECT_FALSE(Match(cc_tensor, m_Shape({1, 224, 224})));
  // Dimension mismatch
  EXPECT_FALSE(Match(cc_tensor, m_Shape({1, 224, 224, 1})));
  // Empty shape check (scalar) - fail
  EXPECT_FALSE(Match(cc_tensor, m_Shape({})));
}

TEST(MatchersTest, ShapeWildcardMatching) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  tensor.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 224, 224, 3}));

  Tensor cc_tensor(ctx, &tensor);
  // Match exact
  EXPECT_TRUE(Match(cc_tensor, m_Shape({1, 224, 224, 3})));
  // Match with wildcards
  EXPECT_TRUE(Match(cc_tensor, m_Shape({1, -1, -1, 3})));
  EXPECT_TRUE(Match(cc_tensor, m_Shape({-1, -1, -1, -1})));

  // Mismatch
  EXPECT_FALSE(Match(cc_tensor, m_Shape({1, -1, 100, 3})));
}

TEST(MatchersTest, RankMatching) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  tensor.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 224, 224, 3}));

  Tensor cc_tensor(ctx, &tensor);
  EXPECT_TRUE(Match(cc_tensor, m_Rank(4)));
  EXPECT_FALSE(Match(cc_tensor, m_Rank(3)));
  EXPECT_FALSE(Match(cc_tensor, m_Rank(5)));
}

TEST(MatchersTest, ElementTypeMatching) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1}));

  Tensor cc_tensor(ctx, &tensor);
  EXPECT_TRUE(Match(cc_tensor, m_ElementType(kLiteRtElementTypeFloat32)));
  EXPECT_FALSE(Match(cc_tensor, m_ElementType(kLiteRtElementTypeInt32)));
  EXPECT_FALSE(Match(cc_tensor, m_ElementType(kLiteRtElementTypeBool)));
}

TEST(MatchersTest, OneUseMatching) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  Tensor cc_tensor(ctx, &tensor);

  // 0 uses
  EXPECT_FALSE(Match(cc_tensor, m_HasOneUse()));

  // 1 use
  auto& op1 = subgraph.EmplaceOp();
  op1.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&tensor, op1);
  EXPECT_TRUE(Match(cc_tensor, m_HasOneUse()));

  // 2 uses
  auto& op2 = subgraph.EmplaceOp();
  op2.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&tensor, op2);
  EXPECT_FALSE(Match(cc_tensor, m_HasOneUse()));
}

TEST(MatchersTest, ConstantValueMatching) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1}));

  // Set value 1.0
  float data = 1.0f;
  std::vector<uint8_t> bytes(sizeof(float));
  std::memcpy(bytes.data(), &data, sizeof(float));
  SetWeightsFromOwnedBuffer(tensor.Weights(),
                            OwningBufferRef<uint8_t>(std::move(bytes)));

  Tensor cc_tensor(ctx, &tensor);
  EXPECT_TRUE(Match(cc_tensor, m_IsConstant()));
  EXPECT_TRUE(Match(cc_tensor, m_ConstantValue<float>(1.0f)));
  EXPECT_FALSE(Match(cc_tensor, m_ConstantValue<float>(0.0f)));

  // Wrong Type
  EXPECT_FALSE(Match(cc_tensor, m_ConstantValue<int32_t>(1)));
}

TEST(MatchersTest, QuantizationMatching) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {1}));

  Tensor cc_tensor(ctx, &tensor);
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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  Tensor cc_tensor(ctx, &tensor);

  EXPECT_TRUE(Match(cc_tensor, m_HasUsers(0)));
  EXPECT_FALSE(Match(cc_tensor, m_HasUsers(1)));

  auto& op1 = subgraph.EmplaceOp();
  AttachInput(&tensor, op1);
  EXPECT_TRUE(Match(cc_tensor, m_HasUsers(1)));
  EXPECT_TRUE(Match(cc_tensor, m_HasOneUse()));

  auto& op2 = subgraph.EmplaceOp();
  AttachInput(&tensor, op2);
  EXPECT_TRUE(Match(cc_tensor, m_HasUsers(2)));
}

TEST(MatchersTest, CaptureOrSameAs_SameAs) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& t1 = subgraph.EmplaceTensor();
  auto& t2 = subgraph.EmplaceTensor();

  Tensor cc_t1(ctx, &t1);
  Tensor cc_t2(ctx, &t2);

  Tensor captured(ctx, nullptr);

  // Capture t1, match t1 against captured (true)
  EXPECT_TRUE(Match(cc_t1, m_AllOf(m_CaptureOrSameAs(&captured, m_Any()),
                                   m_CaptureOrSameAs(&captured, m_Any()))));

  // Capture t1, match t2 against captured (false)
  EXPECT_FALSE(Match(cc_t2, m_AllOf(m_CaptureOrSameAs(&captured, m_Any()),
                                    m_CaptureOrSameAs(&cc_t1, m_Any()))));

  // Reset captured
  captured = Tensor(ctx, nullptr);

  Match(cc_t1, m_CaptureOrSameAs(&captured, m_Any()));
  EXPECT_TRUE(Match(cc_t1, m_CaptureOrSameAs(&captured, m_Any())));
  EXPECT_FALSE(Match(cc_t2, m_CaptureOrSameAs(&captured, m_Any())));
}

TEST(MatchersTest, OutputIndexMatching) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflSplit);
  auto& out0 = subgraph.EmplaceTensor();
  AttachOutput(&out0, op);
  auto& out1 = subgraph.EmplaceTensor();
  AttachOutput(&out1, op);

  Tensor cc_out0(ctx, &out0);
  Tensor cc_out1(ctx, &out1);

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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& input = subgraph.EmplaceTensor();

  auto& split_op = subgraph.EmplaceOp();
  split_op.SetOpCode(kLiteRtOpCodeTflSplit);
  AttachInput(&input, split_op);
  auto& split_out0 = subgraph.EmplaceTensor();
  AttachOutput(&split_out0, split_op);
  auto& split_out1 = subgraph.EmplaceTensor();
  AttachOutput(&split_out1, split_op);

  auto& conv_op = subgraph.EmplaceOp();
  conv_op.SetOpCode(kLiteRtOpCodeTflConv2d);
  AttachInput(&split_out0, conv_op);
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
  AttachOutput(&conv_out, conv_op);

  auto& add_op = subgraph.EmplaceOp();
  add_op.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&conv_out, add_op);
  AttachInput(&split_out1, add_op);

  Op root(ctx, &add_op);

  auto match_split = m_OpCode<kLiteRtOpCodeTflSplit>();

  auto match_conv_opts = m_Options<Conv2dOptions>([](const Conv2dOptions& o) {
    return o.stride_w == 1 && o.stride_h == 1;
  });

  auto match_conv =
      m_AllOf(m_Op<kLiteRtOpCodeTflConv2d>(m_OutputIndex(0, match_split)),
              match_conv_opts);

  auto match_resnet_add = m_CommutativeOp<kLiteRtOpCodeTflAdd>(
      match_conv, m_OutputIndex(1, match_split));

  EXPECT_TRUE(Match(root, match_resnet_add));
}

TEST(MatchersTest, VariadicTypedConcat) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflConcatenation);

  for (int i = 0; i < 3; ++i) {
    auto& t = subgraph.EmplaceTensor();
    t.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 10}));
    AttachInput(&t, op);
  }

  Op root(ctx, &op);

  auto float_tensor = m_ElementType(kLiteRtElementTypeFloat32);

  EXPECT_TRUE(Match(root, m_OpVariadic<kLiteRtOpCodeTflConcatenation>(
                              float_tensor, float_tensor, float_tensor)));

  subgraph.Tensors().back()->SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt32, {1, 10}));
  EXPECT_FALSE(Match(root, m_OpVariadic<kLiteRtOpCodeTflConcatenation>(
                               float_tensor, float_tensor, float_tensor)));
}

TEST(MatchersTest, AllInOneIntegration) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
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
  AttachInput(&cst, mul_op);
  AttachInput(&input, mul_op);
  auto& t_mul = subgraph.EmplaceTensor();
  t_mul.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 2}));
  AttachOutput(&t_mul, mul_op);

  // 3. Add Op (Options)
  auto& add_op = subgraph.EmplaceOp();
  add_op.SetOpCode(kLiteRtOpCodeTflAdd);
  tflite::AddOptionsT add_opts;
  add_opts.fused_activation_function = tflite::ActivationFunctionType_RELU;
  litert::internal::TflOptions tfl_add_opts;
  tfl_add_opts.type = tflite::BuiltinOptions_AddOptions;
  tfl_add_opts.Set(std::move(add_opts));
  litert::internal::SetTflOptions(add_op, std::move(tfl_add_opts));

  AttachInput(&t_mul, add_op);
  AttachInput(&t_mul, add_op);
  auto& t_add = subgraph.EmplaceTensor();
  t_add.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 2}));
  AttachOutput(&t_add, add_op);

  // 4. Split Op (Output Index)
  auto& split_op = subgraph.EmplaceOp();
  split_op.SetOpCode(kLiteRtOpCodeTflSplit);
  AttachInput(&t_add, split_op);
  auto& t_split_0 = subgraph.EmplaceTensor();
  t_split_0.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 1}));
  AttachOutput(&t_split_0, split_op);
  auto& t_split_1 = subgraph.EmplaceTensor();
  t_split_1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 1}));
  AttachOutput(&t_split_1, split_op);

  // 5. Concat Op (Variadic)
  auto& concat_op = subgraph.EmplaceOp();
  concat_op.SetOpCode(kLiteRtOpCodeTflConcatenation);
  AttachInput(&t_split_0, concat_op);
  AttachInput(&t_split_1, concat_op);
  AttachInput(&t_mul, concat_op);

  Op root(ctx, &concat_op);
  Op captured_add(ctx, nullptr);

  auto match_mul_inputs = m_CommutativeOp<kLiteRtOpCodeTflMul>(
      m_AllOf(m_IsConstant(), m_ElementType(kLiteRtElementTypeFloat32)),
      m_AllOf(m_IsSubgraphInput(), m_Shape({1, 2})));

  auto match_add = m_CaptureOrSameAs(
      &captured_add,
      m_AllOf(m_OpCode<kLiteRtOpCodeTflAdd>(),
              m_Options<AddOptions>([](const AddOptions& o) {
                return o.fused_activation_function ==
                       kActivationFunctionTypeRelu;
              }),
              m_Op<kLiteRtOpCodeTflAdd>(match_mul_inputs, match_mul_inputs)));

  auto match_split_out0 =
      m_OutputIndex(0, m_Op<kLiteRtOpCodeTflSplit>(match_add));
  auto match_split_out1 = m_OutputIndex(1, m_OpCode<kLiteRtOpCodeTflSplit>());

  auto match_root = m_AllOf(m_OpVariadic<kLiteRtOpCodeTflConcatenation>(
                                match_split_out0, match_split_out1,
                                m_AnyOf(match_mul_inputs, m_IsConstant())),
                            m_Not(m_OpCode<kLiteRtOpCodeTflAdd>()));

  EXPECT_TRUE(Match(root, match_root));
  EXPECT_EQ(captured_add.Get(), &add_op);
}

TEST(MatchersTest, LabeledVariadicOpTest) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  Op cc_op(ctx, &op);
  LoggingMatchTracer tracer;

  bool res = Match(cc_op, m_Op<kLiteRtOpCodeTflAdd>(m_Any(), m_Any(), "MyAdd"),
                   &tracer);
  EXPECT_FALSE(res);

  ASSERT_GE(tracer.logs().size(), 1);
  EXPECT_EQ(tracer.logs()[0].type, "Start");
  EXPECT_EQ(tracer.logs()[0].name, "MyAdd");
  EXPECT_EQ(tracer.logs()[1].type, "Fail");
  EXPECT_EQ(tracer.logs()[1].name, "MyAdd");
}

TEST(MatchersTest, NestedTraceScope) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  auto& in = subgraph.EmplaceTensor();
  AttachInput(&in, op);

  Op cc_op(ctx, &op);
  LoggingMatchTracer tracer;

  bool res = Match(cc_op, m_Op<kLiteRtOpCodeTflAdd>(m_Any(), m_Any()), &tracer);
  EXPECT_FALSE(res);

  ASSERT_GE(tracer.logs().size(), 2);
  EXPECT_EQ(tracer.logs()[0].type, "Start");
  EXPECT_EQ(tracer.logs()[0].name, "OpMatcher");
  EXPECT_EQ(tracer.logs()[1].type, "Fail");
  EXPECT_EQ(tracer.logs()[1].name, "OpMatcher");
  EXPECT_EQ(tracer.logs()[1].reason, "Input count mismatch");
}

TEST(MatchersTest, AllMatchersFailureLogging) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;

  auto& add_op = subgraph.EmplaceOp();
  add_op.SetOpCode(kLiteRtOpCodeTflAdd);

  auto& tensor_float = subgraph.EmplaceTensor();
  tensor_float.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 2}));
  tensor_float.SetName("FloatTensor");

  auto& tensor_const = subgraph.EmplaceTensor();
  tensor_const.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));
  int32_t val = 42;
  std::vector<uint8_t> buf(sizeof(val));
  std::memcpy(buf.data(), &val, sizeof(val));
  SetWeightsFromOwnedBuffer(tensor_const.Weights(),
                            OwningBufferRef<uint8_t>(std::move(buf)));

  AttachInput(&tensor_float, add_op);

  Op cc_op(ctx, &add_op);
  Tensor cc_tensor_float(ctx, &tensor_float);
  Tensor cc_tensor_const(ctx, &tensor_const);

  auto VerifyFailure = [&](bool debug_res, const auto& matcher, const auto& val,
                           absl::string_view expected_label,
                           absl::string_view expected_reason) {
    EXPECT_FALSE(debug_res);
    LoggingMatchTracer tracer;
    EXPECT_FALSE(Match(val, matcher, &tracer));
    bool found = false;
    for (const auto& log : tracer.logs()) {
      if (log.type == "Fail" && log.name == expected_label &&
          log.reason == expected_reason) {
        found = true;
        break;
      }
    }
    if (!found) {
      for (const auto& log : tracer.logs()) {
        std::cerr << "Log: " << log.type << " " << log.name << ": "
                  << log.reason << "\n";
      }
    }
    EXPECT_TRUE(found) << "Expected failure [" << expected_label
                       << "]: " << expected_reason;
  };

  // 1. m_OpCode
  VerifyFailure(DebugMatch(cc_op, m_OpCode<kLiteRtOpCodeTflMul>("LblOpCode")),
                m_OpCode<kLiteRtOpCodeTflMul>("LblOpCode"), cc_op, "LblOpCode",
                "OpCode mismatch");

  // 2. m_Op (OpCode mismatch)
  VerifyFailure(DebugMatch(cc_op, m_Op<kLiteRtOpCodeTflMul>("LblOp")),
                m_Op<kLiteRtOpCodeTflMul>("LblOp"), cc_op, "LblOp",
                "OpCode mismatch");

  // 3. m_Op (Input count mismatch)
  VerifyFailure(DebugMatch(cc_op, m_Op<kLiteRtOpCodeTflAdd>("LblOpCount")),
                m_Op<kLiteRtOpCodeTflAdd>("LblOpCount"), cc_op, "LblOpCount",
                "Input count mismatch");

  // 4. m_OpVariadic (Insufficient inputs)
  VerifyFailure(DebugMatch(cc_op, m_OpVariadic<kLiteRtOpCodeTflAdd>(
                                      m_Any(), m_Any(), "LblVar")),
                m_OpVariadic<kLiteRtOpCodeTflAdd>(m_Any(), m_Any(), "LblVar"),
                cc_op, "LblVar", "Input count mismatch (insufficient inputs)");

  // 5. m_CommutativeOp (Input count mismatch)
  VerifyFailure(
      DebugMatch(cc_op, m_CommutativeOp<kLiteRtOpCodeTflAdd>(m_Any(), m_Any(),
                                                             "LblComm")),
      m_CommutativeOp<kLiteRtOpCodeTflAdd>(m_Any(), m_Any(), "LblComm"), cc_op,
      "LblComm", "Input count mismatch (expected 2)");

  // 6. m_Options
  auto& conv_op = subgraph.EmplaceOp();
  conv_op.SetOpCode(kLiteRtOpCodeTflConv2d);
  tflite::Conv2DOptionsT opts;
  opts.stride_w = 1;
  litert::internal::TflOptions tfl_opts;
  tfl_opts.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_opts.Set(std::move(opts));
  litert::internal::SetTflOptions(conv_op, std::move(tfl_opts));
  Op cc_conv(ctx, &conv_op);

  VerifyFailure(
      DebugMatch(cc_conv,
                 m_Options<Conv2dOptions>(
                     [](const auto& o) { return o.stride_w == 2; }, "LblOpt")),
      m_Options<Conv2dOptions>([](const auto& o) { return o.stride_w == 2; },
                               "LblOpt"),
      cc_conv, "LblOpt", "Predicate returned false");

  // 7. m_Shape
  VerifyFailure(DebugMatch(cc_tensor_float, m_Shape({1, 3}, "LblShape")),
                m_Shape({1, 3}, "LblShape"), cc_tensor_float, "LblShape",
                "Dimension mismatch");

  // 8. m_Rank
  VerifyFailure(DebugMatch(cc_tensor_float, m_Rank(3, "LblRank")),
                m_Rank(3, "LblRank"), cc_tensor_float, "LblRank",
                "Rank mismatch");

  // 9. m_ElementType
  VerifyFailure(DebugMatch(cc_tensor_float,
                           m_ElementType(kLiteRtElementTypeInt32, "LblType")),
                m_ElementType(kLiteRtElementTypeInt32, "LblType"),
                cc_tensor_float, "LblType", "Type mismatch");

  // 10. m_OutputIndex
  VerifyFailure(
      DebugMatch(cc_tensor_float, m_OutputIndex(0, m_AnyOp(), "LblOutIdx")),
      m_OutputIndex(0, m_AnyOp(), "LblOutIdx"), cc_tensor_float, "LblOutIdx",
      "No defining op found");

  // 11. m_Capture (sub-matcher fail)
  Op captured(ctx, nullptr);
  VerifyFailure(
      DebugMatch(cc_op,
                 m_CaptureOrSameAs(&captured, m_OpCode<kLiteRtOpCodeTflMul>(),
                                   "LblCap")),
      m_CaptureOrSameAs(&captured, m_OpCode<kLiteRtOpCodeTflMul>(), "LblCap"),
      cc_op, "OpCodeMatcher", "OpCode mismatch");

  // 12. m_IsConstant
  VerifyFailure(DebugMatch(cc_tensor_float, m_IsConstant("LblConst")),
                m_IsConstant("LblConst"), cc_tensor_float, "LblConst",
                "Tensor is not constant");

  // 13. m_IsSubgraphInput
  VerifyFailure(DebugMatch(cc_tensor_const, m_IsSubgraphInput("LblInput")),
                m_IsSubgraphInput("LblInput"), cc_tensor_const, "LblInput",
                "Tensor is not subgraph input");

  // 14. m_Predicate
  VerifyFailure(
      DebugMatch(cc_op,
                 m_Predicate<Op>([](const Op&) { return false; }, "LblPred")),
      m_Predicate<Op>([](const Op&) { return false; }, "LblPred"), cc_op,
      "LblPred", "Predicate returned false");

  // 15. m_Custom
  VerifyFailure(
      DebugMatch(cc_op, m_Custom([](const Op&) { return false; }, "LblCustom")),
      m_Custom([](const Op&) { return false; }, "LblCustom"), cc_op,
      "LblCustom", "Predicate returned false");

  // 16. m_AllOf
  VerifyFailure(DebugMatch(cc_op, m_AllOf(m_OpCode<kLiteRtOpCodeTflMul>(),
                                          m_AnyOp(), "LblAllOf")),
                m_AllOf(m_OpCode<kLiteRtOpCodeTflMul>(), m_AnyOp(), "LblAllOf"),
                cc_op, "OpCodeMatcher", "OpCode mismatch");

  // 17. m_AnyOf (All fail)
  VerifyFailure(
      DebugMatch(cc_op, m_AnyOf(m_OpCode<kLiteRtOpCodeTflMul>(),
                                m_OpCode<kLiteRtOpCodeTflSub>(), "LblAnyOf")),
      m_AnyOf(m_OpCode<kLiteRtOpCodeTflMul>(), m_OpCode<kLiteRtOpCodeTflSub>(),
              "LblAnyOf"),
      cc_op, "LblAnyOf", "All sub-matchers failed");

  // 18. m_Not (Sub-matcher matched)
  VerifyFailure(
      DebugMatch(cc_op, m_Not(m_OpCode<kLiteRtOpCodeTflAdd>(), "LblNot")),
      m_Not(m_OpCode<kLiteRtOpCodeTflAdd>(), "LblNot"), cc_op, "LblNot",
      "Sub-matcher matched (expected failure)");

  // 19. m_SameAs
  Tensor cc_tensor_const_copy(ctx, &tensor_const);
  VerifyFailure(
      DebugMatch(cc_tensor_float,
                 m_CaptureOrSameAs(&cc_tensor_const_copy, m_Any(), "LblSame")),
      m_CaptureOrSameAs(&cc_tensor_const_copy, m_Any(), "LblSame"),
      cc_tensor_float, "LblSame", "Object mismatch");

  // 20. m_ConstantValue
  VerifyFailure(
      DebugMatch(cc_tensor_const, m_ConstantValue<int32_t>(99, "LblVal")),
      m_ConstantValue<int32_t>(99, "LblVal"), cc_tensor_const, "LblVal",
      "Value mismatch");

  // 21. m_CustomOpCode
  VerifyFailure(DebugMatch(cc_op, m_CustomOpCode("MyOp", "LblCustCode")),
                m_CustomOpCode("MyOp", "LblCustCode"), cc_op, "LblCustCode",
                "Not a custom op");

  // 22. m_CustomOp
  VerifyFailure(DebugMatch(cc_op, m_CustomOp("MyOp", "LblCustOp")),
                m_CustomOp("MyOp", "LblCustOp"), cc_op, "LblCustOp",
                "Not a custom op");

  // 23. m_Name
  VerifyFailure(DebugMatch(cc_tensor_float, m_Name("WrongName", "LblName")),
                m_Name("WrongName", "LblName"), cc_tensor_float, "LblName",
                "Name mismatch");

  // 24. m_IsQuantized
  VerifyFailure(DebugMatch(cc_tensor_float, m_IsQuantized("LblQuant")),
                m_IsQuantized("LblQuant"), cc_tensor_float, "LblQuant",
                "Tensor is not quantized");

  // 25. m_QType
  VerifyFailure(DebugMatch(cc_tensor_float,
                           m_QType(kLiteRtQuantizationPerTensor, "LblQType")),
                m_QType(kLiteRtQuantizationPerTensor, "LblQType"),
                cc_tensor_float, "LblQType", "Quantization type mismatch");

  // 26. m_HasUsers
  VerifyFailure(DebugMatch(cc_tensor_float, m_HasUsers(2, "LblUsers")),
                m_HasUsers(2, "LblUsers"), cc_tensor_float, "LblUsers",
                "User count mismatch");
}

}  // namespace
}  // namespace litert::compiler

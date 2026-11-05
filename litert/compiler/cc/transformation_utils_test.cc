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

#include "litert/compiler/cc/transformation_utils.h"

#include <cstdint>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_op_options.h"
#include "litert/core/model/model.h"

namespace litert::compiler {
namespace {

using ::testing::ElementsAre;

TEST(TransformationUtilsTest, GetTensorNamespace) {
  EXPECT_EQ(GetTensorNamespace(""), "");
  EXPECT_EQ(GetTensorNamespace("foo"), "");
  EXPECT_EQ(GetTensorNamespace("/foo"), "");
  EXPECT_EQ(GetTensorNamespace("foo/bar"), "foo");
  EXPECT_EQ(GetTensorNamespace("foo/bar/"), "foo/bar");
  EXPECT_EQ(GetTensorNamespace("model/layers.0/self_attn/q_proj/MatMul"),
            "model/layers.0/self_attn/q_proj");
  EXPECT_EQ(GetTensorNamespace("a/b/c;d/e/f"), "a/b");
  EXPECT_EQ(GetTensorNamespace("a/b;c/d;e/f"), "a");
  EXPECT_EQ(GetTensorNamespace(";"), "");
}

TEST(TransformationUtilsTest, MakeNamespacedTensorName) {
  EXPECT_EQ(MakeNamespacedTensorName("model/layers.0/self_attn", "logits"),
            "model/layers.0/self_attn/logits");
  EXPECT_EQ(MakeNamespacedTensorName("model/layers.0/self_attn", "head", 0),
            "model/layers.0/self_attn/head_0");
  EXPECT_EQ(MakeNamespacedTensorName("", "logits"), "logits");
  EXPECT_EQ(MakeNamespacedTensorName("", "head", 3), "head_3");
}

TEST(TransformationUtilsTest, MakeNamespacedTensorNameFromTensor) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  Builder builder(ctx, &builder_impl);

  RankedTensorType type(ElementType::Float32, Layout(BuildLayout({2, 4})));
  auto tensor_with_ns =
      builder.BuildTensor(RankedTensorSpecBuilder(type)
                              .WithTensorName("model/layers.0/attn/input")
                              .Build());
  ASSERT_TRUE(tensor_with_ns.HasValue());
  EXPECT_EQ(MakeNamespacedTensorName(*tensor_with_ns, "output"),
            "model/layers.0/attn/output");
  EXPECT_EQ(MakeNamespacedTensorName(*tensor_with_ns, "split", 2),
            "model/layers.0/attn/split_2");

  auto tensor_without_ns = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("input").Build());
  ASSERT_TRUE(tensor_without_ns.HasValue());
  EXPECT_EQ(MakeNamespacedTensorName(*tensor_without_ns, "output"), "output");
  EXPECT_EQ(MakeNamespacedTensorName(*tensor_without_ns, "split", 1),
            "split_1");
}

TEST(TransformationUtilsTest, NewLikeWithNamespace) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  Builder builder(ctx, &builder_impl);

  RankedTensorType type(ElementType::Float32, Layout(BuildLayout({2, 4})));
  auto orig_spec = RankedTensorSpecBuilder(type)
                       .WithTensorName("model/transformer/layer_0/input_tensor")
                       .Build();
  auto orig_tensor = builder.BuildTensor(orig_spec);
  ASSERT_TRUE(orig_tensor.HasValue());
  EXPECT_EQ(orig_tensor->Name(), "model/transformer/layer_0/input_tensor");
  EXPECT_EQ(GetTensorNamespace(*orig_tensor), "model/transformer/layer_0");

  // NewLike with tag copies namespace
  auto new_tensor = NewLike(builder, *orig_tensor, {2, 8}, "Softmax");
  ASSERT_TRUE(new_tensor.HasValue());
  EXPECT_EQ(new_tensor->Name(), "model/transformer/layer_0/Softmax");
  EXPECT_TRUE(HasDims(*new_tensor, {2, 8}));
  EXPECT_EQ(new_tensor->ElementType(), ElementType::Float32);

  // NewLike without tag creates tensor with empty name
  auto anon_tensor = NewLike(builder, *orig_tensor, {3, 5});
  ASSERT_TRUE(anon_tensor.HasValue());
  EXPECT_EQ(anon_tensor->Name(), "");
  EXPECT_TRUE(HasDims(*anon_tensor, {3, 5}));
}

TEST(TransformationUtilsTest, NewLikePreservesPerTensorQuantization) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  Builder builder(ctx, &builder_impl);

  RankedTensorType type(ElementType::Int8, Layout(BuildLayout({2, 4})));
  LiteRtQuantizationPerTensor q_params{/*scale=*/0.5f, /*zero_point=*/-10};
  auto orig_spec = RankedTensorSpecBuilder(type)
                       .WithTensorName("quant_layer/input")
                       .WithPerTensorQuantization(q_params)
                       .Build();
  auto orig_tensor = builder.BuildTensor(orig_spec);
  ASSERT_TRUE(orig_tensor.HasValue());
  EXPECT_EQ(orig_tensor->QTypeId(), kLiteRtQuantizationPerTensor);
  EXPECT_FLOAT_EQ(orig_tensor->PerTensorQuantization().scale, 0.5f);
  EXPECT_EQ(orig_tensor->PerTensorQuantization().zero_point, -10);

  auto new_tensor = NewLike(builder, *orig_tensor, {4, 8}, "output");
  ASSERT_TRUE(new_tensor.HasValue());
  EXPECT_EQ(new_tensor->ElementType(), ElementType::Int8);
  EXPECT_EQ(new_tensor->QTypeId(), kLiteRtQuantizationPerTensor);
  EXPECT_FLOAT_EQ(new_tensor->PerTensorQuantization().scale, 0.5f);
  EXPECT_EQ(new_tensor->PerTensorQuantization().zero_point, -10);
  EXPECT_EQ(new_tensor->Name(), "quant_layer/output");
}

TEST(TransformationUtilsTest, ConstI32AndHasDims) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  Builder builder(ctx, &builder_impl);

  std::vector<int32_t> data = {1, 2, 3, 4};
  auto c = ConstI32(builder, data);
  ASSERT_TRUE(c.HasValue());
  EXPECT_TRUE(HasDims(*c, {4}));
  EXPECT_FALSE(HasDims(*c, {2, 2}));
  EXPECT_FALSE(HasDims(*c, {5}));
  EXPECT_FALSE(HasDims(*c, {}));

  EXPECT_TRUE(c->IsConstant());
  auto weights_data = c->WeightsData<int32_t>();
  ASSERT_TRUE(weights_data.HasValue());
  EXPECT_THAT(*weights_data, ElementsAre(1, 2, 3, 4));
}

TEST(TransformationUtilsTest, HasPerm) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  Builder builder(ctx, &builder_impl);

  RankedTensorType type(ElementType::Float32,
                        Layout(BuildLayout({1, 2, 3, 4})));
  auto in_tensor = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("in").Build());
  ASSERT_TRUE(in_tensor.HasValue());

  // Non-transpose op returns false
  auto add_out = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("add_out").Build());
  ASSERT_TRUE(add_out.HasValue());
  auto add_op = builder.BuildOp(kLiteRtOpCodeTflAdd, {*in_tensor, *in_tensor},
                                {*add_out});
  ASSERT_TRUE(add_op.HasValue());
  EXPECT_FALSE(HasPerm(*add_op, {0, 2, 1, 3}));

  // Transpose op with non-constant perm returns false
  RankedTensorType perm_type(ElementType::Int32, Layout(BuildLayout({4})));
  auto non_const_perm = builder.BuildTensor(
      RankedTensorSpecBuilder(perm_type).WithTensorName("perm").Build());
  ASSERT_TRUE(non_const_perm.HasValue());
  auto non_const_out = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("non_const_out").Build());
  ASSERT_TRUE(non_const_out.HasValue());
  auto non_const_transpose =
      builder.BuildOp(kLiteRtOpCodeTflTranspose, {*in_tensor, *non_const_perm},
                      {*non_const_out});
  ASSERT_TRUE(non_const_transpose.HasValue());
  EXPECT_FALSE(HasPerm(*non_const_transpose, {0, 2, 1, 3}));

  // Transpose op with constant perm matching expected returns true
  auto perm_const = ConstI32(builder, {0, 2, 1, 3});
  ASSERT_TRUE(perm_const.HasValue());
  auto valid_out = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("valid_out").Build());
  ASSERT_TRUE(valid_out.HasValue());
  auto valid_transpose = builder.BuildOp(
      kLiteRtOpCodeTflTranspose, {*in_tensor, *perm_const}, {*valid_out});
  ASSERT_TRUE(valid_transpose.HasValue());
  EXPECT_TRUE(HasPerm(*valid_transpose, {0, 2, 1, 3}));
  EXPECT_FALSE(HasPerm(*valid_transpose, {0, 1, 2, 3}));
}

TEST(TransformationUtilsTest, BuildSlice) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  Builder builder(ctx, &builder_impl);

  RankedTensorType type(ElementType::Float32, Layout(BuildLayout({4, 8})));
  auto orig = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("layer/input").Build());
  ASSERT_TRUE(orig.HasValue());

  auto slice = BuildSlice(builder, *orig, {0, 0}, {2, 4}, *orig, "Slice");
  ASSERT_TRUE(slice.HasValue());
  EXPECT_EQ(slice->Name(), "layer/Slice");
  EXPECT_TRUE(HasDims(*slice, {2, 4}));

  auto def_op = slice->GetDefiningOp();
  ASSERT_TRUE(def_op.HasValue());
  EXPECT_EQ(def_op->Code(), kLiteRtOpCodeTflSlice);
  EXPECT_EQ(def_op->Inputs().size(), 3);
  EXPECT_EQ(def_op->Inputs()[0], *orig);

  auto begin_weights = def_op->Inputs()[1].WeightsData<int32_t>();
  ASSERT_TRUE(begin_weights.HasValue());
  EXPECT_THAT(*begin_weights, ElementsAre(0, 0));

  auto size_weights = def_op->Inputs()[2].WeightsData<int32_t>();
  ASSERT_TRUE(size_weights.HasValue());
  EXPECT_THAT(*size_weights, ElementsAre(2, 4));
}

TEST(TransformationUtilsTest, BuildConcatAndBuildConcatInto) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  Builder builder(ctx, &builder_impl);

  RankedTensorType type(ElementType::Float32, Layout(BuildLayout({2, 3})));
  auto in1 = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("model/in1").Build());
  auto in2 = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("model/in2").Build());
  ASSERT_TRUE(in1.HasValue());
  ASSERT_TRUE(in2.HasValue());

  // Test BuildConcat
  auto concat =
      BuildConcat(builder, {*in1, *in2}, /*axis=*/1, *in1, {2, 6}, "Concat");
  ASSERT_TRUE(concat.HasValue());
  EXPECT_EQ(concat->Name(), "model/Concat");
  EXPECT_TRUE(HasDims(*concat, {2, 6}));

  auto def_op = concat->GetDefiningOp();
  ASSERT_TRUE(def_op.HasValue());
  EXPECT_EQ(def_op->Code(), kLiteRtOpCodeTflConcatenation);
  EXPECT_EQ(def_op->Inputs().size(), 2);
  EXPECT_EQ(def_op->Outputs().size(), 1);

  auto opts = GetOptionsAs<ConcatenationOptions>(ctx, def_op->Get());
  ASSERT_TRUE(opts.HasValue());
  EXPECT_EQ(opts->axis, 1);
  EXPECT_EQ(opts->fused_activation_function, 0);

  // Test BuildConcatInto
  RankedTensorType out_type(ElementType::Float32, Layout(BuildLayout({4, 3})));
  auto custom_out =
      builder.BuildTensor(RankedTensorSpecBuilder(out_type)
                              .WithTensorName("model/custom_concat")
                              .Build());
  ASSERT_TRUE(custom_out.HasValue());

  auto status = BuildConcatInto(builder, {*in1, *in2}, /*axis=*/0, *custom_out);
  ASSERT_TRUE(status.HasValue());
  auto into_def_op = custom_out->GetDefiningOp();
  ASSERT_TRUE(into_def_op.HasValue());
  EXPECT_EQ(into_def_op->Code(), kLiteRtOpCodeTflConcatenation);
  auto into_opts = GetOptionsAs<ConcatenationOptions>(ctx, into_def_op->Get());
  ASSERT_TRUE(into_opts.HasValue());
  EXPECT_EQ(into_opts->axis, 0);
}

TEST(TransformationUtilsTest, BuildSoftmax) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  Builder builder(ctx, &builder_impl);

  RankedTensorType type(ElementType::Float32, Layout(BuildLayout({2, 16})));
  auto in = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("attn/scores").Build());
  ASSERT_TRUE(in.HasValue());

  auto sm = BuildSoftmax(builder, *in, 2.5f, *in, {2, 16}, "Softmax");
  ASSERT_TRUE(sm.HasValue());
  EXPECT_EQ(sm->Name(), "attn/Softmax");
  EXPECT_TRUE(HasDims(*sm, {2, 16}));

  auto def_op = sm->GetDefiningOp();
  ASSERT_TRUE(def_op.HasValue());
  EXPECT_EQ(def_op->Code(), kLiteRtOpCodeTflSoftmax);
  EXPECT_EQ(def_op->Inputs().size(), 1);
  EXPECT_EQ(def_op->Inputs()[0], *in);

  auto opts = GetOptionsAs<SoftmaxOptions>(ctx, def_op->Get());
  ASSERT_TRUE(opts.HasValue());
  EXPECT_FLOAT_EQ(opts->beta, 2.5f);
}

TEST(TransformationUtilsTest, BuildAdd) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  Builder builder(ctx, &builder_impl);

  RankedTensorType type(ElementType::Float32, Layout(BuildLayout({2, 4})));
  auto lhs = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("residual/lhs").Build());
  auto rhs = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("residual/rhs").Build());
  ASSERT_TRUE(lhs.HasValue());
  ASSERT_TRUE(rhs.HasValue());

  auto add = BuildAdd(builder, *lhs, *rhs, *lhs, {2, 4}, "Add");
  ASSERT_TRUE(add.HasValue());
  EXPECT_EQ(add->Name(), "residual/Add");
  EXPECT_TRUE(HasDims(*add, {2, 4}));

  auto def_op = add->GetDefiningOp();
  ASSERT_TRUE(def_op.HasValue());
  EXPECT_EQ(def_op->Code(), kLiteRtOpCodeTflAdd);
  EXPECT_EQ(def_op->Inputs().size(), 2);
  EXPECT_EQ(def_op->Inputs()[0], *lhs);
  EXPECT_EQ(def_op->Inputs()[1], *rhs);

  auto opts = GetOptionsAs<AddOptions>(ctx, def_op->Get());
  ASSERT_TRUE(opts.HasValue());
  EXPECT_EQ(opts->fused_activation_function, 0);
}

TEST(TransformationUtilsTest, BuildReshapeAndBuildReshapeInto) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  Builder builder(ctx, &builder_impl);

  RankedTensorType type(ElementType::Float32, Layout(BuildLayout({2, 4})));
  auto in = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("layer/in").Build());
  ASSERT_TRUE(in.HasValue());

  // Test BuildReshape
  auto reshaped = BuildReshape(builder, *in, *in, {8}, "Reshape");
  ASSERT_TRUE(reshaped.HasValue());
  EXPECT_EQ(reshaped->Name(), "layer/Reshape");
  EXPECT_TRUE(HasDims(*reshaped, {8}));

  auto def_op = reshaped->GetDefiningOp();
  ASSERT_TRUE(def_op.HasValue());
  EXPECT_EQ(def_op->Code(), kLiteRtOpCodeTflReshape);
  EXPECT_EQ(def_op->Inputs().size(), 2);
  EXPECT_EQ(def_op->Inputs()[0], *in);

  auto shape_weights = def_op->Inputs()[1].WeightsData<int32_t>();
  ASSERT_TRUE(shape_weights.HasValue());
  EXPECT_THAT(*shape_weights, ElementsAre(8));

  auto opts = GetOptionsAs<ReshapeOptions>(ctx, def_op->Get());
  ASSERT_TRUE(opts.HasValue());
  EXPECT_THAT(opts->new_shape, ElementsAre(8));

  // Test BuildReshapeInto
  RankedTensorType out_type(ElementType::Float32, Layout(BuildLayout({1, 8})));
  auto custom_out =
      builder.BuildTensor(RankedTensorSpecBuilder(out_type)
                              .WithTensorName("layer/custom_reshape")
                              .Build());
  ASSERT_TRUE(custom_out.HasValue());

  auto status = BuildReshapeInto(builder, *in, {1, 8}, *custom_out);
  ASSERT_TRUE(status.HasValue());
  auto into_def_op = custom_out->GetDefiningOp();
  ASSERT_TRUE(into_def_op.HasValue());
  EXPECT_EQ(into_def_op->Code(), kLiteRtOpCodeTflReshape);
  auto into_opts = GetOptionsAs<ReshapeOptions>(ctx, into_def_op->Get());
  ASSERT_TRUE(into_opts.HasValue());
  EXPECT_THAT(into_opts->new_shape, ElementsAre(1, 8));
}

TEST(TransformationUtilsTest, BuildBatchMatmul) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  Builder builder(ctx, &builder_impl);

  RankedTensorType lhs_type(ElementType::Float32,
                            Layout(BuildLayout({2, 3, 4})));
  RankedTensorType rhs_type(ElementType::Float32,
                            Layout(BuildLayout({2, 4, 5})));
  auto lhs = builder.BuildTensor(
      RankedTensorSpecBuilder(lhs_type).WithTensorName("attn/q").Build());
  auto rhs = builder.BuildTensor(
      RankedTensorSpecBuilder(rhs_type).WithTensorName("attn/k").Build());
  ASSERT_TRUE(lhs.HasValue());
  ASSERT_TRUE(rhs.HasValue());

  // Test adj_y = false
  auto bmm1 = BuildBatchMatmul(builder, *lhs, *rhs, *lhs, {2, 3, 5},
                               /*asymmetric_quantize_input=*/true,
                               /*adj_y=*/false, "BMM1");
  ASSERT_TRUE(bmm1.HasValue());
  EXPECT_EQ(bmm1->Name(), "attn/BMM1");
  EXPECT_TRUE(HasDims(*bmm1, {2, 3, 5}));

  auto def_op1 = bmm1->GetDefiningOp();
  ASSERT_TRUE(def_op1.HasValue());
  EXPECT_EQ(def_op1->Code(), kLiteRtOpCodeTflBatchMatmul);
  auto opts1 = GetOptionsAs<BatchMatmulOptions>(ctx, def_op1->Get());
  ASSERT_TRUE(opts1.HasValue());
  EXPECT_FALSE(opts1->adj_x);
  EXPECT_FALSE(opts1->adj_y);
  EXPECT_TRUE(opts1->asymmetric_quantize_input);

  // Test adj_y = true
  RankedTensorType rhs_t_type(ElementType::Float32,
                              Layout(BuildLayout({2, 5, 4})));
  auto rhs_t = builder.BuildTensor(
      RankedTensorSpecBuilder(rhs_t_type).WithTensorName("attn/kt").Build());
  ASSERT_TRUE(rhs_t.HasValue());

  auto bmm2 = BuildBatchMatmul(builder, *lhs, *rhs_t, *lhs, {2, 3, 5},
                               /*asymmetric_quantize_input=*/false,
                               /*adj_y=*/true, "BMM2");
  ASSERT_TRUE(bmm2.HasValue());
  auto def_op2 = bmm2->GetDefiningOp();
  ASSERT_TRUE(def_op2.HasValue());
  auto opts2 = GetOptionsAs<BatchMatmulOptions>(ctx, def_op2->Get());
  ASSERT_TRUE(opts2.HasValue());
  EXPECT_FALSE(opts2->adj_x);
  EXPECT_TRUE(opts2->adj_y);
  EXPECT_FALSE(opts2->asymmetric_quantize_input);
}

TEST(TransformationUtilsTest, SingleUser) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  Builder builder(ctx, &builder_impl);

  RankedTensorType type(ElementType::Float32, Layout(BuildLayout({2, 2})));
  auto t = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("tensor_t").Build());
  auto other = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("other").Build());
  auto out1 = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("out1").Build());
  auto out2 = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("out2").Build());
  ASSERT_TRUE(t.HasValue());
  ASSERT_TRUE(other.HasValue());
  ASSERT_TRUE(out1.HasValue());
  ASSERT_TRUE(out2.HasValue());

  // 0 users
  EXPECT_EQ(SingleUser(*t, kLiteRtOpCodeTflAdd, 0), std::nullopt);

  // 1 user (Add, t is input 0)
  auto add_op = builder.BuildOp(kLiteRtOpCodeTflAdd, {*t, *other}, {*out1});
  ASSERT_TRUE(add_op.HasValue());
  auto user = SingleUser(*t, kLiteRtOpCodeTflAdd, 0);
  ASSERT_TRUE(user.has_value());
  EXPECT_EQ(*user, *add_op);

  // Wrong opcode
  EXPECT_EQ(SingleUser(*t, kLiteRtOpCodeTflMul, 0), std::nullopt);

  // Wrong input index
  EXPECT_EQ(SingleUser(*t, kLiteRtOpCodeTflAdd, 1), std::nullopt);

  // 2 users
  auto mul_op = builder.BuildOp(kLiteRtOpCodeTflMul, {*t, *other}, {*out2});
  ASSERT_TRUE(mul_op.HasValue());
  EXPECT_EQ(SingleUser(*t, kLiteRtOpCodeTflAdd, 0), std::nullopt);
}

TEST(TransformationUtilsTest, SingleOutput) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  Builder builder(ctx, &builder_impl);

  RankedTensorType type(ElementType::Float32, Layout(BuildLayout({2, 2})));
  auto in = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("in").Build());
  auto out1 = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("out1").Build());
  auto split_out1 = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("split_out1").Build());
  auto split_out2 = builder.BuildTensor(
      RankedTensorSpecBuilder(type).WithTensorName("split_out2").Build());
  ASSERT_TRUE(in.HasValue());
  ASSERT_TRUE(out1.HasValue());
  ASSERT_TRUE(split_out1.HasValue());
  ASSERT_TRUE(split_out2.HasValue());

  // Op with 1 output
  auto single_out_op = builder.BuildOp(kLiteRtOpCodeTflRelu, {*in}, {*out1});
  ASSERT_TRUE(single_out_op.HasValue());
  auto output = SingleOutput(*single_out_op);
  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(*output, *out1);

  // Op with 2 outputs
  auto two_out_op = builder.BuildOp(kLiteRtOpCodeTflSplit, {*in, *in},
                                    {*split_out1, *split_out2});
  ASSERT_TRUE(two_out_op.HasValue());
  EXPECT_EQ(SingleOutput(*two_out_op), std::nullopt);

  // Op with 0 outputs
  auto zero_out_op = builder.BuildOp(kLiteRtOpCodeTflCustom, {*in}, {});
  ASSERT_TRUE(zero_out_op.HasValue());
  EXPECT_EQ(SingleOutput(*zero_out_op), std::nullopt);
}

}  // namespace
}  // namespace litert::compiler

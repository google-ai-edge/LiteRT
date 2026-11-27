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
#include <limits>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_model.h"
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
  const Builder builder(ctx, &builder_impl);

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

TEST(TransformationUtilsTest, NewLikeAndQuantizationPreservation) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  const Builder builder(ctx, &builder_impl);

  RankedTensorType type(ElementType::Float32, Layout(BuildLayout({2, 4})));
  auto orig = builder.BuildTensor(
      RankedTensorSpecBuilder(type)
          .WithTensorName("model/transformer/layer_0/input_tensor")
          .Build());
  ASSERT_TRUE(orig.HasValue());

  auto new_tensor = NewLike(builder, *orig, {2, 8}, "Softmax");
  ASSERT_TRUE(new_tensor.HasValue());
  EXPECT_EQ(new_tensor->Name(), "model/transformer/layer_0/Softmax");
  EXPECT_TRUE(HasDims(*new_tensor, {2, 8}));
  EXPECT_EQ(new_tensor->ElementType(), ElementType::Float32);

  auto anon_tensor = NewLike(builder, *orig, {3, 5});
  ASSERT_TRUE(anon_tensor.HasValue());
  EXPECT_EQ(anon_tensor->Name(), "");
  EXPECT_TRUE(HasDims(*anon_tensor, {3, 5}));

  RankedTensorType q_type(ElementType::Int8, Layout(BuildLayout({2, 4})));
  LiteRtQuantizationPerTensor q_params{/*scale=*/0.5f, /*zero_point=*/-10};
  auto q_orig = builder.BuildTensor(RankedTensorSpecBuilder(q_type)
                                        .WithTensorName("quant_layer/input")
                                        .WithPerTensorQuantization(q_params)
                                        .Build());
  ASSERT_TRUE(q_orig.HasValue());
  auto q_new = NewLike(builder, *q_orig, {4, 8}, "output");
  ASSERT_TRUE(q_new.HasValue());
  EXPECT_EQ(q_new->ElementType(), ElementType::Int8);
  EXPECT_EQ(q_new->QTypeId(), kLiteRtQuantizationPerTensor);
  EXPECT_FLOAT_EQ(q_new->PerTensorQuantization().scale, 0.5f);
  EXPECT_EQ(q_new->PerTensorQuantization().zero_point, -10);
  EXPECT_EQ(q_new->Name(), "quant_layer/output");
}

TEST(TransformationUtilsTest, ConstI32HasDimsAndHasPerm) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  Builder builder(ctx, &builder_impl);

  std::vector<int32_t> data = {1, 2, 3, 4};
  auto c = ConstI32(builder, data);
  ASSERT_TRUE(c.HasValue());
  EXPECT_TRUE(HasDims(*c, {4}));
  EXPECT_FALSE(HasDims(*c, {2, 2}));
  EXPECT_TRUE(c->IsConstant());
  auto weights_data = c->WeightsData<int32_t>();
  ASSERT_TRUE(weights_data.HasValue());
  EXPECT_THAT(*weights_data, ElementsAre(1, 2, 3, 4));

  auto in_tensor =
      BuildTypedTensor(builder, ElementType::Float32, {1, 2, 3, 4}, "in");
  ASSERT_TRUE(in_tensor.HasValue());
  auto add_out =
      BuildAdd(builder, *in_tensor, *in_tensor, *in_tensor, {1, 2, 3, 4});
  ASSERT_TRUE(add_out.HasValue());
  auto add_def = add_out->GetDefiningOp();
  ASSERT_TRUE(add_def.HasValue());
  EXPECT_FALSE(HasPerm(*add_def, {0, 2, 1, 3}));

  auto non_const_perm =
      BuildTypedTensor(builder, ElementType::Int32, {4}, "perm");
  auto non_const_out =
      BuildTypedTensor(builder, ElementType::Float32, {1, 3, 2, 4});
  auto non_const_transpose =
      builder.BuildOp(kLiteRtOpCodeTflTranspose, {*in_tensor, *non_const_perm},
                      {*non_const_out});
  ASSERT_TRUE(non_const_transpose.HasValue());
  EXPECT_FALSE(HasPerm(*non_const_transpose, {0, 2, 1, 3}));

  auto perm_const = ConstI32(builder, {0, 2, 1, 3});
  auto valid_out =
      BuildTypedTensor(builder, ElementType::Float32, {1, 3, 2, 4});
  auto valid_transpose = builder.BuildOp(
      kLiteRtOpCodeTflTranspose, {*in_tensor, *perm_const}, {*valid_out});
  ASSERT_TRUE(valid_transpose.HasValue());
  EXPECT_TRUE(HasPerm(*valid_transpose, {0, 2, 1, 3}));
  EXPECT_FALSE(HasPerm(*valid_transpose, {0, 1, 2, 3}));
}

TEST(TransformationUtilsTest,
     BuildSliceConcatSoftmaxReshapeAndBatchMatmulWithDefaults) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  const Builder builder(ctx, &builder_impl);

  auto orig =
      BuildTypedTensor(builder, ElementType::Float32, {4, 8}, "layer/input");
  ASSERT_TRUE(orig.HasValue());

  auto slice = BuildSlice(builder, *orig, {0, 0}, {2, 4}, *orig, "Slice");
  ASSERT_TRUE(slice.HasValue());
  EXPECT_EQ(slice->Name(), "layer/Slice");
  EXPECT_TRUE(HasDims(*slice, {2, 4}));

  auto concat =
      BuildConcat(builder, {*slice, *slice}, /*axis=*/1, *orig, {2, 8});
  ASSERT_TRUE(concat.HasValue());
  EXPECT_TRUE(HasDims(*concat, {2, 8}));

  auto sm = BuildSoftmax(builder, *concat, 2.5f, *orig, {2, 8});
  ASSERT_TRUE(sm.HasValue());
  auto sm_def = sm->GetDefiningOp();
  ASSERT_TRUE(sm_def.HasValue());
  auto sm_opts = GetOptionsAs<SoftmaxOptions>(ctx, sm_def->Get());
  ASSERT_TRUE(sm_opts.HasValue());
  EXPECT_FLOAT_EQ(sm_opts->beta, 2.5f);

  auto reshaped = BuildReshape(builder, *sm, *orig, {16});
  ASSERT_TRUE(reshaped.HasValue());
  EXPECT_TRUE(HasDims(*reshaped, {16}));

  // Exercise BuildBatchMatmul default `asymmetric_quantize_input = false` via
  // both litert::compiler and litert::qnn compatibility alias:
  auto lhs = BuildTypedTensor(builder, ElementType::Float32, {2, 3, 4}, "q/a");
  auto rhs = BuildTypedTensor(builder, ElementType::Float32, {2, 4, 5}, "q/b");
  ASSERT_TRUE(lhs.HasValue());
  ASSERT_TRUE(rhs.HasValue());
  auto bmm_default =
      ::litert::qnn::BuildBatchMatmul(builder, *lhs, *rhs, *lhs, {2, 3, 5});
  ASSERT_TRUE(bmm_default.HasValue());
  auto bmm_def = bmm_default->GetDefiningOp();
  ASSERT_TRUE(bmm_def.HasValue());
  auto bmm_opts = GetOptionsAs<BatchMatmulOptions>(ctx, bmm_def->Get());
  ASSERT_TRUE(bmm_opts.HasValue());
  EXPECT_FALSE(bmm_opts->adj_x);
  EXPECT_FALSE(bmm_opts->adj_y);
  EXPECT_FALSE(bmm_opts->asymmetric_quantize_input);
}

TEST(TransformationUtilsTest, SingleUserDefaultIndexAndSingleOutput) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  const Builder builder(ctx, &builder_impl);

  auto t = BuildTypedTensor(builder, ElementType::Float32, {2, 2}, "t");
  auto other = BuildTypedTensor(builder, ElementType::Float32, {2, 2}, "other");
  ASSERT_TRUE(t.HasValue());
  ASSERT_TRUE(other.HasValue());

  EXPECT_EQ(SingleUser(*t, kLiteRtOpCodeTflAdd), std::nullopt);

  auto add_out = BuildAdd(builder, *t, *other, *t, {2, 2});
  ASSERT_TRUE(add_out.HasValue());
  // Default input_index = 0:
  auto user0 = ::litert::qnn::SingleUser(*t, kLiteRtOpCodeTflAdd);
  ASSERT_TRUE(user0.has_value());
  EXPECT_EQ(SingleUser(*other, kLiteRtOpCodeTflAdd, 1), user0);
  EXPECT_EQ(SingleUser(*other, kLiteRtOpCodeTflAdd), std::nullopt);

  auto single_out = SingleOutput(*user0);
  ASSERT_TRUE(single_out.has_value());
  EXPECT_EQ(*single_out, *add_out);
}

TEST(TransformationUtilsTest,
     StaticPositiveDimsCheckedNumElementsAndBroadcastShapes) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;

  auto& valid_t = subgraph.EmplaceTensor();
  valid_t.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 3, 4}));
  auto dims = GetStaticPositiveDims(Tensor(ctx, &valid_t));
  ASSERT_TRUE(dims.has_value());
  EXPECT_THAT(*dims, ElementsAre(2, 3, 4));

  auto& scalar_t = subgraph.EmplaceTensor();
  scalar_t.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  auto scalar_dims = GetStaticPositiveDims(Tensor(ctx, &scalar_t));
  ASSERT_TRUE(scalar_dims.has_value());
  EXPECT_TRUE(scalar_dims->empty());

  auto& dyn_t = subgraph.EmplaceTensor();
  dyn_t.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, -1}));
  EXPECT_EQ(GetStaticPositiveDims(Tensor(ctx, &dyn_t)), std::nullopt);

  auto& zero_t = subgraph.EmplaceTensor();
  zero_t.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 0}));
  EXPECT_EQ(GetStaticPositiveDims(Tensor(ctx, &zero_t)), std::nullopt);

  LiteRtBuilderT builder_impl;
  const Builder builder(ctx, &builder_impl);
  auto unranked_t = builder.BuildScalar(kLiteRtElementTypeFloat32);
  ASSERT_TRUE(unranked_t.HasValue());
  EXPECT_EQ(GetStaticPositiveDims(*unranked_t), std::nullopt);
  EXPECT_EQ(GetStaticPositiveDims(Tensor()), std::nullopt);

  EXPECT_EQ(CheckedNumElements({}), 1u);
  EXPECT_EQ(CheckedNumElements({2, 3, 4}), 24u);
  EXPECT_EQ(CheckedNumElements({2, 0}), std::nullopt);
  EXPECT_EQ(CheckedNumElements({2, -1}), std::nullopt);
  const int32_t big = (std::numeric_limits<int32_t>::max)();
  EXPECT_EQ(CheckedNumElements({big, big, big, big}), std::nullopt);

  auto bcast1 = BroadcastShapes({2, 1, 4}, {3, 1});
  ASSERT_TRUE(bcast1.has_value());
  EXPECT_THAT(*bcast1, ElementsAre(2, 3, 4));
  auto bcast_scalar = BroadcastShapes({}, {2, 5});
  ASSERT_TRUE(bcast_scalar.has_value());
  EXPECT_THAT(*bcast_scalar, ElementsAre(2, 5));
  auto bcast_both_scalar = BroadcastShapes({}, {});
  ASSERT_TRUE(bcast_both_scalar.has_value());
  EXPECT_TRUE(bcast_both_scalar->empty());
  EXPECT_EQ(BroadcastShapes({2, 3}, {4, 3}), std::nullopt);
  EXPECT_EQ(BroadcastShapes({0, 3}, {1, 3}), std::nullopt);
  EXPECT_EQ(BroadcastShapes({-1, 3}, {1, 3}), std::nullopt);
}

TEST(TransformationUtilsTest, ReadIntConstValuesAndConstBuilders) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  const Builder builder(ctx, &builder_impl);

  auto i32_const =
      BuildConst<int32_t>(builder, {3}, std::vector<int32_t>{-2, 0, 7}, "i32");
  ASSERT_TRUE(i32_const.HasValue());
  EXPECT_EQ(i32_const->Name(), "i32");
  auto i32_vals = ReadIntConstValues(*i32_const);
  ASSERT_TRUE(i32_vals.has_value());
  EXPECT_THAT(*i32_vals, ElementsAre(-2, 0, 7));

  const int64_t large = (1LL << 40);
  auto i64_const =
      BuildConst<int64_t>(builder, {2}, std::vector<int64_t>{1, large}, "i64");
  ASSERT_TRUE(i64_const.HasValue());
  auto i64_vals = ReadIntConstValues(*i64_const);
  ASSERT_TRUE(i64_vals.has_value());
  EXPECT_THAT(*i64_vals, ElementsAre(1, large));

  auto scalar_f32 = BuildScalarConst<float>(builder, 1.0f, "one");
  ASSERT_TRUE(scalar_f32.HasValue());
  EXPECT_TRUE(HasDims(*scalar_f32, {}));
  EXPECT_EQ(ReadIntConstValues(*scalar_f32), std::nullopt);

  // Shape/element count mismatch rejected:
  auto bad_const =
      BuildConst<float>(builder, {2, 2}, std::vector<float>{1.0f, 2.0f});
  EXPECT_FALSE(bad_const.HasValue());

  // Quantized Int32 rejected by ReadIntConstValues:
  LiteRtQuantizationPerTensor q_params{/*scale=*/1.0f, /*zero_point=*/0};
  auto q_i32 = builder.BuildTensor(
      RankedTensorSpecBuilder(
          RankedTensorType(ElementType::Int32, Layout(BuildLayout({1}))))
          .WithPerTensorQuantization(q_params)
          .Build());
  ASSERT_TRUE(q_i32.HasValue());
  ASSERT_TRUE(builder.BuildWeights<int32_t>(std::vector<int32_t>{5}, *q_i32)
                  .HasValue());
  EXPECT_EQ(ReadIntConstValues(*q_i32), std::nullopt);
}

TEST(TransformationUtilsTest, BuildCastBinaryOpsAndCopyOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  const Builder builder(ctx, &builder_impl);

  auto f_in = BuildTypedTensor(builder, ElementType::Float32, {2, 3}, "f_in");
  ASSERT_TRUE(f_in.HasValue());

  auto b_cast = BuildCast(builder, *f_in, ElementType::Bool, {2, 3}, "b_cast");
  ASSERT_TRUE(b_cast.HasValue());
  EXPECT_EQ(b_cast->ElementType(), ElementType::Bool);
  auto b_cast_def = b_cast->GetDefiningOp();
  ASSERT_TRUE(b_cast_def.HasValue());
  EXPECT_EQ(b_cast_def->Code(), kLiteRtOpCodeTflCast);

  auto f_back = BuildTypedTensor(builder, ElementType::Float32, {2, 3});
  ASSERT_TRUE(f_back.HasValue());
  ASSERT_TRUE(BuildCastInto(builder, *b_cast, *f_back).HasValue());
  auto f_back_def = f_back->GetDefiningOp();
  ASSERT_TRUE(f_back_def.HasValue());
  EXPECT_EQ(f_back_def->Code(), kLiteRtOpCodeTflCast);

  // Binary ops with fused activation options (Sub, Mul, Div) + optionless
  // Maximum:
  auto sub_out = BuildBinaryOp(builder, kLiteRtOpCodeTflSub, *f_in, *f_back,
                               ElementType::Float32, {2, 3},
                               /*fused_activation=*/1, "sub_relu");
  ASSERT_TRUE(sub_out.HasValue());
  auto sub_def = sub_out->GetDefiningOp();
  ASSERT_TRUE(sub_def.HasValue());
  auto sub_opts = GetOptionsAs<SubOptions>(ctx, sub_def->Get());
  ASSERT_TRUE(sub_opts.HasValue());
  EXPECT_EQ(sub_opts->fused_activation_function, 1u);

  auto mul_out = BuildBinaryOp(builder, kLiteRtOpCodeTflMul, *f_in, *f_back,
                               ElementType::Float32, {2, 3});
  ASSERT_TRUE(mul_out.HasValue());
  auto div_out = BuildBinaryOp(builder, kLiteRtOpCodeTflDiv, *f_in, *f_back,
                               ElementType::Float32, {2, 3});
  ASSERT_TRUE(div_out.HasValue());
  auto max_out = BuildBinaryOp(builder, kLiteRtOpCodeTflMaximum, *f_in, *f_back,
                               ElementType::Float32, {2, 3});
  ASSERT_TRUE(max_out.HasValue());

  // Optionless op rejects non-zero fused_activation without mutating output:
  auto unused_out = BuildTypedTensor(builder, ElementType::Float32, {2, 3});
  ASSERT_TRUE(unused_out.HasValue());
  EXPECT_FALSE(BuildBinaryOpInto(builder, kLiteRtOpCodeTflMaximum, *f_in,
                                 *f_back, *unused_out,
                                 /*fused_activation=*/1)
                   .HasValue());
  EXPECT_EQ(unused_out->DefiningOp(), std::nullopt);
  EXPECT_FALSE(BuildBinaryOp(builder, kLiteRtOpCodeTflMaximum, *f_in, *f_back,
                             ElementType::Float32, {2, 3},
                             /*fused_activation=*/1)
                   .HasValue());

  // Rank > LITERT_TENSOR_MAX_RANK rejected before BuildLayout:
  const std::vector<int32_t> over_max_rank(LITERT_TENSOR_MAX_RANK + 1, 1);
  EXPECT_FALSE(NewLike(builder, *f_in, over_max_rank).HasValue());
  EXPECT_FALSE(BuildTypedTensor(builder, ElementType::Float32, over_max_rank)
                   .HasValue());
  EXPECT_FALSE(
      BuildConst<float>(builder, over_max_rank, std::vector<float>{1.0f})
          .HasValue());

  // CopyOpOptions copies SubOptions cleanly onto a newly built Sub op:
  auto dst_out = BuildTypedTensor(builder, ElementType::Float32, {2, 3});
  ASSERT_TRUE(dst_out.HasValue());
  auto dst_sub =
      builder.BuildOp(kLiteRtOpCodeTflSub, {*f_in, *f_back}, {*dst_out});
  ASSERT_TRUE(dst_sub.HasValue());
  ASSERT_TRUE(
      CopyOpOptions<SubOptions>(builder, *sub_def, *dst_sub).HasValue());
  auto copied_opts = GetOptionsAs<SubOptions>(ctx, dst_sub->Get());
  ASSERT_TRUE(copied_opts.HasValue());
  EXPECT_EQ(copied_opts->fused_activation_function, 1u);
}

TEST(TransformationUtilsTest, NullHandleSafetyWithLiveCompilerContext) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder_impl;
  const Builder builder(ctx, &builder_impl);

  const Tensor null_t(ctx, nullptr);
  const Op null_op(ctx, nullptr);

  EXPECT_EQ(GetTensorNamespace(null_t), "");
  EXPECT_EQ(MakeNamespacedTensorName(null_t, "tag"), "tag");
  EXPECT_FALSE(HasDims(null_t, {1}));
  EXPECT_FALSE(HasPerm(null_op, {0}));
  EXPECT_FALSE(NewLike(builder, null_t, {1}).HasValue());
  EXPECT_EQ(SingleUser(null_t, kLiteRtOpCodeTflAdd), std::nullopt);
  EXPECT_EQ(SingleOutput(null_op), std::nullopt);
  EXPECT_EQ(GetStaticPositiveDims(null_t), std::nullopt);
  EXPECT_EQ(ReadIntConstValues(null_t), std::nullopt);

  auto t = BuildTypedTensor(builder, ElementType::Float32, {2});
  ASSERT_TRUE(t.HasValue());
  auto sub_out = BuildBinaryOp(builder, kLiteRtOpCodeTflSub, *t, *t,
                               ElementType::Float32, {2});
  ASSERT_TRUE(sub_out.HasValue());
  auto valid_op = sub_out->GetDefiningOp();
  ASSERT_TRUE(valid_op.HasValue());
  EXPECT_FALSE(
      CopyOpOptions<SubOptions>(builder, null_op, *valid_op).HasValue());
  EXPECT_FALSE(
      CopyOpOptions<SubOptions>(builder, *valid_op, null_op).HasValue());
}

}  // namespace
}  // namespace litert::compiler

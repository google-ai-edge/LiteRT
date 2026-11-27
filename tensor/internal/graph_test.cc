/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensor/internal/graph.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/source_location.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/type_id.h"
#include "tensor/utils/matchers.h"
#include "tensor/utils/parallel.h"

namespace litert::tensor::graph {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using testing::ElementsAre;
using testing::ElementsAreArray;
using testing::FloatEq;
using testing::IsEmpty;
using testing::Not;
using testing::SizeIs;
using testing::StrEq;

MATCHER_P(IsLocation, loc, "") {
  return testing::ExplainMatchResult(testing::Eq(loc.line()), arg.line(),
                                     result_listener) &&
         testing::ExplainMatchResult(testing::StrEq(loc.file_name()),
                                     arg.file_name(), result_listener);
}

TEST(TensorTest, NewTensorGroupWorks) {
  const auto loc = absl::SourceLocation::current();
  const std::shared_ptr<const TensorGroup> g = NewTensorGroup(3, loc);
  EXPECT_NE(g, nullptr);
  EXPECT_THAT(g->tensor_infos, SizeIs(3));
  EXPECT_THAT(g->status, IsOk());
  EXPECT_EQ(g->producer, nullptr);
  EXPECT_THAT(g->loc, IsLocation(loc));
}

TEST(TensorTest, NewTensorCreatesAValidTensor) {
  const auto loc = absl::SourceLocation::current();
  Tensor a = NewTensor(loc);
  EXPECT_EQ(a.index, 0);
  ASSERT_NE(a.group, nullptr);
  EXPECT_THAT(GetStatus(a), IsOk());
  EXPECT_THAT(GetLocation(a), IsLocation(loc));
}

TEST(TensorTest, SetAndGetTensorName) {
  Tensor a = NewTensor(absl::SourceLocation::current());
  EXPECT_THAT(SetName(a, "my tensor name"), IsOk());
  EXPECT_THAT(GetName(a), IsOkAndHolds(StrEq("my tensor name")));
}

TEST(TensorTest, SetAndGetBuffer) {
  Tensor a = NewTensor(absl::SourceLocation::current());
  auto buffer = OwningCpuBuffer::Copy<Type::kI32>({1, 2, 3, 4});
  EXPECT_THAT(SetBuffer(a, buffer), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & retrieved_buffer, GetBuffer(a));
  EXPECT_THAT(retrieved_buffer.Lock(), ElementsAreArray(buffer->Lock()));
}

TEST(TensorTest, DefaultTensorIsInvalid) {
  Tensor a;
  EXPECT_THAT(GetStatus(a), Not(IsOk()));
  EXPECT_THAT(GetLocation(a), IsLocation(absl::SourceLocation()));
  EXPECT_THAT(GetProducer(a), Not(IsOk()));
  EXPECT_THAT(GetConsumers(a), Not(IsOk()));
}

TEST(TensorTest, TensorWithWringIndexIsInvalid) {
  Tensor a = NewTensor(absl::SourceLocation::current());
  a.index = 3;
  EXPECT_THAT(GetStatus(a), Not(IsOk()));
}

TEST(QuantizationTest, GetTypeId) {
  PerChannelAffineQuantization quant({1.0f}, {0}, 0);
  EXPECT_EQ(quant.GetTypeId(),
            internal::TypeId::Get<PerChannelAffineQuantization>());
}

TEST(QuantizationTest, IsA) {
  PerChannelAffineQuantization quant({1.0f}, {0}, 0);
  EXPECT_TRUE(quant.IsA(internal::TypeId::Get<PerChannelAffineQuantization>()));
}

struct DummyQuantization : Quantization {
  internal::TypeId GetTypeId() const override {
    return internal::TypeId::Get<DummyQuantization>();
  }
  bool IsA(internal::TypeId id) const override {
    return id == internal::TypeId::Get<DummyQuantization>();
  }
};

TEST(QuantizationTest, As) {
  PerChannelAffineQuantization quant({1.0f}, std::vector<int64_t>{0}, 0);
  Quantization& quant_ref = quant;

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(PerChannelAffineQuantization & quant_as,
                                  quant_ref.As<PerChannelAffineQuantization>());
  EXPECT_EQ(&quant_as, &quant);
  EXPECT_THAT(quant_ref.As<DummyQuantization>(), Not(IsOk()));

  const Quantization& const_quant_ref = quant;
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      const PerChannelAffineQuantization& const_quant_as,
      const_quant_ref.As<PerChannelAffineQuantization>());
  EXPECT_EQ(&const_quant_as, &quant);
  EXPECT_THAT(const_quant_ref.As<DummyQuantization>(), Not(IsOk()));
}

TEST(QuantizationTest, DefaultsToSymmetricZeroPoints) {
  PerChannelAffineQuantization pcq;
  EXPECT_TRUE(pcq.zero_points.empty());
  EXPECT_TRUE(pcq.zero_points.IsSymmetric());
  EXPECT_EQ(pcq.zero_points.size(), 0);
  EXPECT_THAT(pcq.zero_points.Data<int64_t>(), IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(pcq.zero_points.Data<float>(),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(pcq.zero_points.Scalar<int64_t>(), IsOkAndHolds(0));
  EXPECT_THAT(pcq.zero_points.Scalar<float>(), IsOkAndHolds(FloatEq(0.0f)));
  EXPECT_THAT(pcq.zero_points.At<float>(0), IsOkAndHolds(FloatEq(0.0f)));

  BlockwiseQuantization bwq;
  EXPECT_TRUE(bwq.zero_points.empty());
  EXPECT_TRUE(bwq.zero_points.IsSymmetric());
  EXPECT_EQ(bwq.zero_points.size(), 0);
  EXPECT_THAT(bwq.zero_points.Data<int64_t>(), IsOkAndHolds(IsEmpty()));
}

TEST(QuantizationTest, PreservesZeroPointsWithoutNormalizingAllZeros) {
  const ZeroPoints empty;
  EXPECT_EQ(ZeroPoints::Int({}), empty);
  EXPECT_TRUE(ZeroPoints::Float({}).empty());
  EXPECT_TRUE(ZeroPoints::Float({}).IsSymmetric());
  EXPECT_THAT(ZeroPoints::Float({}).Data<float>(), IsOkAndHolds(IsEmpty()));

  const ZeroPoints zero_ints = ZeroPoints::From(std::vector<int64_t>{0, 0});
  EXPECT_FALSE(zero_ints.empty());
  EXPECT_TRUE(zero_ints.IsSymmetric());
  EXPECT_TRUE(zero_ints.IsUniform());
  EXPECT_EQ(zero_ints.size(), 2);
  EXPECT_NE(zero_ints, empty);
  EXPECT_THAT(zero_ints.Data<int64_t>(), IsOkAndHolds(ElementsAre(0, 0)));
  EXPECT_THAT(zero_ints.Scalar<int32_t>(), IsOkAndHolds(0));
  // Wrongly sized all-zero vectors are still rejected by Validate.
  EXPECT_THAT(zero_ints.Validate<int32_t>(3),
              StatusIs(absl::StatusCode::kInvalidArgument));

  const ZeroPoints zero_floats =
      ZeroPoints::From(std::vector<float>{0.0f, -0.0f});
  EXPECT_FALSE(zero_floats.empty());
  EXPECT_TRUE(zero_floats.IsSymmetric());
  EXPECT_EQ(zero_floats.size(), 2);
  EXPECT_THAT(zero_floats.Data<float>(),
              IsOkAndHolds(ElementsAre(FloatEq(0.0f), FloatEq(-0.0f))));
  EXPECT_THAT(zero_floats.Data<int64_t>(),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(zero_floats.Scalar<int32_t>(), IsOkAndHolds(0));

  const ZeroPoints uniform_ints = ZeroPoints::Int({8, 8, 8});
  EXPECT_FALSE(uniform_ints.IsSymmetric());
  EXPECT_TRUE(uniform_ints.IsUniform());
  EXPECT_THAT(uniform_ints.Scalar<int32_t>(), IsOkAndHolds(8));
  EXPECT_THAT(ZeroPoints::Float({-0.5f, -0.5f}).Scalar<float>(),
              IsOkAndHolds(FloatEq(-0.5f)));
  EXPECT_THAT(ZeroPoints::Float({-0.5f, -0.5f}).Scalar<int32_t>(),
              StatusIs(absl::StatusCode::kUnimplemented));

  EXPECT_NE(ZeroPoints::Int({1}), empty);
  EXPECT_NE(ZeroPoints::Int({1}), ZeroPoints::Float({1.0f}));
}

TEST(QuantizationTest, IntegerZeroPoints) {
  PerChannelAffineQuantization pcq({0.5f, 0.25f},
                                   /*zero_points=*/ZeroPoints::Int({0, 3}),
                                   /*quantized_dimension=*/0);
  EXPECT_FALSE(pcq.zero_points.IsSymmetric());
  EXPECT_FALSE(pcq.zero_points.IsUniform());
  EXPECT_EQ(pcq.zero_points.size(), 2);
  EXPECT_THAT(pcq.zero_points.Data<int64_t>(), IsOkAndHolds(ElementsAre(0, 3)));
  EXPECT_THAT(pcq.zero_points.Data<float>(),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(pcq.zero_points.Scalar<int64_t>(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(pcq.zero_points.At<int64_t>(0), IsOkAndHolds(0));
  EXPECT_THAT(pcq.zero_points.At<int64_t>(1), IsOkAndHolds(3));
  EXPECT_THAT(pcq.zero_points.At<float>(2),
              StatusIs(absl::StatusCode::kInvalidArgument));

  std::vector<int32_t> copied(2);
  EXPECT_THAT(pcq.zero_points.Validate<int32_t>(2), ::testing::tensor::IsOk());
  EXPECT_THAT(pcq.zero_points.CopyTo<int32_t>(absl::MakeSpan(copied)),
              ::testing::tensor::IsOk());
  EXPECT_THAT(copied, ElementsAre(0, 3));
  EXPECT_THAT(pcq.zero_points.CopyAs<int32_t>(2),
              IsOkAndHolds(ElementsAre(0, 3)));
  EXPECT_THAT(pcq.zero_points.Validate<int32_t>(3),
              StatusIs(absl::StatusCode::kInvalidArgument));

  BlockwiseQuantization bwq({0.5f, 0.25f},
                            /*zero_points=*/ZeroPoints::Int({2}),
                            /*block_size=*/32);
  EXPECT_EQ(bwq.zero_points.size(), 1);
  EXPECT_THAT(bwq.zero_points.Scalar<int64_t>(), IsOkAndHolds(2));
  EXPECT_THAT(bwq.zero_points.At<float>(0), IsOkAndHolds(FloatEq(2.0f)));
  EXPECT_THAT(bwq.zero_points.At<float>(1), IsOkAndHolds(FloatEq(2.0f)));
  std::vector<int32_t> broadcast(3);
  EXPECT_THAT(bwq.zero_points.CopyTo<int32_t>(absl::MakeSpan(broadcast)),
              ::testing::tensor::IsOk());
  EXPECT_THAT(broadcast, ElementsAre(2, 2, 2));
}

TEST(QuantizationTest, FloatZeroPoints) {
  PerChannelAffineQuantization pcq({0.5f, 0.25f},
                                   /*zero_points=*/ZeroPoints::Float({-0.5f}),
                                   /*quantized_dimension=*/0);
  EXPECT_FALSE(pcq.zero_points.IsSymmetric());
  EXPECT_EQ(pcq.zero_points.size(), 1);
  EXPECT_THAT(pcq.zero_points.Data<int64_t>(),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(pcq.zero_points.Data<float>(),
              IsOkAndHolds(ElementsAre(FloatEq(-0.5f))));
  EXPECT_THAT(pcq.zero_points.Scalar<float>(), IsOkAndHolds(FloatEq(-0.5f)));
  EXPECT_THAT(pcq.zero_points.Scalar<int64_t>(),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(pcq.zero_points.Validate<int32_t>(2),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(pcq.zero_points.At<float>(0), IsOkAndHolds(FloatEq(-0.5f)));
  EXPECT_THAT(pcq.zero_points.At<float>(1), IsOkAndHolds(FloatEq(-0.5f)));

  BlockwiseQuantization bwq(
      {0.5f, 0.25f},
      /*zero_points=*/ZeroPoints::Float({-0.5f, 0.25f}),
      /*block_size=*/32, /*quantized_dimension=*/0);
  EXPECT_EQ(bwq.zero_points.size(), 2);
  EXPECT_THAT(bwq.zero_points.At<float>(0), IsOkAndHolds(FloatEq(-0.5f)));
  EXPECT_THAT(bwq.zero_points.At<float>(1), IsOkAndHolds(FloatEq(0.25f)));
  EXPECT_THAT(bwq.zero_points.At<float>(2),
              StatusIs(absl::StatusCode::kInvalidArgument));

  const ZeroPoints whole_float = ZeroPoints::Float({2.0f, -3.0f});
  EXPECT_THAT(whole_float.Validate<int32_t>(2), ::testing::tensor::IsOk());
  std::vector<int32_t> ints(2);
  EXPECT_THAT(whole_float.CopyTo<int32_t>(absl::MakeSpan(ints)),
              ::testing::tensor::IsOk());
  EXPECT_THAT(ints, ElementsAre(2, -3));

  const ZeroPoints nan_zp =
      ZeroPoints::Float({std::numeric_limits<float>::quiet_NaN()});
  EXPECT_THAT(nan_zp.Scalar<int64_t>(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(QuantizationTest, ZeroPointsConversionRangeLimits) {
  const ZeroPoints pos_inf =
      ZeroPoints::Float({std::numeric_limits<float>::infinity()});
  const ZeroPoints neg_inf =
      ZeroPoints::Float({-std::numeric_limits<float>::infinity()});
  EXPECT_THAT(pos_inf.Scalar<int32_t>(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(neg_inf.Scalar<int64_t>(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  const ZeroPoints above_i32 = ZeroPoints::Float({2147483648.0f});
  EXPECT_THAT(above_i32.Scalar<int32_t>(),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(above_i32.Scalar<int64_t>(), IsOkAndHolds(int64_t{2147483648}));

  const ZeroPoints min_i32 = ZeroPoints::Float({-2147483648.0f});
  EXPECT_THAT(min_i32.Scalar<int32_t>(),
              IsOkAndHolds(std::numeric_limits<int32_t>::min()));

  const ZeroPoints below_i32 = ZeroPoints::Float({-4294967296.0f});
  EXPECT_THAT(below_i32.Scalar<int32_t>(),
              StatusIs(absl::StatusCode::kUnimplemented));

  const ZeroPoints above_i64 = ZeroPoints::Float({9223372036854775808.0f});
  EXPECT_THAT(above_i64.Scalar<int64_t>(),
              StatusIs(absl::StatusCode::kUnimplemented));

  const ZeroPoints neg_int = ZeroPoints::Int({-1});
  const ZeroPoints neg_float = ZeroPoints::Float({-1.0f});
  EXPECT_THAT(neg_int.Scalar<uint32_t>(),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(neg_float.Scalar<uint32_t>(),
              StatusIs(absl::StatusCode::kUnimplemented));

  const ZeroPoints u8_fit = ZeroPoints::Int({255});
  const ZeroPoints u8_overflow = ZeroPoints::Int({256});
  EXPECT_THAT(u8_fit.Scalar<uint8_t>(), IsOkAndHolds(uint8_t{255}));
  EXPECT_THAT(u8_overflow.Scalar<uint8_t>(),
              StatusIs(absl::StatusCode::kUnimplemented));

  const ZeroPoints i64_above_i32 = ZeroPoints::Int(
      {static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1});
  const ZeroPoints i64_below_i32 = ZeroPoints::Int(
      {static_cast<int64_t>(std::numeric_limits<int32_t>::min()) - 1});
  EXPECT_THAT(i64_above_i32.Scalar<int32_t>(),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(i64_below_i32.Scalar<int32_t>(),
              StatusIs(absl::StatusCode::kUnimplemented));

  const ZeroPoints mixed_fractional = ZeroPoints::Float({2.0f, 0.5f});
  std::vector<int32_t> copy_dst(2);
  EXPECT_THAT(mixed_fractional.CopyTo<int32_t>(absl::MakeSpan(copy_dst)),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(QuantizationTest, ZeroPointsStridedVisitView) {
  std::vector<float> visited;
  auto collect = [&visited](size_t n) {
    return [&visited, n](auto view) {
      visited.clear();
      for (size_t i = 0; i < n; ++i) {
        visited.push_back(static_cast<float>(view[i]));
      }
    };
  };

  EXPECT_THAT(ZeroPoints{}.Visit(3, collect(3)), ::testing::tensor::IsOk());
  EXPECT_THAT(visited,
              ElementsAre(FloatEq(0.0f), FloatEq(0.0f), FloatEq(0.0f)));

  EXPECT_THAT(ZeroPoints::Int({4}).Visit(3, collect(3)),
              ::testing::tensor::IsOk());
  EXPECT_THAT(visited,
              ElementsAre(FloatEq(4.0f), FloatEq(4.0f), FloatEq(4.0f)));

  EXPECT_THAT(ZeroPoints::Float({-0.5f, 1.25f, 2.0f}).Visit(3, collect(3)),
              ::testing::tensor::IsOk());
  EXPECT_THAT(visited,
              ElementsAre(FloatEq(-0.5f), FloatEq(1.25f), FloatEq(2.0f)));

  EXPECT_THAT(ZeroPoints::Int({1, 2}).Visit(3, collect(3)),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(ZeroPoints::Int({5}).Visit(
                  2,
                  [](auto view) -> absl::Status {
                    return view[0] == 5 && view[1] == 5
                               ? absl::OkStatus()
                               : absl::InternalError("unexpected value");
                  }),
              ::testing::tensor::IsOk());
}

MATCHER(IsEmptySymmetric, "") {
  return testing::ExplainMatchResult(testing::IsTrue(), arg.empty(),
                                     result_listener) &&
         testing::ExplainMatchResult(testing::Eq(0u), arg.size(),
                                     result_listener) &&
         testing::ExplainMatchResult(testing::IsTrue(), arg.IsSymmetric(),
                                     result_listener) &&
         testing::ExplainMatchResult(testing::IsTrue(), arg.IsUniform(),
                                     result_listener) &&
         testing::ExplainMatchResult(IsOkAndHolds(0),
                                     arg.template Scalar<int64_t>(),
                                     result_listener) &&
         testing::ExplainMatchResult(IsOkAndHolds(FloatEq(0.0f)),
                                     arg.template Scalar<float>(),
                                     result_listener) &&
         testing::ExplainMatchResult(testing::tensor::IsOk(),
                                     arg.template Validate<int32_t>(0),
                                     result_listener);
}

TEST(QuantizationTest, ZeroPointsMovePreservesMovedFromInvariants) {
  // Move construction from uniform non-zero integers.
  ZeroPoints int_src = ZeroPoints::Int({5, 5});
  ZeroPoints dst(std::move(int_src));
  EXPECT_FALSE(dst.IsSymmetric());
  EXPECT_TRUE(dst.IsUniform());
  EXPECT_THAT(dst.Data<int64_t>(), IsOkAndHolds(ElementsAre(5, 5)));
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_THAT(int_src, IsEmptySymmetric());

  // Move assignment across representations (Float -> Int destination).
  ZeroPoints float_src = ZeroPoints::Float({-0.5f, 1.25f});
  dst = std::move(float_src);
  EXPECT_FALSE(dst.IsSymmetric());
  EXPECT_FALSE(dst.IsUniform());
  EXPECT_THAT(dst.Data<float>(),
              IsOkAndHolds(ElementsAre(FloatEq(-0.5f), FloatEq(1.25f))));
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_THAT(float_src, IsEmptySymmetric());

  // Move assignment across representations (Int -> Float destination).
  ZeroPoints int_src2 = ZeroPoints::Int({3});
  dst = std::move(int_src2);
  EXPECT_FALSE(dst.IsSymmetric());
  EXPECT_TRUE(dst.IsUniform());
  EXPECT_THAT(dst.Data<int64_t>(), IsOkAndHolds(ElementsAre(3)));
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_THAT(int_src2, IsEmptySymmetric());

  // Self-move-assignment preserves state.
  ZeroPoints self = ZeroPoints::Int({7, 7});
  ZeroPoints& self_ref = self;
  self = std::move(self_ref);
  EXPECT_FALSE(self.IsSymmetric());
  EXPECT_TRUE(self.IsUniform());
  EXPECT_THAT(self.Data<int64_t>(), IsOkAndHolds(ElementsAre(7, 7)));
}

struct DummyOp : Operation {
  absl::string_view GetName() const override { return "DummyOp"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};

TEST(ScopeTest, NestedScopeHierarchyAndAttributesAndBoundaries) {
  EXPECT_EQ(CurrentScope(), nullptr);

  Tensor leaf_input = NewTensor(absl::SourceLocation::current());
  Tensor leaf_weight = NewTensor(absl::SourceLocation::current());

  std::shared_ptr<DummyOp> inner_op;
  Tensor inner_output;
  {
    ScopeGuard layer_scope("transformer_layer", "layer_3");
    layer_scope.Attr("layer_idx", 3).Bind("input", leaf_input);

    {
      ScopeGuard attn_scope("gated_attention", "attn");
      attn_scope.Attr("num_heads", 8)
          .Attr("scale", 0.0625)
          .Attr("mode", "gqa")
          .Bind("input", leaf_input)
          .Bind("q_weight", leaf_weight);

      inner_op = std::make_shared<DummyOp>();
      auto group = NewTensorGroup(1, absl::SourceLocation::current());
      inner_op->outputs_group = group;
      group->producer = inner_op;
      inner_output = GetTensor(0, group);
      attn_scope.Bind("output", inner_output);
    }

    EXPECT_EQ(CurrentScope(), layer_scope.scope());
  }
  EXPECT_EQ(CurrentScope(), nullptr);

  ASSERT_NE(inner_op->scope, nullptr);
  EXPECT_EQ(inner_op->scope->kind, "gated_attention");
  EXPECT_EQ(inner_op->scope->name, "attn");
  EXPECT_EQ(inner_op->scope->FullPath(), "layer_3/attn");
  EXPECT_EQ(inner_op->scope->GetIntAttr("num_heads"), 8);
  EXPECT_DOUBLE_EQ(*inner_op->scope->GetFloatAttr("scale"), 0.0625);
  EXPECT_EQ(inner_op->scope->GetStringAttr("mode"), "gqa");

  ASSERT_NE(inner_op->scope->parent, nullptr);
  EXPECT_EQ(inner_op->scope->parent->kind, "transformer_layer");
  EXPECT_EQ(inner_op->scope->parent->name, "layer_3");
  EXPECT_EQ(inner_op->scope->parent->GetIntAttr("layer_idx"), 3);
  EXPECT_EQ(inner_op->scope->FindAncestor("transformer_layer"),
            inner_op->scope->parent.get());

  auto bound_in = inner_op->scope->GetBoundaryTensor("input");
  ASSERT_TRUE(bound_in.has_value());
  EXPECT_EQ(*bound_in, leaf_input);

  auto bound_w = inner_op->scope->GetBoundaryTensor("q_weight");
  ASSERT_TRUE(bound_w.has_value());
  EXPECT_EQ(*bound_w, leaf_weight);

  auto bound_out = inner_op->scope->GetBoundaryTensor("output");
  ASSERT_TRUE(bound_out.has_value());
  EXPECT_EQ(*bound_out, inner_output);
}

TEST(ScopeTest, BindOptionalTensorWrapper) {
  struct FakeTensorWrapper {
    Tensor raw;
    const Tensor& GetRaw() const { return raw; }
  };

  FakeTensorWrapper present{NewTensor(absl::SourceLocation::current())};
  std::optional<FakeTensorWrapper> opt_present = present;
  std::optional<FakeTensorWrapper> opt_empty = std::nullopt;

  std::shared_ptr<Scope> captured_scope;
  {
    ScopeGuard scope("test_scope");
    scope.Bind("present", opt_present).Bind("empty", opt_empty);
    captured_scope = scope.scope();
  }

  ASSERT_NE(captured_scope, nullptr);
  auto bound_present = captured_scope->GetBoundaryTensor("present");
  ASSERT_TRUE(bound_present.has_value());
  EXPECT_EQ(*bound_present, present.raw);
  EXPECT_FALSE(captured_scope->GetBoundaryTensor("empty").has_value());
}

TEST(QuantizationTest, BlockwiseFloatZeroPointsAndPredicates) {
  BlockwiseQuantization sym({0.25f, 0.5f}, {}, /*block_size=*/32);
  EXPECT_TRUE(sym.zero_points.IsSymmetric());
  EXPECT_THAT(sym.zero_points.At<float>(0), IsOkAndHolds(FloatEq(0.0f)));
  EXPECT_THAT(sym.zero_points.At<float>(1), IsOkAndHolds(FloatEq(0.0f)));

  BlockwiseQuantization all_zero_fzp({0.25f, 0.5f},
                                     ZeroPoints::Float({0.0f, 0.0f}),
                                     /*block_size=*/32);
  EXPECT_TRUE(all_zero_fzp.zero_points.IsSymmetric());

  BlockwiseQuantization broadcast_fzp({0.25f, 0.5f}, ZeroPoints::Float({-2.5f}),
                                      /*block_size=*/32);
  EXPECT_FALSE(broadcast_fzp.zero_points.IsSymmetric());
  EXPECT_THAT(broadcast_fzp.zero_points.At<float>(0),
              IsOkAndHolds(FloatEq(-2.5f)));
  EXPECT_THAT(broadcast_fzp.zero_points.At<float>(7),
              IsOkAndHolds(FloatEq(-2.5f)));

  BlockwiseQuantization per_block_fzp({0.25f, 0.5f},
                                      ZeroPoints::Float({0.0f, -6.25f}),
                                      /*block_size=*/32);
  EXPECT_FALSE(per_block_fzp.zero_points.IsSymmetric());
  EXPECT_THAT(per_block_fzp.zero_points.At<float>(0),
              IsOkAndHolds(FloatEq(0.0f)));
  EXPECT_THAT(per_block_fzp.zero_points.At<float>(1),
              IsOkAndHolds(FloatEq(-6.25f)));
  EXPECT_FALSE(per_block_fzp.zero_points.At<float>(2).ok());

  BlockwiseQuantization int_zp({0.25f, 0.5f}, {0, -3}, /*block_size=*/32);
  EXPECT_FALSE(int_zp.zero_points.IsSymmetric());
  EXPECT_THAT(int_zp.zero_points.At<float>(0), IsOkAndHolds(FloatEq(0.0f)));
  EXPECT_THAT(int_zp.zero_points.At<float>(1), IsOkAndHolds(FloatEq(-3.0f)));
}

TEST(QuantizationTest,
     BlockwiseCpuSymmetricFallbackI8AndI4AndCacheInvalidation) {
  constexpr int kRows = 2;
  constexpr int kCols = 64;
  constexpr int kBlockSize = 32;

  // 1. Blockwise INT8 (`Q8_0`) -> per-channel symmetric `qcint8`.
  std::vector<int8_t> raw_i8(kRows * kCols);
  for (int i = 0; i < kRows * kCols; ++i) {
    raw_i8[i] = static_cast<int8_t>((i % 25) - 12);
  }
  BlockwiseQuantization bq_i8({0.125f, 0.5f, 0.25f, 0.0625f}, {}, kBlockSize);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto fb_i8, bq_i8.GetOrCreateCpuSymmetricFallback(
                      Type::kI8, kRows, kCols, raw_i8.data(), raw_i8.size()));
  ASSERT_NE(fb_i8, nullptr);
  EXPECT_EQ(fb_i8->type, Type::kI8);
  ASSERT_EQ(fb_i8->scales.size(), static_cast<size_t>(kRows));
  EXPECT_EQ(fb_i8->bf16_scales, nullptr);

  // Subsequent call returns the exact cached pointer.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto fb_i8_cached,
      bq_i8.GetOrCreateCpuSymmetricFallback(Type::kI8, kRows, kCols,
                                            raw_i8.data(), raw_i8.size()));
  EXPECT_EQ(fb_i8_cached.get(), fb_i8.get());

  const auto sym_i8 = fb_i8->data->Span<const int8_t>();
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols; ++c) {
      const int blk = r * (kCols / kBlockSize) + (c / kBlockSize);
      const float ref =
          bq_i8.scales[blk] * static_cast<float>(raw_i8[r * kCols + c]);
      const float deq =
          fb_i8->scales[r] * static_cast<float>(sym_i8[r * kCols + c]);
      EXPECT_NEAR(deq, ref, 0.05f);
    }
  }

  bq_i8.InvalidateBf16ScalesCache();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto fb_i8_rebuilt,
      bq_i8.GetOrCreateCpuSymmetricFallback(Type::kI8, kRows, kCols,
                                            raw_i8.data(), raw_i8.size()));
  EXPECT_NE(fb_i8_rebuilt.get(), fb_i8.get());

  // 2. Affine blockwise INT4 (`Q4_1` / `Q4_K`) -> symmetric `qbint4`.
  auto buf_i4 = OwningCpuBuffer::Allocate<Type::kI4>(kRows * kCols);
  int4_t* packed_i4 = buf_i4->Span<int4_t>().data();
  for (int idx = 0; idx < (kRows * kCols) / 2; ++idx) {
    packed_i4[idx].a = static_cast<int8_t>((idx % 15) - 7);
    packed_i4[idx].b = static_cast<int8_t>(((idx * 3 + 5) % 15) - 7);
  }
  BlockwiseQuantization bq_i4({0.25f, 0.125f, 0.5f, 0.0625f},
                              ZeroPoints::Float({-2.0f, 1.5f, 0.0f, -3.5f}),
                              kBlockSize);
  auto raw_i4_lock = buf_i4->Lock();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto fb_i4,
      bq_i4.GetOrCreateCpuSymmetricFallback(
          Type::kI4, kRows, kCols, raw_i4_lock.data(), raw_i4_lock.size()));
  ASSERT_NE(fb_i4, nullptr);
  EXPECT_EQ(fb_i4->type, Type::kI4);
  EXPECT_TRUE(fb_i4->scales.empty());
  ASSERT_NE(fb_i4->bf16_scales, nullptr);
  EXPECT_EQ(fb_i4->bf16_scales->size(), 4u);

  const auto sym_i4 = fb_i4->data->Span<const int4_t>();
  for (int blk = 0; blk < 4; ++blk) {
    const float max_abs = 10.0f * bq_i4.scales[blk];
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const float expected_zp,
                                    bq_i4.zero_points.At<float>(blk));
    for (int j = 0; j < kBlockSize; j += 2) {
      const int idx = (blk * kBlockSize + j) / 2;
      const float ref_a =
          bq_i4.scales[blk] *
          (static_cast<float>(static_cast<int>(packed_i4[idx].a)) -
           expected_zp);
      const float ref_b =
          bq_i4.scales[blk] *
          (static_cast<float>(static_cast<int>(packed_i4[idx].b)) -
           expected_zp);
      const float deq_a = static_cast<float>((*fb_i4->bf16_scales)[blk]) *
                          static_cast<float>(static_cast<int>(sym_i4[idx].a));
      const float deq_b = static_cast<float>((*fb_i4->bf16_scales)[blk]) *
                          static_cast<float>(static_cast<int>(sym_i4[idx].b));
      // Symmetric 4-bit re-quantization step is at most max(|w|) / 8.
      EXPECT_NEAR(deq_a, ref_a, max_abs / 7.0f + 0.05f);
      EXPECT_NEAR(deq_b, ref_b, max_abs / 7.0f + 0.05f);
    }
  }

  // Error validation checks.
  EXPECT_THAT(bq_i4.GetOrCreateCpuSymmetricFallback(
                  Type::kI4, kRows, /*cols=*/48, raw_i4_lock.data(),
                  raw_i4_lock.size()),
              Not(IsOk()));
  EXPECT_THAT(bq_i4.GetOrCreateCpuSymmetricFallback(
                  Type::kI4, kRows, kCols, raw_i4_lock.data(), /*raw_bytes=*/4),
              Not(IsOk()));
  EXPECT_THAT(
      bq_i4.GetOrCreateCpuSymmetricFallback(
          Type::kFP32, kRows, kCols, raw_i4_lock.data(), raw_i4_lock.size()),
      Not(IsOk()));
}

TEST(QuantizationTest, LazyBlockwiseMaterializationAndRelease) {
  std::atomic<int> materialize_calls{0};
  auto bq = BlockwiseQuantization::CreateLazily(
      /*block_size=*/32, /*quantized_dimension=*/0, /*num_blocks=*/4,
      [&]() -> BlockwiseQuantization::MaterializedBlockwiseParams {
        materialize_calls.fetch_add(1, std::memory_order_relaxed);
        return {
            .scales = {0.25f, 0.5f, 0.125f, 1.0f},
            .float_zero_points = {0.0f, -2.5f, 0.0f, 1.25f},
        };
      },
      BlockwiseQuantization::LazyOptions{
          .zero_points =
              BlockwiseQuantization::ZeroPointPresence::kMayHaveFloat,
          .pre_sanitized = true,
      });

  EXPECT_FALSE(bq->IsMaterialized());
  EXPECT_TRUE(bq->IsPreSanitized());
  EXPECT_EQ(bq->NumBlocks(), 4u);
  EXPECT_TRUE(bq->MayHaveFloatZeroPoints());
  EXPECT_FALSE(bq->HasIntegerZeroPoints());
  EXPECT_EQ(materialize_calls.load(std::memory_order_relaxed), 0);

  bq->EnsureMaterialized();
  {
    auto zp1 = bq->zero_points.At<float>(1);
    ASSERT_OK(zp1);
    EXPECT_FLOAT_EQ(*zp1, -2.5f);
  }
  EXPECT_TRUE(bq->IsMaterialized());
  EXPECT_EQ(materialize_calls.load(std::memory_order_relaxed), 1);
  ASSERT_EQ(bq->scales.size(), 4u);
  EXPECT_FLOAT_EQ(bq->scales[2], 0.125f);

  auto bf16 = bq->GetOrCreateBf16Scales();
  ASSERT_NE(bf16, nullptr);
  EXPECT_EQ(bf16->size(), 4u);
  EXPECT_EQ(materialize_calls.load(std::memory_order_relaxed), 1);

  bq->ReleaseMaterializedData();
  EXPECT_FALSE(bq->IsMaterialized());
  EXPECT_TRUE(bq->scales.empty());
  EXPECT_EQ(bq->NumBlocks(), 4u);

  // Re-materializes cleanly if requested again.
  bq->EnsureMaterialized();
  {
    auto zp3 = bq->zero_points.At<float>(3);
    ASSERT_OK(zp3);
    EXPECT_FLOAT_EQ(*zp3, 1.25f);
  }
  EXPECT_TRUE(bq->IsMaterialized());
  EXPECT_EQ(materialize_calls.load(std::memory_order_relaxed), 2);
}

TEST(QuantizationTest, LazyBlockwiseCopyMoveSetParamsFailureAndConcurrency) {
  // 1. Failed materialization does not poison `IsMaterialized()`.
  auto failing_bq = BlockwiseQuantization::CreateLazily(
      /*block_size=*/32, /*quantized_dimension=*/0, /*num_blocks=*/4,
      []() -> BlockwiseQuantization::MaterializedBlockwiseParams {
        return {};
      });
  failing_bq->EnsureMaterialized();
  EXPECT_FALSE(failing_bq->IsMaterialized());
  EXPECT_TRUE(failing_bq->scales.empty());

  // Wrong-sized `SetMaterializedParams` is also ignored.
  failing_bq->SetMaterializedParams({1.0f, 2.0f});
  EXPECT_FALSE(failing_bq->IsMaterialized());

  // Valid `SetMaterializedParams` populates without invoking `materialize_fn`.
  int called = 0;
  auto direct_bq = BlockwiseQuantization::CreateLazily(
      /*block_size=*/32, /*quantized_dimension=*/0, /*num_blocks=*/2,
      [&]() -> BlockwiseQuantization::MaterializedBlockwiseParams {
        ++called;
        return {.scales = {0.5f, 0.25f}};
      });
  direct_bq->SetMaterializedParams({0.125f, 0.75f}, {0.5f, -0.5f});
  EXPECT_TRUE(direct_bq->IsMaterialized());
  EXPECT_EQ(called, 0);
  EXPECT_THAT(direct_bq->scales, ElementsAre(FloatEq(0.125f), FloatEq(0.75f)));

  // 2. Copy and move preserve lazy spec & materialization state.
  std::atomic<int> copy_calls{0};
  auto orig = BlockwiseQuantization::CreateLazily(
      /*block_size=*/32, /*quantized_dimension=*/0, /*num_blocks=*/3,
      [&]() -> BlockwiseQuantization::MaterializedBlockwiseParams {
        copy_calls.fetch_add(1, std::memory_order_relaxed);
        return {.scales = {1.0f, 2.0f, 3.0f}};
      });
  BlockwiseQuantization copied(*orig);
  EXPECT_FALSE(copied.IsMaterialized());
  EXPECT_EQ(copied.NumBlocks(), 3u);
  copied.EnsureMaterialized();
  EXPECT_TRUE(copied.IsMaterialized());
  EXPECT_FALSE(orig->IsMaterialized());
  EXPECT_THAT(copied.scales, ElementsAre(FloatEq(1.0f), FloatEq(2.0f), 3.0f));

  BlockwiseQuantization copy_assigned;
  copy_assigned = *orig;
  EXPECT_FALSE(copy_assigned.IsMaterialized());
  EXPECT_EQ(copy_assigned.NumBlocks(), 3u);

  BlockwiseQuantization moved(std::move(*orig));
  EXPECT_FALSE(moved.IsMaterialized());
  EXPECT_EQ(moved.NumBlocks(), 3u);
  BlockwiseQuantization move_assigned;
  move_assigned = std::move(moved);
  EXPECT_FALSE(move_assigned.IsMaterialized());
  move_assigned.EnsureMaterialized();
  EXPECT_TRUE(move_assigned.IsMaterialized());
  EXPECT_THAT(move_assigned.scales,
              ElementsAre(FloatEq(1.0f), FloatEq(2.0f), 3.0f));

  // 3. Concurrent `EnsureMaterialized()` invokes `materialize_fn` once.
  std::atomic<int> concurrent_calls{0};
  auto shared_bq = BlockwiseQuantization::CreateLazily(
      /*block_size=*/32, /*quantized_dimension=*/0, /*num_blocks=*/4,
      [&]() -> BlockwiseQuantization::MaterializedBlockwiseParams {
        concurrent_calls.fetch_add(1, std::memory_order_relaxed);
        return {.scales = {0.25f, 0.5f, 0.75f, 1.0f},
                .float_zero_points = {0.5f, -0.5f, 1.0f, -1.0f}};
      },
      BlockwiseQuantization::LazyOptions{
          .zero_points =
              BlockwiseQuantization::ZeroPointPresence::kMayHaveFloat,
      });
  ParallelFor(
      /*count=*/16, /*min_grain=*/1,
      [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
          shared_bq->EnsureMaterialized();
          EXPECT_TRUE(shared_bq->IsMaterialized());
          EXPECT_EQ(shared_bq->NumBlocks(), 4u);
        }
      },
      /*max_threads=*/8);
  EXPECT_EQ(concurrent_calls.load(std::memory_order_relaxed), 1);
  EXPECT_THAT(shared_bq->scales, ElementsAre(0.25f, 0.5f, 0.75f, 1.0f));
}

TEST(QuantizationTest, CpuSymmetricFallbackLargeTensorInt8AndInt4) {
  // Large tensor (262144 elements) verifying symmetric fallback conversion.
  constexpr int kRows = 128;
  constexpr int kCols = 2048;
  constexpr int kBlockSize = 32;
  constexpr size_t kTotalElements = static_cast<size_t>(kRows) * kCols;
  constexpr size_t kBlocksPerRow = kCols / kBlockSize;
  constexpr size_t kTotalBlocks = static_cast<size_t>(kRows) * kBlocksPerRow;

  std::vector<float> scales(kTotalBlocks);
  std::vector<float> float_zps(kTotalBlocks);
  for (size_t b = 0; b < kTotalBlocks; ++b) {
    scales[b] = 0.05f * static_cast<float>((b % 7) + 1);
    float_zps[b] = (b % 2 == 0) ? -1.5f : 0.75f;
  }

  // INT8 parallel fallback.
  std::vector<int8_t> raw_i8(kTotalElements);
  for (size_t i = 0; i < kTotalElements; ++i) {
    raw_i8[i] = static_cast<int8_t>((i * 13 + 7) % 101 - 50);
  }
  BlockwiseQuantization bq_i8(scales, ZeroPoints::Float(float_zps), kBlockSize);
  ASSERT_OK_AND_ASSIGN(
      auto fb_i8, bq_i8.GetOrCreateCpuSymmetricFallback(
                      Type::kI8, kRows, kCols, raw_i8.data(), raw_i8.size()));
  ASSERT_NE(fb_i8, nullptr);
  ASSERT_EQ(fb_i8->scales.size(), static_cast<size_t>(kRows));
  const int8_t* fb_i8_data = fb_i8->data->Span<const int8_t>().data();
  for (int r : {0, 37, kRows - 1}) {
    const float row_scale = fb_i8->scales[r];
    for (int c : {0, 31, 1024, kCols - 1}) {
      const size_t idx = static_cast<size_t>(r) * kCols + c;
      const size_t blk = static_cast<size_t>(r) * kBlocksPerRow +
                         static_cast<size_t>(c / kBlockSize);
      const float expected =
          (static_cast<float>(raw_i8[idx]) - float_zps[blk]) * scales[blk];
      const float actual = static_cast<float>(fb_i8_data[idx]) * row_scale;
      EXPECT_NEAR(actual, expected, row_scale + 1e-4f);
    }
  }

  // INT4 parallel fallback.
  auto raw_i4 = OwningCpuBuffer::Allocate<Type::kI4>(kTotalElements);
  ASSERT_NE(raw_i4, nullptr);
  int4_t* raw_i4_ptr = raw_i4->Span<int4_t>().data();
  for (size_t i = 0; i < kTotalElements / 2; ++i) {
    raw_i4_ptr[i].a = static_cast<int8_t>((i % 15) - 7);
    raw_i4_ptr[i].b = static_cast<int8_t>(((i * 3) % 15) - 7);
  }
  BlockwiseQuantization bq_i4(scales, ZeroPoints::Float(float_zps), kBlockSize);
  auto raw_i4_lock = raw_i4->Lock();
  ASSERT_OK_AND_ASSIGN(
      auto fb_i4,
      bq_i4.GetOrCreateCpuSymmetricFallback(
          Type::kI4, kRows, kCols, raw_i4_lock.data(), raw_i4_lock.size()));
  ASSERT_NE(fb_i4, nullptr);
  ASSERT_NE(fb_i4->bf16_scales, nullptr);
  EXPECT_EQ(fb_i4->bf16_scales->size(), kTotalBlocks);
  const int4_t* fb_i4_ptr = fb_i4->data->Span<const int4_t>().data();
  for (size_t b : {size_t{0}, kTotalBlocks / 2, kTotalBlocks - 1}) {
    const float fb_s = static_cast<float>((*fb_i4->bf16_scales)[b]);
    EXPECT_GT(fb_s, 0.0f);
    const size_t pair_base = b * (kBlockSize / 2);
    const float expected0 =
        (static_cast<float>(raw_i4_ptr[pair_base].a) - float_zps[b]) *
        scales[b];
    const float actual0 = static_cast<float>(fb_i4_ptr[pair_base].a) * fb_s;
    EXPECT_NEAR(actual0, expected0, fb_s + 1e-2f);
  }
}

TEST(ScopeTest, ScopeBoundaryRetentionGuardUsesWeakRefForLeaves) {
  std::shared_ptr<Scope> captured_scope;
  std::weak_ptr<TensorGroup> weak_leaf;
  {
    ScopeBoundaryRetentionGuard guard(/*retain_leaf_tensors=*/false);
    ScopeGuard scope("test_scope");
    Tensor leaf = NewTensor(absl::SourceLocation::current());
    weak_leaf = leaf.group;
    scope.Bind("weight", leaf);
    captured_scope = scope.scope();
    EXPECT_TRUE(captured_scope->GetBoundaryTensor("weight").has_value());
  }
  EXPECT_TRUE(weak_leaf.expired());
  EXPECT_FALSE(captured_scope->GetBoundaryTensor("weight").has_value());
}

}  // namespace
}  // namespace litert::tensor::graph

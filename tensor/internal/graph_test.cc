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

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/source_location.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/type_id.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::graph {
namespace {

using testing::ElementsAre;
using testing::ElementsAreArray;
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
  PerChannelAffineQuantization quant({1.0f}, std::vector<int64_t>{0}, 0);
  EXPECT_EQ(quant.GetTypeId(),
            internal::TypeId::Get<PerChannelAffineQuantization>());
}

TEST(QuantizationTest, IsA) {
  PerChannelAffineQuantization quant({1.0f}, std::vector<int64_t>{0}, 0);
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

TEST(QuantizationTest, DefaultsToNoIntegerZeroPoints) {
  PerChannelAffineQuantization pcq;
  ASSERT_NE(pcq.IntegerZeroPoints(), nullptr);
  EXPECT_THAT(*pcq.IntegerZeroPoints(), IsEmpty());
  EXPECT_EQ(pcq.FloatZeroPoints(), nullptr);
  EXPECT_FALSE(pcq.HasFloatZeroPoints());
  EXPECT_EQ(pcq.NumZeroPoints(), 0);
  EXPECT_FALSE(pcq.HasNonZeroZeroPoints());
  EXPECT_FLOAT_EQ(pcq.ZeroPointAt(0), 0.0f);

  BlockwiseQuantization bwq;
  ASSERT_NE(bwq.IntegerZeroPoints(), nullptr);
  EXPECT_THAT(*bwq.IntegerZeroPoints(), IsEmpty());
  EXPECT_FALSE(bwq.HasFloatZeroPoints());
  EXPECT_EQ(bwq.NumZeroPoints(), 0);
}

TEST(QuantizationTest, IntegerZeroPoints) {
  PerChannelAffineQuantization pcq({0.5f, 0.25f},
                                   /*zero_points=*/std::vector<int64_t>{0, 3},
                                   /*quantized_dimension=*/0);
  ASSERT_NE(pcq.IntegerZeroPoints(), nullptr);
  EXPECT_THAT(*pcq.IntegerZeroPoints(), ElementsAre(0, 3));
  EXPECT_EQ(pcq.FloatZeroPoints(), nullptr);
  EXPECT_FALSE(pcq.HasFloatZeroPoints());
  EXPECT_EQ(pcq.NumZeroPoints(), 2);
  EXPECT_TRUE(pcq.HasNonZeroZeroPoints());
  EXPECT_FLOAT_EQ(pcq.ZeroPointAt(0), 0.0f);
  EXPECT_FLOAT_EQ(pcq.ZeroPointAt(1), 3.0f);
  EXPECT_FLOAT_EQ(pcq.ZeroPointAt(2), 0.0f);

  BlockwiseQuantization bwq({0.5f, 0.25f},
                            /*zero_points=*/std::vector<int64_t>{2},
                            /*block_size=*/32);
  EXPECT_EQ(bwq.NumZeroPoints(), 1);
  EXPECT_FLOAT_EQ(bwq.ZeroPointAt(0), 2.0f);
  EXPECT_FLOAT_EQ(bwq.ZeroPointAt(1), 2.0f);
}

TEST(QuantizationTest, FloatZeroPoints) {
  PerChannelAffineQuantization pcq({0.5f, 0.25f},
                                   /*zero_points=*/std::vector<float>{-0.5f},
                                   /*quantized_dimension=*/0);
  EXPECT_EQ(pcq.IntegerZeroPoints(), nullptr);
  ASSERT_NE(pcq.FloatZeroPoints(), nullptr);
  EXPECT_THAT(*pcq.FloatZeroPoints(), ElementsAre(-0.5f));
  EXPECT_TRUE(pcq.HasFloatZeroPoints());
  EXPECT_EQ(pcq.NumZeroPoints(), 1);
  EXPECT_TRUE(pcq.HasNonZeroZeroPoints());
  EXPECT_FLOAT_EQ(pcq.ZeroPointAt(0), -0.5f);
  EXPECT_FLOAT_EQ(pcq.ZeroPointAt(1), -0.5f);

  BlockwiseQuantization bwq({0.5f, 0.25f},
                            /*zero_points=*/std::vector<float>{-0.5f, 0.25f},
                            /*block_size=*/32, /*quantized_dimension=*/0);
  EXPECT_TRUE(bwq.HasFloatZeroPoints());
  EXPECT_EQ(bwq.NumZeroPoints(), 2);
  EXPECT_TRUE(bwq.HasNonZeroZeroPoints());
  EXPECT_FLOAT_EQ(bwq.ZeroPointAt(0), -0.5f);
  EXPECT_FLOAT_EQ(bwq.ZeroPointAt(1), 0.25f);
}

TEST(QuantizationTest, AllZeroFloatZeroPointsAreStillFloat) {
  PerChannelAffineQuantization pcq(
      {0.5f, 0.25f},
      /*zero_points=*/std::vector<float>{0.0f, 0.0f},
      /*quantized_dimension=*/0);
  EXPECT_TRUE(pcq.HasFloatZeroPoints());
  EXPECT_FALSE(pcq.HasNonZeroZeroPoints());
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

}  // namespace
}  // namespace litert::tensor::graph

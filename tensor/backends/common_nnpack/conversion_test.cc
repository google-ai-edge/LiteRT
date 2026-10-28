/* Copyright 2026 Google LLC.

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

#include "tensor/backends/common_nnpack/conversion.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/source_location.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/datatypes.h"
#include "tensor/internal/arithmetic_helpers.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/type_id.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

TEST(CommonNnpackConversionTest, TopologicalSortLeafTensorReturnsEmpty) {
  Tensor<> a(TensorInit{.name = "a", .type = Type::kFP32, .shape = {2}});

  absl::flat_hash_set<graph::Tensor> inlined_inputs;
  absl::flat_hash_set<graph::Tensor> visited_tensors;
  absl::flat_hash_set<const graph::Operation*> visited_ops;
  std::vector<const graph::Operation*> ordered_ops;

  EXPECT_THAT(TopologicalSort(a.GetRaw(), inlined_inputs, visited_tensors,
                              visited_ops, ordered_ops),
              IsOk());
  EXPECT_TRUE(ordered_ops.empty());
}

TEST(CommonNnpackConversionTest,
     TopologicalSortOrdersOperationsInDependencyOrder) {
  Tensor<> a(TensorInit{.name = "a", .type = Type::kFP32, .shape = {2}});
  Tensor<> b(TensorInit{.name = "b", .type = Type::kFP32, .shape = {2}});
  Tensor<> d(TensorInit{.name = "d", .type = Type::kFP32, .shape = {2}});

  Tensor<> c = Add(a, b);
  Tensor<> e = Add(c, d);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto c_producer,
                                  graph::GetProducer(c.GetRaw()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto e_producer,
                                  graph::GetProducer(e.GetRaw()));

  absl::flat_hash_set<graph::Tensor> inlined_inputs;
  absl::flat_hash_set<graph::Tensor> visited_tensors;
  absl::flat_hash_set<const graph::Operation*> visited_ops;
  std::vector<const graph::Operation*> ordered_ops;

  EXPECT_THAT(TopologicalSort(e.GetRaw(), inlined_inputs, visited_tensors,
                              visited_ops, ordered_ops),
              IsOk());
  EXPECT_THAT(ordered_ops, ElementsAre(c_producer.get(), e_producer.get()));
}

TEST(CommonNnpackConversionTest, TopologicalSortStopsAtInlinedInputs) {
  Tensor<> a(TensorInit{.name = "a", .type = Type::kFP32, .shape = {2}});
  Tensor<> b(TensorInit{.name = "b", .type = Type::kFP32, .shape = {2}});
  Tensor<> d(TensorInit{.name = "d", .type = Type::kFP32, .shape = {2}});

  Tensor<> c = Add(a, b);
  Tensor<> e = Add(c, d);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto e_producer,
                                  graph::GetProducer(e.GetRaw()));

  absl::flat_hash_set<graph::Tensor> inlined_inputs = {c.GetRaw()};
  absl::flat_hash_set<graph::Tensor> visited_tensors;
  absl::flat_hash_set<const graph::Operation*> visited_ops;
  std::vector<const graph::Operation*> ordered_ops;

  EXPECT_THAT(TopologicalSort(e.GetRaw(), inlined_inputs, visited_tensors,
                              visited_ops, ordered_ops),
              IsOk());
  EXPECT_THAT(ordered_ops, ElementsAre(e_producer.get()));
}

struct DummyOpExtension : public graph::BackendExtension {
  internal::TypeId GetTypeId() const override {
    return internal::TypeId::Get<DummyOpExtension>();
  }
};

struct DummyGraph : public NnpackGraph {};

class DummyBuildContext : public NnpackBuildContext {
 public:
  using NnpackBuildContext::NnpackBuildContext;

  absl::string_view BackendName() const override { return "Dummy"; }
  uint32_t FlagExternalInput() const override { return 1; }
  uint32_t FlagExternalOutput() const override { return 2; }

 protected:
  absl::Status EnsureInitialized() override { return absl::OkStatus(); }
  std::unique_ptr<NnpackGraph> CreateEmptyGraph() override {
    return std::make_unique<DummyGraph>();
  }
  absl::Status CreateSubgraph(size_t, uint32_t) override {
    return absl::OkStatus();
  }
  absl::Status DefineTensorValue(const graph::Tensor&, NnpackValue&) override {
    return absl::OkStatus();
  }
  absl::Status DefineConstantTensor(Type, absl::Span<const size_t>, const void*,
                                    uint32_t* id) override {
    *id = 0;
    return absl::OkStatus();
  }
  absl::Status LowerOp(const graph::Operation&) override {
    return absl::OkStatus();
  }
};

struct DummyOp : public graph::Operation {
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION;
  absl::string_view GetName() const override { return "DummyOp"; }
};

template <class... Mixins>
TensorHandle DummyArithmeticOp(
    TensorHandle a,
    absl::SourceLocation loc = absl::SourceLocation::current()) {
  return ElementwiseOp<DummyOp>(loc, a);
}

TEST(CommonNnpackConversionTest,
     InlineImplementationGraphForFailsIfOutputsMismatch) {
  TensorHandle in(TensorInit{.name = "in", .type = Type::kFP32, .shape = {2}});
  TensorHandle out = DummyArithmeticOp(in).SetName("out");

  auto op = out.GetRaw().group->producer;

  TensorHandle extra_output(
      TensorInit{.name = "extra", .type = Type::kFP32, .shape = {2}});

  DummyBuildContext ctx(/*outputs=*/{out, extra_output});
  ASSERT_THAT(ctx.Init(), IsOk());

  auto status = InlineImplementationGraphFor(
      *op, /*inlined_inputs=*/{in.GetRaw()},
      /*inlined_outputs=*/{out.GetRaw(), extra_output.GetRaw()}, ctx);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::tensor

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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/utils/matchers.h"
#include "tensor/utils/source_location.h"

namespace litert::tensor::graph {
namespace {

using testing::ElementsAreArray;
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
  const auto loc = source_location::current();
  const std::shared_ptr<const TensorGroup> g = NewTensorGroup(3, loc);
  EXPECT_NE(g, nullptr);
  EXPECT_THAT(g->tensor_infos, SizeIs(3));
  EXPECT_THAT(g->status, IsOk());
  EXPECT_EQ(g->producer, nullptr);
  EXPECT_THAT(g->loc, IsLocation(loc));
}

TEST(TensorTest, NewTensorCreatesAValidTensor) {
  const auto loc = source_location::current();
  Tensor a = NewTensor(loc);
  EXPECT_EQ(a.index, 0);
  ASSERT_NE(a.group, nullptr);
  EXPECT_THAT(GetStatus(a), IsOk());
  EXPECT_THAT(GetLocation(a), IsLocation(loc));
}

TEST(TensorTest, SetAndGetTensorName) {
  Tensor a = NewTensor(source_location::current());
  EXPECT_THAT(SetName(a, "my tensor name"), IsOk());
  EXPECT_THAT(GetName(a), IsOkAndHolds(StrEq("my tensor name")));
}

TEST(TensorTest, SetAndGetBuffer) {
  Tensor a = NewTensor(source_location::current());
  auto buffer = OwningCpuBuffer::Copy<Type::kI32>({1, 2, 3, 4});
  EXPECT_THAT(SetBuffer(a, buffer), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & retrieved_buffer, GetBuffer(a));
  EXPECT_THAT(retrieved_buffer.Lock(), ElementsAreArray(buffer->Lock()));
}

TEST(TensorTest, DefaultTensorIsInvalid) {
  Tensor a;
  EXPECT_THAT(GetStatus(a), Not(IsOk()));
  EXPECT_THAT(GetLocation(a), IsLocation(source_location()));
  EXPECT_THAT(GetProducer(a), Not(IsOk()));
  EXPECT_THAT(GetConsumers(a), Not(IsOk()));
}

TEST(TensorTest, TensorWithWringIndexIsInvalid) {
  Tensor a = NewTensor(source_location::current());
  a.index = 3;
  EXPECT_THAT(GetStatus(a), Not(IsOk()));
}

}  // namespace
}  // namespace litert::tensor::graph

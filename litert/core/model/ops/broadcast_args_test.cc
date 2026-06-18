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

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference.h"

namespace litert::internal {
namespace {

TEST(BroadcastArgsOpTest, BasicBroadcast) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflBroadcastArgs);

  auto make_shape_tensor = [&](std::vector<int32_t> dims) -> LiteRtTensorT& {
    auto& t = subgraph.EmplaceTensor();
    OwningBufferRef<uint8_t> buf(dims.size() * sizeof(int32_t));
    std::memcpy(buf.Data(), dims.data(), buf.Size());
    SetWeightsFromOwnedBuffer(t.Weights(), std::move(buf));
    t.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32,
                                   {static_cast<int32_t>(dims.size())}));
    return t;
  };

  auto& s1 = make_shape_tensor({2, 1, 4});
  auto& s2 = make_shape_tensor({3, 4});

  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {}));

  AttachInput(&s1, op);
  AttachInput(&s2, op);
  AttachOutput(&output, op);

  ShapeInferenceEngine engine(&model);
  ASSERT_EQ(engine.InferShapes(), kLiteRtStatusOk);

  const auto& shape = output.Type().second.ranked_tensor_type.layout;
  EXPECT_EQ(shape.rank, 1);
  EXPECT_EQ(shape.dimensions[0], 3);

  ASSERT_GT(output.Weights().Buffer().Size(), 0);
  const int32_t* data =
      reinterpret_cast<const int32_t*>(output.Weights().Buffer().Data());
  EXPECT_THAT(std::vector<int32_t>(data, data + 3),
              testing::ElementsAre(2, 3, 4));
}

}  // namespace
}  // namespace litert::internal

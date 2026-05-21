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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference.h"

namespace litert::internal {
namespace {

TEST(RangeOpTest, RangeInt32) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflRange);

  auto make_scalar_int = [&](int32_t val) -> LiteRtTensorT& {
    auto& t = subgraph.EmplaceTensor();
    SetWeightsFromOwnedBuffer(
        t.Weights(),
        OwningBufferRef<uint8_t>(absl::string_view(
            reinterpret_cast<const char*>(&val), sizeof(int32_t))));
    t.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {}));
    return t;
  };

  auto& start = make_scalar_int(0);
  auto& limit = make_scalar_int(10);
  auto& delta = make_scalar_int(2);

  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {}));

  AttachInput(&start, op);
  AttachInput(&limit, op);
  AttachInput(&delta, op);
  AttachOutput(&output, op);

  ShapeInferenceEngine engine(&model);
  ASSERT_EQ(engine.InferShapes(), kLiteRtStatusOk);

  const auto& shape = output.Type().second.ranked_tensor_type.layout;
  EXPECT_EQ(shape.rank, 1);
  EXPECT_EQ(shape.dimensions[0], 5);

  ASSERT_GT(output.Weights().Buffer().Size(), 0);
  const int32_t* data =
      reinterpret_cast<const int32_t*>(output.Weights().Buffer().Data());
  EXPECT_THAT(std::vector<int32_t>(data, data + 5),
              testing::ElementsAre(0, 2, 4, 6, 8));
}

}  // namespace
}  // namespace litert::internal

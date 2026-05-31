// Copyright 2024 Google LLC.
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
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/core/model/model.h"
#include "litert/tools/batch_utils.h"

namespace litert {
namespace {

using ::MakeRankedTensorType;

TEST(FixBatchTest, SuccessDynamicBatch) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& tensor = subgraph.EmplaceTensor();
  tensor.SetType(::MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 128}));

  EXPECT_EQ(tools::ValidateModelForBatchFix(model), kLiteRtStatusOk);
  tools::FixBatchDimension(model, 8);

  EXPECT_EQ(tensor.Type().second.ranked_tensor_type.layout.dimensions[0], 8);
}

TEST(FixBatchTest, RejectDynamicNonBatchDim) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& tensor = subgraph.EmplaceTensor();
  // Dynamic at index 1
  tensor.SetType(::MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, -1}));

  EXPECT_EQ(tools::ValidateModelForBatchFix(model),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(FixBatchTest, AllowTransposePreservingBatch) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& in = subgraph.EmplaceTensor();
  in.SetType(::MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 10, 20}));
  auto& out = subgraph.EmplaceTensor();
  out.SetType(::MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 20, 10}));

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflTranspose);
  ::litert::internal::AttachInput(&in, op);
  ::litert::internal::AttachOutput(&out, op);

  EXPECT_EQ(tools::ValidateModelForBatchFix(model), kLiteRtStatusOk);
  tools::FixBatchDimension(model, 4);

  EXPECT_EQ(in.Type().second.ranked_tensor_type.layout.dimensions[0], 4);
  EXPECT_EQ(out.Type().second.ranked_tensor_type.layout.dimensions[0], 4);
}

TEST(FixBatchTest, RejectTransposeMovingBatch) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& in = subgraph.EmplaceTensor();
  in.SetType(::MakeRankedTensorType(kLiteRtElementTypeFloat32, {-1, 10, 20}));
  auto& out = subgraph.EmplaceTensor();
  // Batch moved to index 1
  out.SetType(::MakeRankedTensorType(kLiteRtElementTypeFloat32, {10, -1, 20}));

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflTranspose);
  ::litert::internal::AttachInput(&in, op);
  ::litert::internal::AttachOutput(&out, op);

  EXPECT_EQ(tools::ValidateModelForBatchFix(model),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(FixBatchTest, SuccessFixedBatch) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& tensor = subgraph.EmplaceTensor();
  tensor.SetType(::MakeRankedTensorType(kLiteRtElementTypeFloat32, {4, 64}));

  EXPECT_EQ(tools::ValidateModelForBatchFix(model), kLiteRtStatusOk);
  tools::FixBatchDimension(model, 16);

  EXPECT_EQ(tensor.Type().second.ranked_tensor_type.layout.dimensions[0], 16);
}

}  // namespace
}  // namespace litert

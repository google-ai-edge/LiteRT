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

#include "tensor/runners/ml_drift/compiled_ml_drift_model_runner_util.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "tensor/arithmetic.h"
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/graph_traversal.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"  // IWYU pragma: keep (OSS ASSERT_OK*)

namespace litert::tensor::ml_drift_runner_internal {
namespace {

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;
using TensorMLD = Tensor<MlDriftMixinTag>;

TEST(CollectFcAndEmbeddingWeightsTest, OrdersByFirstUseAndFlagsFcWeights) {
  constexpr int kVocab = 8;
  constexpr int kDim = 16;
  constexpr int kOut = 4;
  TensorMLD ids({.name = "ids", .type = Type::kI32, .shape = {1, 2}});
  TensorMLD embedding(
      {.name = "embedding", .type = Type::kFP32, .shape = {kVocab, kDim}});
  TensorMLD w({.name = "w", .type = Type::kFP32, .shape = {kOut, kDim}});
  TensorMLD tied({.name = "tied", .type = Type::kFP32, .shape = {kDim, kDim}});
  // `embedding` is only used by an EmbeddingLookup; `tied` by an
  // EmbeddingLookup and a FullyConnected.
  TensorMLD h = FullyConnected(EmbeddingLookup(ids, embedding), w);
  TensorMLD z = FullyConnected(EmbeddingLookup(ids, tied), tied);
  const std::vector<TensorHandle> outputs = {h, z};
  ASSERT_OK_AND_ASSIGN(const std::vector<const graph::Operation*> plan,
                       GetExecutionPlan(outputs));

  const FcAndEmbeddingWeights weights = CollectFcAndEmbeddingWeights(plan);

  // Each tensor once, `embedding` (first used by the EmbeddingLookup that
  // feeds the FullyConnected of `w`) before `w`.
  ASSERT_THAT(weights.ordered, UnorderedElementsAre(embedding.GetRaw(),
                                                    w.GetRaw(), tied.GetRaw()));
  const auto index_of = [&](const graph::Tensor& t) {
    return std::find(weights.ordered.begin(), weights.ordered.end(), t) -
           weights.ordered.begin();
  };
  EXPECT_LT(index_of(embedding.GetRaw()), index_of(w.GetRaw()));
  EXPECT_THAT(weights.used_by_fully_connected,
              UnorderedElementsAre(w.GetRaw(), tied.GetRaw()));
}

TEST(GpuWeightConversionBatchEndTest, FillsBatchesUpToTheLimit) {
  std::vector<GpuWeightConversion> conversions(5);
  const size_t raw_bytes[] = {100, 50, 60, 300, 10};
  for (size_t i = 0; i < conversions.size(); ++i) {
    conversions[i].raw_bytes = raw_bytes[i];
  }
  std::vector<size_t> ends;
  for (size_t begin = 0; begin < conversions.size();) {
    begin = GpuWeightConversionBatchEnd(conversions, begin,
                                        /*max_batch_bytes=*/160);
    ends.push_back(begin);
  }
  // A tensor larger than the limit gets a batch of its own.
  EXPECT_THAT(ends, ElementsAre(2, 3, 4, 5));
}

TEST(GpuWeightsSumIDescriptorTest, IsALinearInt32Buffer) {
  const ml_drift::TensorDescriptor desc = GpuWeightsSumIDescriptor(12);
  EXPECT_EQ(desc.GetDataType(), ml_drift::DataType::kInt32);
  EXPECT_EQ(desc.GetStorageType(), ml_drift::TensorStorageType::kBuffer);
  EXPECT_EQ(desc.GetLayout(), ml_drift::Layout::kLinear);
  EXPECT_EQ(desc.GetBHWCShape(), ml_drift::BHWC(1, 1, 1, 12));
}

}  // namespace
}  // namespace litert::tensor::ml_drift_runner_internal

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

// Backend-independent helpers of `CompiledMlDriftModelRunner`.

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_COMPILED_ML_DRIFT_MODEL_RUNNER_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_COMPILED_ML_DRIFT_MODEL_RUNNER_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "tensor/buffer.h"
#include "tensor/internal/graph.h"

namespace litert::tensor::ml_drift_runner_internal {

// The weights (input 1) of the FullyConnected and EmbeddingLookup ops of an
// execution plan.
struct FcAndEmbeddingWeights {
  // Each weights tensor once, in the order of its first use. A deterministic
  // order (unlike iterating a hash set) creates the model tensors and forms
  // the GPU conversion batches the same way on every run.
  std::vector<graph::Tensor> ordered;
  // The weights used by at least one FullyConnected; the others are only used
  // by EmbeddingLookup ops. Only FullyConnected consumers need `sum_i`.
  absl::flat_hash_set<graph::Tensor> used_by_fully_connected;
};

FcAndEmbeddingWeights CollectFcAndEmbeddingWeights(
    absl::Span<const graph::Operation* const> execution_plan);

// How the GPU rearranges one constant weights tensor.
struct GpuWeightConversionSpec {
  ml_drift::OHWI ohwi;
  // The kernel layout, as the CPU path selects it.
  ml_drift::WeightsDescription desc;
  // kInt4 / kInt8 / kInt2: the type of the raw OHWI values (packed for INT4 /
  // INT2).
  ml_drift::DataType src_type = ml_drift::DataType::kUnknown;
  // Bytes of the raw OHWI values.
  size_t raw_bytes = 0;
  // The converted tensor (without data).
  ml_drift::TensorDescriptor dst_desc;
  // Whether the GPU also computes `sum_i` (like the CPU path, for INT4 and
  // INT8 weights with a FullyConnected consumer and 8-bit convolutions).
  bool needs_sum_i = false;
};

// Returns how the GPU rearranges the constant 2D INT4 / INT8 / INT2 weights
// `info` (`wants_sum_i`: a FullyConnected consumer needs `sum_i`) into exactly
// the tensor that the CPU path creates, or nullopt if the weights keep the CPU
// path: other types or shapes, INT4 / INT2 rows that aren't a multiple of 4
// values, INT4 `sum_i` with `i % 8 != 0 || o % 4 != 0`, and kernel layouts
// other than the linear (buffer) ones and the I4O4 2D texture one.
std::optional<GpuWeightConversionSpec> GetGpuWeightConversionSpec(
    ml_drift::GpuModelBuilder& builder, const graph::TensorInformation& info,
    bool wants_sum_i);

// The `sum_i` tensor (INT32, one value per output channel, in a buffer) that
// the GPU computes for weights with `out_channels` output channels.
ml_drift::TensorDescriptor GpuWeightsSumIDescriptor(int out_channels);

// A constant FullyConnected / EmbeddingLookup weights tensor rearranged on the
// GPU (`CompiledMlDriftModelRunner::SetConvertWeightsOnGpu`).
struct GpuWeightConversion {
  graph::Tensor weights_tensor;
  std::string name;
  ml_drift::OHWI ohwi;
  ml_drift::WeightsDescription desc;
  // kInt4 / kInt8 / kInt2: the raw OHWI values (packed for INT4 / INT2).
  ml_drift::DataType src_type = ml_drift::DataType::kUnknown;
  std::shared_ptr<Buffer> buffer;  // Keeps `raw` alive.
  LockedBufferSpan<const int8_t> raw = LockedBufferSpan<const int8_t>::Empty();
  size_t raw_bytes = 0;
  // The converted weights and `sum_i` in the main model.
  ml_drift::GpuModelBuilder::TensorHandle handle;
  std::optional<ml_drift::GpuModelBuilder::TensorHandle> sum_i;
};

// The end of the batch of `conversions` that starts at `begin`: at least one
// conversion, then as many as fit in `max_batch_bytes` of raw weights.
size_t GpuWeightConversionBatchEnd(
    absl::Span<const GpuWeightConversion> conversions, size_t begin,
    size_t max_batch_bytes);

// An ML Drift model that converts a batch of weights on the GPU.
struct GpuWeightConversionModel {
  ml_drift::GpuModel model;
  // Value id in `model` -> value id in the main model of each converted
  // tensor (the outputs of `model`).
  absl::flat_hash_map<ml_drift::ValueId, ml_drift::ValueId> io_mapping;
  // The raw weights to upload as the inputs of `model`. They point to the
  // `raw` values of the conversions.
  std::vector<ml_drift::WeightsManager::UploadWeightsInfo> uploads;
};

// Creates the model that converts all the weights (and `sum_i`) of `batch`
// together.
absl::StatusOr<GpuWeightConversionModel> CreateGpuWeightConversionModel(
    const ml_drift::GpuInfo& gpu_info,
    absl::Span<const GpuWeightConversion> batch);

// The descriptor, with the data, of the raw weights input `upload` of
// `conversion_model`.
absl::StatusOr<ml_drift::TensorDescriptor> RawWeightsDescriptor(
    const ml_drift::GpuModel& conversion_model,
    const ml_drift::WeightsManager::UploadWeightsInfo& upload);

}  // namespace litert::tensor::ml_drift_runner_internal

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_COMPILED_ML_DRIFT_MODEL_RUNNER_UTIL_H_

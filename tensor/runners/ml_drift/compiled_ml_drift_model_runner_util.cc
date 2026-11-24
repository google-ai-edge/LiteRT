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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_conversion.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "tensor/backends/ml_drift/ml_drift_helpers.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::ml_drift_runner_internal {

FcAndEmbeddingWeights CollectFcAndEmbeddingWeights(
    absl::Span<const graph::Operation* const> execution_plan) {
  FcAndEmbeddingWeights weights;
  absl::flat_hash_set<graph::Tensor> seen;
  for (const graph::Operation* op : execution_plan) {
    const bool fully_connected = op->GetName() == "FullyConnected";
    if (!fully_connected && op->GetName() != "EmbeddingLookup") continue;
    const graph::Tensor& weights_tensor = op->inputs[1];
    if (seen.insert(weights_tensor).second) {
      weights.ordered.push_back(weights_tensor);
    }
    if (fully_connected) {
      weights.used_by_fully_connected.insert(weights_tensor);
    }
  }
  return weights;
}

std::optional<GpuWeightConversionSpec> GetGpuWeightConversionSpec(
    ml_drift::GpuModelBuilder& builder, const graph::TensorInformation& info,
    bool wants_sum_i) {
  // Named constant 2D (output x input channels) weights.
  if (info.buffer == nullptr || info.name.empty() || info.shape.size() != 2 ||
      info.shape[0] <= 0 || info.shape[1] <= 0) {
    return std::nullopt;
  }
  GpuWeightConversionSpec spec;
  spec.ohwi = ml_drift::OHWI(info.shape[0], 1, 1, info.shape[1]);
  const size_t num_values = spec.ohwi.DimensionsProduct();
  // The raw type and the kernel layout that the CPU path selects for it.
  if (info.type == Type::kI4) {
    // ML Drift's raw INT4 converter needs whole input bytes per row group, and
    // `CreateAccumulateInputChannels` (`sum_i`) needs `i % 8 == 0 &&
    // o % 4 == 0`.
    spec.needs_sum_i = wants_sum_i;
    if (spec.ohwi.i % 4 != 0) return std::nullopt;
    if (spec.needs_sum_i && (spec.ohwi.i % 8 != 0 || spec.ohwi.o % 4 != 0)) {
      return std::nullopt;
    }
    spec.src_type = ml_drift::DataType::kInt4;
    spec.desc = builder.GetFullyConnectedInt4WeightsDesc(spec.ohwi);
    spec.raw_bytes = (num_values + 1) / 2;
  } else if (info.type == Type::kI8) {
    spec.src_type = ml_drift::DataType::kInt8;
    spec.desc = builder.GetFullyConnectedInt8WeightsDesc(spec.ohwi);
    spec.raw_bytes = num_values;
  } else if (info.type == Type::kI2) {
    if (spec.ohwi.i % 4 != 0) return std::nullopt;
    spec.src_type = ml_drift::DataType::kInt2;
    spec.desc = builder.GetFullyConnectedInt2WeightsDesc(spec.ohwi);
    spec.raw_bytes = (num_values + 3) / 4;
  } else {
    return std::nullopt;
  }
  // Same storage as the CPU paths: linear layouts in a buffer, the I4O4
  // layout in one 2D texture (`GetTensorDescriptorsForWeightsLayout` gives
  // the same storage, size and bytes as the CPU paths for these layouts). The
  // CPU paths store the other layouts differently (e.g. in a buffer for
  // embeddings), so they keep the CPU paths.
  const bool texture_layout =
      spec.desc.layout ==
      ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
  if (!spec.desc.IsLinearLayout() && !texture_layout) return std::nullopt;
  if (spec.desc.type != ml_drift::DataType::kUint4 &&
      spec.desc.type != ml_drift::DataType::kUint8 &&
      spec.desc.type != ml_drift::DataType::kUint2) {
    return std::nullopt;
  }
  std::vector<ml_drift::TensorDescriptor> dst_descs =
      ml_drift::GetTensorDescriptorsForWeightsLayout(spec.ohwi, spec.desc);
  if (dst_descs.size() != 1) return std::nullopt;
  spec.dst_desc = std::move(dst_descs[0]);
  return spec;
}

ml_drift::TensorDescriptor GpuWeightsSumIDescriptor(int out_channels) {
  ml_drift::TensorDescriptor desc(ml_drift::DataType::kInt32,
                                  ml_drift::TensorStorageType::kBuffer,
                                  ml_drift::Layout::kLinear);
  desc.SetBHWCShape(ml_drift::BHWC(1, 1, 1, out_channels));
  return desc;
}

size_t GpuWeightConversionBatchEnd(
    absl::Span<const GpuWeightConversion> conversions, size_t begin,
    size_t max_batch_bytes) {
  size_t end = begin;
  size_t batch_bytes = 0;
  while (end < conversions.size() &&
         (end == begin ||
          batch_bytes + conversions[end].raw_bytes <= max_batch_bytes)) {
    batch_bytes += conversions[end].raw_bytes;
    ++end;
  }
  return end;
}

absl::StatusOr<GpuWeightConversionModel> CreateGpuWeightConversionModel(
    const ml_drift::GpuInfo& gpu_info,
    absl::Span<const GpuWeightConversion> batch) {
  // The weights and their `sum_i` share one raw input (same source pointer).
  ml_drift::WeightsManager weights_manager;
  for (const GpuWeightConversion& c : batch) {
    weights_manager.RegisterWeightsConversion({c.handle.id}, c.desc, c.ohwi,
                                              c.src_type, c.raw.data());
    if (c.sum_i.has_value()) {
      weights_manager.RegisterWeightsSumIConversion({c.sum_i->id}, c.ohwi,
                                                    c.src_type, c.raw.data());
    }
  }
  GpuWeightConversionModel conversion;
  LRT_TENSOR_RETURN_IF_ERROR(weights_manager.CreateConversionGpuModel(
      gpu_info, &conversion.model, &conversion.io_mapping,
      &conversion.uploads));
  return conversion;
}

absl::StatusOr<ml_drift::TensorDescriptor> RawWeightsDescriptor(
    const ml_drift::GpuModel& conversion_model,
    const ml_drift::WeightsManager::UploadWeightsInfo& upload) {
  auto desc_it = conversion_model.tensors.find(upload.input_id);
  if (desc_it == conversion_model.tensors.end()) {
    return absl::InternalError("Missing raw weights tensor");
  }
  ml_drift::TensorDescriptor raw_desc =
      CopyDescriptorWithoutData(desc_it->second);
  raw_desc.UploadDataRaw(absl::MakeConstSpan(
      static_cast<const uint8_t*>(upload.data), upload.size));
  return raw_desc;
}

}  // namespace litert::tensor::ml_drift_runner_internal

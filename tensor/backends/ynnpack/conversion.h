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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_YNNPACK_CONVERSION_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_YNNPACK_CONVERSION_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/conversion.h"
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/backends/ynnpack/graph.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/type_id.h"
#include "tensor/tensor.h"

namespace litert::tensor {

class YnnpackBuildContext;

// Base class for YNNPACK operations.
class YnnpackOperation : public graph::BackendExtension {
 public:
  internal::TypeId GetTypeId() const override {
    return internal::TypeId::Get<YnnpackOperation>();
  }
  // Converts the operation to YNNPACK.
  virtual absl::Status ToYnnpack(const graph::Operation& op,
                                 YnnpackBuildContext& ctx) const = 0;
};

// Maps a Tensor API element type to the matching YNNPACK type.
//
// Returns `ynn_type_invalid` for types YNNPACK doesn't handle.
ynn_type ToYnnType(Type type);

class YnnpackBuildContext : public NnpackBuildContext {
 public:
  using NnpackBuildContext::NnpackBuildContext;

  absl::string_view BackendName() const override { return "YNNPACK"; }
  uint32_t FlagExternalInput() const override {
    return YNN_VALUE_FLAG_EXTERNAL_INPUT;
  }
  uint32_t FlagExternalOutput() const override {
    return YNN_VALUE_FLAG_EXTERNAL_OUTPUT;
  }

  ynn_subgraph_t subgraph() {
    return static_cast<YnnpackGraph&>(*graph_).GetSubgraph();
  }

  // Defines a new internal value that YNNPACK will infer the shape and type of.
  //
  // YNNPACK node definitions take an in/out `output_id`. Passing
  // `YNN_INVALID_VALUE_ID` lets YNNPACK create the value itself, which is what
  // we want for every value that isn't an external input or output.
  static constexpr uint32_t kInferredValueId = YNN_INVALID_VALUE_ID;

 protected:
  absl::Status EnsureInitialized() override;
  std::unique_ptr<NnpackGraph> CreateEmptyGraph() override;
  absl::Status CreateSubgraph(size_t external_value_ids,
                              uint32_t flags) override;
  absl::Status DefineTensorValue(const graph::Tensor& tensor,
                                 NnpackValue& value) override;
  absl::Status DefineConstantTensor(Type datatype,
                                    absl::Span<const size_t> shape,
                                    const void* data, uint32_t* id) override;
  absl::Status LowerOp(const graph::Operation& op) override;

 private:
  // Defines a constant fp32 scale tensor of shape `dims`.
  absl::StatusOr<uint32_t> DefineQuantizationScales(
      absl::Span<const float> scales, absl::Span<const size_t> dims);

  // Defines a constant int32 zero point tensor of shape `dims`, or returns
  // `YNN_INVALID_VALUE_ID` for symmetric quantization.
  absl::StatusOr<uint32_t> DefineQuantizationZeroPoints(
      absl::Span<const int64_t> zero_points, absl::Span<const size_t> dims);

  // Defines the dequantize node turning the packed `quantized_id` weights of a
  // per-channel quantized tensor into the fp32 value the graph consumes.
  absl::StatusOr<uint32_t> DefinePerChannelDequantize(
      const graph::TensorInformation& info, absl::Span<const size_t> dims,
      const graph::PerChannelAffineQuantization& quantization,
      uint32_t quantized_id);

  // Same as `DefinePerChannelDequantize`, for blockwise quantized tensors.
  absl::StatusOr<uint32_t> DefineBlockwiseDequantize(
      const graph::TensorInformation& info, absl::Span<const size_t> dims,
      const graph::BlockwiseQuantization& quantization, uint32_t quantized_id);

  // Grows a blockwise quantization parameter of shape
  // [channels, num_blocks, 1] into one of shape [channels, reduction] that is
  // elementwise compatible with the weights it applies to.
  absl::StatusOr<uint32_t> ExpandBlockwiseParam(uint32_t param_id,
                                                size_t block_size,
                                                absl::string_view name);
};

// Builds a YNNPACK graph from the given outputs.
absl::StatusOr<std::unique_ptr<YnnpackGraph>> BuildYnnpackGraph(
    std::vector<TensorHandle> outputs);

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_YNNPACK_CONVERSION_H_

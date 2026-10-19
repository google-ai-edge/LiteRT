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
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
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

  // Returns the id of the integer data of the quantized tensor `tensor`, or
  // `YNN_INVALID_VALUE_ID` if it has none.
  //
  // YNNPACK represents quantized tensors using 3 values: int data, scales and
  // zero points. `DefineValue()` creates an extra fp32 value for the
  // dequantized tensor, which will be pruned if unused.
  //
  // > Note: `DefineValue()` must have been called first.
  // >
  // > - Quantized constants always have integer data.
  // > - A quantized tensor produced by an op only has it if the op's lowering
  // >   records it with `DefineDequantizedValue()`.
  uint32_t FindQuantizedDataId(const graph::Tensor& tensor) const {
    auto it = quantized_data_ids_.find(tensor);
    return it == quantized_data_ids_.end() ? YNN_INVALID_VALUE_ID : it->second;
  }

  // Defines the fp32 value of a quantized tensor from its integer data, and
  // records the integer data for `FindQuantizedDataId()`.
  //
  // - `tensor`: the quantized graph tensor, whose quantization parameters the
  //   dequantize uses.
  // - `data_id`: the YNNPACK value holding the integer data of `tensor`.
  // - `fp32_id`: the YNNPACK value to write the dequantized data to, e.g. the
  //   already defined output of the op producing `tensor`. By default, YNNPACK
  //   creates a new value.
  //
  // Returns the id of the dequantized value, i.e. `fp32_id` if one is given.
  //
  // Note: See `FindQuantizedDataId()` for more info.
  absl::StatusOr<uint32_t> DefineDequantizedValue(
      const graph::Tensor& tensor, uint32_t data_id,
      uint32_t fp32_id = kInferredValueId);

  // Checks that the quantization of `info` is supported by the backend and is
  // consistent with its shape.
  static absl::Status ValidateQuantization(
      const graph::TensorInformation& info);

  // Returns the YNNPACK type of the integer data of a quantized tensor.
  static ynn_type GetQuantizedStorageType(const graph::TensorInformation& info);

  // Returns the block size of blockwise quantization, or 0.
  static size_t GetBlockSize(const Quantization& quantization);

  // Returns the quantized dimension, or -1 for per-tensor quantization.
  static int GetQuantizedDimension(const Quantization& quantization);

  // Returns the scales of `quantization`.
  //
  // For blockwise quantization they are [channels, num_blocks] row-major,
  // whatever the dimension of the channels.
  static absl::Span<const float> GetScales(const Quantization& quantization);

  // Returns one zero point per scale. Missing zero points mean 0, and a single
  // zero point applies to every scale.
  //
  // Warning: Assumes that `ValidateQuantization()` checked the zero point
  // count.
  static std::vector<int32_t> GetZeroPoints(const Quantization& quantization);

  // Returns whether every zero point of `quantization` is 0.
  static bool IsSymmetric(const Quantization& quantization);

  // Returns how many elements of `type` are packed into a byte, or 0 if
  // `type` isn't an integer type the quantized lowerings support.
  static size_t ElementsPerByte(ynn_type type);

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
      absl::Span<const int32_t> zero_points, absl::Span<const size_t> dims);

  // Helper for `DefineDequantizedValue()` for per channel quantization.
  //
  // Warning: `info` must have passed `ValidateQuantization()`.
  absl::StatusOr<uint32_t> DefinePerChannelDequantize(
      const graph::TensorInformation& info, uint32_t data_id, uint32_t fp32_id);

  // Helper for `DefineDequantizedValue()` for blockwise quantization.
  //
  // Warning: `info` must have passed `ValidateQuantization()`.
  absl::StatusOr<uint32_t> DefineBlockwiseDequantize(
      const graph::TensorInformation& info, uint32_t data_id, uint32_t fp32_id);

  // The integer data of quantized tensors, see `FindQuantizedDataId`.
  absl::flat_hash_map<graph::Tensor, uint32_t> quantized_data_ids_;
};

// Builds a YNNPACK graph from the given outputs.
absl::StatusOr<std::unique_ptr<YnnpackGraph>> BuildYnnpackGraph(
    std::vector<TensorHandle> outputs);

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_YNNPACK_CONVERSION_H_

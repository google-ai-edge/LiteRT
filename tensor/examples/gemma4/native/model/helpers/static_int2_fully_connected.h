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

// Local experiment: preserve original static INT2 MLP coefficients and scales.
#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_MODEL_HELPERS_STATIC_INT2_FULLY_CONNECTED_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_MODEL_HELPERS_STATIC_INT2_FULLY_CONNECTED_H_

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "xnnpack.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/backends/xnnpack/conversion.h"
#include "tensor/backends/xnnpack/utils.h"
#include "tensor/tensor.h"
#include "xnnpack/operator.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/subgraph.h"

namespace litert::tensor::examples::gemma4::native {

// This setting is consumed at load time only. Existing loaded tensors retain
// their representation even if a later load selects the other setting.
inline bool preserve_static_int2_weights = true;
inline void SetPreserveStaticInt2Weights(bool enabled) {
  preserve_static_int2_weights = enabled;
}

struct StaticInt2Provenance {
  std::string name;
  std::string source_dtype;
  std::string source_data_sha256;
  int source_section = -1;
  int source_subgraph = -1;
  int source_tensor = -1;
  Shape shape;
  size_t source_bytes = 0;
};

// An immutable, genuinely two-bit Buffer. The tensor's Type::kI2, matrix shape,
// ByteSize(), and Lock() span all agree. Its provenance record is an explicit
// marker: numeric values fitting two bits alone never select the QC2 path.
// Extra bytes are owned once per weight and never copied into stage graphs.
class StaticInt2WeightBuffer final : public Buffer {
 public:
  static absl::StatusOr<std::shared_ptr<StaticInt2WeightBuffer>> FromWidenedI4(
      const uint8_t* source, size_t source_bytes,
      StaticInt2Provenance provenance) {
    const auto& shape = provenance.shape;
    if (provenance.source_dtype != "INT2" || shape.size() != 2 ||
        shape[0] <= 1 || shape[1] <= 0 || shape[1] % 4 != 0 ||
        static_cast<size_t>(shape[0]) >
            std::numeric_limits<size_t>::max() / shape[1])
      return absl::InvalidArgumentError("Invalid original INT2 provenance/shape");
    const size_t elements = static_cast<size_t>(shape[0]) * shape[1];
    if (source == nullptr || source_bytes != elements / 2 ||
        provenance.source_bytes != elements / 4)
      return absl::InvalidArgumentError("INT2 source/export byte count mismatch");
    auto result = std::shared_ptr<StaticInt2WeightBuffer>(
        new StaticInt2WeightBuffer(std::move(provenance), elements / 4));
    for (size_t i = 0; i < elements; ++i) {
      const int nibble = (source[i / 2] >> (4 * (i % 2))) & 15;
      const int code = nibble >= 8 ? nibble - 16 : nibble;
      if (code < -2 || code > 1)
        return absl::InvalidArgumentError(
            "Original INT2 coefficient outside [-2,1]: " + result->provenance.name);
      result->storage_[i / 4] |=
          static_cast<uint8_t>((code & 3) << (2 * (i % 4)));
    }
    return result;
  }
  internal::TypeId GetTypeId() const override {
    return internal::TypeId::Get<StaticInt2WeightBuffer>();
  }
  bool IsA(internal::TypeId id) const override { return id == GetTypeId(); }
  absl::StatusOr<size_t> ByteSize() const override { return payload_bytes_; }
  LockedBufferSpan<const std::byte> Lock() override {
    return {reinterpret_cast<const std::byte*>(storage_.data()),
            [](const std::byte*) {}, payload_bytes_};
  }
  const uint8_t* data() const { return storage_.data(); }
  size_t payload_bytes() const { return payload_bytes_; }
  size_t allocation_bytes() const { return storage_.capacity(); }
  const StaticInt2Provenance provenance;
  mutable std::atomic<size_t> lowering_count{0};
 private:
  StaticInt2WeightBuffer(StaticInt2Provenance source, size_t bytes)
      : provenance(std::move(source)), payload_bytes_(bytes),
        storage_(bytes + XNN_EXTRA_BYTES, 0) {}
  const size_t payload_bytes_;
  std::vector<uint8_t> storage_;
};

inline absl::StatusOr<const StaticInt2WeightBuffer&> GetStaticInt2Weight(
    const TensorHandle& weight) {
  if (weight.GetType() != Type::kI2 || !weight.GetBufferPtr())
    return absl::InvalidArgumentError("Expected an owned original INT2 weight");
  return weight.GetBufferPtr()->As<const StaticInt2WeightBuffer>();
}

struct StaticInt2WeightAudit {
  size_t tensor_count = 0;
  size_t widened_bytes = 0;
  size_t compact_bytes = 0;
  size_t allocation_bytes = 0;
  size_t lowering_count = 0;
};
template <class WeightMap>
inline StaticInt2WeightAudit GetStaticInt2WeightAudit(const WeightMap& weights) {
  StaticInt2WeightAudit result;
  for (const auto& entry : weights) {
    auto buffer = GetStaticInt2Weight(entry.second);
    if (!buffer.ok()) continue;
    ++result.tensor_count;
    result.compact_bytes += buffer->payload_bytes();
    result.widened_bytes += 2 * buffer->payload_bytes();
    result.allocation_bytes += buffer->allocation_bytes();
    result.lowering_count += buffer->lowering_count.load();
  }
  return result;
}

// Counts executed static QS8/QC2W operators, excluding the separate dynamic
// QC2 head. Inspect after PrepareRuntime() (or Run()), not just authored nodes.
inline absl::StatusOr<size_t> CountStaticInt2RuntimeOperators(xnn_runtime_t rt) {
  if (!rt) return absl::InvalidArgumentError("Missing runtime for QC2 audit");
  size_t count = 0;
  for (size_t i = 0; i < rt->num_ops; ++i) {
    const auto& op = rt->opdata[i];
    for (const auto* object : op.operator_objects) {
      if (!object || object->type != xnn_operator_type_fully_connected_nc_qs8_qc2w)
        continue;
      if (op.num_inputs < 2 || op.num_outputs != 1 ||
          op.inputs[0] >= rt->num_values || op.inputs[1] >= rt->num_values ||
          op.outputs[0] >= rt->num_values ||
          rt->values[op.inputs[0]].datatype != xnn_datatype_qint8 ||
          rt->values[op.inputs[1]].datatype != xnn_datatype_qcint2 ||
          rt->values[op.outputs[0]].datatype != xnn_datatype_qint8)
        return absl::FailedPreconditionError("Static QC2 runtime edge mismatch");
      ++count;
    }
  }
  return count;
}

inline absl::Status ValidateStaticInt2FullyConnected(
    const TensorHandle& input, const TensorHandle& weight) {
  LRT_TENSOR_RETURN_IF_ERROR(input.GetStatus());
  LRT_TENSOR_RETURN_IF_ERROR(weight.GetStatus());
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& buffer, GetStaticInt2Weight(weight));
  const auto& shape = weight.GetShape();
  if (shape != buffer.provenance.shape || input.GetShape().empty() ||
      input.GetShape().back() != shape[1] || input.GetType() != Type::kI8 ||
      !input.GetQuantization() || !weight.GetQuantization())
    return absl::InvalidArgumentError("Static INT2 FC input/weight mismatch");
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const auto& iq, input.GetQuantization()->As<PerChannelAffineQuantization>());
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const auto& wq, weight.GetQuantization()->As<PerChannelAffineQuantization>());
  if (iq.scales.size() != 1 || iq.zero_points != std::vector<int64_t>{0} ||
      !std::isfinite(iq.scales[0]) || iq.scales[0] <= 0 ||
      wq.quantized_dimension != 0 || wq.scales.size() != shape[0] ||
      wq.zero_points != std::vector<int64_t>{0})
    return absl::InvalidArgumentError("Static INT2 FC quantization mismatch");
  for (float scale : wq.scales)
    if (!std::isfinite(scale) || scale <= 0)
      return absl::InvalidArgumentError("Static INT2 FC invalid weight scale");
  return absl::OkStatus();
}
}  // namespace litert::tensor::examples::gemma4::native

namespace litert::tensor::graph {
struct StaticInt2FullyConnectedOperation : Operation {
  absl::string_view GetName() const override { return "StaticInt2FullyConnected"; }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};
template <>
class OpMixin<StaticInt2FullyConnectedOperation, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(const Operation& op,
                        XnnpackBuildContext& ctx) const override {
    using namespace examples::gemma4::native;
    if (op.inputs.size() != 2)
      return absl::InvalidArgumentError("Static INT2 FC expects two inputs");
    const TensorHandle input(op.inputs[0]), weight(op.inputs[1]);
    LRT_TENSOR_RETURN_IF_ERROR(ValidateStaticInt2FullyConnected(input, weight));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, GetOutputs(op));
    if (outputs.size() != 1)
      return absl::InvalidArgumentError("Static INT2 FC expects one output");
    const TensorHandle output(outputs[0]);
    auto shape = input.GetShape();
    shape.back() = weight.GetShape()[0];
    if (output.GetType() != Type::kI8 || output.GetShape() != shape ||
        !output.GetQuantization())
      return absl::InvalidArgumentError("Static INT2 FC output mismatch");
    LRT_TENSOR_ASSIGN_OR_RETURN(
        const auto& oq, output.GetQuantization()->As<PerChannelAffineQuantization>());
    if (oq.scales.size() != 1 || oq.zero_points != std::vector<int64_t>{0} ||
        !std::isfinite(oq.scales[0]) || oq.scales[0] <= 0)
      return absl::InvalidArgumentError("Static INT2 FC output scale mismatch");
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto& buffer, GetStaticInt2Weight(weight));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        const auto& wq, weight.GetQuantization()->As<PerChannelAffineQuantization>());
    // Keep original compact storage alive without copying it per signature or
    // stage. Scale copies are tiny and remain graph-owned like ordinary values.
    ctx.graph().keep_alive_buffers().push_back(weight.GetBufferPtr());
    auto& scales = ctx.graph().dequantized_buffers().emplace_back(wq.scales);
    const size_t dimensions[] = {static_cast<size_t>(weight.GetShape()[0]),
                                 static_cast<size_t>(weight.GetShape()[1])};
    uint32_t weight_id;
    LRT_TENSOR_RETURN_IF_ERROR(xnn_define_channelwise_quantized_tensor_value_v3(
        ctx.subgraph(), xnn_datatype_qcint2, 0, scales.data(), 2, 0,
        dimensions, buffer.data(), XNN_INVALID_VALUE_ID, 0, &weight_id, nullptr));
    LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
    LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(outputs[0]));
    LRT_TENSOR_RETURN_IF_ERROR(xnn_define_fully_connected(
        ctx.subgraph(), -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), input_id, weight_id,
        XNN_INVALID_VALUE_ID, output_id, 0));
    ++buffer.lowering_count;
    return absl::OkStatus();
  }
};
}  // namespace litert::tensor::graph

namespace litert::tensor::examples::gemma4::native {
template <class... Mixins>
Tensor<Mixins...> StaticInt2FullyConnected(
    Tensor<Mixins...> input, Tensor<Mixins...> weight,
    source_location loc = source_location::current()) {
  LRT_TENSOR_RETURN_IF_ERROR(ValidateStaticInt2FullyConnected(input, weight));
  auto op = std::make_shared<graph::StaticInt2FullyConnectedOperation>();
  RegisterMixins<Mixins...>(op);
  AddInputs(op, input, weight);
  Tensor<Mixins...> output = AddOutput(op, loc);
  LRT_TENSOR_ASSIGN_OR_RETURN(auto& info, graph::GetInfo(output.GetRaw()));
  info.shape = input.GetShape();
  info.shape.back() = weight.GetShape()[0];
  info.type = Type::kI8;
  graph::OpDebugger::DebugOp(*op);
  return output;
}
}  // namespace litert::tensor::examples::gemma4::native
#endif  // LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_MODEL_HELPERS_STATIC_INT2_FULLY_CONNECTED_H_

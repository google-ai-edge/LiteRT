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

#include "ml_drift_delegate/delegate/composite/ir/block_ffw_parser.h"

#include <optional>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/block_ffw_parser.h"
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {
namespace {

using ::litert::ml_drift::BlockFfwQuantizedFcOpSpec;
using ::litert::ml_drift::BlockFfwWeightOperandSpec;
using ::litert::ml_drift::BuildBlockFfwQuantizedFcOpSpec;
using ::litert::ml_drift::ResolvedBlockFfwPlan;
using ::ml_drift::BHWC;
using ::ml_drift::DataType;
using ::ml_drift::OperationType;
using ::ml_drift::QuantizeAndDequantizeAttributes;
using ::ml_drift::ir::IrModel;
using ::ml_drift::ir::IrOp;
using ::ml_drift::ir::IrTensor;
using ::ml_drift::ir::IrTensorId;

bool SameScale(std::optional<float> a, std::optional<float> b) {
  return a.has_value() && b.has_value() && *a == *b;
}

// Inserts a zero-point-0 signed `int8_t` (`[-128, 127]`) QuantizeAndDequantize
// op when `scale` is present, matching `PopulateQuantParams` in
// `model_builder_helper.cc` (`min = -128 * scale`, `max = 127 * scale`).
IrTensorId MaybeAddInt8QuantizeAndDequantize(IrModel& ir_model,
                                             IrTensorId src_id,
                                             std::optional<float> scale) {
  if (!scale.has_value()) {
    return src_id;
  }
  const IrTensor* src_tensor = ir_model.tensor(src_id);
  const IrTensor* dst_tensor = ir_model.add_tensor(
      src_tensor->desc.GetDataType(), src_tensor->desc.GetBHWCShape());
  IrOp* qdq = ir_model.add_op();
  qdq->name = ToString(OperationType::kQuantizeAndDequantize);
  QuantizeAndDequantizeAttributes attr;
  attr.min = -128.0f * (*scale);
  attr.max = 127.0f * (*scale);
  attr.scale = *scale;
  qdq->attr = attr;
  ir_model.AddConsumer(src_id, qdq->id);
  ir_model.SetProducer(dst_tensor->id, qdq->id);
  return dst_tensor->id;
}

void AddSharedQuantizedFcOp(IrModel& ir_model, IrTensorId src_id,
                            IrTensorId dst_id, IrTensorId raw_w_id,
                            const BlockFfwWeightOperandSpec& spec) {
  IrTensorId w_id = raw_w_id;
  IrTensor* w_tensor = ir_model.GetMutableTensor(w_id);
  const BHWC ohwi_as_bhwc(spec.shape.o, spec.shape.h, spec.shape.w,
                          spec.shape.i);
  if (!w_tensor->consumers.empty()) {
    // SharedMemoryManager requires a single consumer op per shared weight
    // tensor handle so each op can attach its own scale/zero-point inputs.
    // Reset per-consumer flags so settings from an earlier non-quantized
    // consumer cannot leak onto this quantized weight handle.
    ::ml_drift::ir::BufferSource buffer_source = w_tensor->buffer_source;
    buffer_source.dequant_forced = false;
    buffer_source.force_linear_layout = false;
    const DataType w_dtype = w_tensor->desc.GetDataType();
    w_tensor = ir_model.add_tensor(w_dtype, ohwi_as_bhwc);
    w_tensor->buffer_source = buffer_source;
    w_id = w_tensor->id;
  } else {
    w_tensor->desc.SetBHWCShape(ohwi_as_bhwc);
  }

  IrOp* op = ir_model.add_op();
  ir_model.AddConsumer(src_id, op->id);
  ir_model.AddConsumer(w_id, op->id);
  BlockFfwQuantizedFcOpSpec fc_spec = BuildBlockFfwQuantizedFcOpSpec(spec);
  op->name = std::move(fc_spec.op_type);
  op->attr = std::move(fc_spec.attributes);
  ir_model.SetProducer(dst_id, op->id);
}

void AddHostFloatFcOp(IrModel& ir_model, IrTensorId src_id, IrTensorId dst_id,
                      const BlockFfwWeightOperandSpec& spec,
                      std::vector<float> host_fallback) {
  IrOp* op = ir_model.add_op();
  op->name = ToString(OperationType::kFullyConnected);
  ::ml_drift::FullyConnectedAttributes attr;
  attr.weights.shape = spec.shape;
  attr.weights.data = std::move(host_fallback);
  // `ml_drift::RearrangeWeights` expects a flat linear float buffer with
  // `XNN_EXTRA_BYTES` of trailing padding (no per-row strided padding) for
  // XNNPACK transpose kernels.
  attr.weights.data.resize(
      attr.weights.shape.DimensionsProduct() + XNN_EXTRA_BYTES / sizeof(float),
      0.0f);
  op->attr = std::move(attr);
  ir_model.AddConsumer(src_id, op->id);
  ir_model.SetProducer(dst_id, op->id);
}

void AddBatchedMatmulFallbackOp(IrModel& ir_model, DataType dtype,
                                IrTensorId src_id, IrTensorId dst_id,
                                const BlockFfwWeightOperandSpec& spec,
                                std::vector<float> host_fallback) {
  // Non-shared batched `[1, H, I, O]` fallback via Constant + BatchedMatmul so
  // `different_weights_for_height` holds for arbitrary sequence lengths T.
  const BHWC rhs_shape(1, spec.shape.h, spec.shape.i, spec.shape.o);
  const IrTensorId rhs_id = ir_model.add_tensor(dtype, rhs_shape)->id;
  IrOp* const_op = ir_model.add_op();
  const_op->name = ToString(OperationType::kConstant);
  ::ml_drift::ConstTensorAttributes const_attr;
  auto& rhs_tensor =
      const_attr.tensor.emplace<::ml_drift::Tensor<BHWC, DataType::kFloat32>>();
  rhs_tensor.shape = rhs_shape;
  rhs_tensor.data = std::move(host_fallback);
  const_op->attr = std::move(const_attr);
  ir_model.SetProducer(rhs_id, const_op->id);

  IrOp* bmm_op = ir_model.add_op();
  bmm_op->name = ToString(OperationType::kBatchedMatmul);
  bmm_op->attr = ::ml_drift::BatchedMatMulAttributes{};
  ir_model.AddConsumer(src_id, bmm_op->id);
  ir_model.AddConsumer(rhs_id, bmm_op->id);
  ir_model.SetProducer(dst_id, bmm_op->id);
}

IrTensorId AddLinearProjection(
    const TfLiteContext& context, IrModel& ir_model,
    const absl::flat_hash_map<int, IrTensorId>& tensor_map, IrTensorId src_id,
    const BlockFfwWeightOperandSpec& spec, const BHWC& dst_shape) {
  const DataType dtype = ir_model.tensor(src_id)->desc.GetDataType();
  const IrTensorId dst_id = ir_model.add_tensor(dtype, dst_shape)->id;
  const IrTensorId raw_w_id = tensor_map.at(spec.tensor_id);
  const bool is_shared =
      ::litert::ml_drift::IsBlockFfwQuantizedWeightType(spec.type) &&
      ir_model.tensor(raw_w_id)->buffer_source.is_shared;

  if (is_shared) {
    AddSharedQuantizedFcOp(ir_model, src_id, dst_id, raw_w_id, spec);
    return dst_id;
  }

  std::vector<float> host_fallback =
      ::litert::ml_drift::ExtractBlockFfwHostFallback(
          context.tensors, context.tensors_size, spec);
  if (!spec.is_batched) {
    AddHostFloatFcOp(ir_model, src_id, dst_id, spec, std::move(host_fallback));
  } else {
    AddBatchedMatmulFallbackOp(ir_model, dtype, src_id, dst_id, spec,
                               std::move(host_fallback));
  }
  return dst_id;
}

// Emits an edge stage (Stage 1 U1 or Stage 4 D2): `K1` parallel branches of
// UnpackPart -> optional QDQ -> LinearProjection -> optional QDQ -> PackParts.
void BuildEdgeStage(const TfLiteContext& context, IrModel& ir_model,
                    const absl::flat_hash_map<int, IrTensorId>& tensor_map,
                    IrTensorId src_id, IrTensorId dst_id,
                    absl::Span<const BlockFfwWeightOperandSpec> weights,
                    int tokens, int part_in_channels, int part_out_channels,
                    std::optional<float> in_scale,
                    std::optional<float> out_scale) {
  const DataType dtype = ir_model.tensor(src_id)->desc.GetDataType();
  const int k1 = static_cast<int>(weights.size());
  std::vector<IrTensorId> parts(k1);
  for (int k = 0; k < k1; ++k) {
    const IrTensorId part_in_id =
        ir_model.add_tensor(dtype, BHWC(1, 1, tokens, part_in_channels))->id;
    IrOp* unpack = ir_model.add_op();
    unpack->name = ::litert::ml_drift::kBlockFfwUnpackPartType;
    unpack->attr = ::litert::ml_drift::BlockFfwUnpackPartAttributes{
        .part = k, .num_parts = k1};
    ir_model.AddConsumer(src_id, unpack->id);
    ir_model.SetProducer(part_in_id, unpack->id);

    const IrTensorId q_in_id =
        MaybeAddInt8QuantizeAndDequantize(ir_model, part_in_id, in_scale);
    const IrTensorId fc_out_id = AddLinearProjection(
        context, ir_model, tensor_map, q_in_id, weights[k],
        BHWC(1, 1, tokens, part_out_channels));
    parts[k] =
        MaybeAddInt8QuantizeAndDequantize(ir_model, fc_out_id, out_scale);
  }
  IrOp* pack_op = ir_model.add_op();
  pack_op->name = ::litert::ml_drift::kBlockFfwPackPartsType;
  pack_op->attr =
      ::litert::ml_drift::BlockFfwPackPartsAttributes{.num_parts = k1};
  for (IrTensorId part_id : parts) {
    ir_model.AddConsumer(part_id, pack_op->id);
  }
  ir_model.SetProducer(dst_id, pack_op->id);
}

// Emits Stage 2 (U2) + SiLU activation, deduplicating `u2.in_scale` when it
// matches `u1.out_scale`.
IrTensorId BuildU2AndSiluStage(
    const TfLiteContext& context, IrModel& ir_model,
    const absl::flat_hash_map<int, IrTensorId>& tensor_map, IrTensorId u2_in_id,
    const ResolvedBlockFfwPlan& plan) {
  if (plan.attrs.u2.in_scale.has_value() &&
      !SameScale(plan.attrs.u2.in_scale, plan.attrs.u1.out_scale)) {
    u2_in_id = MaybeAddInt8QuantizeAndDequantize(ir_model, u2_in_id,
                                                 plan.attrs.u2.in_scale);
  }
  const DataType dtype = ir_model.tensor(u2_in_id)->desc.GetDataType();
  const BHWC u2_out_shape(1, plan.k2, plan.tokens, plan.o2);
  const IrTensorId u2_raw_id = AddLinearProjection(
      context, ir_model, tensor_map, u2_in_id, plan.u2_weight, u2_out_shape);
  const IrTensorId u2_out_id = MaybeAddInt8QuantizeAndDequantize(
      ir_model, u2_raw_id, plan.attrs.u2.out_scale);
  const IrTensorId act_out_id = ir_model.add_tensor(dtype, u2_out_shape)->id;
  IrOp* silu_op = ir_model.add_op();
  silu_op->name = ::litert::ml_drift::kBlockFfwSiluType;
  ir_model.AddConsumer(u2_out_id, silu_op->id);
  ir_model.SetProducer(act_out_id, silu_op->id);
  return act_out_id;
}

// Emits Stage 3 (D1): optional QDQ -> LinearProjection -> optional QDQ.
IrTensorId BuildD1Stage(const TfLiteContext& context, IrModel& ir_model,
                        const absl::flat_hash_map<int, IrTensorId>& tensor_map,
                        IrTensorId act_out_id,
                        const ResolvedBlockFfwPlan& plan) {
  const IrTensorId d1_in_id = MaybeAddInt8QuantizeAndDequantize(
      ir_model, act_out_id, plan.attrs.d1.in_scale);
  const IrTensorId d1_raw_id = AddLinearProjection(
      context, ir_model, tensor_map, d1_in_id, plan.d1_weight,
      BHWC(1, plan.k2, plan.tokens, plan.k1 * plan.s));
  return MaybeAddInt8QuantizeAndDequantize(ir_model, d1_raw_id,
                                           plan.attrs.d1.out_scale);
}

absl::Status BlockFfwIsSupported(const TfLiteContext* context,
                                 const TfLiteNode* tflite_node,
                                 const TfLiteRegistration* registration) {
  return ::litert::ml_drift::BlockFfwOperationParser().IsSupported(
      context, tflite_node, registration);
}

void BlockFfwConvert(const TfLiteContext& context,
                     const TfLiteNode& tflite_node,
                     const TfLiteRegistration& /*registration*/,
                     absl::flat_hash_map<int, IrTensorId>& tensor_map,
                     const IrModelBuilderOptions& /*options*/,
                     IrModel& ir_model) {
  const absl::StatusOr<ResolvedBlockFfwPlan> plan =
      ::litert::ml_drift::ResolveBlockFfwPlan(
          context.tensors, context.tensors_size, tflite_node);
  ABSL_CHECK_OK(plan.status());

  const IrTensorId x_id = tensor_map.at(tflite_node.inputs->data[0]);
  const IrTensorId out_id = tensor_map.at(tflite_node.outputs->data[0]);
  const DataType dtype = ir_model.tensor(x_id)->desc.GetDataType();
  const int tokens = plan->tokens;
  const int i2 = plan->k1 * plan->s;
  const int i4 = plan->k2 * plan->s;

  // Stage 1 (U1) + permutation into U2's input `[1, K2, T, K1 * S]`.
  const IrTensorId u2_in_id =
      ir_model.add_tensor(dtype, BHWC(1, plan->k2, tokens, i2))->id;
  BuildEdgeStage(context, ir_model, tensor_map, x_id, u2_in_id,
                 plan->u1_weights, tokens, plan->i1, plan->o1,
                 plan->attrs.u1.in_scale, plan->attrs.u1.out_scale);

  // Stage 2 (U2) + SiLU activation.
  const IrTensorId act_out_id =
      BuildU2AndSiluStage(context, ir_model, tensor_map, u2_in_id, *plan);

  // Stage 3 (D1).
  const IrTensorId d1_out_id =
      BuildD1Stage(context, ir_model, tensor_map, act_out_id, *plan);

  // Stage 4 (D2) + merge into `[1, 1, T, D]`, deduplicating `d2.in_scale`
  // when it matches `d1.out_scale`.
  const std::optional<float> d2_in_scale =
      (plan->attrs.d2.in_scale.has_value() &&
       !SameScale(plan->attrs.d2.in_scale, plan->attrs.d1.out_scale))
          ? plan->attrs.d2.in_scale
          : std::nullopt;
  BuildEdgeStage(context, ir_model, tensor_map, d1_out_id, out_id,
                 plan->d2_weights, tokens, i4, plan->o4, d2_in_scale,
                 plan->attrs.d2.out_scale);
}

}  // namespace

CustomIrOpParser GetBlockFfwParser() {
  return {
      .is_supported = BlockFfwIsSupported,
      .convert = BlockFfwConvert,
  };
}

}  // namespace litert::ml_drift::ir

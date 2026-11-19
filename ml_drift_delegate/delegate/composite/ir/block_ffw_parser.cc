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

using ::ml_drift::BHWC;
using ::ml_drift::DataType;
using ::ml_drift::OHWI;
using ::ml_drift::OperationType;
using ::ml_drift::QuantizeAndDequantizeAttributes;
using ::ml_drift::ir::IrModel;
using ::ml_drift::ir::IrOp;
using ::ml_drift::ir::IrTensor;
using ::ml_drift::ir::IrTensorId;

bool SameScale(std::optional<float> a, std::optional<float> b) {
  return a.has_value() && b.has_value() && *a == *b;
}

IrTensorId MaybeAddSymmetricInt8Qdq(IrModel& ir_model, IrTensorId src_id,
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

IrTensorId AddLinearStage(
    const TfLiteContext& context, IrModel& ir_model,
    const absl::flat_hash_map<int, IrTensorId>& tensor_map, IrTensorId src_id,
    const ::litert::ml_drift::BlockFfwWeightOperandSpec& spec,
    const BHWC& dst_shape) {
  const DataType dtype = ir_model.tensor(src_id)->desc.GetDataType();
  const IrTensorId dst_id = ir_model.add_tensor(dtype, dst_shape)->id;
  const IrTensorId raw_w_id = tensor_map.at(spec.tensor_id);
  const bool is_shared =
      ::litert::ml_drift::IsBlockFfwQuantizedWeightType(spec.type) &&
      ir_model.tensor(raw_w_id)->buffer_source.is_shared;

  if (is_shared) {
    IrTensorId w_id = raw_w_id;
    IrTensor* w_tensor = ir_model.GetMutableTensor(w_id);
    const BHWC ohwi_as_bhwc(spec.shape.o, spec.shape.h, spec.shape.w,
                            spec.shape.i);
    if (!w_tensor->consumers.empty()) {
      // SharedMemoryManager requires a single consumer op per shared weight
      // tensor handle so each op can attach its own scale/zero-point inputs.
      const ::ml_drift::ir::BufferSource buffer_source =
          w_tensor->buffer_source;
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
    auto set_quant_shapes = [&](auto& attr) {
      attr.scale.shape = spec.scale_shape;
      if (spec.has_zero_point) {
        attr.zero_point.shape = spec.scale_shape;
      }
    };
    if (spec.type == kTfLiteInt8) {
      ::ml_drift::FullyConnectedInt8Attributes attr;
      attr.weights.shape = spec.shape;
      set_quant_shapes(attr);
      op->name = ToString(OperationType::kFullyConnectedInt8);
      op->attr = std::move(attr);
    } else if (spec.type == kTfLiteInt4) {
      ::ml_drift::FullyConnectedInt4Attributes attr;
      ::ml_drift::Tensor<OHWI, DataType::kInt4> w;
      w.shape = spec.shape;
      attr.weights = w;
      set_quant_shapes(attr);
      op->name = ToString(OperationType::kFullyConnectedInt4);
      op->attr = std::move(attr);
    } else {
      ::ml_drift::FullyConnectedInt2Attributes attr;
      ::ml_drift::Tensor<OHWI, DataType::kInt2> w;
      w.shape = spec.shape;
      attr.weights = w;
      set_quant_shapes(attr);
      op->name = ToString(OperationType::kFullyConnectedInt2);
      op->attr = std::move(attr);
    }
    ir_model.SetProducer(dst_id, op->id);
    return dst_id;
  }

  std::vector<float> host_fallback =
      ::litert::ml_drift::ExtractBlockFfwHostFallback(
          context.tensors, context.tensors_size, spec);

  if (spec.shape.h == 1) {
    IrOp* op = ir_model.add_op();
    op->name = ToString(OperationType::kFullyConnected);
    ::ml_drift::FullyConnectedAttributes attr;
    attr.weights.shape = spec.shape;
    attr.weights.data = std::move(host_fallback);
    // `ml_drift::RearrangeWeights` expects `XNN_EXTRA_BYTES` of trailing
    // padding on float weight buffers for XNNPACK transpose kernels.
    attr.weights.data.resize(
        attr.weights.shape.DimensionsProduct() +
            XNN_EXTRA_BYTES / sizeof(float),
        0.0f);
    op->attr = std::move(attr);
    ir_model.AddConsumer(src_id, op->id);
    ir_model.SetProducer(dst_id, op->id);
    return dst_id;
  }

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
  return dst_id;
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
  ::litert::ml_drift::ResolvedBlockFfwPlan plan;
  ABSL_CHECK_OK(::litert::ml_drift::ResolveBlockFfwPlan(
      context.tensors, context.tensors_size, tflite_node, &plan));

  const IrTensorId x_id = tensor_map[tflite_node.inputs->data[0]];
  const IrTensorId out_id = tensor_map[tflite_node.outputs->data[0]];
  const DataType dtype = ir_model.tensor(x_id)->desc.GetDataType();
  const int T = plan.tokens;
  const int i2 = plan.k1 * plan.s;
  const int i4 = plan.k2 * plan.s;

  // Stage 1 (U1) + permutation into U2's input `[1, K2, T, K1 * S]`.
  IrTensorId u2_in_id = 0;
  {
    std::vector<IrTensorId> parts(plan.k1);
    for (int k = 0; k < plan.k1; ++k) {
      const IrTensorId part_in_id =
          ir_model.add_tensor(dtype, BHWC(1, 1, T, plan.i1))->id;
      IrOp* unpack = ir_model.add_op();
      unpack->name = ::litert::ml_drift::kBlockFfwUnpackPartType;
      unpack->attr = ::litert::ml_drift::BlockFfwUnpackPartAttributes{
          .part = k, .num_parts = plan.k1};
      ir_model.AddConsumer(x_id, unpack->id);
      ir_model.SetProducer(part_in_id, unpack->id);

      const IrTensorId q_in_id = MaybeAddSymmetricInt8Qdq(
          ir_model, part_in_id, plan.attrs.u1.in_scale);
      const IrTensorId fc_out_id =
          AddLinearStage(context, ir_model, tensor_map, q_in_id,
                         plan.u1_weights[k], BHWC(1, 1, T, plan.o1));
      parts[k] = MaybeAddSymmetricInt8Qdq(ir_model, fc_out_id,
                                          plan.attrs.u1.out_scale);
    }
    const IrTensorId packed_id =
        ir_model.add_tensor(dtype, BHWC(1, plan.k2, T, i2))->id;
    IrOp* pack_op = ir_model.add_op();
    pack_op->name = ::litert::ml_drift::kBlockFfwPackPartsType;
    pack_op->attr =
        ::litert::ml_drift::BlockFfwPackPartsAttributes{.num_parts = plan.k1};
    for (IrTensorId part_id : parts) {
      ir_model.AddConsumer(part_id, pack_op->id);
    }
    ir_model.SetProducer(packed_id, pack_op->id);
    u2_in_id = packed_id;
  }

  // Deduplicate adjacent identical scales (`u2_in_scale == u1_out_scale`).
  if (plan.attrs.u2.in_scale.has_value() &&
      !SameScale(plan.attrs.u2.in_scale, plan.attrs.u1.out_scale)) {
    u2_in_id =
        MaybeAddSymmetricInt8Qdq(ir_model, u2_in_id, plan.attrs.u2.in_scale);
  }

  // Stage 2 (U2) + SiLU activation.
  const IrTensorId u2_raw_id =
      AddLinearStage(context, ir_model, tensor_map, u2_in_id, plan.u2_weight,
                     BHWC(1, plan.k2, T, plan.o2));
  const IrTensorId u2_out_id =
      MaybeAddSymmetricInt8Qdq(ir_model, u2_raw_id, plan.attrs.u2.out_scale);

  const IrTensorId act_out_id =
      ir_model.add_tensor(dtype, BHWC(1, plan.k2, T, plan.o2))->id;
  IrOp* silu_op = ir_model.add_op();
  silu_op->name = ::litert::ml_drift::kBlockFfwSiluType;
  ir_model.AddConsumer(u2_out_id, silu_op->id);
  ir_model.SetProducer(act_out_id, silu_op->id);

  // Stage 3 (D1).
  const IrTensorId d1_in_id =
      MaybeAddSymmetricInt8Qdq(ir_model, act_out_id, plan.attrs.d1.in_scale);
  const IrTensorId d1_raw_id =
      AddLinearStage(context, ir_model, tensor_map, d1_in_id, plan.d1_weight,
                     BHWC(1, plan.k2, T, i2));
  const IrTensorId d1_out_id =
      MaybeAddSymmetricInt8Qdq(ir_model, d1_raw_id, plan.attrs.d1.out_scale);

  // Stage 4 (D2) + merge into `[1, 1, T, D]`.
  const std::optional<float> d2_in_scale =
      (plan.attrs.d2.in_scale.has_value() &&
       !SameScale(plan.attrs.d2.in_scale, plan.attrs.d1.out_scale))
          ? plan.attrs.d2.in_scale
          : std::nullopt;

  std::vector<IrTensorId> parts(plan.k1);
  for (int k = 0; k < plan.k1; ++k) {
    const IrTensorId part_in_id =
        ir_model.add_tensor(dtype, BHWC(1, 1, T, i4))->id;
    IrOp* unpack = ir_model.add_op();
    unpack->name = ::litert::ml_drift::kBlockFfwUnpackPartType;
    unpack->attr = ::litert::ml_drift::BlockFfwUnpackPartAttributes{
        .part = k, .num_parts = plan.k1};
    ir_model.AddConsumer(d1_out_id, unpack->id);
    ir_model.SetProducer(part_in_id, unpack->id);

    const IrTensorId q_in_id =
        MaybeAddSymmetricInt8Qdq(ir_model, part_in_id, d2_in_scale);
    const IrTensorId fc_out_id =
        AddLinearStage(context, ir_model, tensor_map, q_in_id,
                       plan.d2_weights[k], BHWC(1, 1, T, plan.o4));
    parts[k] =
        MaybeAddSymmetricInt8Qdq(ir_model, fc_out_id, plan.attrs.d2.out_scale);
  }
  IrOp* pack_op = ir_model.add_op();
  pack_op->name = ::litert::ml_drift::kBlockFfwPackPartsType;
  pack_op->attr =
      ::litert::ml_drift::BlockFfwPackPartsAttributes{.num_parts = plan.k1};
  for (IrTensorId part_id : parts) {
    ir_model.AddConsumer(part_id, pack_op->id);
  }
  ir_model.SetProducer(out_id, pack_op->id);
}

}  // namespace

CustomIrOpParser GetBlockFfwParser() {
  return {
      .is_supported = BlockFfwIsSupported,
      .convert = BlockFfwConvert,
  };
}

}  // namespace litert::ml_drift::ir

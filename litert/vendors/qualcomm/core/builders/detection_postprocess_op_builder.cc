// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/detection_postprocess_op_builder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {

namespace {
// TFLite_Detection_PostProcess input indices
constexpr size_t kTFLiteBoxEncodings = 0;
constexpr size_t kTFLiteClassPredictions = 1;
constexpr size_t kTFLiteAnchors = 2;

// TFLite_Detection_PostProcess output indices
constexpr size_t kTFLiteOutputBoxes = 0;
constexpr size_t kTFLiteOutputClasses = 1;
constexpr size_t kTFLiteOutputScores = 2;
constexpr size_t kTFLiteOutputNumDetections = 3;
}  // namespace

LiteRtStatus BuildDetectionPostprocessOp(
    absl::Span<const uint8_t> custom_options, TensorPool& tensor_pool,
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    std::vector<OpWrapper>& op_wrappers) {
  const auto root =
      flexbuffers::GetRoot(custom_options.data(), custom_options.size());
  if (!root.IsMap()) {
    QNN_LOG_ERROR(
        "TFLite_Detection_PostProcess: custom options are not a flexbuffer "
        "map.");
    return kLiteRtStatusErrorInvalidArgument;
  }

  const auto m = root.AsMap();
  const float nms_score_threshold = m["nms_score_threshold"].AsFloat();
  const float nms_iou_threshold = m["nms_iou_threshold"].AsFloat();
  const float y_scale = m["y_scale"].AsFloat();
  const float x_scale = m["x_scale"].AsFloat();
  const float h_scale = m["h_scale"].AsFloat();
  const float w_scale = m["w_scale"].AsFloat();
  const std::int32_t max_detections = m["max_detections"].AsInt32();
  // background_label_id defaults to num_classes (background is at the last
  // index when not explicitly provided).
  const std::uint32_t background_class_idx =
      m["background_label_id"].IsNull()
          ? static_cast<std::uint32_t>(m["num_classes"].AsInt32())
          : static_cast<std::uint32_t>(m["background_label_id"].AsInt32());

  // QNN expects reciprocals of the TFLite scale values as delta scaling
  // factors.
  const float delta_scaling_y = (y_scale != 0.0f) ? 1.0f / y_scale : 0.0f;
  const float delta_scaling_x = (x_scale != 0.0f) ? 1.0f / x_scale : 0.0f;
  const float delta_scaling_h = (h_scale != 0.0f) ? 1.0f / h_scale : 0.0f;
  const float delta_scaling_w = (w_scale != 0.0f) ? 1.0f / w_scale : 0.0f;

  // TFLite stores all four detection outputs as float32[1] placeholders because
  // its kernel resizes them dynamically in Prepare(). Fix shape and type in-
  // place so QNN sees the correct spec without requiring intermediate tensors.
  //
  // QNN spec (MasterOpDef):
  //   out[0] scores:           float32 [batch, detection_limit]
  //   out[1] boxes:            float32 [batch, detection_limit, 4]
  //   out[2] classes (labels): INT_32  [batch, detection_limit]
  //   out[3] num_detections:   UINT_32 [batch]
  const std::uint32_t batch = 1;
  const auto d = static_cast<std::uint32_t>(max_detections);

  TensorWrapper& scores_tensor = outputs[kTFLiteOutputScores].get();
  scores_tensor.OverrideDimensions({batch, d});

  TensorWrapper& boxes_tensor = outputs[kTFLiteOutputBoxes].get();
  boxes_tensor.OverrideDimensions({batch, d, 4});

  TensorWrapper& classes_tensor = outputs[kTFLiteOutputClasses].get();
  classes_tensor.OverrideDimensions({batch, d});
  classes_tensor.OverrideDataType(QNN_DATATYPE_INT_32);

  TensorWrapper& num_det_tensor = outputs[kTFLiteOutputNumDetections].get();
  num_det_tensor.OverrideDimensions({batch});
  num_det_tensor.OverrideDataType(QNN_DATATYPE_UINT_32);

  auto& detection_op = CreateOpWrapper(op_wrappers, QNN_OP_DETECTION_OUTPUT);

  // QNN input[0] = scores (class predictions), input[1] = box encodings,
  // input[2] = anchors.  TFLite order is [box_encodings, class_predictions,
  // anchors], so we swap 0 and 1.
  detection_op.AddInputTensor(inputs[kTFLiteClassPredictions]);
  detection_op.AddInputTensor(inputs[kTFLiteBoxEncodings]);
  detection_op.AddInputTensor(inputs[kTFLiteAnchors]);

  // QNN output order: scores, boxes, classes, num_detections.
  // TFLite order:     boxes,  classes, scores, num_detections.
  detection_op.AddOutputTensor(scores_tensor);
  detection_op.AddOutputTensor(boxes_tensor);
  detection_op.AddOutputTensor(classes_tensor);
  detection_op.AddOutputTensor(num_det_tensor);

  // delta_scaling_factors tensor: [dy, dx, dh, dw].
  const std::array<float, 4> delta_factors = {
      delta_scaling_y, delta_scaling_x, delta_scaling_h, delta_scaling_w};
  const std::vector<std::uint32_t> delta_shape{4};
  auto& delta_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, QuantizeParamsWrapperVariant{}, delta_shape,
      sizeof(float) * delta_factors.size(), delta_factors.data());
  detection_op.AddTensorParam(
      QNN_OP_DETECTION_OUTPUT_PARAM_DELTA_SCALING_FACTORS, delta_tensor);

  detection_op.AddScalarParam<float>(
      QNN_OP_DETECTION_OUTPUT_PARAM_CONFIDENCE_THRESHOLD, nms_score_threshold);
  detection_op.AddScalarParam<float>(
      QNN_OP_DETECTION_OUTPUT_PARAM_IOU_THRESHOLD, nms_iou_threshold);

  // Always use REGULAR NMS: HTP's FAST NMS is disabled on current hardware,
  // and qairt-converter also unconditionally selects REGULAR.
  detection_op.AddScalarParam<std::uint32_t>(
      QNN_OP_DETECTION_OUTPUT_PARAM_NMS_TYPE,
      static_cast<std::uint32_t>(QNN_OP_DETECTION_OUTPUT_NMS_TYPE_REGULAR));

  detection_op.AddScalarParam<std::uint32_t>(
      QNN_OP_DETECTION_OUTPUT_PARAM_BACKGROUND_CLASS_IDX, background_class_idx);

  detection_op.AddScalarParam<bool>(
      QNN_OP_DETECTION_OUTPUT_PARAM_USE_BG_IN_NMS, false);
  detection_op.AddScalarParam<bool>(
      QNN_OP_DETECTION_OUTPUT_PARAM_OUTPUT_BACKGROUND, true);
  detection_op.AddScalarParam<bool>(
      QNN_OP_DETECTION_OUTPUT_PARAM_SHARE_LOCATION, true);
  detection_op.AddScalarParam<float>(QNN_OP_DETECTION_OUTPUT_PARAM_NMS_ETA,
                                     1.0f);
  detection_op.AddScalarParam<std::int32_t>(
      QNN_OP_DETECTION_OUTPUT_PARAM_DETECTION_LIMIT, max_detections);

  return kLiteRtStatusOk;
}

}  // namespace qnn

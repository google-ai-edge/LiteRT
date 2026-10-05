// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/vendors/qualcomm/core/builders/detection_postprocess_op_builder.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {

using testing::ElementsAre; // NOLINT
using testing::FloatNear;   // NOLINT
using testing::Pointwise;   // NOLINT

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

// Builds a flexbuffer custom_options blob matching TFLite's
// TFLite_Detection_PostProcess custom op metadata.
std::vector<uint8_t> BuildCustomOptions(int num_classes, int max_detections,
                                        float score_threshold,
                                        float iou_threshold, float y_scale,
                                        float x_scale, float h_scale,
                                        float w_scale,
                                        int background_label_id = -1) {
  flexbuffers::Builder fbb;
  fbb.Map([&] {
    fbb.Int("num_classes", num_classes);
    fbb.Int("max_detections", max_detections);
    fbb.Float("nms_score_threshold", score_threshold);
    fbb.Float("nms_iou_threshold", iou_threshold);
    fbb.Float("y_scale", y_scale);
    fbb.Float("x_scale", x_scale);
    fbb.Float("h_scale", h_scale);
    fbb.Float("w_scale", w_scale);
    if (background_label_id >= 0) {
      fbb.Int("background_label_id", background_label_id);
    }
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

// Verifies that BuildDetectionPostprocessOp produces a non-empty op list and
// that the graph finalizes without error on x86 (compilation only).
// Mirrors the TFLite_Detection_PostProcess custom op with:
//   box_encodings:     [1, 6, 4]  (6 anchors, 4 box deltas)
//   class_predictions: [1, 6, 3]  (6 anchors, 3 classes)
//   anchors:           [6, 4]
//   max_detections: 3, num_classes: 2
TEST_P(QnnModelTest, DetectionPostprocessBuildsAndFinalizes) {
  static constexpr int kNumAnchors = 6;
  static constexpr int kNumClasses = 2;
  static constexpr int kMaxDetections = 3;

  auto custom_opts =
      BuildCustomOptions(kNumClasses, kMaxDetections,
                         /*score_threshold=*/0.3f,
                         /*iou_threshold=*/0.6f,
                         /*y_scale=*/10.0f, /*x_scale=*/10.0f,
                         /*h_scale=*/5.0f, /*w_scale=*/5.0f);

  // TFLite input order: [box_encodings, class_predictions, anchors]
  auto& box_encodings = tensor_pool_.CreateInputTensorWithName(
      "box_encodings", QNN_DATATYPE_FLOAT_32, {},
      {1, static_cast<uint32_t>(kNumAnchors), 4});
  auto& class_predictions = tensor_pool_.CreateInputTensorWithName(
      "class_predictions", QNN_DATATYPE_FLOAT_32, {},
      {1, static_cast<uint32_t>(kNumAnchors),
       static_cast<uint32_t>(kNumClasses + 1)});
  auto& anchors = tensor_pool_.CreateInputTensorWithName(
      "anchors", QNN_DATATYPE_FLOAT_32, {},
      {static_cast<uint32_t>(kNumAnchors), 4});

  // TFLite output order: [boxes, classes, scores, num_detections]
  // The builder resizes these in-place from TFLite's float32[1] placeholders.
  auto& out_boxes = tensor_pool_.CreateOutputTensorWithName(
      "out_boxes", QNN_DATATYPE_FLOAT_32, {}, {1});
  auto& out_classes = tensor_pool_.CreateOutputTensorWithName(
      "out_classes", QNN_DATATYPE_FLOAT_32, {}, {1});
  auto& out_scores = tensor_pool_.CreateOutputTensorWithName(
      "out_scores", QNN_DATATYPE_FLOAT_32, {}, {1});
  auto& out_num_det = tensor_pool_.CreateOutputTensorWithName(
      "out_num_det", QNN_DATATYPE_FLOAT_32, {}, {1});

  std::vector<::qnn::TensorWrapperRef> inputs = {box_encodings,
                                                 class_predictions, anchors};
  std::vector<::qnn::TensorWrapperRef> outputs = {out_boxes, out_classes,
                                                  out_scores, out_num_det};
  std::vector<::qnn::OpWrapper> op_wrappers;

  const auto status = ::qnn::BuildDetectionPostprocessOp(
      {custom_opts.data(), custom_opts.size()}, tensor_pool_, inputs, outputs,
      op_wrappers);

  ASSERT_EQ(status, kLiteRtStatusOk);
  ASSERT_FALSE(op_wrappers.empty());

  qnn_model_.MoveOpsToGraph(std::move(op_wrappers));
  ASSERT_TRUE(qnn_model_.Finalize());
}

// Verifies that output tensors are reshaped and retyped correctly by the
// builder:
//   scores:         float32 [1, max_detections]
//   boxes:          float32 [1, max_detections, 4]
//   classes:        INT_32  [1, max_detections]
//   num_detections: UINT_32 [1]
TEST_P(QnnModelTest, DetectionPostprocessOutputShapesAreFixed) {
  static constexpr int kNumAnchors = 4;
  static constexpr int kNumClasses = 3;
  static constexpr int kMaxDetections = 2;

  auto custom_opts =
      BuildCustomOptions(kNumClasses, kMaxDetections, 0.2f, 0.5f, 10.0f, 10.0f,
                         5.0f, 5.0f);

  auto& box_encodings = tensor_pool_.CreateInputTensorWithName(
      "box_encodings", QNN_DATATYPE_FLOAT_32, {},
      {1, static_cast<uint32_t>(kNumAnchors), 4});
  auto& class_predictions = tensor_pool_.CreateInputTensorWithName(
      "class_predictions", QNN_DATATYPE_FLOAT_32, {},
      {1, static_cast<uint32_t>(kNumAnchors),
       static_cast<uint32_t>(kNumClasses + 1)});
  auto& anchors = tensor_pool_.CreateInputTensorWithName(
      "anchors", QNN_DATATYPE_FLOAT_32, {},
      {static_cast<uint32_t>(kNumAnchors), 4});

  auto& out_boxes = tensor_pool_.CreateOutputTensorWithName(
      "out_boxes", QNN_DATATYPE_FLOAT_32, {}, {1});
  auto& out_classes = tensor_pool_.CreateOutputTensorWithName(
      "out_classes", QNN_DATATYPE_FLOAT_32, {}, {1});
  auto& out_scores = tensor_pool_.CreateOutputTensorWithName(
      "out_scores", QNN_DATATYPE_FLOAT_32, {}, {1});
  auto& out_num_det = tensor_pool_.CreateOutputTensorWithName(
      "out_num_det", QNN_DATATYPE_FLOAT_32, {}, {1});

  std::vector<::qnn::TensorWrapperRef> inputs = {box_encodings,
                                                 class_predictions, anchors};
  std::vector<::qnn::TensorWrapperRef> outputs = {out_boxes, out_classes,
                                                  out_scores, out_num_det};
  std::vector<::qnn::OpWrapper> op_wrappers;

  ASSERT_EQ(::qnn::BuildDetectionPostprocessOp(
                {custom_opts.data(), custom_opts.size()}, tensor_pool_, inputs,
                outputs, op_wrappers),
            kLiteRtStatusOk);

  // scores: float32 [1, max_detections]
  EXPECT_EQ(out_scores.GetDataType(), QNN_DATATYPE_FLOAT_32);
  EXPECT_EQ(out_scores.GetDimensions(),
            (std::vector<uint32_t>{1, kMaxDetections}));

  // boxes: float32 [1, max_detections, 4]
  EXPECT_EQ(out_boxes.GetDataType(), QNN_DATATYPE_FLOAT_32);
  EXPECT_EQ(out_boxes.GetDimensions(),
            (std::vector<uint32_t>{1, kMaxDetections, 4}));

  // classes: INT_32 [1, max_detections]
  EXPECT_EQ(out_classes.GetDataType(), QNN_DATATYPE_INT_32);
  EXPECT_EQ(out_classes.GetDimensions(),
            (std::vector<uint32_t>{1, kMaxDetections}));

  // num_detections: UINT_32 [1]
  EXPECT_EQ(out_num_det.GetDataType(), QNN_DATATYPE_UINT_32);
  EXPECT_EQ(out_num_det.GetDimensions(), (std::vector<uint32_t>{1}));
}

// Verifies that a missing or zero-value y_scale does not cause a divide-by-zero
// crash. The builder should clamp to 0.0 and still return kLiteRtStatusOk.
TEST_P(QnnModelTest, DetectionPostprocessZeroScaleDoesNotCrash) {
  static constexpr int kNumAnchors = 4;
  static constexpr int kNumClasses = 2;
  static constexpr int kMaxDetections = 2;

  // y_scale = 0 triggers the (y_scale != 0) guard in the builder.
  auto custom_opts = BuildCustomOptions(kNumClasses, kMaxDetections, 0.2f, 0.5f,
                                        /*y_scale=*/0.0f, /*x_scale=*/10.0f,
                                        /*h_scale=*/5.0f, /*w_scale=*/5.0f);

  auto& box_encodings = tensor_pool_.CreateInputTensorWithName(
      "box_encodings", QNN_DATATYPE_FLOAT_32, {},
      {1, static_cast<uint32_t>(kNumAnchors), 4});
  auto& class_predictions = tensor_pool_.CreateInputTensorWithName(
      "class_predictions", QNN_DATATYPE_FLOAT_32, {},
      {1, static_cast<uint32_t>(kNumAnchors),
       static_cast<uint32_t>(kNumClasses + 1)});
  auto& anchors = tensor_pool_.CreateInputTensorWithName(
      "anchors", QNN_DATATYPE_FLOAT_32, {},
      {static_cast<uint32_t>(kNumAnchors), 4});

  auto& out_boxes = tensor_pool_.CreateOutputTensorWithName(
      "out_boxes", QNN_DATATYPE_FLOAT_32, {}, {1});
  auto& out_classes = tensor_pool_.CreateOutputTensorWithName(
      "out_classes", QNN_DATATYPE_FLOAT_32, {}, {1});
  auto& out_scores = tensor_pool_.CreateOutputTensorWithName(
      "out_scores", QNN_DATATYPE_FLOAT_32, {}, {1});
  auto& out_num_det = tensor_pool_.CreateOutputTensorWithName(
      "out_num_det", QNN_DATATYPE_FLOAT_32, {}, {1});

  std::vector<::qnn::TensorWrapperRef> inputs = {box_encodings,
                                                 class_predictions, anchors};
  std::vector<::qnn::TensorWrapperRef> outputs = {out_boxes, out_classes,
                                                  out_scores, out_num_det};
  std::vector<::qnn::OpWrapper> op_wrappers;

  EXPECT_EQ(::qnn::BuildDetectionPostprocessOp(
                {custom_opts.data(), custom_opts.size()}, tensor_pool_, inputs,
                outputs, op_wrappers),
            kLiteRtStatusOk);
}

// End-to-end accuracy test (on-device only).
//
// Setup — 6 anchors, 2 foreground classes, background at index 2:
//   box_encodings:     all zeros → decoded box == anchor (no delta applied)
//   class_predictions: [class0, class1, bg] per anchor
//   anchors (cy,cx,h,w): two clusters (cx=0.5 and cx=10.5) plus one isolated
//                         anchor at cx=100.5 to avoid cross-cluster suppression
//
// REGULAR per-class NMS result — overlapping anchors within the same class are
// suppressed, but different classes at the same location are independent:
//   class0 survivors: anchor4(0.93), anchor1(0.90), anchor6(0.30)
//   class1 survivors: anchor4(0.95), anchor1(0.80), anchor6(0.20)
//
// Global top-3 sorted by score:
//   rank 0: anchor4 class1 score=0.95  box=[0,10,1,11]
//   rank 1: anchor4 class0 score=0.93  box=[0,10,1,11]
//   rank 2: anchor1 class0 score=0.90  box=[0, 0,1, 1]
TEST_P(QnnModelTest, DetectionPostprocessAccuracy) {
  static constexpr int kNumAnchors = 6;
  static constexpr int kNumClasses = 2;
  static constexpr int kMaxDetections = 3;

  auto custom_opts =
      BuildCustomOptions(kNumClasses, kMaxDetections,
                         /*score_threshold=*/0.0f,
                         /*iou_threshold=*/0.5f,
                         /*y_scale=*/10.0f, /*x_scale=*/10.0f,
                         /*h_scale=*/5.0f, /*w_scale=*/5.0f);

  auto& box_encodings = tensor_pool_.CreateInputTensorWithName(
      "box_encodings", QNN_DATATYPE_FLOAT_32, {},
      {1, kNumAnchors, 4});
  // scores layout: [background, class0, class1] per anchor
  auto& class_predictions = tensor_pool_.CreateInputTensorWithName(
      "class_predictions", QNN_DATATYPE_FLOAT_32, {},
      {1, kNumAnchors, kNumClasses + 1});
  auto& anchors = tensor_pool_.CreateInputTensorWithName(
      "anchors", QNN_DATATYPE_FLOAT_32, {},
      {kNumAnchors, 4});

  auto& out_boxes = tensor_pool_.CreateOutputTensorWithName(
      "out_boxes", QNN_DATATYPE_FLOAT_32, {}, {1});
  auto& out_classes = tensor_pool_.CreateOutputTensorWithName(
      "out_classes", QNN_DATATYPE_FLOAT_32, {}, {1});
  auto& out_scores = tensor_pool_.CreateOutputTensorWithName(
      "out_scores", QNN_DATATYPE_FLOAT_32, {}, {1});
  auto& out_num_det = tensor_pool_.CreateOutputTensorWithName(
      "out_num_det", QNN_DATATYPE_FLOAT_32, {}, {1});

  std::vector<::qnn::TensorWrapperRef> inputs = {box_encodings,
                                                 class_predictions, anchors};
  std::vector<::qnn::TensorWrapperRef> outputs = {out_boxes, out_classes,
                                                  out_scores, out_num_det};
  std::vector<::qnn::OpWrapper> op_wrappers;

  ASSERT_EQ(::qnn::BuildDetectionPostprocessOp(
                {custom_opts.data(), custom_opts.size()}, tensor_pool_, inputs,
                outputs, op_wrappers),
            kLiteRtStatusOk);
  qnn_model_.MoveOpsToGraph(std::move(op_wrappers));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto boxes_idx = qnn_model_.AddInputTensor(box_encodings);
  auto scores_idx = qnn_model_.AddInputTensor(class_predictions);
  auto anchors_idx = qnn_model_.AddInputTensor(anchors);
  auto out_boxes_idx = qnn_model_.AddOutputTensor(out_boxes);
  auto out_classes_idx = qnn_model_.AddOutputTensor(out_classes);
  auto out_scores_idx = qnn_model_.AddOutputTensor(out_scores);
  auto out_num_det_idx = qnn_model_.AddOutputTensor(out_num_det);

  // All zero box deltas → decoded box equals anchor converted to corner form.
  qnn_model_.SetInputData<float>(boxes_idx, {
      0.f, 0.f, 0.f, 0.f,  // anchor 1
      0.f, 0.f, 0.f, 0.f,  // anchor 2
      0.f, 0.f, 0.f, 0.f,  // anchor 3
      0.f, 0.f, 0.f, 0.f,  // anchor 4
      0.f, 0.f, 0.f, 0.f,  // anchor 5
      0.f, 0.f, 0.f, 0.f,  // anchor 6
  });
  // [class0, class1, bg] per anchor — background at index num_classes=2,
  // matching background_class_idx = num_classes set by the builder.
  qnn_model_.SetInputData<float>(scores_idx, {
      .90f, .80f, 0.f,   // anchor 1
      .75f, .72f, 0.f,   // anchor 2  (suppressed by anchor1, IoU≈1)
      .60f, .50f, 0.f,   // anchor 3  (suppressed by anchor1, IoU≈1)
      .93f, .95f, 0.f,   // anchor 4
      .50f, .40f, 0.f,   // anchor 5  (suppressed by anchor4, IoU≈1)
      .30f, .20f, 0.f,   // anchor 6  (different location, survives)
  });
  // Anchors (cy, cx, h, w) — anchors 1-3 overlap, 4-5 overlap, 6 is isolated.
  qnn_model_.SetInputData<float>(anchors_idx, {
      0.5f,   0.5f, 1.f, 1.f,  // anchor 1
      0.5f,   0.5f, 1.f, 1.f,  // anchor 2
      0.5f,   0.5f, 1.f, 1.f,  // anchor 3
      0.5f,  10.5f, 1.f, 1.f,  // anchor 4
      0.5f,  10.5f, 1.f, 1.f,  // anchor 5
      0.5f, 100.5f, 1.f, 1.f,  // anchor 6
  });

  //   Class 0 NMS (sorted by score: anchor4=0.93, anchor1=0.90, anchor2=0.75, anchor3=0.60, anchor5=0.50, anchor6=0.30):
  //   - anchor4 (0.93) → keeper → suppresses anchor5 (IoU≈1)
  //   - anchor1 (0.90) → keeper → suppresses anchor2, anchor3 (IoU≈1)
  //   - anchor6 (0.30) → keeper (isolated)
  //   Class 1 NMS (sorted: anchor4=0.95, anchor1=0.80, anchor2=0.72, anchor3=0.50, anchor5=0.40, anchor6=0.20):
  //   - anchor4 (0.95) → keeper → suppresses anchor5 (IoU≈1)
  //   - anchor1 (0.80) → keeper → suppresses anchor2, anchor3 (IoU≈1)
  //   - anchor6 (0.20) → keeper (isolated)

  //   All survivors merged and globally sorted:
  //   (anchor4, class1, 0.95)
  //   (anchor4, class0, 0.93)
  //   (anchor1, class0, 0.90)  ← top 3 cutoff (max_detections=3)
  //   (anchor1, class1, 0.80)
  //   (anchor6, class0, 0.30)
  //   (anchor6, class1, 0.20)

  ASSERT_TRUE(qnn_model_.Execute());

  auto num_det_data = qnn_model_.GetOutputData<std::uint32_t>(out_num_det_idx);
  ASSERT_TRUE(num_det_data);
  ASSERT_THAT(num_det_data.value(), ElementsAre(3u));

  // scores: [0.95, 0.93, 0.90]
  auto scores_data = qnn_model_.GetOutputData<float>(out_scores_idx);
  ASSERT_TRUE(scores_data);
  ASSERT_THAT(scores_data.value(),
              Pointwise(FloatNear(1e-2), {0.95f, 0.93f, 0.90f}));

  // input anchors are in (cy, cx, h, w) center-size format, but the
  // output boxes are in (ymin, xmin, ymax, xmax) corner format
  // rank 0: anchor4 [0.5f, 10.5f, 1.f, 1.f] → [0,10,1,11]
  // rank 1: anchor4 [0.5f, 10.5f, 1.f, 1.f] → [0,10,1,11]
  // rank 2: anchor1 [0.5f, 0.5f, 1.f, 1.f]  → [0,0,1,1]
  auto boxes_data = qnn_model_.GetOutputData<float>(out_boxes_idx);
  ASSERT_TRUE(boxes_data);
  ASSERT_THAT(boxes_data.value(),
              Pointwise(FloatNear(1e-1), {
                  0.f, 10.f, 1.f, 11.f,  // rank 0
                  0.f, 10.f, 1.f, 11.f,  // rank 1
                  0.f,  0.f, 1.f,  1.f,  // rank 2
              }));

  // classes (INT_32): class1=1, class0=0, class0=0
  auto classes_data = qnn_model_.GetOutputData<std::int32_t>(out_classes_idx);
  ASSERT_TRUE(classes_data);
  ASSERT_THAT(classes_data.value(), ElementsAre(1, 0, 0));
}

// Verifies that an empty custom_options blob is rejected gracefully.
TEST_P(QnnModelTest, DetectionPostprocessRejectsEmptyCustomOptions) {
  auto& box_encodings = tensor_pool_.CreateInputTensorWithName(
      "box_encodings", QNN_DATATYPE_FLOAT_32, {}, {1, 4, 4});
  auto& class_predictions = tensor_pool_.CreateInputTensorWithName(
      "class_predictions", QNN_DATATYPE_FLOAT_32, {}, {1, 4, 3});
  auto& anchors = tensor_pool_.CreateInputTensorWithName(
      "anchors", QNN_DATATYPE_FLOAT_32, {}, {4, 4});
  auto& out_boxes = tensor_pool_.CreateOutputTensorWithName(
      "out_boxes", QNN_DATATYPE_FLOAT_32, {}, {1});
  auto& out_classes = tensor_pool_.CreateOutputTensorWithName(
      "out_classes", QNN_DATATYPE_FLOAT_32, {}, {1});
  auto& out_scores = tensor_pool_.CreateOutputTensorWithName(
      "out_scores", QNN_DATATYPE_FLOAT_32, {}, {1});
  auto& out_num_det = tensor_pool_.CreateOutputTensorWithName(
      "out_num_det", QNN_DATATYPE_FLOAT_32, {}, {1});

  std::vector<::qnn::TensorWrapperRef> inputs = {box_encodings,
                                                 class_predictions, anchors};
  std::vector<::qnn::TensorWrapperRef> outputs = {out_boxes, out_classes,
                                                  out_scores, out_num_det};
  std::vector<::qnn::OpWrapper> op_wrappers;

  EXPECT_EQ(::qnn::BuildDetectionPostprocessOp({}, tensor_pool_, inputs,
                                               outputs, op_wrappers),
            kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
}  // namespace litert::qnn

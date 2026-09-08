// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_DETECTION_POSTPROCESS_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_DETECTION_POSTPROCESS_OP_BUILDER_H_

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "litert/c/litert_common.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

// Parses the flexbuffer custom options of a TFLite_Detection_PostProcess op,
// then builds the equivalent QNN DetectionOutput op.
//
// TFLite input order:  [0] box_encodings, [1] class_predictions, [2] anchors
// QNN input order:     [0] scores(class_predictions), [1] box_encodings,
//                      [2] anchors
//
// TFLite output order: [0] boxes, [1] classes, [2] scores, [3] num_detections
// QNN output order:    [0] scores, [1] boxes, [2] classes, [3] num_detections
//
// Output tensors are fixed up in-place (shape + dtype) because TFLite stores
// them as float32[1] placeholders that its kernel resizes dynamically.
// NMS type is always set to REGULAR; HTP's FAST NMS is disabled on current
// hardware.
LiteRtStatus BuildDetectionPostprocessOp(
    absl::Span<const uint8_t> custom_options, TensorPool& tensor_pool,
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    std::vector<OpWrapper>& op_wrappers);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_DETECTION_POSTPROCESS_OP_BUILDER_H_

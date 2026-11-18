// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_COLOR_TRANSFORM_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_COLOR_TRANSFORM_OP_BUILDER_H_

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

// Creates a QNN color-transform op that converts the raw camera-format input
// tensor to RGB/BGR for the first op in the graph.
//
// encoding_in  : "nv21" -> QNN_OP_NV21_TO_RGB
//                "nv12" -> QNN_OP_NV12_TO_RGB
//                "argb32" or "rgba" -> QNN_OP_ARGB_TO_RGB
// encoding_out : "rgb" or "bgr" (controls output_order / reverse_output param)
OpWrapper BuildColorTransformOp(
    const TensorWrapper& input, const TensorWrapper& output,
    const DesiredInputColorEncodingEntry& entry);

// Computes the raw input dimensions expected by the color-transform op given
// the original model input dimensions and the source encoding.
//
// NV21/NV12  : [B, H*W*3/2]  (flat YUV buffer)
// ARGB32/RGBA: [B, H, W, 4]  (4-channel)
std::vector<std::uint32_t> ColorTransformInputDims(
    const std::vector<std::uint32_t>& original_dims,
    const std::string& encoding_in);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_COLOR_TRANSFORM_OP_BUILDER_H_

// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/color_transform_op_builder.h"

#include <cstdint>
#include <string>
#include <vector>

#include "QnnOpDef.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

OpWrapper BuildColorTransformOp(const TensorWrapper& input,
                                const TensorWrapper& output,
                                const DesiredInputColorEncodingEntry& entry) {
  const bool out_bgr = (entry.encoding_out == "bgr");

  if (entry.encoding_in == "nv21") {
    OpWrapper op(GetUniqueOpName(QNN_OP_NV21_TO_RGB), QNN_OP_NV21_TO_RGB,
                 QnnOpCode::kNv21ToRgb);
    op.AddInputTensor(input);
    op.AddOutputTensor(output);
    const std::uint32_t order = out_bgr ? QNN_OP_NV21_TO_RGB_OUTPUT_ORDER_BGR
                                        : QNN_OP_NV21_TO_RGB_OUTPUT_ORDER_RGB;
    op.AddScalarParam<std::uint32_t>(QNN_OP_NV21_TO_RGB_PARAM_OUTPUT_ORDER,
                                     order);
    return op;
  }

  if (entry.encoding_in == "nv12") {
    OpWrapper op(GetUniqueOpName(QNN_OP_NV12_TO_RGB), QNN_OP_NV12_TO_RGB,
                 QnnOpCode::kNv12ToRgb);
    op.AddInputTensor(input);
    op.AddOutputTensor(output);
    const std::uint32_t order = out_bgr ? QNN_OP_NV12_TO_RGB_OUTPUT_ORDER_BGR
                                        : QNN_OP_NV12_TO_RGB_OUTPUT_ORDER_RGB;
    op.AddScalarParam<std::uint32_t>(QNN_OP_NV12_TO_RGB_PARAM_OUTPUT_ORDER,
                                     order);
    return op;
  }

  // argb32 or rgba -> QNN_OP_ARGB_TO_RGB
  OpWrapper op(GetUniqueOpName(QNN_OP_ARGB_TO_RGB), QNN_OP_ARGB_TO_RGB,
               QnnOpCode::kArgbToRgb);
  op.AddInputTensor(input);
  op.AddOutputTensor(output);
  const std::uint32_t input_order = (entry.encoding_in == "rgba")
                                        ? QNN_OP_ARGB_TO_RGB_INPUT_ORDER_RGBA
                                        : QNN_OP_ARGB_TO_RGB_INPUT_ORDER_ARGB;
  op.AddScalarParam<std::uint32_t>(QNN_OP_ARGB_TO_RGB_PARAM_INPUT_ORDER,
                                   input_order);
  op.AddScalarParam<bool>(QNN_OP_ARGB_TO_RGB_PARAM_REVERSE_OUTPUT, out_bgr);
  return op;
}

std::vector<std::uint32_t> ColorTransformInputDims(
    const std::vector<std::uint32_t>& original_dims,
    const std::string& encoding_in) {
  if (encoding_in == "nv21" || encoding_in == "nv12") {
    // Original dims are [B, H, W, 3]. Flat YUV buffer: [B, H*W*3/2].
    if (original_dims.size() == 4) {
      const std::uint32_t b = original_dims[0];
      const std::uint32_t h = original_dims[1];
      const std::uint32_t w = original_dims[2];
      return {b, h * w * 3 / 2};
    }
  } else if (encoding_in == "argb32" || encoding_in == "rgba") {
    // Original dims are [B, H, W, 3]. 4-channel input: [B, H, W, 4].
    if (original_dims.size() == 4) {
      return {original_dims[0], original_dims[1], original_dims[2], 4};
    }
  }
  // Fallback: return original dims unchanged.
  return original_dims;
}

}  // namespace qnn

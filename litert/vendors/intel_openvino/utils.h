// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_OPENVINO_UTILS_H_
#define ODML_LITERT_LITERT_VENDORS_OPENVINO_UTILS_H_
#include "openvino/frontend/tensorflow_lite/decoder.hpp"

#include "litert/c/litert_logging.h"
#include "litert/cc/litert_model.h"

using namespace litert;
static const ov::element::Type MapLiteTypeToOV(const LiteRtElementType element_type) {
    ov::element::Type ov_type;
    switch (element_type) {
        case kLiteRtElementTypeBool:
            ov_type = ov::element::boolean;
            break;
        case kLiteRtElementTypeInt4:
            ov_type = ov::element::i4;
            break;
        case kLiteRtElementTypeInt8:
            ov_type = ov::element::i8;
            break;
        case kLiteRtElementTypeInt16:
            ov_type = ov::element::i16;
            break;
        case kLiteRtElementTypeInt32:
            ov_type = ov::element::i32;
            break;
        case kLiteRtElementTypeInt64:
            ov_type = ov::element::i64;
            break;
        case kLiteRtElementTypeUInt8:
            ov_type = ov::element::u8;
            break;
        case kLiteRtElementTypeUInt16:
            ov_type = ov::element::u16;
            break;
        case kLiteRtElementTypeUInt32:
            ov_type = ov::element::u32;
            break;
        case kLiteRtElementTypeUInt64:
            ov_type = ov::element::u64;
            break;
        case kLiteRtElementTypeFloat16:
            ov_type = ov::element::f16;
            break;
        case kLiteRtElementTypeFloat32:
            ov_type = ov::element::f32;
            break;
        case kLiteRtElementTypeFloat64:
            ov_type = ov::element::f64;
            break;
        case kLiteRtElementTypeBFloat16:
            ov_type = ov::element::bf16;
            break;
        default:
            ov_type = ov::element::undefined;
    }
    return ov_type;
}

static const LiteRtStatus GetOVTensorShape(const litert::Tensor& litert_tensor,
                                           std::vector<int64_t>& ov_shape_vec) {
    if (litert_tensor.TypeId() != kLiteRtRankedTensorType) return kLiteRtStatusErrorInvalidArgument;

    const auto ranked_tensor_type = litert_tensor.RankedTensorType();
    if (!ranked_tensor_type) {
        LITERT_LOG(LITERT_ERROR, "%s", ranked_tensor_type.Error().Message().data());
        return ranked_tensor_type.Error().Status();
    }

    const auto tensor_layout = ranked_tensor_type->Layout();
    if (tensor_layout.Rank() == 0)
        return kLiteRtStatusErrorUnsupported;
    else {
        ov_shape_vec.resize(tensor_layout.Rank());
        for (int i = 0; i < ov_shape_vec.size(); i++)
            ov_shape_vec[i] = tensor_layout.Dimensions()[i];
    }
    return kLiteRtStatusOk;
}
#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_UTILS_H_

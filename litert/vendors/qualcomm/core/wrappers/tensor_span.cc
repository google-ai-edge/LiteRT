// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/wrappers/tensor_span.h"

#include <numeric>

#include "QnnTypes.h"  // from @qairt
#include "absl/strings/match.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/utils/miscs.h"

namespace qnn {
bool TensorSpan::IsMarkedDump() const {
  return absl::EndsWith(GetTensorName(), kDumpSuffix) &&
         src_->v1.type == QNN_TENSOR_TYPE_APP_READ;
}

Qnn_ScaleOffset_t TensorSpan::GetScaleOffset() const {
  if (src_->v1.quantizeParams.encodingDefinition == QNN_DEFINITION_DEFINED &&
      src_->v1.quantizeParams.quantizationEncoding ==
          QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    return src_->v1.quantizeParams.scaleOffsetEncoding;
  }
  return {1.0f, 0};
}

std::uint32_t TensorSpan::GetTensorNumElements() const {
  return std::accumulate(src_->v1.dimensions,
                         src_->v1.dimensions + src_->v1.rank, 1,
                         std::multiplies<>());
}

std::uint32_t TensorSpan::GetTensorBytes() const {
  return GetDataTypeSize(src_->v1.dataType) * GetTensorNumElements();
}

}  // namespace qnn

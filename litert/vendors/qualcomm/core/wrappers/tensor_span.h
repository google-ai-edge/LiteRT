// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_TENSOR_SPAN_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_TENSOR_SPAN_H_

#include <cstdint>

#include "QnnTypes.h"  // from @qairt
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/utils/miscs.h"

namespace qnn {

class TensorSpan {
 public:
  explicit TensorSpan(Qnn_Tensor_t* src) : src_(src) {}

  bool IsQUInt16() const {
    return src_->v1.dataType == QNN_DATATYPE_UFIXED_POINT_16;
  }

  absl::string_view GetName() const { return src_->v1.name; }

  bool IsMarkedDump() const;

  Qnn_ScaleOffset_t GetScaleOffset() const;

  std::uint32_t GetNumElements() const;

  std::uint32_t GetBytes() const;

  void SetDataSize(std::uint32_t data_size) {
    src_->v1.clientBuf.dataSize = data_size;
  }

  void SetData(void* data) { src_->v1.clientBuf.data = data; }

  Qnn_Tensor_t* Get() { return src_; }

  const Qnn_Tensor_t* Get() const { return src_; }

 private:
  Qnn_Tensor_t* src_;
};
}  // namespace qnn

#endif // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_TENSOR_SPAN_H_

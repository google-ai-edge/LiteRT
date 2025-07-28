// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_MODEL_TEST_QNN_MODEL_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_MODEL_TEST_QNN_MODEL_H_

#include <cstddef>
#include <vector>

#include "QnnCommon.h"
#include "QnnInterface.h"
#include "absl/types/span.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

class QnnModel {
 public:
  QnnModel(const Qnn_BackendHandle_t backend_handle,
           const Qnn_ContextHandle_t context_handle,
           const QNN_INTERFACE_VER_TYPE* api)
      : backend_handle_(backend_handle),
        context_handle_(context_handle),
        api_(api){};
  ~QnnModel() = default;

  template <typename T>
  bool SetInputData(size_t idx, absl::Span<const T> data) {
    if (idx >= input_tensors_.size()) {
      return false;
    }
    return input_tensors_[idx]->SetTensorData(data);
  }

  size_t SetInputTensor(TensorWrapper& tensor) {
    input_tensors_.emplace_back(&tensor);
    return input_tensors_.size() - 1;
  }

  template <typename T>
  std::optional<absl::Span<const T>> GetOutputData(size_t idx) {
    if (idx >= output_tensors_.size()) {
      return std::nullopt;
    }
    return output_tensors_[idx]->GetTensorData<T>();
  }

  size_t SetOutputTensor(TensorWrapper& tensor) {
    tensor.AllocateOutputTensorBuffer();
    output_tensors_.emplace_back(&tensor);
    return output_tensors_.size() - 1;
  }

  bool ValidateOpConfig(std::vector<qnn::OpWrapper>& ops);

  bool Finalize(std::vector<qnn::OpWrapper>& ops);

  bool Execute();

 private:
  const QNN_INTERFACE_VER_TYPE* api_{nullptr};
  Qnn_BackendHandle_t backend_handle_ = nullptr;
  Qnn_ContextHandle_t context_handle_ = nullptr;
  Qnn_ContextHandle_t graph_handle_ = nullptr;
  std::vector<qnn::TensorWrapper*> input_tensors_;
  std::vector<qnn::TensorWrapper*> output_tensors_;
  absl::flat_hash_set<const ::qnn::TensorWrapper*> created_tensors_;
};

}  // namespace qnn
#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_MODEL_TEST_QNN_MODEL_H_
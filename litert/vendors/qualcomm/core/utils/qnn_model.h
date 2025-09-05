// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_QNN_MODEL_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_QNN_MODEL_H_

#include <cstddef>
#include <vector>

#include "QnnCommon.h"
#include "QnnInterface.h"
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

class QnnModel {
 public:
  QnnModel(const Qnn_BackendHandle_t backend_handle,
           const QNN_INTERFACE_VER_TYPE* api,
           const Qnn_ContextHandle_t context_handle)
      : backend_handle_(backend_handle),
        api_(api),
        context_handle_(context_handle){};

  QnnModel() = default;
  ~QnnModel() = default;

  template <typename T>
  bool SetInputData(size_t idx, absl::Span<const T> data) {
    if (idx >= input_tensors_.size()) {
      return false;
    }
    return input_tensors_[idx]->SetTensorData(data);
  }

  // TODO (chunhsue-qti): Add another SetInputData which gets input in float and
  // quantize.

  size_t AddInputTensor(TensorWrapper& tensor) {
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

  size_t AddOutputTensor(TensorWrapper& tensor) {
    output_tensors_.emplace_back(&tensor);
    return output_tensors_.size() - 1;
  }

  bool ValidateOpConfig();

  bool Finalize();

  bool Execute();

  void MoveOpsToGraph(std::vector<::qnn::OpWrapper>&& ops) {
    std::move(ops.begin(), ops.end(), std::back_inserter(op_wrappers_));
  }

 private:
  Qnn_BackendHandle_t backend_handle_ = nullptr;
  const QNN_INTERFACE_VER_TYPE* api_{nullptr};
  Qnn_ContextHandle_t context_handle_ = nullptr;
  Qnn_GraphHandle_t graph_handle_ = nullptr;

  std::vector<OpWrapper> op_wrappers_;

  std::vector<TensorWrapper*> input_tensors_;
  std::vector<TensorWrapper*> output_tensors_;
};

}  // namespace qnn
#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_QNN_MODEL_H_

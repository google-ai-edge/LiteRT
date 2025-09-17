// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/utils/qnn_model.h"

#include "HTP/QnnHtpGraph.h"               // from @qairt
#include "QnnGraph.h"                      // from @qairt
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/types/span.h"               // from @com_google_absl
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/utils/log.h"

namespace qnn {
namespace {
#define QNN_RETURN_STATUS_IF_NOT_OK(expr) \
  if (QNN_SUCCESS != (expr)) {            \
    return false;                         \
  }

inline absl::Span<const QnnGraph_Config_t*> DefaultGraphConfigs() {
  static std::array<QnnHtpGraph_CustomConfig_t, 3> graph_custom_configs;
  // Default use O3 for now.
  graph_custom_configs[0] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[0].option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
  graph_custom_configs[0].optimizationOption.type =
      QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
  // Change to 2 if you want to use O2 (default).
  graph_custom_configs[0].optimizationOption.floatValue = 3;
  // VTCM
  graph_custom_configs[1] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[1].option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
  graph_custom_configs[1].vtcmSizeInMB = QNN_HTP_GRAPH_CONFIG_OPTION_MAX;
  // FoldRelu Off
  graph_custom_configs[2] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[2].option =
      QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF;
  graph_custom_configs[2].foldReluActivationIntoConvOff = true;

  static std::array<QnnGraph_Config_t, 3> graph_configs;
  graph_configs[0] = QNN_GRAPH_CONFIG_INIT;
  graph_configs[0].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graph_configs[0].customConfig = &graph_custom_configs[0];

  graph_configs[1] = QNN_GRAPH_CONFIG_INIT;
  graph_configs[1].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graph_configs[1].customConfig = &graph_custom_configs[1];

  graph_configs[2] = QNN_GRAPH_CONFIG_INIT;
  graph_configs[2].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graph_configs[2].customConfig = &graph_custom_configs[2];

  static std::array<const QnnGraph_Config_t*, 4> result = {
      &graph_configs[0], &graph_configs[1], &graph_configs[2], nullptr};

  return absl::MakeSpan(result.data(), result.size());
}
}  // namespace
bool QnnModel::ValidateOpConfig() {
  return std::all_of(op_wrappers_.begin(), op_wrappers_.end(),
                     [this](OpWrapper& op_wrapper) -> bool {
                       return QNN_SUCCESS ==
                              api_->backendValidateOpConfig(
                                  backend_handle_, op_wrapper.GetOpConfig());
                     });
}

bool QnnModel::Finalize() {
  absl::flat_hash_set<const ::qnn::TensorWrapper*> created_tensors;
  QNN_RETURN_STATUS_IF_NOT_OK(api_->graphCreate(
      context_handle_, "test", DefaultGraphConfigs().data(), &graph_handle_));
  for (auto& op_wrapper : op_wrappers_) {
    for (const auto& tensor_wrapper_ref : op_wrapper.GetAllTensors()) {
      if (!created_tensors.count(&(tensor_wrapper_ref.get()))) {
        QNN_RETURN_STATUS_IF_NOT_OK(api_->tensorCreateGraphTensor(
            graph_handle_, &(tensor_wrapper_ref.get().GetQnnTensor())));
        created_tensors.emplace(&(tensor_wrapper_ref.get()));
      }
    }
    api_->graphAddNode(graph_handle_, op_wrapper.GetOpConfig());
  }
  QNN_RETURN_STATUS_IF_NOT_OK(
      api_->graphFinalize(graph_handle_, nullptr, nullptr));
  return true;
}

bool QnnModel::Execute() {
  if (graph_handle_ == nullptr) {
    QNN_LOG_ERROR("Finalize() should be called before Execute()")
    return false;
  }
  std::vector<Qnn_Tensor_t> input_tensors;
  std::vector<Qnn_Tensor_t> output_tensors;
  for (auto* tensor_wrapper : input_tensors_) {
    input_tensors.emplace_back(tensor_wrapper->GetQnnTensor());
  }
  for (auto* tensor_wrapper : output_tensors_) {
    tensor_wrapper->AllocateOutputTensorBuffer();
    output_tensors.emplace_back(tensor_wrapper->GetQnnTensor());
  }
  QNN_RETURN_STATUS_IF_NOT_OK(api_->graphExecute(
      graph_handle_, input_tensors.data(), input_tensors.size(),
      output_tensors.data(), output_tensors.size(), nullptr, nullptr));
  return true;
}
}  // namespace qnn

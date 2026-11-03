// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/htp_quick_response.h"

#include <cstddef>
#include <chrono>
#include <system_error>

#include "QnnBackend.h" // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnContext.h" // from @qairt
#include "QnnGraph.h"   // from @qairt
#include "QnnOpDef.h"   // from @qairt
#include "litert/vendors/qualcomm/core/backends/htp_backend.h"
#include "litert/vendors/qualcomm/core/utils/log.h"

namespace qnn {
namespace {

bool HasRequiredApi(const QNN_INTERFACE_VER_TYPE *api) {
  return api != nullptr && api->contextCreate != nullptr &&
         api->contextFree != nullptr && api->graphCreate != nullptr &&
         api->tensorCreateGraphTensor != nullptr &&
         api->backendValidateOpConfig != nullptr &&
         api->graphAddNode != nullptr && api->graphFinalize != nullptr &&
         api->graphExecute != nullptr;
}

Qnn_Tensor_t CreateTensor(const char *name, Qnn_TensorType_t tensor_type,
                          std::array<std::uint32_t, 4> &dimensions,
                          float scale) {
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  tensor.version = QNN_TENSOR_VERSION_2;
  tensor.v2.name = name;
  tensor.v2.rank = static_cast<std::uint32_t>(dimensions.size());
  tensor.v2.dimensions = dimensions.data();
  tensor.v2.type = tensor_type;
  tensor.v2.dataFormat = 0;
  tensor.v2.dataType = QNN_DATATYPE_SFIXED_POINT_8;
  tensor.v2.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
  tensor.v2.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
  tensor.v2.quantizeParams.quantizationEncoding =
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  tensor.v2.quantizeParams.scaleOffsetEncoding = {scale, 0};
  tensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
  tensor.v2.clientBuf.dataSize = sizeof(std::int8_t);
  return tensor;
}

} // namespace

HtpQuickResponse::HtpQuickResponse(const QNN_INTERFACE_VER_TYPE *api,
                                   HtpBackend &backend)
    : api_(api), backend_(backend) {}

std::unique_ptr<HtpQuickResponse>
HtpQuickResponse::Create(const QNN_INTERFACE_VER_TYPE *api,
                         HtpBackend &backend) {
  auto quick_response =
      std::unique_ptr<HtpQuickResponse>(new HtpQuickResponse(api, backend));
  if (!quick_response->Init()) {
    return nullptr;
  }
  return quick_response;
}

HtpQuickResponse::~HtpQuickResponse() {
  Stop();
  FreeContext();
}

bool HtpQuickResponse::Init() {
  if (!HasRequiredApi(api_)) {
    QNN_LOG_WARNING(
        "HTP quick response cannot start because required QNN APIs are "
        "missing.");
    return false;
  }

  Qnn_ErrorHandle_t error = api_->contextCreate(backend_.GetBackendHandle(),
                                                backend_.GetDeviceHandle(),
                                                nullptr, &context_handle_);
  if (error != QNN_SUCCESS) {
    QNN_LOG_WARNING("HTP quick response failed to create QNN context. Error %d",
                    QNN_GET_ERROR_CODE(error));
    return false;
  }

  if (!CreateGraph()) {
    QNN_LOG_WARNING("HTP quick response failed to create graph.");
    return false;
  }

  try {
    thread_ = std::thread(&HtpQuickResponse::Run, this);
  } catch (const std::system_error &error) {
    QNN_LOG_WARNING("HTP quick response failed to start thread: %s",
                    error.what());
    return false;
  }
  return true;
}

bool HtpQuickResponse::CreateGraph() {
  QnnGraph_Config_t priority_config = QNN_GRAPH_CONFIG_INIT;
  priority_config.option = QNN_GRAPH_CONFIG_OPTION_PRIORITY;
  priority_config.priority = QNN_PRIORITY_LOW;
  const QnnGraph_Config_t *graph_configs[] = {&priority_config, nullptr};
  Qnn_ErrorHandle_t error = api_->graphCreate(
      context_handle_, "HtpQuickResponse", graph_configs, &graph_handle_);
  if (error != QNN_SUCCESS) {
    QNN_LOG_WARNING("HTP quick response failed to create graph. Error %d",
                    QNN_GET_ERROR_CODE(error));
    return false;
  }

  static constexpr float kTensorScale = 0.001f;
  input_tensors_[0] =
      CreateTensor("quick_response_input_0", QNN_TENSOR_TYPE_APP_WRITE,
                   tensor_dims_, kTensorScale);
  input_tensors_[1] =
      CreateTensor("quick_response_input_1", QNN_TENSOR_TYPE_APP_WRITE,
                   tensor_dims_, kTensorScale);
  output_tensors_[0] =
      CreateTensor("quick_response_output", QNN_TENSOR_TYPE_APP_READ,
                   tensor_dims_, 2 * kTensorScale);

  for (Qnn_Tensor_t &tensor : input_tensors_) {
    error = api_->tensorCreateGraphTensor(graph_handle_, &tensor);
    if (error != QNN_SUCCESS) {
      QNN_LOG_WARNING(
          "HTP quick response failed to create input tensor. Error %d",
          QNN_GET_ERROR_CODE(error));
      return false;
    }
  }

  error = api_->tensorCreateGraphTensor(graph_handle_, &output_tensors_[0]);
  if (error != QNN_SUCCESS) {
    QNN_LOG_WARNING(
        "HTP quick response failed to create output tensor. Error %d",
        QNN_GET_ERROR_CODE(error));
    return false;
  }

  Qnn_OpConfig_t add_op = QNN_OPCONFIG_INIT;
  add_op.version = QNN_OPCONFIG_VERSION_1;
  add_op.v1 = {"quick_response_add",
               QNN_OP_PACKAGE_NAME_QTI_AISW,
               QNN_OP_ELEMENT_WISE_ADD,
               /*numOfParams=*/0,
               nullptr,
               static_cast<std::uint32_t>(input_tensors_.size()),
               input_tensors_.data(),
               static_cast<std::uint32_t>(output_tensors_.size()),
               output_tensors_.data()};

  error = api_->backendValidateOpConfig(backend_.GetBackendHandle(), add_op);
  if (error != QNN_SUCCESS) {
    QNN_LOG_WARNING("HTP quick response failed to validate ADD op. Error %d",
                    QNN_GET_ERROR_CODE(error));
    return false;
  }

  error = api_->graphAddNode(graph_handle_, add_op);
  if (error != QNN_SUCCESS) {
    QNN_LOG_WARNING("HTP quick response failed to add ADD node. Error %d",
                    QNN_GET_ERROR_CODE(error));
    return false;
  }

  error = api_->graphFinalize(graph_handle_, nullptr, nullptr);
  if (error != QNN_SUCCESS) {
    QNN_LOG_WARNING("HTP quick response failed to finalize graph. Error %d",
                    QNN_GET_ERROR_CODE(error));
    return false;
  }

  input_tensors_[0].v2.clientBuf.data = input_data_0_.data();
  input_tensors_[1].v2.clientBuf.data = input_data_1_.data();
  output_tensors_[0].v2.clientBuf.data = output_data_.data();
  return true;
}

void HtpQuickResponse::Run() {
  static constexpr std::chrono::milliseconds kQuickResponseInterval{10};

  while (true) {
    if (!ExecuteOnce()) {
      return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    if (stop_cv_.wait_for(lock, kQuickResponseInterval,
                          [this] { return stop_; })) {
      return;
    }
  }
}

bool HtpQuickResponse::ExecuteOnce() {
  Qnn_ErrorHandle_t error = api_->graphExecute(
      graph_handle_, input_tensors_.data(), input_tensors_.size(),
      output_tensors_.data(), output_tensors_.size(), nullptr, nullptr);
  if (error != QNN_SUCCESS) {
    QNN_LOG_WARNING("HTP quick response graph execution failed. Error %d",
                    QNN_GET_ERROR_CODE(error));
    return false;
  }
  return true;
}

void HtpQuickResponse::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  stop_cv_.notify_all();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void HtpQuickResponse::FreeContext() {
  if (context_handle_ == nullptr) {
    return;
  }
  Qnn_ErrorHandle_t error = api_->contextFree(context_handle_, nullptr);
  if (error != QNN_SUCCESS) {
    QNN_LOG_WARNING("HTP quick response failed to free context. Error %d",
                    QNN_GET_ERROR_CODE(error));
  }
  context_handle_ = nullptr;
  graph_handle_ = nullptr;
}

} // namespace qnn

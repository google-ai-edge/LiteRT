// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/qnn_backend_test/qnn_backend_creator.h"

#include <string_view>

#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/qnn_manager.h"

namespace litert::qnn {

QnnBackendCreator::QnnBackendCreator(::qnn::Options options,
                                     std::string_view soc_model_name) {
  auto configs = QnnManager::DefaultBackendConfigs();
  auto context_configs = QnnManager::DefaultContextConfigs();
  qnn_manager_ = QnnManager::Create(configs, options, std::nullopt,
                                    ::qnn::FindSocModel(soc_model_name));
  if (!qnn_manager_) {
    throw std::runtime_error("Failed to create QnnManager");
  }
  context_handle_ = (**qnn_manager_).CreateContextHandle(context_configs);
}
}  // namespace litert::qnn

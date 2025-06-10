// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_BACKEND_TEST_TEST_UTILS_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_BACKEND_TEST_TEST_UTILS_H_

#include <gtest/gtest.h>

#include <string_view>

#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/utils/qnn_model.h"
#include "litert/vendors/qualcomm/qnn_manager.h"

namespace litert::qnn {

class QnnModelTest : public testing::Test {
 protected:
  QnnManager::Ptr qnn_manager_ptr_;
  QnnManager::ContextHandle context_handle_;
  ::qnn::QnnModel qnn_model_;
  ::qnn::TensorPool tensor_pool_;

  void SetUpQnnModel(const ::qnn::Options& options,
                     std::string_view soc_model_name) {
    auto configs = QnnManager::DefaultBackendConfigs();
    auto qnn_manager = QnnManager::Create(configs, options, std::nullopt,
                                          ::qnn::FindSocModel(soc_model_name));
    ASSERT_TRUE(qnn_manager) << "Failed to create QnnManager";
    auto context_configs = QnnManager::DefaultContextConfigs();
    auto context_handle = (**qnn_manager).CreateContextHandle(context_configs);
    ASSERT_TRUE(context_handle) << "Failed to create Context Handle";

    std::swap(qnn_manager_ptr_, *qnn_manager);
    context_handle_ = std::move(context_handle.Value());

    auto qnn_model =
        ::qnn::QnnModel(qnn_manager_ptr_->BackendHandle(),
                        qnn_manager_ptr_->Api(), context_handle_.get());

    std::swap(qnn_model_, qnn_model);
  }
};

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_BACKEND_TEST_TEST_UTILS_H_

// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_MODEL_TEST_QNN_BACKEND_CREATOR_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_MODEL_TEST_QNN_BACKEND_CREATOR_H_

#include <string_view>

#include "QnnCommon.h"  // from @qairt
#include "litert/cc/litert_expected.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/qnn_manager.h"

namespace litert::qnn {
class QnnBackendCreator {
 public:
  QnnBackendCreator(::qnn::Options options, std::string_view soc_model_name);

  const QNN_INTERFACE_VER_TYPE* GetQnnApi() const {
    return (**qnn_manager_).Api();
  }
  const Qnn_BackendHandle_t GetBackendHandle() {
    return (**qnn_manager_).BackendHandle();
  }
  const Qnn_ContextHandle_t GetContextHandle() const {
    return context_handle_->get();
  }

 private:
  Expected<QnnManager::Ptr> qnn_manager_;
  Expected<QnnManager::ContextHandle> context_handle_;
};
}  // namespace litert::qnn
#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_MODEL_TEST_QNN_BACKEND_CREATOR_H_

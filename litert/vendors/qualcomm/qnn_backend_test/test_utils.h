// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_BACKEND_TEST_TEST_UTILS_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_BACKEND_TEST_TEST_UTILS_H_

#include <string_view>

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/utils/qnn_model.h"
#include "litert/vendors/qualcomm/qnn_manager.h"

namespace litert::qnn {
const ::qnn::Options kTestingDefaultQnnOptions{};

std::string QnnTestPrinter(
    const ::testing::TestParamInfo<
        std::tuple<::qnn::Options, std::string_view>>& param_info);

class QnnModelTest : public testing::TestWithParam<
                         std::tuple<::qnn::Options, std::string_view>> {
 protected:
  QnnManager::Ptr qnn_manager_ptr_;
  QnnManager::ContextHandle context_handle_;
  ::qnn::QnnModel qnn_model_;
  ::qnn::TensorPool tensor_pool_;

  void SetUp() override {
    const auto& [options, soc_model_name] = GetParam();
    SetUpQnnModel(options, soc_model_name);
  }

 private:
  void SetUpQnnModel(const ::qnn::Options& options,
                     std::string_view soc_model_name);
};

inline auto GetDefaultQnnModelParams() {
  return ::testing::Combine(::testing::Values(kTestingDefaultQnnOptions),
                            ::testing::Values("SM8650", "SM8750", "SM8850"));
}

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_BACKEND_TEST_TEST_UTILS_H_

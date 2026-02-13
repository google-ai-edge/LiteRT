// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_BACKEND_TEST_TEST_UTILS_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_BACKEND_TEST_TEST_UTILS_H_

#include <string_view>
#include <vector>

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
  QnnManager::Ptr qnn_manager_ptr_{};
  QnnManager::ContextHandle context_handle_{};
  ::qnn::QnnModel qnn_model_{};
  ::qnn::TensorPool tensor_pool_{};
  bool is_fp16_supported_{false};

  void SetUp() override {
    const auto& [options, soc_model_name] = GetParam();
    SetUpQnnModel(options, soc_model_name);
  }

 private:
  void SetUpQnnModel(const ::qnn::Options& options,
                     std::string_view soc_model_name);
};

inline auto GetDefaultQnnModelParams() {
#if defined(__x86_64__) || defined(_M_X64)
  std::vector<std::string_view> socs = {"SM8650", "SM8750", "SM8850"};
#else
  // On device, qnn manager will use online soc for compilation.
  std::vector<std::string_view> socs = {"SOC_UNKNOWN"};
#endif
  return ::testing::Combine(::testing::Values(kTestingDefaultQnnOptions),
                            ::testing::ValuesIn(socs));
}
}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_BACKEND_TEST_TEST_UTILS_H_

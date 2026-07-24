// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_BACKEND_TEST_TEST_UTILS_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_BACKEND_TEST_TEST_UTILS_H_

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>

#include <gtest/gtest.h>
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/qnn_model.h"
#include "litert/vendors/qualcomm/core/utils/test_utils.h"
#include "litert/vendors/qualcomm/qnn_manager.h"

namespace litert::qnn {

inline const ::qnn::Options& GetTestingDefaultQnnOptions() {
  static const absl::NoDestructor<::qnn::Options> options;
  return *options;
}

std::string QnnTestPrinter(
    const ::testing::TestParamInfo<std::tuple<::qnn::Options, const char*>>&
        param_info);

class QnnModelTest
    : public testing::TestWithParam<std::tuple<::qnn::Options, const char*>> {
 protected:
  QnnManager::Ptr qnn_manager_ptr_{};
  std::unique_ptr<::qnn::QnnBackend> qnn_backend_ptr_{};
  QnnManager::ContextHandle context_handle_{};
  ::qnn::QnnModel qnn_model_{};
  ::qnn::TensorPool tensor_pool_{};
  bool is_fp16_supported_{false};

  void SetUp() override {
    const auto& [options, soc_model_name] = GetParam();
    if (!::qnn::IsTestHtpBackend()) {
      GTEST_SKIP() << "Skipping test because targeted backend is not supported";
    }
    SetUpQnnModel(options, soc_model_name);
  }

 private:
  void SetUpQnnModel(const ::qnn::Options& options, const char* soc_model_name);
};

inline auto GetDefaultQnnModelParams() {
#if defined(__x86_64__) || defined(_M_X64)
  static constexpr std::array<const char*, 3> kSocs = {"SM8650", "SM8750",
                                                       "SM8850"};
#else
  // On device, qnn manager will use online soc for compilation.
  static constexpr std::array<const char*, 1> kSocs = {nullptr};
#endif
  return ::testing::Combine(::testing::Values(GetTestingDefaultQnnOptions()),
                            ::testing::ValuesIn(kSocs));
}
}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_BACKEND_TEST_TEST_UTILS_H_

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
    // TODO (chunhsue-qti) get rid of QnnManager and move to core/
    auto qnn_manager = QnnManager::Create(options, std::nullopt,
                                          ::qnn::FindSocModel(soc_model_name));
    ASSERT_TRUE(qnn_manager) << "Failed to create QnnManager";
    auto context_configs = QnnManager::DefaultContextConfigs();
    auto context_handle =
        (**qnn_manager)
            .CreateContextHandle(context_configs, options.GetProfiling());
    ASSERT_TRUE(context_handle) << "Failed to create Context Handle";

    std::swap(qnn_manager_ptr_, *qnn_manager);
    context_handle_ = std::move(context_handle.Value());

    auto qnn_model =
        ::qnn::QnnModel(qnn_manager_ptr_->BackendHandle(),
                        qnn_manager_ptr_->Api(), context_handle_.get());

    std::swap(qnn_model_, qnn_model);
  }
};

void ConvertDataFromInt8ToInt2(const std::vector<std::int8_t>& src,
                               std::vector<std::uint8_t>& dst) {
  // The source vector size must be a multiple of 4.
  assert(src.size() % 4 == 0);

  dst.clear();
  dst.reserve(src.size() / 4);

  // Process the source vector in chunks of 4.
  for (size_t i = 0; i < src.size(); i += 4) {
    // Mask each int8_t to get its 2-bit representation, discarding sign bits.
    // Mask: 0000 0011
    std::uint8_t num1 = src[i] & 0x03;
    std::uint8_t num2 = src[i + 1] & 0x03;
    std::uint8_t num3 = src[i + 2] & 0x03;
    std::uint8_t num4 = src[i + 3] & 0x03;

    // Combine the four 2-bit numbers into a single byte.
    // num4 is placed in the most significant bits, num1 in the least.
    std::uint8_t byte = num1 | (num2 << 2) | (num3 << 4) | (num4 << 6);
    dst.push_back(byte);
  }
}
}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_BACKEND_TEST_TEST_UTILS_H_

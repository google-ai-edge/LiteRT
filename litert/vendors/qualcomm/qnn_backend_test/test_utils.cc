// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/utils/qnn_model.h"
#include "litert/vendors/qualcomm/qnn_manager.h"
namespace litert::qnn {

std::string QnnTestPrinter(
    const ::testing::TestParamInfo<
        std::tuple<::qnn::Options, std::string_view>>& param_info) {
  std::stringstream ss;
  ss << param_info.index << "_";

  const auto& [options, soc_model_name] = param_info.param;
  ss << soc_model_name << "_";

  // TODO (chunhsue-qti): Add more backend once it expands.
  switch (options.GetBackendType()) {
    case ::qnn::BackendType::kHtpBackend:
      ss << "Htp";
      break;
    case ::qnn::BackendType::kIrBackend:
      ss << "Ir";
      break;
    default:
      ss << "UndefinedBackend";
      break;
  }

  return ss.str();
}

void QnnModelTest::SetUpQnnModel(const ::qnn::Options& options,
                                 std::string_view soc_model_name) {
  // TODO (chunhsue-qti) get rid of QnnManager and move to core/
  auto qnn_manager = QnnManager::Create(options, std::nullopt,
                                        ::qnn::FindSocModel(soc_model_name));
  ASSERT_TRUE(qnn_manager) << qnn_manager.Error();
  auto context_configs = QnnManager::DefaultContextConfigs();
  auto context_handle =
      (**qnn_manager)
          .CreateContextHandle(context_configs, options.GetProfiling());
  ASSERT_TRUE(context_handle) << context_handle.Error();

  std::swap(qnn_manager_ptr_, *qnn_manager);
  context_handle_ = std::move(context_handle.Value());

  auto qnn_model =
      ::qnn::QnnModel(qnn_manager_ptr_->BackendHandle(),
                      qnn_manager_ptr_->Api(), context_handle_.get());

  std::swap(qnn_model_, qnn_model);
}

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
    dst.emplace_back(byte);
  }
}
}  // namespace litert::qnn

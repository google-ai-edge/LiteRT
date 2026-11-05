// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include "HTP/QnnHtpDevice.h"  // from @qairt
// QAIRT >= 2.47 moves some HTP device definitions from QnnHtpDevice.h to
// QnnHtpDeviceConfigShared.h.
#if __has_include("HTP/QnnHtpDeviceConfigShared.h")
#include "HTP/QnnHtpDeviceConfigShared.h"  // from @qairt
#endif
#include "QnnCommon.h"  // from @qairt
#include "QnnDevice.h"  // from @qairt
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/common.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/test_utils.h"
#include "litert/vendors/qualcomm/qnn_manager.h"

namespace litert::qnn {
namespace {

struct PlatformInfoDeleter {
  const QnnApi* api;

  void operator()(const QnnDevice_PlatformInfo_t* platform_info) const {
    if (platform_info != nullptr) {
      api->deviceFreePlatformInfo(nullptr, platform_info);
    }
  }
};

std::string GetSocNames(std::uint32_t soc_model) {
  std::string soc_names;
  for (const auto& soc_info : ::qnn::kSocInfos) {
    if (soc_info.soc_model == soc_model) {
      if (!soc_names.empty()) {
        soc_names += "/";
      }
      soc_names += std::string(soc_info.soc_name);
    }
  }
  if (soc_names.empty()) {
    return "UNKNOWN_SOC";
  }
  return soc_names;
}

TEST(HtpPlatformInfoTest, PrintsSocModelFromDevicePlatformInfo) {
  if (!::qnn::IsTestHtpBackend()) {
    GTEST_SKIP() << "Skipping test because targeted backend is not HTP";
  }

  ::qnn::Options options;
  options.SetBackendType(::qnn::BackendType::kHtpBackend);

  auto qnn_manager = QnnManager::Create(options);
  if (!qnn_manager) {
    GTEST_SKIP() << "Skipping test because QNN manager could not be created "
                    "(QNN libraries may be missing): "
                 << qnn_manager.Error();
  }

  const QnnApi* api = (**qnn_manager).Api();
  ASSERT_NE(api, nullptr);
  ASSERT_NE(api->deviceGetPlatformInfo, nullptr);
  ASSERT_NE(api->deviceFreePlatformInfo, nullptr);

  const QnnDevice_PlatformInfo_t* platform_info_raw = nullptr;
  const Qnn_ErrorHandle_t status =
      api->deviceGetPlatformInfo(nullptr, &platform_info_raw);
  if (status == QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
    GTEST_SKIP() << "QNN HTP deviceGetPlatformInfo is unsupported";
  }
  ASSERT_EQ(status, QNN_SUCCESS);
  ASSERT_NE(platform_info_raw, nullptr);

  std::unique_ptr<const QnnDevice_PlatformInfo_t, PlatformInfoDeleter>
      platform_info(platform_info_raw, PlatformInfoDeleter{api});

  ASSERT_EQ(platform_info->version, QNN_DEVICE_PLATFORM_INFO_VERSION_1);
  ASSERT_NE(platform_info->v1.hwDevices, nullptr);
  ASSERT_GT(platform_info->v1.numHwDevices, 0u);

  bool found_on_chip_htp_device = false;
  for (std::uint32_t i = 0; i < platform_info->v1.numHwDevices; ++i) {
    const auto& hw_device = platform_info->v1.hwDevices[i];
    if (hw_device.version != QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1 ||
        hw_device.v1.deviceInfoExtension == nullptr) {
      continue;
    }

    const auto* htp_info =
        static_cast<const QnnHtpDevice_DeviceInfoExtension_t*>(
            hw_device.v1.deviceInfoExtension);
    if (htp_info->devType != QNN_HTP_DEVICE_TYPE_ON_CHIP) {
      continue;
    }

    found_on_chip_htp_device = true;
    const auto& on_chip_device = htp_info->onChipDevice;
    std::cout << "HTP device_id=" << hw_device.v1.deviceId
              << " SoC model: " << GetSocNames(on_chip_device.socModel)
              << " for QNN socModel " << on_chip_device.socModel
              << ", HTP arch: V"
              << static_cast<std::uint32_t>(on_chip_device.arch) << "\n";
  }

  EXPECT_TRUE(found_on_chip_htp_device)
      << "No on-chip HTP device was reported by deviceGetPlatformInfo";
}

}  // namespace
}  // namespace litert::qnn

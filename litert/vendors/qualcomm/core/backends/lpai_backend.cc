// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/lpai_backend.h"

#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/graph_config_builder.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "LPAI/QnnLpaiBackend.h"  // from @qairt
#include "LPAI/QnnLpaiGraph.h"  // from @qairt
#include "LPAI/QnnLpaiGraphPrepare.h"  // from @qairt
#include "QnnBackend.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt

namespace qnn {
namespace {

// Cores allowed for the prepare config. Hardcoded until the SDK exposes a
// high-level config for it.
constexpr char kDefaultCoreSelection[] = "0,1";

QnnLpaiBackend_HwVersion_t ToQnnHwVersion(LpaiHardwareVersion version) {
  switch (version) {
    case LpaiHardwareVersion::kV5:
      return QNN_LPAI_BACKEND_HW_VERSION_V5;
    case LpaiHardwareVersion::kV6:
      return QNN_LPAI_BACKEND_HW_VERSION_V6;
    case LpaiHardwareVersion::kUnknown:
      return QNN_LPAI_BACKEND_HW_VERSION_UNKNOWN;
  }
  return QNN_LPAI_BACKEND_HW_VERSION_UNKNOWN;
}

QnnLpaiBackend_Target_t ToQnnTarget(LpaiTarget target_env) {
  switch (target_env) {
    case LpaiTarget::kX86:
      return QNN_LPAI_BACKEND_TARGET_X86;
    case LpaiTarget::kArm:
      return QNN_LPAI_BACKEND_TARGET_ARM;
    case LpaiTarget::kAdsp:
      return QNN_LPAI_BACKEND_TARGET_ADSP;
    case LpaiTarget::kTensilica:
      return QNN_LPAI_BACKEND_TARGET_TENSILICA;
  }
  return QNN_LPAI_BACKEND_TARGET_ADSP;
}

QnnLpaiGraph_ClientPerfType_t ToQnnClientPerfType(LpaiClientPerfType type) {
  switch (type) {
    case LpaiClientPerfType::kRealTime:
      return QNN_LPAI_GRAPH_CLIENT_PERF_TYPE_REAL_TIME;
    case LpaiClientPerfType::kNonRealTime:
      return QNN_LPAI_GRAPH_CLIENT_PERF_TYPE_NON_REAL_TIME;
    case LpaiClientPerfType::kDefault:
      return QNN_LPAI_GRAPH_CLIENT_PERF_TYPE_REAL_TIME;
  }
  return QNN_LPAI_GRAPH_CLIENT_PERF_TYPE_REAL_TIME;
}

QnnLpaiGraph_CoreAffinityType_t ToQnnCoreAffinity(
    LpaiCoreAffinityType affinity) {
  switch (affinity) {
    case LpaiCoreAffinityType::kSoft:
      return QNN_LPAI_GRAPH_CORE_AFFINITY_SOFT;
    case LpaiCoreAffinityType::kHard:
      return QNN_LPAI_GRAPH_CORE_AFFINITY_HARD;
    case LpaiCoreAffinityType::kDefault:
      return QNN_LPAI_GRAPH_CORE_AFFINITY_SOFT;
  }
  return QNN_LPAI_GRAPH_CORE_AFFINITY_SOFT;
}

}  // namespace

LpaiBackend::LpaiBackend(const QNN_INTERFACE_VER_TYPE* qnn_api)
    : QnnBackend(qnn_api) {}

bool LpaiBackend::Init(const Options& options,
                       [[maybe_unused]] std::optional<SocInfo> soc_info) {
  QNN_LOG_WARNING(
      "LPAI backend: the QAIRT SDK will generate an .eaix file regardless of "
      "the requested output. This is a known limitation and will be fixed in a "
      "future version.");

  // Log Handle.
  auto local_log_handle = CreateLogHandle(options.GetLogLevel());
  if (!local_log_handle && options.GetLogLevel() != LogLevel::kOff) {
    QNN_LOG_ERROR("Failed to create log handle.");
    return false;
  }

  std::vector<const QnnBackend_Config_t*> backend_configs;
#if defined(__ANDROID__)
  QNN_LOG_INFO(
      "On-device LPAI backend: ignoring lpai_target option (%d) and "
      "SoC-table lpai_hw_version; the LPAI driver uses the silicon it runs on.",
      static_cast<int>(options.GetLpaiTarget()));
#else
  LpaiHardwareVersion hw_version = LpaiHardwareVersion::kUnknown;
  if (soc_info.has_value()) {
    soc_info_ = *soc_info;
    hw_version = soc_info_.lpai_hw_version;
  }
  if (hw_version == LpaiHardwareVersion::kUnknown) {
    QNN_LOG_ERROR(
        "No LPAI hardware version could be resolved from the SoC table; cannot "
        "initialize the LPAI backend.");
    return false;
  }

  // Backend HW-info custom config.
  auto& hw_info = lpai_hw_infos_.emplace_back();
  hw_info.hwVersion = ToQnnHwVersion(hw_version);
  hw_info.lpaiTarget = ToQnnTarget(options.GetLpaiTarget());

  auto& custom_config = lpai_backend_custom_configs_.emplace_back();
  custom_config.option = QNN_LPAI_BACKEND_CUSTOM_CFG_HW_INFO;
  custom_config.config = &hw_info;

  auto& backend_config = AllocateBackendConfig();
  backend_config.option = QNN_BACKEND_CONFIG_OPTION_CUSTOM;
  backend_config.customConfig = &custom_config;

  backend_configs.emplace_back(&backend_config);
#endif  // defined(__ANDROID__)
  backend_configs.emplace_back(nullptr);

  auto local_backend_handle = CreateBackendHandle(
      local_log_handle.get(), absl::MakeSpan(backend_configs));
  if (!local_backend_handle) {
    QNN_LOG_ERROR("Failed to create backend handle.");
    return false;
  }

  // LPAI does not use a QNN device handle: QnnDevice_create is not supported by
  // the driver and returns an error.

  // Follow RAII pattern to manage handles.
  log_handle_ = std::move(local_log_handle);
  backend_handle_ = std::move(local_backend_handle);

  return true;
}

GraphConfigBuilder LpaiBackend::BuildGraphConfigs(
    const Options& options, absl::string_view /*qnn_graph_name*/) {
  GraphConfigBuilder builder;
#if (QNN_API_VERSION_MAJOR > 2) || \
    (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 29)
  auto& prepare = builder.Store<QnnLpaiGraph_CustomConfigPrepare_t>(
      QNN_LPAI_GRAPH_CUSTOM_CONFIG_PREPARE_INIT);
  prepare.enableCoreSelection = const_cast<char*>(kDefaultCoreSelection);

  QnnLpaiGraph_CustomConfig_t custom_config = QNN_LPAI_GRAPH_CUSTOM_CONFIG_INIT;
  custom_config.option = QNN_LPAI_GRAPH_SET_CFG_PREPARE;
  custom_config.config = &prepare;
  builder.AddCustomConfig(custom_config);
#endif
  return builder;
}

bool LpaiBackend::ConfigureGraphAfterRetrieve(const GraphConfigContext& ctx,
                                              const Options& options) {
  GraphConfigBuilder builder;

  // Performance config.
  auto& perf_cfg =
      builder.Store<QnnLpaiGraph_PerfCfg_t>(QNN_LPAI_GRAPH_PERF_CFG_INIT);
  perf_cfg.fps = options.GetLpaiFps();
  perf_cfg.ftrtRatio = options.GetLpaiFtrtRatio();
  if (options.GetLpaiClientPerfType() != LpaiClientPerfType::kDefault) {
    perf_cfg.clientType = ToQnnClientPerfType(options.GetLpaiClientPerfType());
  }

  QnnLpaiGraph_CustomConfig_t perf_custom_config = QNN_LPAI_GRAPH_CUSTOM_CONFIG_INIT;
  perf_custom_config.option = QNN_LPAI_GRAPH_SET_CFG_PERF_CFG;
  perf_custom_config.config = &perf_cfg;
  builder.AddCustomConfig(perf_custom_config);

  // Core affinity config.
  auto& core_affinity =
      builder.Store<QnnLpaiGraph_CoreAffinity_t>(QNN_LPAI_GRAPH_CORE_AFFINITY_INIT);
  if (options.GetLpaiCoreAffinityType() != LpaiCoreAffinityType::kDefault) {
    core_affinity.affinity =
        ToQnnCoreAffinity(options.GetLpaiCoreAffinityType());
  }
  core_affinity.coreSelection = options.GetLpaiCoreSelection();

  QnnLpaiGraph_CustomConfig_t affinity_custom_config = QNN_LPAI_GRAPH_CUSTOM_CONFIG_INIT;
  affinity_custom_config.option = QNN_LPAI_GRAPH_SET_CFG_CORE_AFFINITY;
  affinity_custom_config.config = &core_affinity;
  builder.AddCustomConfig(affinity_custom_config);

  if (auto status = QnnApi()->graphSetConfig(ctx.graph, builder.GetNullTerminatedConfigs().data());
      status != QNN_SUCCESS) {
    QNN_LOG_ERROR("Failed to set LPAI graph config after retrieve. Error %d",
                  QNN_GET_ERROR_CODE(status));
    return false;
  }

  if (auto status = QnnApi()->graphFinalize(ctx.graph, ctx.profile, nullptr);
      status != QNN_SUCCESS) {
    QNN_LOG_ERROR("Failed to re-finalize LPAI graph after retrieve. Error %d",
                  QNN_GET_ERROR_CODE(status));
    return false;
  }

  return true;
}

}  // namespace qnn

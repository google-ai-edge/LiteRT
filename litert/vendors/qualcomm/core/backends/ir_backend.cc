// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/ir_backend.h"

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/graph_config_builder.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "IR/QnnIrGraph.h"  // from @qairt
#include "QnnBackend.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt

namespace qnn {

IrBackend::IrBackend(const QNN_INTERFACE_VER_TYPE* qnn_api)
    : QnnBackend(qnn_api) {}

bool IrBackend::Init(const Options& options,
                     std::optional<::qnn::SocInfo> soc_info) {
  // Log Handle
  auto local_log_handle = CreateLogHandle(options.GetLogLevel());
  if (!local_log_handle && options.GetLogLevel() != ::qnn::LogLevel::kOff) {
    QNN_LOG_ERROR("Failed to create log handle!");
    return false;
  }

  // Backend Handle
  std::array<const QnnBackend_Config_t*, 1> backend_configs = {nullptr};

  auto local_backend_handle = CreateBackendHandle(
      local_log_handle.get(), absl::MakeSpan(backend_configs));
  if (!local_backend_handle) {
    QNN_LOG_ERROR("Failed to create backend handle!");
    return false;
  }

  log_handle_ = std::move(local_log_handle);
  backend_handle_ = std::move(local_backend_handle);
  return true;
}

GraphConfigBuilder IrBackend::BuildGraphConfigs(
    const Options& options, absl::string_view qnn_graph_name) {
  GraphConfigBuilder mgr;

  // The serialized DLC output path must outlive the graphCreate call, so the
  // builder owns the string the custom config points into.
  std::filesystem::path dlc_dir = std::string(options.GetDlcDir());
  auto& dlc_path = mgr.Store<std::string>(
      (dlc_dir / absl::StrCat(qnn_graph_name, ".dlc")).string());

  auto& custom_config = mgr.AddCustom<QnnIrGraph_CustomConfig_t>();
  custom_config.option = QNN_IR_GRAPH_CONFIG_OPTION_SERIALIZATION;
  custom_config.serializationOption.serializationType =
      QNN_IR_GRAPH_SERIALIZATION_TYPE_FLAT_BUFFER;
  custom_config.serializationOption.outputPath = dlc_path.c_str();

  return mgr;
}
}  // namespace qnn

// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_DSP_BACKEND_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_DSP_BACKEND_H_

#include <list>
#include <memory>
#include <optional>
#include <vector>

#include "QnnDevice.h"     // from @qairt
#include "QnnInterface.h"  // from @qairt
#include "QnnTypes.h"      // from @qairt
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
namespace {
#include "DSP/QnnDspCommon.h"              // from @qairt
#include "DSP/QnnDspDevice.h"              // from @qairt
#include "DSP/QnnDspPerfInfrastructure.h"  // from @qairt
}  // namespace

#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"

namespace qnn {

class DspBackend : public QnnBackend {
 public:
  static const char* GetLibraryName() { return "libQnnDsp.so"; }

  static Qnn_Version_t GetExpectedBackendVersion() {
    Qnn_Version_t backend_version;
    backend_version.major = QNN_DSP_API_VERSION_MAJOR;
    backend_version.minor = QNN_DSP_API_VERSION_MINOR;
    backend_version.patch = QNN_DSP_API_VERSION_PATCH;
    return backend_version;
  }

  explicit DspBackend(const QNN_INTERFACE_VER_TYPE* qnn_api);

  ~DspBackend();

  bool Init(const Options& options, std::optional<SocInfo> soc_info) override;

  struct DspPerfInfraDeleter {
    std::uint32_t power_config_id_ = 0;
    const QnnDspPerfInfrastructure_PowerConfig_t** down_vote_configs_ = nullptr;
    void operator()(QnnDspDevice_Infrastructure_t* infra) const {
      if (infra && power_config_id_ != 0) {
        if (down_vote_configs_) {
          infra->setPowerConfig(power_config_id_, down_vote_configs_);
        }
        infra->destroyPowerConfigId(power_config_id_);
      }
    }
  };

  using DspPerfInfra =
      std::unique_ptr<QnnDspDevice_Infrastructure_t, DspPerfInfraDeleter>;

 private:
  DspPerfInfra CreateDspPerfInfra();

  void PerformanceVote() override;

  inline bool IsPerfModeEnabled() const {
    return performance_mode_ != DspPerformanceMode::kDefault;
  }

  // Performance control
  std::vector<QnnDspPerfInfrastructure_PowerConfig_t> perf_power_configs_;
  std::vector<QnnDspPerfInfrastructure_PowerConfig_t> down_vote_power_configs_;
  std::vector<const QnnDspPerfInfrastructure_PowerConfig_t*>
      perf_power_configs_ptr_;
  std::vector<const QnnDspPerfInfrastructure_PowerConfig_t*>
      down_vote_power_configs_ptr_;
  std::uint32_t powerconfig_client_id_{0};
  PerformanceModeVoteType manual_voting_type_{kNoVote};
  DspPerformanceMode performance_mode_{DspPerformanceMode::kDefault};
  DspPerfInfra dsp_perf_infra_{nullptr, DspPerfInfraDeleter{}};
};

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_DSP_BACKEND_H_

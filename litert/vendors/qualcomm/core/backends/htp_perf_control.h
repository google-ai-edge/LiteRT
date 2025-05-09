// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_PERF_CONTROL_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_PERF_CONTROL_H_

#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "litert/vendors/qualcomm/core/common.h"
#include "HTP/QnnHtpDevice.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt

template <typename T>
std::vector<std::add_pointer_t<std::add_const_t<T>>> ObtainNullTermPtrVector(
    const std::vector<T>& vec) {
  std::vector<std::add_pointer_t<std::add_const_t<T>>> ret(vec.size());
  for (int i = 0; i < vec.size(); ++i) {
    ret[i] = &(vec[i]);
  }
  ret.emplace_back(nullptr);
  return ret;
}

// Defines Qnn performance mode vote types for htp backend
enum PerformanceModeVoteType {
  kNoVote = 0,
  kUpVote = 1,
  kDownVote = 2,
};
class PerfControl {
 public:
  explicit PerfControl(const QNN_INTERFACE_VER_TYPE* api,
                       const ::qnn::HtpPerformanceMode htp_performance_mode);
  PerfControl(const PerfControl&) = delete;
  PerfControl(PerfControl&&) = delete;
  PerfControl& operator=(const PerfControl&) = delete;
  PerfControl& operator=(PerfControl&&) = delete;
  ~PerfControl();

  bool Init(const QnnHtpDevice_Arch_t& arch);
  bool Terminate();
  // Direct vote is only supported in manual mode.
  void PerformanceVote();
  bool CreatePerfPowerConfigPtr(const std::uint32_t power_config_id,
                                const ::qnn::HtpPerformanceMode perf_mode,
                                const PerformanceModeVoteType vote_type);

 private:
  inline bool IsPerfModeEnabled() const {
    return performance_mode_ != ::qnn::HtpPerformanceMode::kDefault;
  }
  const QNN_INTERFACE_VER_TYPE* api_{nullptr};
  struct BackendConfig;
  std::unique_ptr<BackendConfig> backend_config_;
  std::uint32_t powerconfig_client_id_{0};
  PerformanceModeVoteType manual_voting_type_{kNoVote};
  // HTPBackendOptions
  ::qnn::HtpPerformanceMode performance_mode_{
      ::qnn::HtpPerformanceMode::kDefault};
  std::uint32_t device_id_{0};
};

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_PERF_CONTROL_H_

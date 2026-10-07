// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_QUICK_RESPONSE_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_QUICK_RESPONSE_H_

#include <array>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

#include "QnnInterface.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt  // from @qairt

namespace qnn {

class HtpBackend;

// Periodically executes a minimal graph to keep the HTP responsive.
class HtpQuickResponse {
 public:
  static std::unique_ptr<HtpQuickResponse> Create(
      const QNN_INTERFACE_VER_TYPE* api, HtpBackend& backend);

  ~HtpQuickResponse();

  HtpQuickResponse(const HtpQuickResponse&) = delete;
  HtpQuickResponse& operator=(const HtpQuickResponse&) = delete;
  HtpQuickResponse(HtpQuickResponse&&) = delete;
  HtpQuickResponse& operator=(HtpQuickResponse&&) = delete;

 private:
  HtpQuickResponse(const QNN_INTERFACE_VER_TYPE* api, HtpBackend& backend);

  bool Init();

  bool CreateGraph();
  void Run();
  bool ExecuteOnce();
  void Stop();
  void FreeContext();

  const QNN_INTERFACE_VER_TYPE* api_{nullptr};
  HtpBackend& backend_;
  Qnn_ContextHandle_t context_handle_{nullptr};
  Qnn_GraphHandle_t graph_handle_{nullptr};
  std::array<std::uint32_t, 4> tensor_dims_{1, 1, 1, 1};
  static constexpr std::uint32_t kDataBytes = sizeof(std::int8_t);
  std::array<std::int8_t, kDataBytes> input_data_0_{};
  std::array<std::int8_t, kDataBytes> input_data_1_{};
  std::array<std::int8_t, kDataBytes> output_data_{};
  std::array<Qnn_Tensor_t, 2> input_tensors_{};
  std::array<Qnn_Tensor_t, 1> output_tensors_{};
  std::mutex mutex_;
  std::condition_variable stop_cv_;
  bool stop_{false};
  std::thread thread_;
};

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_QUICK_RESPONSE_H_

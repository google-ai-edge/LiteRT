// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_dispatch_delegate.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_dispatch_delegate.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/build_stamp.h"
#include "litert/runtime/dispatch/dispatch_delegate_kernel.h"
#include "litert/runtime/metrics.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "tflite/c/c_api_opaque.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/delegates/utils/simple_opaque_delegate.h"

namespace {

using ::litert::Unexpected;

// A TFL Delegate that can recognize subgraphs that run on Dispatch API capable
// accelerators, e.g. TPU, DSP, ... It replaces such subgraphs and offloads
// their work through the Dispatch API.
class DispatchDelegate : public tflite::SimpleOpaqueDelegateInterface {
 public:
  ~DispatchDelegate() override {
    if (device_context_) {
      (void)LiteRtDispatchDeviceContextDestroy(device_context_);
    }
  }

  static TfLiteOpaqueDelegate* Create(
      LiteRtEnvironmentOptions environment_options, LiteRtOptions options) {
    std::unique_ptr<DispatchDelegate> managed_dispatch_delegate(
        new DispatchDelegate(environment_options, options));
    return tflite::TfLiteOpaqueDelegateFactory::CreateSimpleDelegate(
        std::move(managed_dispatch_delegate),
        kTfLiteDelegateFlagsAllowDynamicTensors);
  }

  bool IsNodeSupportedByDelegate(const TfLiteOperator* op,
                                 const TfLiteOpaqueNode* node,
                                 TfLiteOpaqueContext* context) const override;

  TfLiteStatus Initialize(TfLiteOpaqueContext* context) override;

  const char* Name() const override;

  std::unique_ptr<tflite::SimpleOpaqueDelegateKernelInterface>
  CreateDelegateKernelInterface() override;

  litert::Expected<void> StartMetricsCollection(int detail_level);

  litert::Expected<LiteRtMetricsT> StopMetricsCollection();

 private:
  static constexpr absl::string_view kDelegateName = "DispatchDelegate";

  explicit DispatchDelegate(LiteRtEnvironmentOptions environment_options,
                            LiteRtOptions options)
      : environment_options_(environment_options), options_(options) {}

  litert::Expected<void> InitializeDispatchApi();

  LiteRtEnvironmentOptions environment_options_;
  LiteRtOptions options_;
  bool has_dispatch_runtime_ = false;
  int dispatch_graph_name_id_ = 0;
  std::vector<litert::internal::DispatchDelegateKernel*> kernels_;
  LiteRtDispatchDeviceContext device_context_ = nullptr;
};

bool DispatchDelegate::IsNodeSupportedByDelegate(
    const TfLiteOperator* op, const TfLiteOpaqueNode* node,
    TfLiteOpaqueContext* context) const {
  const char* custom_name = TfLiteOperatorGetCustomName(op);
  return custom_name &&
         absl::StrContains(custom_name,
                           ::litert::internal::kLiteRtDispatchOpCustomName);
}

TfLiteStatus DispatchDelegate::Initialize(TfLiteOpaqueContext* context) {
  if (auto status = InitializeDispatchApi(); !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to initialize Dispatch API: %s",
               status.Error().Message().c_str());
    has_dispatch_runtime_ = false;
  } else {
    has_dispatch_runtime_ = true;
  }
  return kTfLiteOk;
}

const char* DispatchDelegate::Name() const { return kDelegateName.data(); }

std::unique_ptr<tflite::SimpleOpaqueDelegateKernelInterface>
DispatchDelegate::CreateDelegateKernelInterface() {
  if (!has_dispatch_runtime_) {
    LITERT_FATAL(
        "Failed to create a dispatch delegate kernel: No usable Dispatch "
        "runtime found");
    return nullptr;
  }

  std::string dispatch_graph_name =
      absl::StrFormat("DispatchGraph_%d", dispatch_graph_name_id_++);

  auto kernel = litert::internal::DispatchDelegateKernel::Create(
      std::move(dispatch_graph_name), environment_options_, options_,
      device_context_);
  if (kernel) {
    auto* kernel_ptr =
        dynamic_cast<typename litert::internal::DispatchDelegateKernel*>(
            kernel->get());
    kernels_.push_back(kernel_ptr);
    return std::move(*kernel);
  } else {
    LITERT_FATAL("Failed to create a dispatch delegate kernel: %s",
                 kernel.Error().Message().c_str());
    return nullptr;
  }
}

litert::Expected<void> DispatchDelegate::StartMetricsCollection(
    int detail_level) {
  for (auto* kernel : kernels_) {
    LITERT_RETURN_IF_ERROR(kernel->StartMetricsCollection(detail_level));
  }
  return {};
}

litert::Expected<LiteRtMetricsT> DispatchDelegate::StopMetricsCollection() {
  // TODO: b/406154325 - Combine metrics of same type from different kernels.
  std::vector<LiteRtMetricsT::Metric> metrics;
  for (auto* kernel : kernels_) {
    LITERT_ASSIGN_OR_RETURN(auto kernel_metrics,
                            kernel->StopMetricsCollection());
    metrics.insert(metrics.end(),
                   std::make_move_iterator(kernel_metrics.metrics.begin()),
                   std::make_move_iterator(kernel_metrics.metrics.end()));
  }
  return LiteRtMetricsT{.metrics = std::move(metrics)};
}

litert::Expected<void> DispatchDelegate::InitializeDispatchApi() {
  LITERT_RETURN_IF_ERROR(
      LiteRtDispatchInitialize(environment_options_, options_));

  const char* vendor_id;
  LITERT_RETURN_IF_ERROR(LiteRtDispatchGetVendorId(&vendor_id));
  LITERT_LOG(LITERT_INFO, "Dispatch API vendor ID: %s", vendor_id);

  const char* build_id;
  LITERT_RETURN_IF_ERROR(LiteRtDispatchGetBuildId(&build_id));
  LITERT_LOG(LITERT_INFO, "Dispatch API build ID: %s", build_id);

  LiteRtApiVersion api_version;
  LITERT_RETURN_IF_ERROR(LiteRtDispatchGetApiVersion(&api_version));
  LITERT_LOG(LITERT_INFO, "Dispatch API version: %d.%d.%d", api_version.major,
             api_version.minor, api_version.patch);

  // Check if the versions mach.
  if (api_version.major != LITERT_API_VERSION_MAJOR ||
      api_version.minor < LITERT_API_VERSION_MINOR) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Found Dispatch API with an unsupported version");
  }

  int capabilities;
  LITERT_RETURN_IF_ERROR(LiteRtDispatchGetCapabilities(&capabilities));
  LITERT_LOG(LITERT_INFO, "Dispatch API capabilities: %d", capabilities);

  if (!(capabilities & kLiteRtDispatchCapabilitiesBasic)) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("Dispatch API has insufficient capabilities: %d",
                        capabilities));
  }

  LITERT_RETURN_IF_ERROR(LiteRtDispatchDeviceContextCreate(&device_context_));

  return {};
}

}  // namespace

TfLiteOpaqueDelegate* LiteRtCreateDispatchDelegate(
    LiteRtEnvironmentOptions environment_options, LiteRtOptions options) {
  return DispatchDelegate::Create(environment_options, options);
}

void LiteRtDestroyDispatchDelegate(TfLiteOpaqueDelegate* delegate) {
  tflite::TfLiteOpaqueDelegateFactory::DeleteSimpleDelegate(delegate);
}

LiteRtStatus LiteRtDispatchDelegateStartMetricsCollection(
    TfLiteOpaqueDelegate* delegate, int detail_level) {
  if (!delegate) return kLiteRtStatusErrorInvalidArgument;
  auto* dispatch_delegate = reinterpret_cast<DispatchDelegate*>(
      TfLiteOpaqueDelegateGetData(delegate));
  LITERT_RETURN_IF_ERROR(
      dispatch_delegate->StartMetricsCollection(detail_level));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchDelegateStopMetricsCollection(
    TfLiteOpaqueDelegate* delegate, LiteRtMetrics metrics) {
  if (!delegate) return kLiteRtStatusErrorInvalidArgument;
  auto* dispatch_delegate = reinterpret_cast<DispatchDelegate*>(
      TfLiteOpaqueDelegateGetData(delegate));
  LITERT_ASSIGN_OR_RETURN(auto liter_metrics,
                          dispatch_delegate->StopMetricsCollection());
  *metrics = std::move(liter_metrics);
  return kLiteRtStatusOk;
}

namespace litert {

DispatchDelegatePtr CreateDispatchDelegatePtr(
    LiteRtEnvironmentOptions environment_options, LiteRtOptions options) {
  return DispatchDelegatePtr(
      LiteRtCreateDispatchDelegate(environment_options, options),
      LiteRtDestroyDispatchDelegate);
}

}  // namespace litert

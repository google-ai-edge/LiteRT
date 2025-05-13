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

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_dispatch_delegate.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_metrics.h"
#include "litert/cc/litert_dispatch_delegate.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/build_stamp.h"
#include "litert/runtime/dispatch/dispatch_delegate_kernel.h"
#include "litert/runtime/dispatch/dispatch_delegate_options.h"
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

  static TfLiteOpaqueDelegate* Create(LiteRtDispatchDelegateOptions* options_) {
    litert::DispatchDelegateOptionsPtr options(
        options_, LiteRtDestroyDispatchDelegateOptions);
    if (!options) {
      LITERT_LOG(LITERT_ERROR, "Null input");
      return nullptr;
    }

    std::unique_ptr<DispatchDelegate> managed_dispatch_delegate(
        new DispatchDelegate(std::move(options)));
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

  explicit DispatchDelegate(litert::DispatchDelegateOptionsPtr&& options)
      : options_(std::move(options)) {}

  litert::Expected<void> InitializeDispatchApi();

  litert::DispatchDelegateOptionsPtr options_;
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
      std::move(dispatch_graph_name), *options_, device_context_);
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
  auto dispatch_options = options_->GetDispatchOptions();
  if (auto status = LiteRtDispatchInitialize(dispatch_options.data(),
                                             dispatch_options.size());
      status != kLiteRtStatusOk) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("Failed to initialize Dispatch API: %d", status));
  }

  const char* vendor_id;
  if (auto status = LiteRtDispatchGetVendorId(&vendor_id);
      status != kLiteRtStatusOk) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("Failed to get Dispatch API vendor ID: %d", status));
  }
  LITERT_LOG(LITERT_INFO, "Dispatch API vendor ID: %s", vendor_id);

  const char* build_id;
  if (auto status = LiteRtDispatchGetBuildId(&build_id);
      status != kLiteRtStatusOk) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("Failed to get Dispatch API build ID: %d", status));
  }
  LITERT_LOG(LITERT_INFO, "Dispatch API build ID: %s", build_id);

  LiteRtApiVersion api_version;
  if (auto status = LiteRtDispatchGetApiVersion(&api_version);
      status != kLiteRtStatusOk) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("Failed to get LiteRT Dispatch API version: %d",
                        status));
  }
  LITERT_LOG(LITERT_INFO, "Dispatch API version: %d.%d.%d", api_version.major,
             api_version.minor, api_version.patch);

  // Check if the versions mach.
  if (api_version.major != LITERT_API_VERSION_MAJOR ||
      api_version.minor < LITERT_API_VERSION_MINOR) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Found Dispatch API with an unsupported version");
  }

  int capabilities;
  if (auto status = LiteRtDispatchGetCapabilities(&capabilities);
      status != kLiteRtStatusOk) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("Failed to get Dispatch API capabilities: %d", status));
  }
  LITERT_LOG(LITERT_INFO, "Dispatch API capabilities: %d", capabilities);

  if (!(capabilities & kLiteRtDispatchCapabilitiesBasic)) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("Dispatch API has insufficient capabilities: %d",
                        capabilities));
  }

  if (auto status = LiteRtDispatchDeviceContextCreate(&device_context_);
      status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to get Dispatch API device context: %d",
               status);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create Dispatch API device context");
  }

  return {};
}

}  // namespace

LiteRtDispatchDelegateOptions* LiteRtCreateDefaultDispatchDelegateOptions(
    LiteRtEnvironmentOptions environment_options) {
  return new LiteRtDispatchDelegateOptions(environment_options);
}

TfLiteStatus LiteRtAddDispatchDelegateOption(
    LiteRtDispatchDelegateOptions* options, LiteRtDispatchOption option) {
  if (!options) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kTfLiteError;
  }

  options->AddOption(option);
  return kTfLiteOk;
}

TfLiteStatus LiteRtDispatchDelegateAddAllocBaseOption(
    LiteRtDispatchDelegateOptions* options, const void* alloc_base) {
  AddAllocBaseOption(alloc_base, *options);
  return kTfLiteOk;
}

TfLiteStatus LiteRtDispatchDelegateAddAllocFdOption(
    LiteRtDispatchDelegateOptions* options, int alloc_fd) {
  AddAllocFdOption(alloc_fd, *options);
  return kTfLiteOk;
}

void LiteRtDestroyDispatchDelegateOptions(
    LiteRtDispatchDelegateOptions* options) {
  delete options;
}

TfLiteOpaqueDelegate* LiteRtCreateDispatchDelegate(
    LiteRtEnvironmentOptions environment_options,
    LiteRtDispatchDelegateOptions* options) {
  if (!options) {
    options = LiteRtCreateDefaultDispatchDelegateOptions(environment_options);
  }
  return DispatchDelegate::Create(options);
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

DispatchDelegateOptionsPtr CreateDispatchDelegateOptionsPtr(
    LiteRtEnvironmentOptions environment_options) {
  return {LiteRtCreateDefaultDispatchDelegateOptions(environment_options),
          LiteRtDestroyDispatchDelegateOptions};
}

DispatchDelegatePtr CreateDispatchDelegatePtr(
    LiteRtEnvironmentOptions environment_options,
    DispatchDelegateOptionsPtr&& options) {
  return DispatchDelegatePtr(
      LiteRtCreateDispatchDelegate(environment_options, options.release()),
      LiteRtDestroyDispatchDelegate);
}
}  // namespace litert

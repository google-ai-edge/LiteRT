// Copyright 2025 Google LLC.
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

#include "litert/runtime/accelerators/dispatch/dispatch_accelerator.h"

#include <memory>

#include "litert/c/litert_accelerator_registration.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_dispatch_delegate.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_dispatch_delegate.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/environment.h"
#include "litert/runtime/accelerators/accelerator_implementation_helper.h"
#include "tflite/c/c_api_types.h"

namespace litert {
namespace {

constexpr const char kNpuAcceleratorName[] = "NpuAccelerator";

struct NpuAcceleratorVersion {
  static constexpr int kMajor = 1;
  static constexpr int kMinor = 0;
  static constexpr int kPatch = 0;
  static constexpr LiteRtApiVersion version = {kMajor, kMinor, kPatch};  // NOLINT
};

}  // namespace

class NpuAccelerator final
    : public internal::AcceleratorImplementationHelper<
          NpuAccelerator, kNpuAcceleratorName, NpuAcceleratorVersion,
          kLiteRtHwAcceleratorNpu> {
 public:
  NpuAccelerator() = default;

  static Expected<Ptr> Create() { return Allocate(); }

  static LiteRtStatus CreateDelegate(LiteRtAccelerator accelerator,
                                     LiteRtOptions options, void** delegate) {
    LITERT_ENSURE(delegate != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Delegate pointer is null.");
    LITERT_ENSURE(accelerator != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Accelerator handle is invalid.");
    LITERT_ENSURE(accelerator->env != nullptr,
                  kLiteRtStatusErrorInvalidArgument,
                  "Accelerator is not registered to an environment.");

    auto dispatch_delegate = litert::CreateDispatchDelegatePtr(
        &accelerator->env->GetOptions(), options);
    LITERT_ENSURE(dispatch_delegate != nullptr,
                  kLiteRtStatusErrorRuntimeFailure,
                  "Dispatch delegate failed to be created.");

    *delegate = dispatch_delegate.release();
    return kLiteRtStatusOk;
  }

  // Starts collection of HW-specific metrics at a specific level of detail.
  static LiteRtStatus StartMetricsCollection(void* delegate, int detail_level) {
    LITERT_ENSURE(delegate != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Delegate pointer is null.");
    LITERT_ENSURE(detail_level >= 0, kLiteRtStatusErrorInvalidArgument,
                  "Detail level must be >= 0.");
    LITERT_LOG(LITERT_INFO, "Dispatch delegate started metrics collection.");
    return LiteRtDispatchDelegateStartMetricsCollection(
        reinterpret_cast<TfLiteOpaqueDelegate*>(delegate), detail_level);
  }

  // Stops collection of HW-specific metrics and report the collected metrics.
  static LiteRtStatus StopMetricsCollection(void* delegate,
                                            LiteRtMetrics metrics) {
    LITERT_ENSURE(delegate != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Delegate pointer is null.");
    LITERT_ENSURE(metrics != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Metrics pointer is null.");
    LITERT_LOG(LITERT_INFO, "Dispatch delegate stopped metrics collection.");
    return LiteRtDispatchDelegateStopMetricsCollection(
        reinterpret_cast<TfLiteOpaqueDelegate*>(delegate), metrics);
  }

  static void DestroyDelegate(void* delegate) {
    LiteRtDestroyDispatchDelegate(
        reinterpret_cast<TfLiteOpaqueDelegate*>(delegate));
  }
};

}  // namespace litert

extern "C" {

LiteRtStatus LiteRtRegisterNpuAccelerator(LiteRtEnvironment environment) {
  LITERT_ENSURE(environment != nullptr, kLiteRtStatusErrorInvalidArgument,
                "environment handle is null");
  LITERT_ENSURE(
      environment->GetOption(kLiteRtEnvOptionTagDispatchLibraryDir).has_value(),
      kLiteRtStatusErrorInvalidArgument,
      "Dispatch library directory is not set.");

  LiteRtAccelerator accelerator_handle;
  LITERT_RETURN_IF_ERROR(LiteRtCreateAccelerator(&accelerator_handle));
  litert::internal::AcceleratorGuard accelerator(accelerator_handle);

  LITERT_RETURN_IF_ERROR(litert::internal::SetAcceleratorBoilerplateFunctions<
                         litert::NpuAccelerator>(accelerator));
  LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorStartMetricsCollection(
      accelerator.get(), litert::NpuAccelerator::StartMetricsCollection));
  LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorStopMetricsCollection(
      accelerator.get(), litert::NpuAccelerator::StopMetricsCollection));

  LITERT_ASSIGN_OR_RETURN(auto accelerator_impl,
                          litert::NpuAccelerator::Create());

  LITERT_RETURN_IF_ERROR(LiteRtRegisterAccelerator(
      environment, accelerator.release(), accelerator_impl.release(),
      litert::NpuAccelerator::Destroy));

  return kLiteRtStatusOk;
}

}  // extern "C"

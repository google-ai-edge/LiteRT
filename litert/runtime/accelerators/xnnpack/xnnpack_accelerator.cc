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

#include "litert/runtime/accelerators/xnnpack/xnnpack_accelerator.h"

#include <memory>

#include "litert/c/litert_accelerator.h"
#include "litert/c/litert_accelerator_compilation_options.h"
#include "litert/c/litert_accelerator_registration.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/environment.h"
#include "litert/runtime/accelerators/accelerator_implementation_helper.h"
#include "tensorflow/lite/c/c_api_types.h"  // from @org_tensorflow
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"  // from @org_tensorflow

namespace litert {

namespace {
constexpr const char kCpuAcceleratorName[] = "CpuAccelerator";

struct CpuAcceleratorVersion {
  static constexpr int kMajor = 1;
  static constexpr int kMinor = 0;
  static constexpr int kPatch = 0;
  static constexpr LiteRtApiVersion version = {kMajor, kMinor, kPatch};  // NOLINT
};

class CpuAccelerator final
    : public internal::AcceleratorImplementationHelper<
          CpuAccelerator, kCpuAcceleratorName, CpuAcceleratorVersion,
          kLiteRtHwAcceleratorCpu> {
 public:
  CpuAccelerator() = default;

  static Expected<Ptr> Create() { return Allocate(); }

  // C API

  // Creates a Dispatch delegate instance.
  static LiteRtStatus CreateDelegate(
      LiteRtAccelerator accelerator,
      LiteRtAcceleratorCompilationOptions options, void** delegate) {
    LITERT_ENSURE(delegate != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Delegate pointer is null.");
    LITERT_ENSURE(accelerator != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Accelerator handle is invalid.");
    LITERT_ENSURE(accelerator->env != nullptr,
                  kLiteRtStatusErrorInvalidArgument,
                  "Accelerator is not registered to an environment.");

    // TODO: b/403547017 - Make the CPU accelerator configurable using the
    // compilation options.
    auto xnn_options = TfLiteXNNPackDelegateOptionsDefault();
    *delegate = TfLiteXNNPackDelegateCreate(&xnn_options);

    LITERT_ENSURE(*delegate != nullptr, kLiteRtStatusErrorRuntimeFailure,
                  "XNNPack delegate failed to be created.");
    return kLiteRtStatusOk;
  }

  // Destroys an XNNPack delegate instance.
  static void DestroyDelegate(void* delegate) {
    TfLiteXNNPackDelegateDelete(reinterpret_cast<TfLiteDelegate*>(delegate));
  }
};

}  // namespace
}  // namespace litert

extern "C" {

LiteRtStatus LiteRtRegisterCpuAccelerator(
    LiteRtEnvironmentT* environment, LiteRtCpuAcceleratorOptions* options) {
  LITERT_ENSURE(environment != nullptr, kLiteRtStatusErrorInvalidArgument,
                "accelerator handle is invalid");
  LiteRtAccelerator accelerator_handle;
  LITERT_RETURN_IF_ERROR(LiteRtCreateAccelerator(&accelerator_handle));
  litert::internal::AcceleratorGuard accelerator(accelerator_handle);

  LITERT_RETURN_IF_ERROR(litert::internal::SetAcceleratorBoilerplateFunctions<
                         litert::CpuAccelerator>(accelerator));

  LITERT_ASSIGN_OR_RETURN(auto accelerator_impl,
                          litert::CpuAccelerator::Create());

  LITERT_RETURN_IF_ERROR(LiteRtRegisterAccelerator(
      environment, accelerator.release(), accelerator_impl.release(),
      litert::CpuAccelerator::Destroy));
  return kLiteRtStatusOk;
}

}  // extern "C"

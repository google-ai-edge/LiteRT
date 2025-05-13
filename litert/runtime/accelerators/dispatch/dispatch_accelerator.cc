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
#include <string>
#include <utility>

#include "litert/c/litert_accelerator.h"
#include "litert/c/litert_accelerator_registration.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_dispatch_delegate.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_dispatch_delegate.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/environment.h"
#include "litert/runtime/accelerator_model_compilation_data.h"
#include "litert/runtime/accelerators/accelerator_implementation_helper.h"
#include "litert/runtime/metrics.h"
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
  explicit NpuAccelerator(std::string library_folder)
      : library_folder_(std::move(library_folder)) {}

  static Expected<Ptr> Create(std::string library_folder) {
    LITERT_RETURN_IF_ERROR(
        !library_folder.empty(),
        Error(kLiteRtStatusErrorInvalidArgument,
              "Dispatch API implementation library folder was not specified."));
    return Allocate(std::move(library_folder));
  }

  // Creates a Dispatch delegate instance.
  static LiteRtStatus CreateDelegate(LiteRtAccelerator accelerator,
                                     LiteRtOpaqueOptions options,
                                     void** delegate) {
    LITERT_ENSURE(delegate != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Delegate pointer is null.");
    LITERT_ENSURE(accelerator != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Accelerator handle is invalid.");
    LITERT_ENSURE(accelerator->env != nullptr,
                  kLiteRtStatusErrorInvalidArgument,
                  "Accelerator is not registered to an environment.");

    LITERT_ASSIGN_OR_RETURN(
        const litert::internal::ModelCompilationData* compilation_data,
        internal::GetModelCompilationData(options));

    LITERT_ENSURE(compilation_data->allocation_base,
                  kLiteRtStatusErrorRuntimeFailure,
                  "No model allocation was passed by the runtime.");

    auto dispatch_delegate_options = litert::CreateDispatchDelegateOptionsPtr(
        &accelerator->env->GetOptions());
    LITERT_ENSURE(dispatch_delegate_options != nullptr,
                  kLiteRtStatusErrorRuntimeFailure,
                  "Dispatch delegate options failed to be created.");

    LITERT_ENSURE(
        LiteRtDispatchDelegateAddAllocBaseOption(
            dispatch_delegate_options.get(),
            compilation_data->allocation_base) == kTfLiteOk,
        kLiteRtStatusErrorRuntimeFailure,
        "Could not add allocation base to dispatch delegate options.");

    if (compilation_data->allocation_fd != -1) {
      LITERT_ENSURE(LiteRtDispatchDelegateAddAllocFdOption(
                        dispatch_delegate_options.get(),
                        compilation_data->allocation_fd) == kTfLiteOk,
                    kLiteRtStatusErrorRuntimeFailure,
                    "Could not add allocation file descriptor to dispatch "
                    "delegate options.");
    }

    auto dispatch_delegate = litert::CreateDispatchDelegatePtr(
        &accelerator->env->GetOptions(), std::move(dispatch_delegate_options));
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
    LITERT_LOG(LITERT_INFO,
               "__dbg Dispatch delegate started metrics collection.");
    return LiteRtDispatchDelegateStartMetricsCollection(
        reinterpret_cast<TfLiteOpaqueDelegate*>(delegate), detail_level);
  }

  // Stops collection of HW-specific metrics and report the collected metrics.
  static LiteRtStatus StopMetricsCollection(void* delegate,
                                            LiteRtMetricsT* metrics) {
    LITERT_ENSURE(delegate != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Delegate pointer is null.");
    LITERT_ENSURE(metrics != nullptr, kLiteRtStatusErrorInvalidArgument,
                  "Metrics pointer is null.");
    return LiteRtDispatchDelegateStopMetricsCollection(
        reinterpret_cast<TfLiteOpaqueDelegate*>(delegate), metrics);
  }

  // Destroys a Dispatch delegate instance.
  static void DestroyDelegate(void* delegate) {
    LiteRtDestroyDispatchDelegate(
        reinterpret_cast<TfLiteOpaqueDelegate*>(delegate));
  }

 private:
  // Note: we do not directly use the option structure because we want to copy
  // and own all the option data.

  // Folder to the Dispatch API implementation shared library.
  std::string library_folder_;
};

}  // namespace litert

extern "C" {

LiteRtStatus LiteRtRegisterNpuAccelerator(
    LiteRtEnvironmentT* environment, LiteRtNpuAcceleratorOptions* options) {
  LITERT_ENSURE(environment != nullptr, kLiteRtStatusErrorInvalidArgument,
                "accelerator handle is invalid");
  LiteRtAccelerator accelerator_handle;
  LITERT_RETURN_IF_ERROR(LiteRtCreateAccelerator(&accelerator_handle));
  litert::internal::AcceleratorGuard accelerator(accelerator_handle);

  LITERT_RETURN_IF_ERROR(litert::internal::SetAcceleratorBoilerplateFunctions<
                         litert::NpuAccelerator>(accelerator));
  LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorStartMetricsCollection(
      accelerator.get(), litert::NpuAccelerator::StartMetricsCollection));
  LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorStopMetricsCollection(
      accelerator.get(), litert::NpuAccelerator::StopMetricsCollection));

  std::string library_folder;
  if (options && options->library_folder) {
    library_folder = options->library_folder;
  }
  // Check the environment options if the library folder wasn't set in the
  // options.
  if (library_folder.empty()) {
    if (auto env_library_folder =
            environment->GetOption(kLiteRtEnvOptionTagDispatchLibraryDir);
        env_library_folder.has_value()) {
      LITERT_ASSIGN_OR_RETURN(
          library_folder, litert::Get<std::string>(env_library_folder.value()));
    }
  }

  LITERT_ASSIGN_OR_RETURN(
      auto accelerator_impl,
      litert::NpuAccelerator::Create(std::move(library_folder)));

  LITERT_RETURN_IF_ERROR(LiteRtRegisterAccelerator(
      environment, accelerator.release(), accelerator_impl.release(),
      litert::NpuAccelerator::Destroy));
  return kLiteRtStatusOk;
}

}  // extern "C"

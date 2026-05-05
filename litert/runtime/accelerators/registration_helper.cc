// Copyright 2026 Google LLC.
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

#include "litert/runtime/accelerators/registration_helper.h"

#include <cstddef>
#include <utility>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_accelerator_registration.h"
#include "litert/c/internal/litert_custom_tensor_buffer_handlers_def.h"
#include "litert/c/internal/litert_tensor_buffer_registry.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/environment.h"

namespace litert::internal {

LiteRtStatus RegisterAcceleratorFromDef(
    LiteRtEnvironment env, const LiteRtAcceleratorDef* accelerator_def) {
  if (accelerator_def->version != LITERT_ACCELERATOR_DEF_CURRENT_VERSION)
    return kLiteRtStatusErrorWrongVersion;

  if (accelerator_def->get_name == nullptr ||
      accelerator_def->get_version == nullptr ||
      accelerator_def->get_hardware_support == nullptr ||
      accelerator_def->is_tflite_delegate_responsible_for_jit_compilation ==
          nullptr ||
      accelerator_def->create_delegate == nullptr ||
      accelerator_def->destroy_delegate == nullptr ||
      accelerator_def->buffer_handlers.num_supported_buffer_types >=
          LITERT_CUSTOM_BUFFER_HANDLERS_DEF_MAX_SUPPORTED_BUFFER_TYPES) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LiteRtAccelerator accelerator;
  LITERT_RETURN_IF_ERROR(LiteRtCreateAccelerator(&accelerator));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetAcceleratorGetName(accelerator, accelerator_def->get_name));
  LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorGetVersion(
      accelerator, accelerator_def->get_version));
  LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorGetHardwareSupport(
      accelerator, accelerator_def->get_hardware_support));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDelegateFunction(accelerator, accelerator_def->create_delegate,
                                accelerator_def->destroy_delegate));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetIsAcceleratorDelegateResponsibleForJitCompilation(
          accelerator,
          accelerator_def->is_tflite_delegate_responsible_for_jit_compilation));

  LITERT_RETURN_IF_ERROR(
      LiteRtRegisterAccelerator(env, accelerator, nullptr, nullptr));

  for (size_t i = 0;
       i < accelerator_def->buffer_handlers.num_supported_buffer_types; ++i) {
    LITERT_RETURN_IF_ERROR(LiteRtRegisterTensorBufferHandlers(
        env, accelerator_def->buffer_handlers.supported_buffer_types[i],
        accelerator_def->buffer_handlers.create_func,
        accelerator_def->buffer_handlers.destroy_func,
        accelerator_def->buffer_handlers.lock_func,
        accelerator_def->buffer_handlers.unlock_func,
        accelerator_def->buffer_handlers.clear_func,
        accelerator_def->buffer_handlers.import_func,
        accelerator_def->buffer_handlers.device_tag,
        accelerator_def->buffer_handlers.queue_tag));
  }

  return kLiteRtStatusOk;
}

Expected<SharedLibrary> LoadSharedLibrary(absl::string_view shlib_path,
                                          bool try_default_on_failure) {
  auto result = SharedLibrary::Load(shlib_path, RtldFlags::Lazy().Local());
  if (result || !try_default_on_failure) {
    return result;
  }
  return SharedLibrary::Load(RtldFlags::kDefault);
}

Expected<void> RegisterSharedObjectAcceleratorViaFunctionPointer(
    LiteRtEnvironmentT& environment, absl::string_view shlib_path,
    absl::string_view registration_function_name, bool try_default_on_failure) {
  LITERT_ASSIGN_OR_RETURN(
      auto shlib, LoadSharedLibrary(shlib_path, try_default_on_failure));
  LITERT_ASSIGN_OR_RETURN(
      auto registration_function,
      shlib.LookupSymbol<LiteRtStatus (*)(LiteRtEnvironment)>(
          registration_function_name.data()));
  LITERT_RETURN_IF_ERROR(registration_function(&environment));
  environment.GetAcceleratorRegistry().TakeOwnershipOfSharedLibrary(
      std::move(shlib));
  return {};
}

Expected<void> RegisterSharedObjectAcceleratorViaAcceleratorDef(
    LiteRtEnvironmentT& environment, absl::string_view shlib_path,
    absl::string_view accelerator_def_name, bool try_default_on_failure) {
  LITERT_ASSIGN_OR_RETURN(
      auto shlib, LoadSharedLibrary(shlib_path, try_default_on_failure));
  LITERT_ASSIGN_OR_RETURN(
      auto accelerator_def,
      shlib.LookupSymbol<LiteRtAcceleratorDef*>(accelerator_def_name.data()));

  LITERT_RETURN_IF_ERROR(
      RegisterAcceleratorFromDef(&environment, accelerator_def));

  environment.GetAcceleratorRegistry().TakeOwnershipOfSharedLibrary(
      std::move(shlib));
  return {};
}

}  // namespace litert::internal

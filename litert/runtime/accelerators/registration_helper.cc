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
#include <string>
#include <utility>

#include "litert/c/internal/litert_logging.h"
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
  if (accelerator_def == nullptr) {
    LITERT_LOG(LITERT_WARNING, "asc.12 accelerator def is null.");
    return kLiteRtStatusErrorInvalidArgument;
  }

  LITERT_LOG(
      LITERT_INFO,
      "asc.12 accelerator def ptr=%p version=%d current=%d callbacks "
      "name=%d version_fn=%d hardware=%d jit=%d create=%d destroy=%d "
      "buffer_types=%zu",
      accelerator_def, accelerator_def->version,
      LITERT_ACCELERATOR_DEF_CURRENT_VERSION,
      static_cast<int>(accelerator_def->get_name != nullptr),
      static_cast<int>(accelerator_def->get_version != nullptr),
      static_cast<int>(accelerator_def->get_hardware_support != nullptr),
      static_cast<int>(
          accelerator_def
              ->is_tflite_delegate_responsible_for_jit_compilation != nullptr),
      static_cast<int>(accelerator_def->create_delegate != nullptr),
      static_cast<int>(accelerator_def->destroy_delegate != nullptr),
      accelerator_def->buffer_handlers.num_supported_buffer_types);

  if (accelerator_def->version != LITERT_ACCELERATOR_DEF_CURRENT_VERSION) {
    LITERT_LOG(LITERT_WARNING,
               "asc.12 accelerator def version mismatch: got=%d expected=%d",
               accelerator_def->version,
               LITERT_ACCELERATOR_DEF_CURRENT_VERSION);
    return kLiteRtStatusErrorWrongVersion;
  }

  if (accelerator_def->get_name == nullptr ||
      accelerator_def->get_version == nullptr ||
      accelerator_def->get_hardware_support == nullptr ||
      accelerator_def->is_tflite_delegate_responsible_for_jit_compilation ==
          nullptr ||
      accelerator_def->create_delegate == nullptr ||
      accelerator_def->destroy_delegate == nullptr ||
      accelerator_def->buffer_handlers.num_supported_buffer_types >=
          LITERT_CUSTOM_BUFFER_HANDLERS_DEF_MAX_SUPPORTED_BUFFER_TYPES) {
    LITERT_LOG(
        LITERT_WARNING,
        "asc.12 accelerator def invalid callbacks or buffer handler count.");
    return kLiteRtStatusErrorInvalidArgument;
  }

  LiteRtAccelerator accelerator;
  LiteRtStatus status = LiteRtCreateAccelerator(&accelerator);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_WARNING,
               "asc.12 LiteRtCreateAccelerator failed: %s",
               LiteRtGetStatusString(status));
    return status;
  }
  status =
      LiteRtSetAcceleratorGetName(accelerator, accelerator_def->get_name);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_WARNING,
               "asc.12 LiteRtSetAcceleratorGetName failed: %s",
               LiteRtGetStatusString(status));
    return status;
  }
  status = LiteRtSetAcceleratorGetVersion(accelerator,
                                          accelerator_def->get_version);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_WARNING,
               "asc.12 LiteRtSetAcceleratorGetVersion failed: %s",
               LiteRtGetStatusString(status));
    return status;
  }
  status = LiteRtSetAcceleratorGetHardwareSupport(
      accelerator, accelerator_def->get_hardware_support);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_WARNING,
               "asc.12 LiteRtSetAcceleratorGetHardwareSupport failed: %s",
               LiteRtGetStatusString(status));
    return status;
  }
  status = LiteRtSetDelegateFunction(accelerator,
                                     accelerator_def->create_delegate,
                                     accelerator_def->destroy_delegate);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_WARNING,
               "asc.12 LiteRtSetDelegateFunction failed: %s",
               LiteRtGetStatusString(status));
    return status;
  }
  status = LiteRtSetIsAcceleratorDelegateResponsibleForJitCompilation(
      accelerator,
      accelerator_def->is_tflite_delegate_responsible_for_jit_compilation);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(
        LITERT_WARNING,
        "asc.12 LiteRtSetIsAcceleratorDelegateResponsibleForJitCompilation "
        "failed: %s",
        LiteRtGetStatusString(status));
    return status;
  }

  status = LiteRtRegisterAccelerator(env, accelerator, nullptr, nullptr);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_WARNING,
               "asc.12 LiteRtRegisterAccelerator failed: %s",
               LiteRtGetStatusString(status));
    return status;
  }

  for (size_t i = 0;
       i < accelerator_def->buffer_handlers.num_supported_buffer_types; ++i) {
    status = LiteRtRegisterTensorBufferHandlers(
        env, accelerator_def->buffer_handlers.supported_buffer_types[i],
        accelerator_def->buffer_handlers.create_func,
        accelerator_def->buffer_handlers.destroy_func,
        accelerator_def->buffer_handlers.lock_func,
        accelerator_def->buffer_handlers.unlock_func,
        accelerator_def->buffer_handlers.clear_func,
        accelerator_def->buffer_handlers.import_func,
        accelerator_def->buffer_handlers.device_tag,
        accelerator_def->buffer_handlers.queue_tag);
    if (status != kLiteRtStatusOk) {
      LITERT_LOG(LITERT_WARNING,
                 "asc.12 LiteRtRegisterTensorBufferHandlers[%zu] failed: %s",
                 i, LiteRtGetStatusString(status));
      return status;
    }
  }

  LITERT_LOG(LITERT_INFO, "asc.12 accelerator def registered successfully.");
  return kLiteRtStatusOk;
}

Expected<SharedLibrary> LoadSharedLibrary(absl::string_view shlib_path,
                                          bool try_default_on_failure) {
  const std::string path(shlib_path);
  LITERT_LOG(LITERT_INFO,
             "asc.12 LoadSharedLibrary path=%s try_default_on_failure=%d",
             path.c_str(), static_cast<int>(try_default_on_failure));
  auto result = SharedLibrary::Load(shlib_path, RtldFlags::Lazy().Local());
  if (result || !try_default_on_failure) {
    if (result) {
      LITERT_LOG(LITERT_INFO,
                 "asc.12 LoadSharedLibrary direct load succeeded path=%s",
                 path.c_str());
    } else {
      LITERT_LOG(LITERT_WARNING,
                 "asc.12 LoadSharedLibrary direct load failed path=%s "
                 "status=%s message=%s",
                 path.c_str(), LiteRtGetStatusString(result.Error().Status()),
                 result.Error().Message().c_str());
    }
    return result;
  }
  LITERT_LOG(LITERT_WARNING,
             "asc.12 LoadSharedLibrary direct load failed path=%s status=%s "
             "message=%s; trying RTLD_DEFAULT",
             path.c_str(), LiteRtGetStatusString(result.Error().Status()),
             result.Error().Message().c_str());
  auto default_result = SharedLibrary::Load(RtldFlags::kDefault);
  if (default_result) {
    LITERT_LOG(LITERT_INFO,
               "asc.12 LoadSharedLibrary RTLD_DEFAULT succeeded path=%s",
               path.c_str());
  } else {
    LITERT_LOG(LITERT_WARNING,
               "asc.12 LoadSharedLibrary RTLD_DEFAULT failed path=%s "
               "status=%s message=%s",
               path.c_str(),
               LiteRtGetStatusString(default_result.Error().Status()),
               default_result.Error().Message().c_str());
  }
  return default_result;
}

Expected<void> RegisterSharedObjectAcceleratorViaFunctionPointer(
    LiteRtEnvironmentT& environment, absl::string_view shlib_path,
    absl::string_view registration_function_name, bool try_default_on_failure) {
  const std::string path(shlib_path);
  const std::string symbol(registration_function_name);
  auto shlib_result = LoadSharedLibrary(shlib_path, try_default_on_failure);
  if (!shlib_result) {
    LITERT_LOG(LITERT_WARNING,
               "asc.12 function-pointer registration load failed path=%s "
               "symbol=%s status=%s message=%s",
               path.c_str(), symbol.c_str(),
               LiteRtGetStatusString(shlib_result.Error().Status()),
               shlib_result.Error().Message().c_str());
    return shlib_result.Error();
  }
  auto shlib = std::move(*shlib_result);
  auto registration_function_result =
      shlib.LookupSymbol<LiteRtStatus (*)(LiteRtEnvironment)>(
          registration_function_name.data());
  if (!registration_function_result) {
    LITERT_LOG(LITERT_WARNING,
               "asc.12 function-pointer symbol lookup failed path=%s "
               "symbol=%s status=%s message=%s",
               path.c_str(), symbol.c_str(),
               LiteRtGetStatusString(
                   registration_function_result.Error().Status()),
               registration_function_result.Error().Message().c_str());
    return registration_function_result.Error();
  }
  auto registration_function = *registration_function_result;
  LiteRtStatus status = registration_function(&environment);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_WARNING,
               "asc.12 function-pointer registration callback failed path=%s "
               "symbol=%s status=%s",
               path.c_str(), symbol.c_str(), LiteRtGetStatusString(status));
    return Error(status);
  }
  environment.GetAcceleratorRegistry().TakeOwnershipOfSharedLibrary(
      std::move(shlib));
  LITERT_LOG(LITERT_INFO,
             "asc.12 function-pointer registration succeeded path=%s "
             "symbol=%s",
             path.c_str(), symbol.c_str());
  return {};
}

Expected<void> RegisterSharedObjectAcceleratorViaAcceleratorDef(
    LiteRtEnvironmentT& environment, absl::string_view shlib_path,
    absl::string_view accelerator_def_name, bool try_default_on_failure) {
  const std::string path(shlib_path);
  const std::string symbol(accelerator_def_name);
  auto shlib_result = LoadSharedLibrary(shlib_path, try_default_on_failure);
  if (!shlib_result) {
    LITERT_LOG(LITERT_WARNING,
               "asc.12 accelerator-def registration load failed path=%s "
               "symbol=%s status=%s message=%s",
               path.c_str(), symbol.c_str(),
               LiteRtGetStatusString(shlib_result.Error().Status()),
               shlib_result.Error().Message().c_str());
    return shlib_result.Error();
  }
  auto shlib = std::move(*shlib_result);
  auto accelerator_def_result =
      shlib.LookupSymbol<LiteRtAcceleratorDef*>(accelerator_def_name.data());
  if (!accelerator_def_result) {
    LITERT_LOG(LITERT_WARNING,
               "asc.12 accelerator-def symbol lookup failed path=%s "
               "symbol=%s status=%s message=%s",
               path.c_str(), symbol.c_str(),
               LiteRtGetStatusString(accelerator_def_result.Error().Status()),
               accelerator_def_result.Error().Message().c_str());
    return accelerator_def_result.Error();
  }
  auto accelerator_def = *accelerator_def_result;

  LiteRtStatus status = RegisterAcceleratorFromDef(&environment, accelerator_def);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_WARNING,
               "asc.12 accelerator-def registration failed path=%s symbol=%s "
               "status=%s",
               path.c_str(), symbol.c_str(), LiteRtGetStatusString(status));
    return Error(status);
  }

  environment.GetAcceleratorRegistry().TakeOwnershipOfSharedLibrary(
      std::move(shlib));
  LITERT_LOG(LITERT_INFO,
             "asc.12 accelerator-def registration succeeded path=%s "
             "symbol=%s",
             path.c_str(), symbol.c_str());
  return {};
}

}  // namespace litert::internal

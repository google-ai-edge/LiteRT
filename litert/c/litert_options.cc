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

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_op_kernel.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/options.h"

#define LRT_CHECK_NON_NULL(handle)                          \
  if (!(handle)) {                                          \
    LITERT_LOG(LITERT_ERROR, #handle " must not be null."); \
    return kLiteRtStatusErrorInvalidArgument;               \
  }

extern "C" {

LiteRtStatus LiteRtCreateOptions(LiteRtOptions* options) {
  LRT_CHECK_NON_NULL(options);
  *options = new LiteRtOptionsT;
  return kLiteRtStatusOk;
}

void LiteRtDestroyOptions(LiteRtOptions options) {
  if (options && options->options) {
    LiteRtDestroyOpaqueOptions(options->options);
  }
  delete options;
}

LiteRtStatus LiteRtSetOptionsHardwareAccelerators(
    LiteRtOptions options, LiteRtHwAcceleratorSet hardware_accelerators) {
  LRT_CHECK_NON_NULL(options);
  if ((hardware_accelerators &
       (kLiteRtHwAcceleratorCpu | kLiteRtHwAcceleratorGpu |
        kLiteRtHwAcceleratorNpu
#ifdef __EMSCRIPTEN__
        | kLiteRtHwAcceleratorWebNn
#endif  // __EMSCRIPTEN__
        )) != hardware_accelerators) {
    LITERT_LOG(LITERT_ERROR,
               "Invalid bitfield value for hardware accelerator set: %d.",
               hardware_accelerators);
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->hardware_accelerators = hardware_accelerators;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetOptionsHardwareAccelerators(
    LiteRtOptions options, LiteRtHwAcceleratorSet* hardware_accelerators) {
  LRT_CHECK_NON_NULL(options);
  LRT_CHECK_NON_NULL(hardware_accelerators);
  *hardware_accelerators = options->hardware_accelerators;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetOptionsSelectedSignatures(
    LiteRtOptions options, size_t num_signature_keys,
    const char* const* signature_keys) {
  LRT_CHECK_NON_NULL(options);
  if (num_signature_keys == 0 || signature_keys == nullptr) {
    LITERT_LOG(LITERT_ERROR,
               "Selected signature keys must be a non-empty list.");
    return kLiteRtStatusErrorInvalidArgument;
  }

  std::vector<std::string> copied_keys;
  copied_keys.reserve(num_signature_keys);
  absl::flat_hash_set<std::string> unique_keys;
  for (size_t i = 0; i < num_signature_keys; ++i) {
    const char* signature_key = signature_keys[i];
    if (signature_key == nullptr || signature_key[0] == '\0') {
      LITERT_LOG(LITERT_ERROR,
                 "Selected signature key at index %zu must not be null or "
                 "empty.",
                 i);
      return kLiteRtStatusErrorInvalidArgument;
    }
    if (!unique_keys.insert(signature_key).second) {
      LITERT_LOG(LITERT_ERROR, "Duplicate selected signature key: %s.",
                 signature_key);
      return kLiteRtStatusErrorInvalidArgument;
    }
    copied_keys.emplace_back(signature_key);
  }

  options->selected_signature_keys = std::move(copied_keys);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAddOpaqueOptions(LiteRtOptions options,
                                    LiteRtOpaqueOptions opaque_options) {
  LRT_CHECK_NON_NULL(options);
  LRT_CHECK_NON_NULL(opaque_options);
  LITERT_RETURN_IF_ERROR(
      LiteRtAppendOpaqueOptions(&(options->options), opaque_options));
  return kLiteRtStatusOk;
}

// Retrieves the head of the accelerator compilation option list.
//
// Note: The following elements may be retrieved with
// `LiteRtGetNextAcceleratorCompilationOptions`.
LiteRtStatus LiteRtGetOpaqueOptions(LiteRtOptions options,
                                    LiteRtOpaqueOptions* opaque_options) {
  LRT_CHECK_NON_NULL(options);
  LRT_CHECK_NON_NULL(opaque_options);
  *opaque_options = options->options;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAddCustomOpKernelOption(
    LiteRtOptions options, const char* custom_op_name, int custom_op_version,
    const LiteRtCustomOpKernel* custom_op_kernel,
    void* custom_op_kernel_user_data) {
  LRT_CHECK_NON_NULL(options);
  LRT_CHECK_NON_NULL(custom_op_name);
  LRT_CHECK_NON_NULL(custom_op_kernel);
  // TODO(b/330649488): Check if the custom op kernel already exists.
  options->custom_op_options.push_back({
      /*.op_name=*/custom_op_name,
      /*.op_version=*/custom_op_version,
      /*.user_data=*/custom_op_kernel_user_data,
      /*.op_kernel=*/*custom_op_kernel,
  });
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAddExternalTensorBinding(LiteRtOptions options,
                                            const char* signature_name,
                                            const char* tensor_name, void* data,
                                            size_t size_bytes) {
  LRT_CHECK_NON_NULL(options);
  LRT_CHECK_NON_NULL(signature_name);
  LRT_CHECK_NON_NULL(tensor_name);
  LRT_CHECK_NON_NULL(data);
  options->external_tensor_bindings.push_back(
      {/*.signature_name =*/signature_name,
       /*.tensor_name =*/tensor_name,
       /*.data =*/data,
       /*.size_bytes =*/size_bytes});
  return kLiteRtStatusOk;
}

}  // extern "C"

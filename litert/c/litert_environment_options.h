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

#ifndef ODML_LITERT_LITERT_C_LITERT_ENVIRONMENT_OPTIONS_H_
#define ODML_LITERT_LITERT_C_LITERT_ENVIRONMENT_OPTIONS_H_

#include <stdint.h>

#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum {
  kLiteRtEnvOptionTagCompilerPluginLibraryDir = 0,
  kLiteRtEnvOptionTagDispatchLibraryDir = 1,
  kLiteRtEnvOptionTagOpenClDeviceId = 2,
  kLiteRtEnvOptionTagOpenClPlatformId = 3,
  kLiteRtEnvOptionTagOpenClContext = 4,
  kLiteRtEnvOptionTagOpenClCommandQueue = 5,
  kLiteRtEnvOptionTagEglDisplay = 6,
  kLiteRtEnvOptionTagEglContext = 7,
  kLiteRtEnvOptionTagWebGpuDevice = 8,
  kLiteRtEnvOptionTagWebGpuQueue = 9,
  kLiteRtEnvOptionTagMetalDevice = 10,
  kLiteRtEnvOptionTagMetalCommandQueue = 11,
  // WARNING: Vulkan support is experimental.
  kLiteRtEnvOptionTagVulkanEnvironment = 12,
  // kLiteRtEnvOptionTagVulkanCommandPool = 13,  // Deprecated.
  kLiteRtEnvOptionTagCallbackOnGpuEnvDestroy = 14,
  kLiteRtEnvOptionTagCallbackUserDataOnGpuEnvDestroy = 15,
  kLiteRtEnvOptionTagMagicNumberConfigs = 16,
  kLiteRtEnvOptionTagMagicNumberVerifications = 17,
  kLiteRtEnvOptionTagCompilerCacheDir = 18,
  // Singleton ML Drift WebGPU/Dawn instance required for shared libraries not
  // to create their own instances.
  kLiteRtEnvOptionTagWebGpuInstance = 19,
  // Dawn procedure table pointer for shared libraries to populate their tables
  // with the shared procedures instead of their own procedures.
  kLiteRtEnvOptionTagWebGpuProcs = 20,
  kLiteRtEnvOptionTagCustomTensorBufferHandlers = 21,
} LiteRtEnvOptionTag;

typedef struct {
  LiteRtEnvOptionTag tag;
  LiteRtAny value;
} LiteRtEnvOption;

// Arbitrary size of array following the pattern in TfLiteIntArray.
#if defined(_MSC_VER)
#define _LITERT_ARBITRARY_ARRAY_SIZE 1
#elif (!defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
       __GNUC_MINOR__ >= 1) ||                                      \
    defined(HEXAGON) ||                                             \
    (defined(__clang__) && __clang_major__ == 7 && __clang_minor__ == 1)
#define _LITERT_ARBITRARY_ARRAY_SIZE 0
#else
#define _LITERT_ARBITRARY_ARRAY_SIZE
#endif

typedef struct {
  int64_t magic_number;
  int64_t target_number;
  // Prefix of signatures to update magic numbers. If null or empty, all
  // signatures will be updated.
  // This C string is owned by the user of LiteRT runtime and must outlive until
  // model is initialized.
  const char* signature_prefix;
} LiteRtMagicNumberConfig;

// Magic number replacement configs set as an option with a pointer to this
// structure.
typedef struct {
  int64_t num_configs;
  LiteRtMagicNumberConfig configs[_LITERT_ARBITRARY_ARRAY_SIZE];
} LiteRtMagicNumberConfigs;

typedef struct {
  // Signature with magic numbers replaced.
  // This C string is owned by the user of LiteRT runtime and must outlive until
  // model is initialized.
  const char* signature;
  // Signature to test against for verification.
  // This C string is owned by the user of LiteRT runtime and must outlive until
  // model is initialized.
  const char* test_signature;
  // Whether the ops of signature is a superset of the ops of test_signature.
  // If true, it verifies only if all ops in test_signature exists in the
  // signature in the same order and their operands are of the same shape
  // assuming that the extra ops manipulate the input/output tensors of larger
  // dimensions, but not to change the semantics of the graph.
  // If false, the test signature must be exactly the same to the signature in
  // all aspects including # of tensors, # of ops, and their shapes.
  bool is_superset;
} LiteRtMagicNumberVerification;

typedef struct {
  int64_t num_verifications;
  LiteRtMagicNumberVerification verifications[_LITERT_ARBITRARY_ARRAY_SIZE];
} LiteRtMagicNumberVerifications;

typedef struct {
  CreateCustomTensorBuffer create_func;
  DestroyCustomTensorBuffer destroy_func;
  LockCustomTensorBuffer lock_func;
  UnlockCustomTensorBuffer unlock_func;
} LiteRtCustomTensorBufferHandlers;

// Retrieves the value corresponding to the given tag.
//
// Returns kLiteRtStatusErrorNotFound if the option tag is not found.
LiteRtStatus LiteRtGetEnvironmentOptionsValue(LiteRtEnvironmentOptions options,
                                              LiteRtEnvOptionTag tag,
                                              LiteRtAny* value);

LiteRtStatus LiteRtSetEnvironmentOptionsValue(LiteRtEnvironmentOptions options,
                                              LiteRtEnvOption env_option);
#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_ENVIRONMENT_OPTIONS_H_

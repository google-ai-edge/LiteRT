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

#ifndef ODML_LITERT_LITERT_C_LITERT_ENVIRONMENT_H_
#define ODML_LITERT_LITERT_C_LITERT_ENVIRONMENT_H_

#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Create a LiteRT environment with options.
// Used to set the path of the compiler plugin library and dispatch library.
// Caller owns the returned LiteRtEnvironment. The owner is responsible for
// calling LiteRtDestroyEnvironment() to release the environment.
//
// Note: options of kLiteRtEnvOptionTagOpenCl* shouldn't be set with this API.
LiteRtStatus LiteRtCreateEnvironment(int num_options,
                                     const LiteRtEnvOption* options,
                                     LiteRtEnvironment* environment);

// Destroy an owned LiteRT environment object.
void LiteRtDestroyEnvironment(LiteRtEnvironment environment);

// Get the options that the environment was created with.
LiteRtStatus LiteRtGetEnvironmentOptions(LiteRtEnvironment environment,
                                         LiteRtEnvironmentOptions* options);

// Add options to the existing environment. When overwrite is true, the options
// will overwrite all existing options with the same tags. When overwrite is
// false, an error will be returned if the option already exists in the
// environment.
LiteRtStatus LiteRtAddEnvironmentOptions(LiteRtEnvironment environment,
                                         int num_options,
                                         const LiteRtEnvOption* options,
                                         bool overwrite);

// Create a LiteRT GPU environment with options.
// The given `environment` takes ownership of the created GPU environment.
// This API is usually called by the GPU accelerator implementation to set GPU
// environment options which affect the entire LiteRT runtime.
//
// Note: In most cases, users should not call this API directly.
LiteRtStatus LiteRtGpuEnvironmentCreate(LiteRtEnvironment environment,
                                        int num_options,
                                        const LiteRtEnvOption* options);

// Returns whether the environment supports CL/GL interop.
LiteRtStatus LiteRtEnvironmentSupportsClGlInterop(LiteRtEnvironment environment,
                                                  bool* is_supported);

// Returns whether the environment supports AHWB/CL interop.
LiteRtStatus LiteRtEnvironmentSupportsAhwbClInterop(
    LiteRtEnvironment environment, bool* is_supported);

// Returns whether the environment supports AHWB/GL interop.
LiteRtStatus LiteRtEnvironmentSupportsAhwbGlInterop(
    LiteRtEnvironment environment, bool* is_supported);

// Returns whether the environment has a GPU environment.
void LiteRtEnvironmentHasGpuEnvironment(LiteRtEnvironment environment,
                                        bool* has_gpu_environment);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_ENVIRONMENT_H_

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_ACCELERATORS_DISPATCH_DISPATCH_ACCELERATOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_ACCELERATORS_DISPATCH_DISPATCH_ACCELERATOR_H_

#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"

#ifdef __cplusplus
extern "C" {
#endif

struct LiteRtNpuAcceleratorOptions {
  const char* library_folder;
};

// Registers the NPU accelerator to the given environment.
//
// `options` may be null, in which case the accelerator is registered with
// a default configuration.
//
// If `options.library_folder` is not specified, the library folder is replaced
// with the `LiteRtEnvOptionTagDispatchLibraryDir` environment option (that was
// passed upon creation).
//
// Once this function has returned, options may be freed or reused.
LiteRtStatus LiteRtRegisterNpuAccelerator(LiteRtEnvironment environment,
                                          LiteRtNpuAcceleratorOptions* options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_ACCELERATORS_DISPATCH_DISPATCH_ACCELERATOR_H_

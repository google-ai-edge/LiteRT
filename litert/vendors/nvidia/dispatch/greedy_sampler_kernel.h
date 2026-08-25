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

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_GREEDY_SAMPLER_KERNEL_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_GREEDY_SAMPLER_KERNEL_H_

#include <driver_types.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Returns the device workspace required by LiteRtNvidiaLaunchF32ArgMax.
// The result is zero when count is zero or exceeds the supported int32 range.
size_t LiteRtNvidiaF32ArgMaxWorkspaceBytes(size_t count);

// Finds the first index containing the maximum F32 value. workspace and
// device_result must point to device memory. Later NaNs are ignored; a NaN at
// index zero is selected, matching the existing CPU greedy sampler.
cudaError_t LiteRtNvidiaLaunchF32ArgMax(const float* input, size_t count,
                                        void* workspace, size_t workspace_bytes,
                                        int32_t* device_result,
                                        cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_GREEDY_SAMPLER_KERNEL_H_

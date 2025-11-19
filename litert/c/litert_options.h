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

#ifndef ODML_LITERT_LITERT_C_LITERT_OPTIONS_H_
#define ODML_LITERT_LITERT_C_LITERT_OPTIONS_H_

#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_op_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a compilation option object.
LiteRtStatus LiteRtCreateOptions(LiteRtOptions* options);

// Destroys a compilation option object.
void LiteRtDestroyOptions(LiteRtOptions options);

// Sets the requested hardware accelerators to apply during model compilation.
LiteRtStatus LiteRtSetOptionsHardwareAccelerators(
    LiteRtOptions options, LiteRtHwAcceleratorSet hardware_accelerators);

// Gets the hardware accelerators to apply during model compilation.
LiteRtStatus LiteRtGetOptionsHardwareAccelerators(
    LiteRtOptions options, LiteRtHwAcceleratorSet* hardware_accelerators);

// Adds compilation options for a specific accelerator to the accelerator
// compilation option list.
//
// Note: Multiple accelerator options may be added to the options object.
//
// Note: `accelerator_compilation_options`'s ownership is transferred to
// `options`.
LiteRtStatus LiteRtAddOpaqueOptions(LiteRtOptions options,
                                    LiteRtOpaqueOptions opaque_options);

// Retrieves the head of the accelerator compilation option list.
//
// Note: The following elements may be retrieved with
// `LiteRtGetNextAcceleratorCompilationOptions`.
LiteRtStatus LiteRtGetOpaqueOptions(LiteRtOptions options,
                                    LiteRtOpaqueOptions* opaque_options);

// Adds a custom op kernel to the given options.
LiteRtStatus LiteRtAddCustomOpKernelOption(
    LiteRtOptions options, const char* custom_op_name, int custom_op_version,
    const LiteRtCustomOpKernel* custom_op_kernel,
    void* custom_op_kernel_user_data);

// Adds an external tensor binding to the given options.
//
// Note: `data` is owned by the caller and must outlive the lifetime of the
// CompiledModel.
// `size_bytes` must match the tensor's expected size.
LiteRtStatus LiteRtAddExternalTensorBinding(LiteRtOptions options,
                                            const char* signature_name,
                                            const char* tensor_name, void* data,
                                            int size_bytes);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_OPTIONS_H_

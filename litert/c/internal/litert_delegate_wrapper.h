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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_DELEGATE_WRAPPER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_DELEGATE_WRAPPER_H_

// The LiteRtDelegateWrapper type is an abstract type that exists to
// hide the API dependency of the LiteRT API on the TF Lite API.
//
// This internal header file exposes the API dependency on the TF Lite API.
// However, this internal header file has limited visibility in the BUILD file
// -- it is exposed only to the LiteRT runtime implementation, and in
// particular to LiteRT runtime's built-in accelerator implementations that are
// implemented using TF Lite delegates.
//
// Users other than LiteRT runtime itself should not use this header.

#include "litert/c/litert_common.h"
#include "tflite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Wrap a TF Lite opaque delegate into a LiteRT delegate wrapper.
// The lifetime of the wrapper returned in `*wrapper` is the same
// as the lifetime of the delegate passed in.
LiteRtStatus LiteRtWrapDelegate(TfLiteOpaqueDelegate* delegate,
                                LiteRtDelegateWrapper* wrapper);

// Extract a TF Lite opaque delegate from a LiteRT delegate wrapper.
// The lifetime of the delegate returned in `*delegate` is the same as the
// lifetime of the wrapper.
LiteRtStatus LiteRtUnwrapDelegate(LiteRtDelegateWrapper wrapper,
                                  TfLiteOpaqueDelegate** delegate);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_DELEGATE_WRAPPER_H_

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

#include "litert/c/internal/litert_delegate_wrapper.h"

#include "litert/c/litert_common.h"
#include "tflite/c/common.h"

// Definition for LiteRtDelegateWrapperT.
//
// The main reason for using `LiteRtDelegateWrapper` instead of
// `TfLiteOpaqueDelegate*` directly in the LiteRT headers is to avoid an
// interface dependency of LiteRT on the TF Lite API. It's OK if the
// _implementation_ depends on the TF Lite API, but we want to ensure the
// LiteRT public header files don't depend on the TF Lite API headers.
// Using LiteRtDelegateWrapper preserves that property, with the exception of
// the `litert_delegate_wrapper.h` header file, which has limited visibility and
// is considered an implementation detail exposed only to accelerators shipped
// with LiteRT itself.
//
// Another reason for not using TfLiteOpaqueDelegate in the headers is that the
// definition of TfLiteOpaqueDelegate may be conditional depending on whether
// TFLite-in-Play-services is enabled. Consequently, interfaces using it would
// need conditional namespacing to avoid ODR violations for strict C++ standards
// compliance.
//
// We never actually allocate or dereference this struct, instead just casting
// the pointer to this struct to/from to a pointer to the address of its first
// member before invoking any of the functions that would actually dereference
// the pointer.
//
// We define this struct purely for the benefit of automatic ABI checking tools.
//
// This type is defined as a _union_ containing both TfLiteDelegate and
// TfLiteOpaqueDelegate, rather than just as a struct wrapper around
// TfLiteOpaqueDelegate. The reason for this is that using a union ensures
// that ABI checking tools will complain about an ABI change if the definition
// of `TfLiteDelegate` changes in incompatible ways.

struct LiteRtDelegateWrapperT {
  union {
    TfLiteOpaqueDelegate opaque_delegate;
    TfLiteDelegate delegate;
  };
};

LiteRtStatus LiteRtWrapDelegate(TfLiteOpaqueDelegate* delegate,
                                LiteRtDelegateWrapper* wrapper) {
  // Using a cast here avoids needing to allocate a LiteRtDelegateWrapperT
  // struct, and importantly avoids needing to deallocate it afterwards,
  // which would complicate things considerably.
  // It is safe to cast from one pointer type to another pointer type,
  // as long as we always cast back again before dereferencing the pointer.
  *wrapper = reinterpret_cast<LiteRtDelegateWrapperT*>(delegate);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtUnwrapDelegate(LiteRtDelegateWrapper wrapper,
                                  TfLiteOpaqueDelegate** delegate) {
  *delegate = reinterpret_cast<TfLiteOpaqueDelegate*>(wrapper);
  // Or equivalently, we could also write:
  //   *delegate = &wrapper->opaque_delegate;
  return kLiteRtStatusOk;
}

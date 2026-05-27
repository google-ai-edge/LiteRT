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

#include <new>

#include "litert/c/litert_common.h"
#include "tflite/c/common.h"
#include "tflite/core/api/op_resolver.h"

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
// We define this struct purely for the benefit of automatic ABI checking tools
// and to encapsulate the delegate deleter function.
//
// This type includes a _union_ containing pointers to both TfLiteDelegate and
// TfLiteOpaqueDelegate. The reason for this is that using a union ensures
// that ABI checking tools will complain about an ABI change if the definition
// of `TfLiteDelegate` changes in incompatible ways.
//
// In addition, we store the associated deleter function to enable
// self-contained lifecycle management without requiring global maps or static
// variables in accelerators to handle multiple delegates properly.

struct LiteRtDelegateWrapperT {
  tflite::OpResolver::TfLiteOpaqueDelegatePtr delegate;
};

LiteRtStatus LiteRtWrapDelegate(TfLiteOpaqueDelegate* delegate,
                                void (*deleter)(TfLiteOpaqueDelegate*),
                                LiteRtDelegateWrapper* wrapper) {
  if (!delegate || !wrapper) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *wrapper = new (std::nothrow) LiteRtDelegateWrapperT{
      tflite::OpResolver::TfLiteOpaqueDelegatePtr(delegate, deleter)};
  if (*wrapper == nullptr) {
    return kLiteRtStatusErrorMemoryAllocationFailure;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtUnwrapDelegate(LiteRtDelegateWrapper wrapper,
                                  TfLiteOpaqueDelegate** delegate) {
  if (!wrapper || !delegate) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *delegate = wrapper->delegate.get();
  return kLiteRtStatusOk;
}

void LiteRtDestroyDelegateWrapper(LiteRtDelegateWrapper wrapper) {
  delete wrapper;
}

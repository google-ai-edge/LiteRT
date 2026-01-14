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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_DISPATCH_API_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_DISPATCH_API_CONFIG_H_

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/options/litert_darwinn_options.h"

namespace litert::google_tensor {

// Initializes the Google Tensor Dispatch API configuration.
//
// This function must be called prior to calling any functions that query the
// Google Tensor Dispatch API configuration.
//
// NOTE: this function makes SouthBound API calls, and thus requires SouthBound
// to already be initialized.
LiteRtStatus InitializeDispatchApiConfig(
    LiteRtEnvironmentOptions environment_options, LiteRtOptions options);

// The following methods are used to query the Google Tensor Dispatch API
// configuration.
//
// The following methods are thread-safe with respect to each other, and are
// designed to be cheap. Thus, repeated calls should be preferred to caching
// results to avoid stale data.
//
// WARNING: `InitializeDispatchApiConfig` must be called prior to calling any of
// the following methods.

// Returns a pointer to the DarwiNN-specific options provided by the application
// when initializing the LiteRT Dispatch API.
//
// If no DarwiNN-specific options were provided, nullptr is returned.
DarwinnRuntimeOptions* GetTheDarwinnOptions();

// Returns the Google Tensor Dispatch API build ID.
//
// NOTE: the returned pointer is never null.
const char* GetTheBuildId();

// Returns the capabilities of the available SouthBound implementation.
//
// NOTE: the returned value is a bitmask of `LiteRtDispatchCapabilities`.
int GetTheCapabilities();

// Returns the tensor buffer types that are supported by the available
// SouthBound implementation.
absl::Span<const LiteRtTensorBufferType> GetTheSupportedTensorBufferTypes();

// Returns `true` if the tensor buffer type `type` is supported by the available
// SouthBound implementation, else `false`.
inline bool IsTensorBufferTypeSupported(LiteRtTensorBufferType type) {
  absl::Span<const LiteRtTensorBufferType> supported_tensor_buffer_types =
      GetTheSupportedTensorBufferTypes();

  return absl::c_find(supported_tensor_buffer_types, type) !=
           supported_tensor_buffer_types.end();
}

}  // namespace litert::google_tensor

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_DISPATCH_API_CONFIG_H_

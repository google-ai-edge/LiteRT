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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_DISPATCH_API_MACROS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_DISPATCH_API_MACROS_H_

// A collection of utility macros for handling common patterns.

// If `expr` evaluates to `nullptr`, an error message is logged and
// `kLiteRtStatusErrorInvalidArgument` is returned.
#define GT_LOG_RETURN_IF_NULL(expr)                    \
  do {                                                 \
    if ((expr) == nullptr) {                           \
      LITERT_LOG(LITERT_ERROR, "'%s' is null", #expr); \
      return kLiteRtStatusErrorInvalidArgument;        \
    }                                                  \
  } while (false)

// If `expr` does not evaluate to `kThrStatusSuccess`, the provided error
// message is logged.
#define GT_LOG_IF_SB_ERROR(expr, ...)        \
  do {                                       \
    if ((expr) != kThrStatusSuccess) {       \
      LITERT_LOG(LITERT_ERROR, __VA_ARGS__); \
    }                                        \
  } while (0)

// If `expr` does not evaluate to `kThrStatusSuccess`, the provided error
// message is logged and `kLiteRtStatusErrorRuntimeFailure` is returned.
#define GT_LOG_RETURN_IF_SB_ERROR(expr, ...)   \
  do {                                         \
    if ((expr) != kThrStatusSuccess) {         \
      LITERT_LOG(LITERT_ERROR, __VA_ARGS__);   \
      return kLiteRtStatusErrorRuntimeFailure; \
    }                                          \
  } while (0)

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_DISPATCH_API_MACROS_H_

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

#import <Foundation/Foundation.h>

#include "litert/cc/litert_expected.h"
#import "third_party/odml/litert/litert/objc/apis/LRTError.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Creates an @c NSError with the LiteRT error domain, given error code and description.
 *
 * @param code Error code matching @c LRTErrorCode or litert::Status.
 * @param description Description message for the error.
 * @return An @c NSError instance with domain @c LRTErrorDomain.
 */
static inline NSError *LRTErrorWithCode(NSInteger code, NSString *description) {
  return [NSError errorWithDomain:LRTErrorDomain
                             code:code
                         userInfo:@{NSLocalizedDescriptionKey : description}];
}

/**
 * Assigns an @c NSError with the LiteRT error domain to @c error, if the caller asked for it.
 *
 * Callers that are not interested in the failure reason pass @c NULL, so failure paths can call
 * this unconditionally instead of guarding every assignment.
 *
 * @param error Out-parameter to populate, or @c NULL.
 * @param code Error code matching @c LRTErrorCode or litert::Status.
 * @param description Description message for the error.
 */
static inline void LRTSetError(NSError *_Nullable *_Nullable error, NSInteger code,
                               NSString *description) {
  if (error != NULL) {
    *error = LRTErrorWithCode(code, description);
  }
}

/**
 * Assigns an @c NSError describing a failed C++ LiteRT call to @c error, if the caller asked
 * for it.
 *
 * @param error Out-parameter to populate, or @c NULL.
 * @param cppError Error reported by a C++ @c litert::Expected.
 */
static inline void LRTSetErrorFromCppError(NSError *_Nullable *_Nullable error,
                                           const litert::Error &cppError) {
  if (error == NULL) {
    return;
  }
  // @() returns nil for messages that are not valid UTF-8, which would make the userInfo
  // dictionary literal in LRTErrorWithCode() throw.
  NSString *description = @(cppError.Message().c_str());
  if (description.length == 0) {
    description = @"LiteRT reported a failure without a message";
  }
  *error = LRTErrorWithCode(static_cast<NSInteger>(cppError.StatusValue()), description);
}

NS_ASSUME_NONNULL_END

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

#include <string>
#include <vector>

#include "litert/cc/litert_api_types.h"
#include "litert/cc/litert_expected.h"
#import "third_party/odml/litert/litert/objc/apis/LRTError.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Converts an Objective-C @c NSString into a C++ @c std::string.
 *
 * Returns an empty @c std::string if @c string is @c nil or if its UTF-8 conversion fails
 * (e.g. due to an unpaired UTF-16 surrogate), avoiding passing @c nullptr to @c absl::string_view
 * or @c std::string constructors.
 *
 * @param string The Objective-C string to convert, or @c nil.
 * @return A UTF-8 @c std::string representation of @c string.
 */
static inline std::string LRTCPPStringFromNSString(NSString *_Nullable string) {
  const char *utf8String = string.UTF8String;
  return utf8String ? std::string(utf8String) : std::string();
}

/**
 * Converts a C++ @c litert::StringView, which is not guaranteed to be NUL-terminated, into an
 * Objective-C @c NSString.
 *
 * @param stringView The C++ string view to convert.
 * @return The decoded UTF-8 @c NSString, or @c nil if the bytes are not valid UTF-8.
 */
static inline NSString *_Nullable LRTStringFromCPPStringView(litert::StringView stringView) {
  return [[NSString alloc] initWithBytes:stringView.data()
                                  length:stringView.size()
                                encoding:NSUTF8StringEncoding];
}

/**
 * Converts a vector of C++ @c litert::StringView values into an @c NSArray of @c NSString objects.
 *
 * Callers index the returned array by tensor or signature index, so an entry that is not valid
 * UTF-8 is represented as an empty string rather than omitted.
 *
 * @param stringViews The C++ string views to convert.
 * @return An @c NSArray<NSString *> with the same count and order as @c stringViews.
 */
static inline NSArray<NSString *> *LRTStringArrayFromCPPStringViews(
    const std::vector<litert::StringView> &stringViews) {
  NSMutableArray<NSString *> *strings = [NSMutableArray arrayWithCapacity:stringViews.size()];
  for (litert::StringView stringView : stringViews) {
    NSString *string = LRTStringFromCPPStringView(stringView);
    [strings addObject:string ?: @""];
  }
  return strings;
}

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
static inline void LRTSetErrorFromCPPError(NSError *_Nullable *_Nullable error,
                                           const litert::Error &cppError) {
  if (error == NULL) {
    return;
  }
  NSString *description = LRTStringFromCPPStringView(cppError.Message());
  if (description.length == 0) {
    description = @"LiteRT reported a failure without a message";
  }
  *error = LRTErrorWithCode(static_cast<NSInteger>(cppError.StatusValue()), description);
}

NS_ASSUME_NONNULL_END

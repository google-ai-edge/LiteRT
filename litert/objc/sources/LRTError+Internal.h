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

#import "third_party/odml/litert/litert/objc/apis/LRTError.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Creates an @c NSError with the LiteRT error domain, given error code and description.
 *
 * @param code Error code matching @c LRTErrorCode or litert::Status.
 * @param description Description message for the error.
 * @return An @c NSError instance with domain @c LRTErrorDomain.
 */
static inline NSError *CreateLRTError(NSInteger code, NSString *description) {
  return [NSError errorWithDomain:LRTErrorDomain
                             code:code
                         userInfo:@{NSLocalizedDescriptionKey : description}];
}

NS_ASSUME_NONNULL_END

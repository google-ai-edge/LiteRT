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

#import "third_party/odml/litert/litert/objc/apis/LRTTensorBufferRequirements.h"

#include "litert/cc/litert_tensor_buffer_requirements.h"

NS_ASSUME_NONNULL_BEGIN

@interface LRTTensorBufferRequirements (Internal)

/**
 * Creates an Objective-C @c LRTTensorBufferRequirements instance from a C++
 * @c litert::TensorBufferRequirements object.
 */
+ (nullable instancetype)requirementsWithCppRequirements:
                             (const litert::TensorBufferRequirements &)cppRequirements
                                                   error:(NSError **)error;

/** Pointer to the underlying C++ @c litert::TensorBufferRequirements object. */
- (nullable const litert::TensorBufferRequirements *)cppRequirements;

@end

NS_ASSUME_NONNULL_END

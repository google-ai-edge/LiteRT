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

#import "third_party/odml/litert/litert/objc/apis/LRTTensorBuffer.h"

#include "litert/cc/litert_tensor_buffer.h"

NS_ASSUME_NONNULL_BEGIN

@interface LRTTensorBuffer (Internal)

/** Creates an Objective-C @c LRTTensorBuffer wrapping a C++ @c litert::TensorBuffer object. */
+ (nullable instancetype)tensorBufferWithCppTensorBuffer:(litert::TensorBuffer)cppTensorBuffer;

/** Pointer to the underlying C++ @c litert::TensorBuffer object. */
- (nullable litert::TensorBuffer *)cppTensorBuffer;

@end

NS_ASSUME_NONNULL_END

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

#import "third_party/odml/litert/litert/objc/apis/LRTTensorType.h"

#include <memory>

#include "litert/cc/litert_ranked_tensor_type.h"

NS_ASSUME_NONNULL_BEGIN

@interface LRTTensorType ()

/** Creates an Objective-C @c LRTTensorType from a C++ @c litert::RankedTensorType. */
+ (instancetype)tensorTypeWithCPPTensorType:(const litert::RankedTensorType &)cppTensorType;

/**
 * Converts the receiver into a C++ @c litert::RankedTensorType suitable for buffer allocation.
 *
 * @param error Out-parameter populated if the element type is @c LRTElementTypeNone or any
 * dimension is negative.
 * @return The C++ @c litert::RankedTensorType, or @c nullptr on failure.
 */
- (std::unique_ptr<litert::RankedTensorType>)cppTensorTypeWithError:(NSError **)error;

@end

NS_ASSUME_NONNULL_END

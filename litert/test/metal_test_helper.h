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

#import <Metal/Metal.h>
#import <XCTest/XCTest.h>
#import <XCTest/XCTestAssertions.h>

#include "litert/cc/litert_tensor_buffer.h"

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_METAL_TEST_HELPER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_METAL_TEST_HELPER_H_

@interface MetalTestHelper : NSObject

// Returns the file path of the model file in the bundle.
+ (NSString *_Nullable)pathForModelName:(NSString *_Nonnull)modelName;

// Checks the output of a TensorBuffer of float values against the expected output.
//
// @param output_buffer The TensorBuffer to check.
// @param expected_output The expected output values.
// @param elementCount The number of elements in the output buffer and expected output.
// @param tolerance The tolerance for floating point comparison.
+ (void)checkTensorBufferFloatOutput:(litert::TensorBuffer *_Nonnull)output_buffer
                  withExpectedOutput:(const float *_Nonnull)expected_output
                    withElementCount:(size_t)element_count
                       withTolerance:(float)tolerance;

// Checks the output of a TensorBuffer of int32_t values against the expected output.
//
// @param output_buffer The TensorBuffer to check.
// @param expected_output The expected output values.
// @param elementCount The number of elements in the output buffer and expected output.
+ (void)checkTensorBufferInt32Output:(litert::TensorBuffer *_Nonnull)output_buffer
                  withExpectedOutput:(const int32_t *_Nonnull)expected_output
                    withElementCount:(size_t)element_count;
@end

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_METAL_TEST_HELPER_H_

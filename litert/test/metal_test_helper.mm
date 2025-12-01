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

#import "third_party/odml/litert/litert/test/metal_test_helper.h"

#import <Metal/Metal.h>
#import <XCTest/XCTest.h>
#import <XCTest/XCTestAssertions.h>

#include "litert/test/common.h"
#include "litert/test/matchers.h"

using litert::TensorBuffer;
using litert::TensorBufferScopedLock;

@implementation MetalTestHelper

+ (NSString *_Nullable)pathForModelName:(NSString *_Nonnull)modelName {
  // Get the bundle for the current test class
  NSBundle *bundle = [NSBundle bundleForClass:[self class]];
  // Construct the full path to the model file
  NSString *modelFilePath = [bundle pathForResource:modelName ofType:@"tflite"];
  if (!modelFilePath) {
    XCTFail(@"Could not find model file in bundle.");
    return nil;
  }
  return modelFilePath;
}

+ (void)checkTensorBufferFloatOutput:(litert::TensorBuffer *_Nonnull)output_buffer
                  withExpectedOutput:(const float *_Nonnull)expected_output
                    withElementCount:(size_t)element_count
                       withTolerance:(float)tolerance {
  auto lock_and_addr =
      TensorBufferScopedLock::Create<const float>(*output_buffer, TensorBuffer::LockMode::kRead);
  XCTAssertTrue(lock_and_addr);
  auto output = absl::MakeSpan(lock_and_addr->second, element_count);
  for (auto i = 0; i < element_count; ++i) {
    LITERT_LOG(LITERT_INFO, "Result: %f\texpected: %f", output[i], expected_output[i]);
  }
  XCTAssertTrue(testing::Matches(testing::Pointwise(
      testing::FloatNear(tolerance), absl::MakeConstSpan(expected_output, element_count)))(output));
}

+ (void)checkTensorBufferInt32Output:(litert::TensorBuffer *_Nonnull)output_buffer
                  withExpectedOutput:(const int32_t *_Nonnull)expected_output
                    withElementCount:(size_t)element_count {
  auto lock_and_addr =
      TensorBufferScopedLock::Create<const int32_t>(*output_buffer, TensorBuffer::LockMode::kRead);
  XCTAssertTrue(lock_and_addr);
  auto output = absl::MakeSpan(lock_and_addr->second, element_count);
  for (auto i = 0; i < element_count; ++i) {
    LITERT_LOG(LITERT_INFO, "Result: %d\texpected: %d", output[i], expected_output[i]);
  }
  XCTAssertTrue(testing::Matches(testing::Pointwise(
      testing::Eq(), absl::MakeConstSpan(expected_output, element_count)))(output));
}

@end

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

#import <XCTest/XCTest.h>
#import <XCTest/XCTestAssertions.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

namespace {
litert::Expected<litert::Options> CreateGpuOptions(bool external_tensors_mode) {
  LITERT_ASSIGN_OR_RETURN(auto gpu_options, litert::GpuOptions::Create());

  LITERT_RETURN_IF_ERROR(gpu_options.EnableExternalTensorsMode(external_tensors_mode));
  LITERT_ASSIGN_OR_RETURN(litert::Options options, litert::Options::Create());
  options.SetHardwareAccelerators(kLiteRtHwAcceleratorGpu);
  gpu_options.SetDelegatePrecision(kLiteRtDelegatePrecisionFp32);
  options.AddOpaqueOptions(std::move(gpu_options));
  return std::move(options);
}
}  // namespace

const float kTolerance = 1e-5;

@interface LitertCompiledModelMetalTest : XCTestCase
@end

@implementation LitertCompiledModelMetalTest

- (void)testCompiledModelMetalGpuEnvironment {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));

  // Get the bundle for the current test class
  NSBundle *bundle = [NSBundle bundleForClass:[self class]];

  // Construct the full path to the model file
  NSString *modelFilePath = [bundle pathForResource:@"simple_model" ofType:@"tflite"];

  if (!modelFilePath) {
    XCTFail(@"Could not find model file in bundle.");
    return;
  }

  auto model = litert::Model::CreateFromFile(modelFilePath.UTF8String);
  XCTAssertTrue(model);
  XCTAssertTrue(env);

  LITERT_ASSERT_OK_AND_ASSIGN(auto options, CreateGpuOptions(/*external_tensors_mode=*/true));
  XCTAssertTrue(options);
  LITERT_ASSERT_OK_AND_ASSIGN(auto compiled_model,
                              litert::CompiledModel::Create(env, *model, options));
  XCTAssertEqual(model->GetNumSignatures(), 1);
  XCTAssertTrue(compiled_model);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers, compiled_model.CreateInputBuffers());

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers, compiled_model.CreateOutputBuffers());

  // // Fill model inputs.
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names, model->GetSignatureInputNames());
  XCTAssertEqual(input_names.size(), 2);
  XCTAssertEqual(input_names.at(0), "arg0");
  XCTAssertEqual(input_names.at(1), "arg1");
  XCTAssertTrue(input_buffers[0].IsMetalMemory());
  XCTAssertTrue(
      input_buffers[0].Write<float>(absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  XCTAssertTrue(input_buffers[1].IsMetalMemory());
  XCTAssertTrue(
      input_buffers[1].Write<float>(absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model.
  compiled_model.Run(input_buffers, output_buffers);

  // Check model output.
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names, model->GetSignatureOutputNames());
  XCTAssertEqual(output_names.size(), 1);
  XCTAssertEqual(output_names.at(0), "tfl.add");
  XCTAssertTrue(output_buffers[0].IsMetalMemory());
  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create<const float>(
        output_buffers[0], litert::TensorBuffer::LockMode::kRead);
    XCTAssertTrue(lock_and_addr);
    auto output = absl::MakeSpan(lock_and_addr->second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      LITERT_LOG(LITERT_INFO, "Result: %f\t%f", output[i], kTestOutputTensor[i]);
    }
    XCTAssertTrue(testing::Matches(testing::Pointwise(
        testing::FloatNear(kTolerance), absl::MakeConstSpan(kTestOutputTensor, kTestOutputSize)))(
        output));
  }

  return;
}

@end

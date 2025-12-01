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

using litert::TensorBuffer;

namespace {
litert::Expected<litert::Options> CreateGpuOptions(bool external_tensors_mode) {
  LITERT_ASSIGN_OR_RETURN(litert::Options options, litert::Options::Create());
  options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
  LITERT_ASSIGN_OR_RETURN(auto &gpu_options, options.GetGpuOptions());
  LITERT_RETURN_IF_ERROR(gpu_options.EnableExternalTensorsMode(external_tensors_mode));
  LITERT_RETURN_IF_ERROR(gpu_options.SetPrecision(litert::GpuOptions::Precision::kFp32));
  return std::move(options);
}
}  // namespace

const float kTolerance = 1e-5;

@interface BasicMetalTest : NSObject

// Tests the model with the given external tensors mode configuration.
//
// @param externalTensorsMode Whether to use external tensors mode.
+ (void)testBasicMetalTest:(BOOL)externalTensorsMode;

@end

@implementation BasicMetalTest

+ (void)testBasicMetalTest:(BOOL)externalTensorsMode {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  XCTAssertTrue(env);

  NSString *modelFilePath = [MetalTestHelper pathForModelName:@"simple_model"];
  XCTAssertNotNil(modelFilePath);

  LITERT_ASSERT_OK_AND_ASSIGN(auto options, CreateGpuOptions(externalTensorsMode));
  XCTAssertTrue(options);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model, litert::CompiledModel::Create(env, modelFilePath.UTF8String, options));
  XCTAssertEqual(compiled_model.GetNumSignatures(), 1);
  XCTAssertTrue(compiled_model);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers, compiled_model.CreateInputBuffers());

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers, compiled_model.CreateOutputBuffers());

  // // Fill model inputs.
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names, compiled_model.GetSignatureInputNames());
  XCTAssertEqual(input_names.size(), 2);
  XCTAssertEqualObjects([NSString stringWithUTF8String:input_names.at(0).data()], @"arg0");
  XCTAssertEqualObjects([NSString stringWithUTF8String:input_names.at(1).data()], @"arg1");
  XCTAssertTrue(input_buffers[0].IsMetalMemory());
  XCTAssertTrue(
      input_buffers[0].Write<float>(absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  XCTAssertTrue(input_buffers[1].IsMetalMemory());
  XCTAssertTrue(
      input_buffers[1].Write<float>(absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model.
  compiled_model.Run(input_buffers, output_buffers);

  // Check model output.
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names, compiled_model.GetSignatureOutputNames());
  XCTAssertEqual(output_names.size(), 1);
  XCTAssertEqualObjects([NSString stringWithUTF8String:output_names.at(0).data()], @"tfl.add");
  XCTAssertTrue(output_buffers[0].IsMetalMemory());
  litert::TensorBuffer *output_buffer = &output_buffers.at(0);
  [MetalTestHelper checkTensorBufferFloatOutput:output_buffer
                             withExpectedOutput:kTestOutputTensor
                               withElementCount:kTestOutputSize
                                  withTolerance:kTolerance];
}

@end

@interface LitertCompiledModelMetalTest : XCTestCase
@end

@implementation LitertCompiledModelMetalTest

- (void)testCompiledModelGpuBasic {
  [BasicMetalTest testBasicMetalTest:false];
}

- (void)testCompiledModelGpuBasic2nd {
  // Run the test twice to verify that the GPU environment is shared between two CompiledModel
  // instances.
  [BasicMetalTest testBasicMetalTest:false];
}

- (void)testCompiledModelGpuExternalTensorsMode {
  // Test the model with external tensors mode enabled.
  [BasicMetalTest testBasicMetalTest:true];
}

- (void)testCompiledModelGpuEnvironment {
  auto env = litert::Environment::Create({});
  XCTAssertTrue(env);

  NSString *modelFilePath = [MetalTestHelper pathForModelName:@"simple_model"];
  XCTAssertNotNil(modelFilePath);

  LITERT_ASSERT_OK_AND_ASSIGN(auto options, CreateGpuOptions(/*external_tensors_mode=*/false));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model, litert::CompiledModel::Create(*env, modelFilePath.UTF8String, options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto env_options, env->GetOptions());
  LITERT_ASSERT_OK_AND_ASSIGN(auto metal_device_id,
                              env_options.GetOption(litert::EnvironmentOptions::Tag::kMetalDevice));
  id<MTLDevice> metal_device = (__bridge id<MTLDevice>)(std::get<void *>(metal_device_id));
  XCTAssertNotNil(metal_device);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto metal_command_queue_id,
      env_options.GetOption(litert::EnvironmentOptions::Tag::kMetalCommandQueue));
  id<MTLCommandQueue> command_queue =
      (__bridge id<MTLCommandQueue>)(std::get<void *>(metal_command_queue_id));
  XCTAssertNotNil(command_queue);
}

- (void)testCompiledModelGpuPartialDelegation {
  NSString *modelFilePath = [MetalTestHelper pathForModelName:@"simple_cast_and_add_op"];
  XCTAssertNotNil(modelFilePath);

  auto env = litert::Environment::Create({});
  XCTAssertTrue(env);

  litert::HwAcceleratorSet accelerator_flags =
      litert::HwAccelerators::kGpu | litert::HwAccelerators::kCpu;
  auto compilation_options = litert::Options::Create();
  compilation_options->SetHardwareAccelerators(accelerator_flags);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      litert::CompiledModel::Create(*env, modelFilePath.UTF8String, *compilation_options));

  XCTAssertEqual(compiled_model.GetNumSignatures(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers, compiled_model.CreateInputBuffers());

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers, compiled_model.CreateOutputBuffers());

  // Fill model inputs.
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names, compiled_model.GetSignatureInputNames());
  XCTAssertEqual(input_names.size(), 3);
  XCTAssertEqualObjects([NSString stringWithUTF8String:input_names.at(0).data()], @"arg0");
  XCTAssertEqualObjects([NSString stringWithUTF8String:input_names.at(1).data()], @"arg1");
  XCTAssertEqualObjects([NSString stringWithUTF8String:input_names.at(2).data()], @"arg2");
  XCTAssertTrue(input_buffers[0].IsMetalMemory());
  XCTAssertTrue(
      input_buffers[0].Write<float>(absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  XCTAssertTrue(input_buffers[1].IsMetalMemory());
  XCTAssertTrue(
      input_buffers[1].Write<float>(absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));
  int64_t arg2_data[1] = {1};
  XCTAssertTrue(input_buffers[2].Write<int64_t>(absl::MakeConstSpan(arg2_data, 1)));

  // Execute model.
  compiled_model.Run(input_buffers, output_buffers);

  // Check model output.
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names, compiled_model.GetSignatureOutputNames());
  XCTAssertEqual(output_names.size(), 1);
  XCTAssertEqualObjects([NSString stringWithUTF8String:output_names.at(0).data()], @"tfl.add1");
  XCTAssertTrue(output_buffers[0].IsMetalMemory());

  float kExpectedOutput[2] = {12.0f, 23.0f};
  [MetalTestHelper checkTensorBufferFloatOutput:&output_buffers[0]
                             withExpectedOutput:kExpectedOutput
                               withElementCount:2
                                  withTolerance:kTolerance];
}

- (void)testCompiledModelGpuBasicAdd3dCstInt32 {
  constexpr const int32_t kInt32TestInput0Tensor[] = {1, 2, 3, 4, 5, 6};
  constexpr const int32_t kInt32TestOutputTensor[] = {11, 22, 33, 44, 55, 66};
  constexpr const size_t kInt32TestInput0Size = 6;
  constexpr const size_t kInt32TestOutputSize = 6;
  NSString *modelFilePath = [MetalTestHelper pathForModelName:@"simple_add3d_cst_int32"];
  XCTAssertNotNil(modelFilePath);

  auto env = litert::Environment::Create({});
  XCTAssertTrue(env);
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, CreateGpuOptions(/*external_tensors_mode=*/false));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model, litert::CompiledModel::Create(*env, modelFilePath.UTF8String, options));

  XCTAssertEqual(compiled_model.GetNumSignatures(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers, compiled_model.CreateInputBuffers());

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers, compiled_model.CreateOutputBuffers());

  // Fill model inputs.
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names, compiled_model.GetSignatureInputNames());
  XCTAssertEqual(input_names.size(), 1);
  XCTAssertEqualObjects([NSString stringWithUTF8String:input_names.at(0).data()], @"arg0");
  XCTAssertTrue(input_buffers[0].IsMetalMemory());
  XCTAssertTrue(input_buffers[0].Write<int32_t>(
      absl::MakeConstSpan(kInt32TestInput0Tensor, kInt32TestInput0Size)));

  // Execute model.
  compiled_model.Run(input_buffers, output_buffers);

  // Check model output.
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names, compiled_model.GetSignatureOutputNames());
  XCTAssertEqual(output_names.size(), 1);
  XCTAssertEqualObjects([NSString stringWithUTF8String:output_names.at(0).data()], @"tfl.add");
  XCTAssertTrue(output_buffers[0].IsMetalMemory());
  [MetalTestHelper checkTensorBufferInt32Output:&output_buffers[0]
                             withExpectedOutput:kInt32TestOutputTensor
                               withElementCount:kInt32TestOutputSize];
}

@end

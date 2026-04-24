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

// Tests the model with the given execution mode and external tensors mode configuration.
//
// @param asyncMode Whether to use async execution mode.
// @param externalTensorsMode Whether to use external tensors mode.
+ (void)testBasicMetalTest:(BOOL)asyncMode externalTensorsMode:(BOOL)externalTensorsMode;

@end

@implementation BasicMetalTest

+ (void)testBasicMetalTest:(BOOL)asyncMode externalTensorsMode:(BOOL)externalTensorsMode {
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
  if (asyncMode) {
    bool async = false;
    litert::Expected<void> result = compiled_model.RunAsync(input_buffers, output_buffers, async);
    XCTAssertTrue(result);
    XCTAssertTrue(async);
  } else {
    litert::Expected<void> result = compiled_model.Run(input_buffers, output_buffers);
    XCTAssertTrue(result);
  }

  // Check model output.
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names, compiled_model.GetSignatureOutputNames());
  XCTAssertEqual(output_names.size(), 1);
  XCTAssertEqualObjects([NSString stringWithUTF8String:output_names.at(0).data()], @"tfl.add");
  XCTAssertTrue(output_buffers[0].IsMetalMemory());
  if (asyncMode) {
    XCTAssertTrue(output_buffers[0].HasEvent());
    litert::Expected<litert::Event> event = output_buffers[0].GetEvent();
    XCTAssertTrue(event);
    litert::Expected<bool> result = event->IsSignaled();
    XCTAssertTrue(result);
    XCTAssertFalse(*result);  // Not signaled yet.
  }
  litert::TensorBuffer *output_buffer = &output_buffers.at(0);
  [MetalTestHelper checkTensorBufferFloatOutput:output_buffer
                             withExpectedOutput:kTestOutputTensor
                               withElementCount:kTestOutputSize
                                  withTolerance:kTolerance];
  if (asyncMode) {
    litert::Expected<litert::Event> event = output_buffers[0].GetEvent();
    XCTAssertTrue(event);
    litert::Expected<bool> result = event->IsSignaled();
    XCTAssertTrue(result);
    // Buffer lock above lets the event be signaled.
    XCTAssertTrue(*result);
  }
}

@end

@interface MetalPipelineTest : NSObject

// Tests the model with the given execution mode and external tensors mode configuration.
//
// @param asyncMode1stModel Whether to use async execution mode for 1st model.
// @param asyncMode2ndModel Whether to use async execution mode for 2nd model.
// @param externalTensorsMode Whether to use external tensors mode.
+ (void)testMetalPipelineTest:(BOOL)asyncMode1stModel
            asyncMode2ndModel:(BOOL)asyncMode2ndModel
          externalTensorsMode:(BOOL)externalTensorsMode;

@end

@implementation MetalPipelineTest

+ (void)testMetalPipelineTest:(BOOL)asyncMode1stModel
            asyncMode2ndModel:(BOOL)asyncMode2ndModel
          externalTensorsMode:(BOOL)externalTensorsMode {
  constexpr const float kTestOutputTensorForPipelineTest[] = {21, 42};

  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  XCTAssertTrue(env);

  NSString *modelFilePath = [MetalTestHelper pathForModelName:@"simple_model"];
  XCTAssertNotNil(modelFilePath);

  LITERT_ASSERT_OK_AND_ASSIGN(auto options, CreateGpuOptions(externalTensorsMode));
  XCTAssertTrue(options);

  // Create 1st model.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model_1, litert::CompiledModel::Create(env, modelFilePath.UTF8String, options));
  XCTAssertEqual(compiled_model_1.GetNumSignatures(), 1);
  XCTAssertTrue(compiled_model_1);
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers_1, compiled_model_1.CreateInputBuffers());
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers_1, compiled_model_1.CreateOutputBuffers());

  // Create 2nd model.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model_2, litert::CompiledModel::Create(env, modelFilePath.UTF8String, options));
  XCTAssertEqual(compiled_model_2.GetNumSignatures(), 1);
  XCTAssertTrue(compiled_model_2);

  // One of input buffers of 2nd model is same as output of 1st model.
  // Set rest of the input buffers of 2nd model same as 1st model's input
  // buffers.
  std::vector<TensorBuffer> input_buffers_2(2);
  LITERT_ASSERT_OK_AND_ASSIGN(input_buffers_2[0], output_buffers_1[0].Duplicate());
  LITERT_ASSERT_OK_AND_ASSIGN(input_buffers_2[1], input_buffers_1[1].Duplicate());

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers_2, compiled_model_2.CreateOutputBuffers());

  // Fill model inputs for 1st model.
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names, compiled_model_1.GetSignatureInputNames());
  XCTAssertEqual(input_names.size(), 2);
  XCTAssertEqualObjects([NSString stringWithUTF8String:input_names.at(0).data()], @"arg0");
  XCTAssertEqualObjects([NSString stringWithUTF8String:input_names.at(1).data()], @"arg1");
  XCTAssertTrue(input_buffers_1[0].IsMetalMemory());
  XCTAssertTrue(
      input_buffers_1[0].Write<float>(absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  XCTAssertTrue(input_buffers_1[1].IsMetalMemory());
  XCTAssertTrue(
      input_buffers_1[1].Write<float>(absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute 1st model.
  if (asyncMode1stModel) {
    bool async = false;
    litert::Expected<void> result =
        compiled_model_1.RunAsync(input_buffers_1, output_buffers_1, async);
    XCTAssertTrue(result);
    XCTAssertTrue(async);
  } else {
    litert::Expected<void> result = compiled_model_1.Run(input_buffers_1, output_buffers_1);
    XCTAssertTrue(result);
  }

  // Execute 2nd model.
  if (asyncMode2ndModel) {
    bool async = false;
    litert::Expected<void> result =
        compiled_model_2.RunAsync(input_buffers_2, output_buffers_2, async);
    XCTAssertTrue(result);
    XCTAssertTrue(async);
  } else {
    litert::Expected<void> result = compiled_model_2.Run(input_buffers_2, output_buffers_2);
    XCTAssertTrue(result);
  }

  // Check 2nd model output.
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names, compiled_model_2.GetSignatureOutputNames());
  XCTAssertEqual(output_names.size(), 1);
  XCTAssertEqualObjects([NSString stringWithUTF8String:output_names.at(0).data()], @"tfl.add");
  XCTAssertTrue(output_buffers_2[0].IsMetalMemory());
  if (asyncMode2ndModel) {
    XCTAssertTrue(output_buffers_2[0].HasEvent());
    litert::Expected<litert::Event> event = output_buffers_2[0].GetEvent();
    XCTAssertTrue(event);
    litert::Expected<bool> result = event->IsSignaled();
    XCTAssertTrue(result);
    XCTAssertFalse(*result);  // Not signaled yet.
  }
  litert::TensorBuffer *output_buffer = &output_buffers_2.at(0);
  [MetalTestHelper checkTensorBufferFloatOutput:output_buffer
                             withExpectedOutput:kTestOutputTensorForPipelineTest
                               withElementCount:kTestOutputSize
                                  withTolerance:kTolerance];
  if (asyncMode2ndModel) {
    litert::Expected<litert::Event> event = output_buffers_2[0].GetEvent();
    XCTAssertTrue(event);
    litert::Expected<bool> result = event->IsSignaled();
    XCTAssertTrue(result);
    // Buffer lock above lets the event be signaled.
    XCTAssertTrue(*result);
  }
}

@end

@interface LitertCompiledModelMetalTest : XCTestCase
@end

@implementation LitertCompiledModelMetalTest

- (void)testCompiledModelGpuBasic {
  [BasicMetalTest testBasicMetalTest:false externalTensorsMode:false];
}

- (void)testCompiledModelGpuBasicAsync {
  [BasicMetalTest testBasicMetalTest:true externalTensorsMode:false];
}

- (void)testCompiledModelGpuExternalTensorsMode {
  [BasicMetalTest testBasicMetalTest:false externalTensorsMode:true];
}

- (void)testCompiledModelGpuExternalTensorsModeAsync {
  [BasicMetalTest testBasicMetalTest:true externalTensorsMode:true];
}

- (void)testCompiledModelGpuPipeline {
  [MetalPipelineTest testMetalPipelineTest:false asyncMode2ndModel:false externalTensorsMode:false];
}

- (void)testCompiledModelGpuPipelineAsync1stModel {
  [MetalPipelineTest testMetalPipelineTest:true asyncMode2ndModel:false externalTensorsMode:false];
}

- (void)testCompiledModelGpuPipelineAsync2ndModel {
  [MetalPipelineTest testMetalPipelineTest:false asyncMode2ndModel:true externalTensorsMode:false];
}

- (void)testCompiledModelGpuPipelineAsyncBothModels {
  [MetalPipelineTest testMetalPipelineTest:true asyncMode2ndModel:true externalTensorsMode:false];
}

- (void)testCompiledModelGpuPipelineExternalTensorsMode {
  [MetalPipelineTest testMetalPipelineTest:false asyncMode2ndModel:false externalTensorsMode:true];
}

- (void)testCompiledModelGpuPipelineExternalTensorsModeAsync1stModel {
  [MetalPipelineTest testMetalPipelineTest:true asyncMode2ndModel:false externalTensorsMode:true];
}

- (void)testCompiledModelGpuPipelineExternalTensorsModeAsync2ndModel {
  [MetalPipelineTest testMetalPipelineTest:false asyncMode2ndModel:true externalTensorsMode:true];
}

- (void)testCompiledModelGpuPipelineExternalTensorsModeAsyncBothModels {
  [MetalPipelineTest testMetalPipelineTest:true asyncMode2ndModel:true externalTensorsMode:true];
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

- (void)testCompiledModelGpuConstantOutputTensor {
  NSString *modelFilePath = [MetalTestHelper pathForModelName:@"constant_output_tensor"];
  XCTAssertNotNil(modelFilePath);

  auto env = litert::Environment::Create({});
  XCTAssertTrue(env);
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, CreateGpuOptions(/*external_tensors_mode=*/false));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model, litert::CompiledModel::Create(*env, modelFilePath.UTF8String, options));

  XCTAssertEqual(compiled_model.GetNumSignatures(), 1);

  // Create input and output buffers
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers, compiled_model.CreateInputBuffers());
  XCTAssertEqual(input_buffers.size(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers, compiled_model.CreateOutputBuffers());
  XCTAssertEqual(output_buffers.size(), 2);  // normal_output and constant_output

  // Set input values
  const float input_data[] = {5.0f, 10.0f};
  XCTAssertTrue(input_buffers[0].Write<float>(absl::MakeConstSpan(input_data, 2)));

  // Run the model
  LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));

  // Note: TFLite might reorder outputs - check which is which by size
  // The constant output has 4 elements, the normal output has 2 elements
  int constant_output_idx = -1;
  int normal_output_idx = -1;

  // Determine which output is which based on size
  for (int i = 0; i < 2; i++) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto size, output_buffers[i].Size());
    if (size == 4 * sizeof(float)) {
      constant_output_idx = i;
    } else if (size == 2 * sizeof(float)) {
      normal_output_idx = i;
    }
  }

  ASSERT_NE(constant_output_idx, -1) << "Could not find constant output";
  ASSERT_NE(normal_output_idx, -1) << "Could not find normal output";

  {
    const float kNormalExpectedOutput[] = {10.0f, 20.0f};
    [MetalTestHelper checkTensorBufferFloatOutput:&output_buffers[normal_output_idx]
                               withExpectedOutput:kNormalExpectedOutput
                                 withElementCount:2
                                    withTolerance:kTolerance];

    const float kConstantExpectedOutput[] = {1.0f, 2.0f, 3.0f, 4.0f};
    [MetalTestHelper checkTensorBufferFloatOutput:&output_buffers[constant_output_idx]
                               withExpectedOutput:kConstantExpectedOutput
                                 withElementCount:4
                                    withTolerance:kTolerance];
  }

  // Run again with different input to verify constant output doesn't change
  const float input_data2[] = {100.0f, 200.0f};
  XCTAssertTrue(input_buffers[0].Write<float>(absl::MakeConstSpan(input_data2, 2)));
  LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));

  {
    const float kNormalExpectedOutput[] = {200.0f, 400.0f};
    [MetalTestHelper checkTensorBufferFloatOutput:&output_buffers[normal_output_idx]
                               withExpectedOutput:kNormalExpectedOutput
                                 withElementCount:2
                                    withTolerance:kTolerance];

    const float kConstantExpectedOutput[] = {1.0f, 2.0f, 3.0f, 4.0f};
    [MetalTestHelper checkTensorBufferFloatOutput:&output_buffers[constant_output_idx]
                               withExpectedOutput:kConstantExpectedOutput
                                 withElementCount:4
                                    withTolerance:kTolerance];
  }
}

- (void)testCompiledModelexternalTensorBinding {
  auto env = litert::Environment::Create({});
  XCTAssertTrue(env);

  // Create weight tensor buffer.
  alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) float kWeightTensor[] = {1.0f, 2.0f};
  constexpr int kWeightSize = sizeof(kWeightTensor);

  // Create Compilation options and bind weight tensor.
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, CreateGpuOptions(/*external_tensors_mode=*/false));
  LITERT_ASSERT_OK(options.AddExternalTensorBinding(
      /*signature_name=*/"", /*tensor_name=*/"arg1", kWeightTensor, kWeightSize));
  NSString *modelFilePath = [MetalTestHelper pathForModelName:@"simple_model"];
  XCTAssertNotNil(modelFilePath);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model, litert::CompiledModel::Create(*env, modelFilePath.UTF8String, options));

  // Create and fill input and output buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> output_buffers,
                              compiled_model.CreateOutputBuffers());
  absl::flat_hash_map<absl::string_view, TensorBuffer> output_map;
  output_map["tfl.add"] = std::move(output_buffers[0]);

  absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
  float kInputTensor[] = {1.0f, 1.0f};
  LITERT_ASSERT_OK_AND_ASSIGN(litert::TensorBufferRequirements requirements,
                              compiled_model.GetInputBufferRequirements(0));
  LITERT_ASSERT_OK_AND_ASSIGN(auto buffer_type, requirements.SupportedTypes());
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer arg0_buffer,
      TensorBuffer::CreateManaged(*env, buffer_type[0],
                                  litert::RankedTensorType(litert::ElementType::Float32,
                                                           litert::Layout(litert::Dimensions({2}))),
                                  sizeof(kInputTensor)));
  LITERT_ASSERT_OK(arg0_buffer.Write<float>(absl::MakeConstSpan(kInputTensor, 2)));
  input_map["arg0"] = std::move(arg0_buffer);

  // Execute model with input and output buffers.
  LITERT_ASSERT_OK(compiled_model.Run(input_map, output_map));

  // Check model output.
  constexpr float kExpectedOutput[] = {2.0f, 3.0f};
  [MetalTestHelper checkTensorBufferFloatOutput:&output_map["tfl.add"]
                             withExpectedOutput:kExpectedOutput
                               withElementCount:2
                                  withTolerance:kTolerance];
}
@end

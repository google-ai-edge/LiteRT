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

#import "third_party/odml/litert/litert/objc/apis/LRTCompiledModel.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#import "third_party/odml/litert/litert/objc/apis/LRTError.h"
#import "third_party/odml/litert/litert/objc/sources/LRTEnvironment+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTOptions+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTTensorBuffer+Internal.h"

NS_ASSUME_NONNULL_BEGIN

@implementation LRTCompiledModel {
  std::unique_ptr<litert::CompiledModel> _cppCompiledModel;
}

- (instancetype)initInternalWithCppCompiledModel:
                    (std::unique_ptr<litert::CompiledModel>)cppCompiledModel
                                     environment:(LRTEnvironment *)environment
                                         options:(nullable LRTOptions *)options {
  self = [super init];
  if (self) {
    _cppCompiledModel = std::move(cppCompiledModel);
    _environment = environment;
    _options = options;
  }
  return self;
}

+ (nullable instancetype)compiledModelWithModelFilePath:(NSString *)modelFilePath
                                            environment:(LRTEnvironment *)environment
                                                options:(nullable LRTOptions *)options
                                                  error:(NSError **)error {
  if (!modelFilePath) {
    if (error) {
      *error =
          [NSError errorWithDomain:LRTErrorDomain
                              code:LRTErrorCodeInvalidArgument
                          userInfo:@{NSLocalizedDescriptionKey : @"modelFilePath cannot be nil"}];
    }
    return nil;
  }

  if (![environment cppEnvironment]) {
    if (error) {
      *error =
          [NSError errorWithDomain:LRTErrorDomain
                              code:LRTErrorCodeInvalidArgument
                          userInfo:@{NSLocalizedDescriptionKey : @"Valid LRTEnvironment required"}];
    }
    return nil;
  }

  litert::Options emptyOptions;
  litert::Options *cppOpts = options ? [options cppOptions] : &emptyOptions;

  litert::Expected<litert::CompiledModel> createResult = litert::CompiledModel::Create(
      *[environment cppEnvironment], std::string(modelFilePath.UTF8String), *cppOpts);

  if (!createResult.HasValue()) {
    if (error) {
      NSDictionary *userInfo =
          @{NSLocalizedDescriptionKey : @(createResult.Error().Message().c_str())};
      *error = [NSError errorWithDomain:LRTErrorDomain
                                   code:static_cast<NSInteger>(createResult.Error().Status())
                               userInfo:userInfo];
    }
    return nil;
  }

  auto cppPtr = std::make_unique<litert::CompiledModel>(std::move(createResult.Value()));
  return [[LRTCompiledModel alloc] initInternalWithCppCompiledModel:std::move(cppPtr)
                                                        environment:environment
                                                            options:options];
}

/**
 * Helper to convert C++ TensorBuffers result to NSArray of LRTTensorBuffers.
 *
 * @param buffersResult C++ expected vector of TensorBuffers.
 * @param error Out-parameter populated on failure.
 * @return Array of LRTTensorBuffer instances, or nil on failure.
 */
static NSArray<LRTTensorBuffer *> *_Nullable CreateObjCTensorBuffersFromCppResult(
    litert::Expected<std::vector<litert::TensorBuffer>> &buffersResult, NSError **error) {
  if (!buffersResult.HasValue()) {
    if (error) {
      NSDictionary *userInfo =
          @{NSLocalizedDescriptionKey : @(buffersResult.Error().Message().c_str())};
      *error = [NSError errorWithDomain:LRTErrorDomain
                                   code:static_cast<NSInteger>(buffersResult.Error().Status())
                               userInfo:userInfo];
    }
    return nil;
  }

  NSMutableArray<LRTTensorBuffer *> *objcBuffers =
      [NSMutableArray arrayWithCapacity:buffersResult.Value().size()];
  for (auto &cppTensorBuffer : buffersResult.Value()) {
    LRTTensorBuffer *tensorBuffer =
        [LRTTensorBuffer tensorBufferWithCppTensorBuffer:std::move(cppTensorBuffer)];
    if (tensorBuffer) {
      [objcBuffers addObject:tensorBuffer];
    }
  }
  return [objcBuffers copy];
}

/**
 * Helper to duplicate ObjC LRTTensorBuffers to C++ TensorBuffers vector.
 *
 * @param objcBuffers Array of LRTTensorBuffers to duplicate.
 * @param bufferKind String description of buffer kind (e.g. @"input", @"output") for error
 * reporting.
 * @param cppBuffers Destination vector for C++ TensorBuffers.
 * @param error Out-parameter populated on failure.
 * @return YES on success, NO on failure.
 */
static BOOL DuplicateObjCTensorBuffersToCpp(NSArray<LRTTensorBuffer *> *objcBuffers,
                                            NSString *bufferKind,
                                            std::vector<litert::TensorBuffer> &cppBuffers,
                                            NSError **error) {
  cppBuffers.reserve(objcBuffers.count);
  for (LRTTensorBuffer *tensorBuffer in objcBuffers) {
    if (![tensorBuffer cppTensorBuffer]) {
      if (error) {
        NSString *msg = [NSString stringWithFormat:@"Invalid %@ tensor buffer", bufferKind];
        *error = [NSError errorWithDomain:LRTErrorDomain
                                     code:LRTErrorCodeInvalidArgument
                                 userInfo:@{NSLocalizedDescriptionKey : msg}];
      }
      return NO;
    }
    litert::Expected<litert::TensorBuffer> dupResult = [tensorBuffer cppTensorBuffer]->Duplicate();
    if (!dupResult.HasValue()) {
      if (error) {
        NSDictionary *userInfo =
            @{NSLocalizedDescriptionKey : @(dupResult.Error().Message().c_str())};
        *error = [NSError errorWithDomain:LRTErrorDomain
                                     code:static_cast<NSInteger>(dupResult.Error().Status())
                                 userInfo:userInfo];
      }
      return NO;
    }
    cppBuffers.push_back(std::move(dupResult.Value()));
  }
  return YES;
}

- (nullable NSArray<LRTTensorBuffer *> *)createInputTensorBuffersWithError:(NSError **)error {
  if (!_cppCompiledModel) {
    if (error) {
      *error = [NSError
          errorWithDomain:LRTErrorDomain
                     code:LRTErrorCodeRuntimeFailure
                 userInfo:@{NSLocalizedDescriptionKey : @"Compiled model is not initialized"}];
    }
    return nil;
  }

  litert::Expected<std::vector<litert::TensorBuffer>> buffersResult =
      _cppCompiledModel->CreateInputBuffers();
  return CreateObjCTensorBuffersFromCppResult(buffersResult, error);
}

- (nullable NSArray<LRTTensorBuffer *> *)createOutputTensorBuffersWithError:(NSError **)error {
  if (!_cppCompiledModel) {
    if (error) {
      *error = [NSError
          errorWithDomain:LRTErrorDomain
                     code:LRTErrorCodeRuntimeFailure
                 userInfo:@{NSLocalizedDescriptionKey : @"Compiled model is not initialized"}];
    }
    return nil;
  }

  litert::Expected<std::vector<litert::TensorBuffer>> buffersResult =
      _cppCompiledModel->CreateOutputBuffers();
  return CreateObjCTensorBuffersFromCppResult(buffersResult, error);
}

- (BOOL)runWithInputs:(NSArray<LRTTensorBuffer *> *)inputs
              outputs:(NSArray<LRTTensorBuffer *> *)outputs
                error:(NSError **)error {
  if (!_cppCompiledModel) {
    if (error) {
      *error = [NSError
          errorWithDomain:LRTErrorDomain
                     code:LRTErrorCodeRuntimeFailure
                 userInfo:@{NSLocalizedDescriptionKey : @"Compiled model is not initialized"}];
    }
    return NO;
  }

  std::vector<litert::TensorBuffer> inputCppBuffers;
  if (!DuplicateObjCTensorBuffersToCpp(inputs, @"input", inputCppBuffers, error)) {
    return NO;
  }

  std::vector<litert::TensorBuffer> outputCppBuffers;
  if (!DuplicateObjCTensorBuffersToCpp(outputs, @"output", outputCppBuffers, error)) {
    return NO;
  }

  litert::Expected<void> runResult = _cppCompiledModel->Run(inputCppBuffers, outputCppBuffers);
  if (!runResult.HasValue()) {
    if (error) {
      NSDictionary *userInfo =
          @{NSLocalizedDescriptionKey : @(runResult.Error().Message().c_str())};
      *error = [NSError errorWithDomain:LRTErrorDomain
                                   code:static_cast<NSInteger>(runResult.Error().Status())
                               userInfo:userInfo];
    }
    return NO;
  }

  return YES;
}

@end

NS_ASSUME_NONNULL_END

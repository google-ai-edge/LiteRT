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
#import "third_party/odml/litert/litert/objc/sources/LRTEnvironment+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTError+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTOptions+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTTensorBuffer+Internal.h"

NS_ASSUME_NONNULL_BEGIN

namespace {

/**
 * Helper to convert C++ TensorBuffers result to NSArray of LRTTensorBuffers.
 *
 * @param buffersResult C++ expected vector of TensorBuffers.
 * @param error Out-parameter populated on failure.
 * @return Array of LRTTensorBuffer instances, or nil on failure.
 */
NSArray<LRTTensorBuffer *> *_Nullable CreateObjCTensorBuffersFromCppResult(
    litert::Expected<std::vector<litert::TensorBuffer>> &buffersResult, NSError **error) {
  if (!buffersResult.HasValue()) {
    LRTSetErrorFromCppError(error, buffersResult.Error());
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
  return objcBuffers;
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
BOOL DuplicateObjCTensorBuffersToCpp(NSArray<LRTTensorBuffer *> *objcBuffers,
                                     NSString *bufferKind,
                                     std::vector<litert::TensorBuffer> &cppBuffers,
                                     NSError **error) {
  cppBuffers.reserve(objcBuffers.count);
  for (LRTTensorBuffer *tensorBuffer in objcBuffers) {
    if (![tensorBuffer cppTensorBuffer]) {
      LRTSetError(error, LRTErrorCodeInvalidArgument,
                  [NSString stringWithFormat:@"Invalid %@ tensor buffer", bufferKind]);
      return NO;
    }
    litert::Expected<litert::TensorBuffer> dupResult = [tensorBuffer cppTensorBuffer]->Duplicate();
    if (!dupResult.HasValue()) {
      LRTSetErrorFromCppError(error, dupResult.Error());
      return NO;
    }
    cppBuffers.push_back(std::move(dupResult.Value()));
  }
  return YES;
}

}  // namespace

@implementation LRTCompiledModel {
  std::unique_ptr<litert::CompiledModel> _cppCompiledModel;
  /** Model bytes the compiled model was built from, or nil when it was compiled from a file. */
  NSData *_Nullable _modelData;
}

- (instancetype)initInternalWithCppCompiledModel:
                    (std::unique_ptr<litert::CompiledModel>)cppCompiledModel
                                     environment:(LRTEnvironment *)environment
                                         options:(nullable LRTOptions *)options
                                       modelData:(nullable NSData *)modelData {
  self = [super init];
  if (self) {
    _cppCompiledModel = std::move(cppCompiledModel);
    _environment = environment;
    _options = options;
    _modelData = modelData;
  }
  return self;
}

+ (nullable instancetype)compiledModelWithModelFilePath:(NSString *)modelFilePath
                                            environment:(LRTEnvironment *)environment
                                                options:(nullable LRTOptions *)options
                                                  error:(NSError **)error {
  if (!modelFilePath) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"modelFilePath cannot be nil");
    return nil;
  }

  litert::Environment *cppEnvironment = [environment cppEnvironment];
  if (cppEnvironment == nullptr) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Valid LRTEnvironment required");
    return nil;
  }

  litert::Options defaultOptions;
  litert::Options *cppOptions = options ? [options cppOptions] : &defaultOptions;

  litert::Expected<litert::CompiledModel> createResult = litert::CompiledModel::Create(
      *cppEnvironment, std::string(modelFilePath.UTF8String), *cppOptions);

  if (!createResult.HasValue()) {
    LRTSetErrorFromCppError(error, createResult.Error());
    return nil;
  }

  auto cppPtr = std::make_unique<litert::CompiledModel>(std::move(createResult.Value()));
  return [[LRTCompiledModel alloc] initInternalWithCppCompiledModel:std::move(cppPtr)
                                                        environment:environment
                                                            options:options
                                                          modelData:nil];
}

+ (nullable instancetype)compiledModelWithModelData:(NSData *)modelData
                                        environment:(LRTEnvironment *)environment
                                            options:(nullable LRTOptions *)options
                                              error:(NSError **)error {
  if (!modelData || modelData.length == 0) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"modelData cannot be empty");
    return nil;
  }

  litert::Environment *cppEnvironment = [environment cppEnvironment];
  if (cppEnvironment == nullptr) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Valid LRTEnvironment required");
    return nil;
  }

  litert::Options defaultOptions;
  litert::Options *cppOptions = options ? [options cppOptions] : &defaultOptions;

  // LiteRT does not copy the model bytes, it reads them for as long as the compiled model is
  // alive. Hold on to an immutable copy so that callers may release their buffer, or pass an
  // NSMutableData and keep mutating it, without corrupting inference.
  NSData *ownedModelData = [modelData copy];
  litert::BufferRef<uint8_t> bufferRef(static_cast<const uint8_t *>(ownedModelData.bytes),
                                       ownedModelData.length);
  auto createResult = litert::CompiledModel::Create(*cppEnvironment, bufferRef, *cppOptions);

  if (!createResult.HasValue()) {
    LRTSetErrorFromCppError(error, createResult.Error());
    return nil;
  }

  auto cppPtr = std::make_unique<litert::CompiledModel>(std::move(createResult.Value()));
  return [[LRTCompiledModel alloc] initInternalWithCppCompiledModel:std::move(cppPtr)
                                                        environment:environment
                                                            options:options
                                                          modelData:ownedModelData];
}

+ (NSString *)defaultSignatureKey {
  return @(litert::CompiledModel::DefaultSignatureKey().data());
}

- (nullable NSArray<LRTTensorBuffer *> *)createInputTensorBuffersWithError:(NSError **)error {
  return [self createInputTensorBuffersForSignatureIndex:0 error:error];
}

- (nullable NSArray<LRTTensorBuffer *> *)
    createInputTensorBuffersForSignatureIndex:(NSUInteger)signatureIndex
                                        error:(NSError **)error {
  if (!_cppCompiledModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Compiled model is not initialized");
    return nil;
  }

  litert::Expected<std::vector<litert::TensorBuffer>> buffersResult =
      _cppCompiledModel->CreateInputBuffers(signatureIndex);
  return CreateObjCTensorBuffersFromCppResult(buffersResult, error);
}

- (nullable NSArray<LRTTensorBuffer *> *)createInputTensorBuffersForSignatureKey:
                                             (NSString *)signatureKey
                                                                           error:(NSError **)error {
  if (!signatureKey) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"signatureKey cannot be nil");
    return nil;
  }

  if (!_cppCompiledModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Compiled model is not initialized");
    return nil;
  }

  litert::Expected<std::vector<litert::TensorBuffer>> buffersResult =
      _cppCompiledModel->CreateInputBuffers(signatureKey.UTF8String);
  return CreateObjCTensorBuffersFromCppResult(buffersResult, error);
}

- (nullable NSArray<LRTTensorBuffer *> *)createOutputTensorBuffersWithError:(NSError **)error {
  return [self createOutputTensorBuffersForSignatureIndex:0 error:error];
}

- (nullable NSArray<LRTTensorBuffer *> *)
    createOutputTensorBuffersForSignatureIndex:(NSUInteger)signatureIndex
                                         error:(NSError **)error {
  if (!_cppCompiledModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Compiled model is not initialized");
    return nil;
  }

  litert::Expected<std::vector<litert::TensorBuffer>> buffersResult =
      _cppCompiledModel->CreateOutputBuffers(signatureIndex);
  return CreateObjCTensorBuffersFromCppResult(buffersResult, error);
}

- (nullable NSArray<LRTTensorBuffer *> *)
    createOutputTensorBuffersForSignatureKey:(NSString *)signatureKey
                                       error:(NSError **)error {
  if (!signatureKey) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"signatureKey cannot be nil");
    return nil;
  }

  if (!_cppCompiledModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Compiled model is not initialized");
    return nil;
  }

  litert::Expected<std::vector<litert::TensorBuffer>> buffersResult =
      _cppCompiledModel->CreateOutputBuffers(signatureKey.UTF8String);
  return CreateObjCTensorBuffersFromCppResult(buffersResult, error);
}

- (BOOL)runWithInputs:(NSArray<LRTTensorBuffer *> *)inputs
              outputs:(NSArray<LRTTensorBuffer *> *)outputs
                error:(NSError **)error {
  return [self runWithInputs:inputs outputs:outputs signatureIndex:0 error:error];
}

- (BOOL)runWithInputs:(NSArray<LRTTensorBuffer *> *)inputs
              outputs:(NSArray<LRTTensorBuffer *> *)outputs
       signatureIndex:(NSUInteger)signatureIndex
                error:(NSError **)error {
  if (!_cppCompiledModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Compiled model is not initialized");
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

  litert::Expected<void> runResult =
      _cppCompiledModel->Run(signatureIndex, inputCppBuffers, outputCppBuffers);
  if (!runResult.HasValue()) {
    LRTSetErrorFromCppError(error, runResult.Error());
    return NO;
  }

  return YES;
}

- (BOOL)runWithInputs:(NSArray<LRTTensorBuffer *> *)inputs
              outputs:(NSArray<LRTTensorBuffer *> *)outputs
         signatureKey:(NSString *)signatureKey
                error:(NSError **)error {
  NSNumber *signatureIndex = [self signatureIndexForSignatureKey:signatureKey error:error];
  if (signatureIndex == nil) {
    return NO;
  }
  return [self runWithInputs:inputs
                     outputs:outputs
              signatureIndex:signatureIndex.unsignedIntegerValue
                       error:error];
}

- (BOOL)resizeInputTensorAtIndex:(NSUInteger)inputIndex
                   newDimensions:(NSArray<NSNumber *> *)dimensions
                           error:(NSError **)error {
  return [self resizeInputTensorAtIndex:inputIndex
                         signatureIndex:0
                          newDimensions:dimensions
                                  error:error];
}

- (BOOL)resizeInputTensorAtIndex:(NSUInteger)inputIndex
                  signatureIndex:(NSUInteger)signatureIndex
                   newDimensions:(NSArray<NSNumber *> *)dimensions
                           error:(NSError **)error {
  if (!_cppCompiledModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Compiled model is not initialized");
    return NO;
  }

  std::vector<int> cppDims;
  cppDims.reserve(dimensions.count);
  for (NSNumber *dim in dimensions) {
    cppDims.push_back(dim.intValue);
  }

  litert::Expected<void> resizeResult =
      _cppCompiledModel->ResizeInputTensor(signatureIndex, inputIndex, cppDims);

  if (!resizeResult.HasValue()) {
    LRTSetErrorFromCppError(error, resizeResult.Error());
    return NO;
  }

  return YES;
}

- (BOOL)resizeInputTensorAtIndex:(NSUInteger)inputIndex
                    signatureKey:(NSString *)signatureKey
                   newDimensions:(NSArray<NSNumber *> *)dimensions
                           error:(NSError **)error {
  NSNumber *signatureIndex = [self signatureIndexForSignatureKey:signatureKey error:error];
  if (signatureIndex == nil) {
    return NO;
  }
  return [self resizeInputTensorAtIndex:inputIndex
                         signatureIndex:signatureIndex.unsignedIntegerValue
                          newDimensions:dimensions
                                  error:error];
}

/**
 * Returns the index of the signature named @c signatureKey, or nil if the model does not have
 * such a signature.
 *
 * @param signatureKey The name/key of the signature in the model.
 * @param error Out-parameter populated on failure.
 * @return The signature index boxed in an @c NSNumber, or @c nil on failure.
 */
- (nullable NSNumber *)signatureIndexForSignatureKey:(NSString *)signatureKey
                                               error:(NSError **)error {
  if (!signatureKey) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"signatureKey cannot be nil");
    return nil;
  }

  if (!_cppCompiledModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Compiled model is not initialized");
    return nil;
  }

  litert::Expected<size_t> signatureIndexResult =
      _cppCompiledModel->GetSignatureIndex(signatureKey.UTF8String);
  if (!signatureIndexResult.HasValue()) {
    LRTSetErrorFromCppError(error, signatureIndexResult.Error());
    return nil;
  }
  return @(signatureIndexResult.Value());
}

@end

NS_ASSUME_NONNULL_END

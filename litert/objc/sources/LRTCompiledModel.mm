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

#include "litert/cc/litert_api_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#import "third_party/odml/litert/litert/objc/sources/LRTEnvironment+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTError+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTOptions+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTTensorBuffer+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTTensorBufferRequirements+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTTensorType+Internal.h"

namespace {

/**
 * Helper to convert a C++ RankedTensorType result into an Objective-C LRTTensorType instance.
 */
LRTTensorType *CreateObjCTensorTypeFromCPPResult(
    const litert::Expected<litert::RankedTensorType> &tensorTypeResult, NSError **error) {
  if (!tensorTypeResult.HasValue()) {
    LRTSetErrorFromCPPError(error, tensorTypeResult.Error());
    return nil;
  }
  return [LRTTensorType tensorTypeWithCPPTensorType:tensorTypeResult.Value()];
}

/**
 * Helper to convert a C++ StringView vector result into an Objective-C NSArray of NSStrings.
 */
NSArray<NSString *> *CreateObjCStringsFromCPPResult(
    const litert::Expected<std::vector<litert::StringView>> &stringsResult, NSError **error) {
  if (!stringsResult.HasValue()) {
    LRTSetErrorFromCPPError(error, stringsResult.Error());
    return nil;
  }
  return LRTStringArrayFromCPPStringViews(stringsResult.Value());
}

/**
 * Helper to convert a C++ TensorBufferRequirements result into an Objective-C
 * LRTTensorBufferRequirements instance.
 */
LRTTensorBufferRequirements *CreateObjCBufferRequirementsFromCPPResult(
    const litert::Expected<litert::TensorBufferRequirements> &requirementsResult, NSError **error) {
  if (!requirementsResult.HasValue()) {
    LRTSetErrorFromCPPError(error, requirementsResult.Error());
    return nil;
  }
  return [LRTTensorBufferRequirements requirementsWithCPPRequirements:requirementsResult.Value()
                                                                error:error];
}

/**
 * Helper to convert C++ TensorBuffers result to NSArray of LRTTensorBuffers.
 *
 * @param buffersResult C++ expected vector of TensorBuffers.
 * @param error Out-parameter populated on failure.
 * @return Array of LRTTensorBuffer instances, or nil on failure.
 */
NSArray<LRTTensorBuffer *> *CreateObjCTensorBuffersFromCPPResult(
    litert::Expected<std::vector<litert::TensorBuffer>> &buffersResult, NSError **error) {
  if (!buffersResult.HasValue()) {
    LRTSetErrorFromCPPError(error, buffersResult.Error());
    return nil;
  }

  NSMutableArray<LRTTensorBuffer *> *objcBuffers =
      [NSMutableArray arrayWithCapacity:buffersResult.Value().size()];
  for (auto &cppTensorBuffer : buffersResult.Value()) {
    LRTTensorBuffer *tensorBuffer =
        [LRTTensorBuffer tensorBufferWithCPPTensorBuffer:std::move(cppTensorBuffer)];
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
BOOL DuplicateObjCTensorBuffersToCPP(NSArray<LRTTensorBuffer *> *objcBuffers, NSString *bufferKind,
                                     std::vector<litert::TensorBuffer> &cppBuffers,
                                     NSError **error) {
  cppBuffers.reserve(objcBuffers.count);
  for (LRTTensorBuffer *tensorBuffer in objcBuffers) {
    litert::TensorBuffer *cppTensorBuffer = [tensorBuffer cppTensorBuffer];
    if (cppTensorBuffer == nullptr) {
      LRTSetError(error, LRTErrorCodeInvalidArgument,
                  [NSString stringWithFormat:@"Invalid %@ tensor buffer", bufferKind]);
      return NO;
    }
    litert::Expected<litert::TensorBuffer> duplicateResult = cppTensorBuffer->Duplicate();
    if (!duplicateResult.HasValue()) {
      LRTSetErrorFromCPPError(error, duplicateResult.Error());
      return NO;
    }
    cppBuffers.push_back(std::move(duplicateResult.Value()));
  }
  return YES;
}

}  // namespace

@implementation LRTCompiledModel {
  std::unique_ptr<litert::CompiledModel> _cppCompiledModel;
  /** Model bytes the compiled model was built from, or nil when it was compiled from a file. */
  NSData *_modelData;
}

+ (instancetype)compiledModelWithModelFilePath:(NSString *)modelFilePath
                                   environment:(LRTEnvironment *)environment
                                       options:(LRTOptions *)options
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
      *cppEnvironment, LRTCPPStringFromNSString(modelFilePath), *cppOptions);

  if (!createResult.HasValue()) {
    LRTSetErrorFromCPPError(error, createResult.Error());
    return nil;
  }

  auto cppCompiledModel = std::make_unique<litert::CompiledModel>(std::move(createResult.Value()));
  return [[LRTCompiledModel alloc] initInternalWithCPPCompiledModel:std::move(cppCompiledModel)
                                                        environment:environment
                                                            options:options
                                                          modelData:nil];
}

+ (instancetype)compiledModelWithModelData:(NSData *)modelData
                               environment:(LRTEnvironment *)environment
                                   options:(LRTOptions *)options
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
    LRTSetErrorFromCPPError(error, createResult.Error());
    return nil;
  }

  auto cppCompiledModel = std::make_unique<litert::CompiledModel>(std::move(createResult.Value()));
  return [[LRTCompiledModel alloc] initInternalWithCPPCompiledModel:std::move(cppCompiledModel)
                                                        environment:environment
                                                            options:options
                                                          modelData:ownedModelData];
}

+ (NSString *)defaultSignatureKey {
  NSString *signatureKey = LRTStringFromCPPStringView(litert::CompiledModel::DefaultSignatureKey());
  NSAssert(signatureKey != nil, @"LiteRT default signature key is not valid UTF-8");
  return signatureKey;
}

- (instancetype)initInternalWithCPPCompiledModel:
                    (std::unique_ptr<litert::CompiledModel>)cppCompiledModel
                                     environment:(LRTEnvironment *)environment
                                         options:(LRTOptions *)options
                                       modelData:(NSData *)modelData {
  self = [super init];
  if (self) {
    _cppCompiledModel = std::move(cppCompiledModel);
    _environment = environment;
    _options = [options copy];
    _modelData = [modelData copy];
  }
  return self;
}

- (NSString *)description {
  return [NSString stringWithFormat:@"<%@: %p, signatureKeys=%@, options=%@>",
                                    NSStringFromClass([self class]), self, self.signatureKeys,
                                    _options];
}

#pragma mark - Properties

- (NSArray<NSString *> *)signatureKeys {
  if (!_cppCompiledModel) {
    return @[];
  }
  litert::Expected<std::vector<litert::StringView>> keysResult =
      _cppCompiledModel->GetSignatureKeys();
  if (!keysResult.HasValue()) {
    return @[];
  }
  return LRTStringArrayFromCPPStringViews(keysResult.Value());
}

#pragma mark - Public

- (NSArray<NSString *> *)inputNamesForSignatureIndex:(NSUInteger)signatureIndex
                                               error:(NSError **)error {
  if (![self validateSignatureIndex:signatureIndex error:error]) {
    return nil;
  }
  litert::Expected<std::vector<litert::StringView>> namesResult =
      _cppCompiledModel->GetSignatureInputNames(signatureIndex);
  return CreateObjCStringsFromCPPResult(namesResult, error);
}

- (NSArray<NSString *> *)inputNamesForSignatureKey:(NSString *)signatureKey
                                             error:(NSError **)error {
  if (![self validateSignatureKey:signatureKey error:error]) {
    return nil;
  }
  litert::Expected<std::vector<litert::StringView>> namesResult =
      _cppCompiledModel->GetSignatureInputNames(LRTCPPStringFromNSString(signatureKey));
  return CreateObjCStringsFromCPPResult(namesResult, error);
}

- (NSArray<NSString *> *)outputNamesForSignatureIndex:(NSUInteger)signatureIndex
                                                error:(NSError **)error {
  if (![self validateSignatureIndex:signatureIndex error:error]) {
    return nil;
  }
  litert::Expected<std::vector<litert::StringView>> namesResult =
      _cppCompiledModel->GetSignatureOutputNames(signatureIndex);
  return CreateObjCStringsFromCPPResult(namesResult, error);
}

- (NSArray<NSString *> *)outputNamesForSignatureKey:(NSString *)signatureKey
                                              error:(NSError **)error {
  if (![self validateSignatureKey:signatureKey error:error]) {
    return nil;
  }
  litert::Expected<std::vector<litert::StringView>> namesResult =
      _cppCompiledModel->GetSignatureOutputNames(LRTCPPStringFromNSString(signatureKey));
  return CreateObjCStringsFromCPPResult(namesResult, error);
}

- (LRTTensorType *)inputTensorTypeAtIndex:(NSUInteger)inputIndex error:(NSError **)error {
  return [self inputTensorTypeAtIndex:inputIndex signatureIndex:0 error:error];
}

- (LRTTensorType *)inputTensorTypeAtIndex:(NSUInteger)inputIndex
                           signatureIndex:(NSUInteger)signatureIndex
                                    error:(NSError **)error {
  if (![self validateSignatureIndex:signatureIndex error:error]) {
    return nil;
  }
  litert::Expected<litert::RankedTensorType> tensorTypeResult =
      _cppCompiledModel->GetInputTensorType(signatureIndex, inputIndex);
  return CreateObjCTensorTypeFromCPPResult(tensorTypeResult, error);
}

- (LRTTensorType *)inputTensorTypeForName:(NSString *)inputName
                             signatureKey:(NSString *)signatureKey
                                    error:(NSError **)error {
  if (!inputName) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"inputName cannot be nil");
    return nil;
  }
  if (![self validateSignatureKey:signatureKey error:error]) {
    return nil;
  }
  litert::Expected<litert::RankedTensorType> tensorTypeResult =
      _cppCompiledModel->GetInputTensorType(LRTCPPStringFromNSString(signatureKey),
                                            LRTCPPStringFromNSString(inputName));
  return CreateObjCTensorTypeFromCPPResult(tensorTypeResult, error);
}

- (LRTTensorType *)outputTensorTypeAtIndex:(NSUInteger)outputIndex error:(NSError **)error {
  return [self outputTensorTypeAtIndex:outputIndex signatureIndex:0 error:error];
}

- (LRTTensorType *)outputTensorTypeAtIndex:(NSUInteger)outputIndex
                            signatureIndex:(NSUInteger)signatureIndex
                                     error:(NSError **)error {
  if (![self validateSignatureIndex:signatureIndex error:error]) {
    return nil;
  }
  litert::Expected<litert::RankedTensorType> tensorTypeResult =
      _cppCompiledModel->GetOutputTensorType(signatureIndex, outputIndex);
  return CreateObjCTensorTypeFromCPPResult(tensorTypeResult, error);
}

- (LRTTensorType *)outputTensorTypeForName:(NSString *)outputName
                              signatureKey:(NSString *)signatureKey
                                     error:(NSError **)error {
  if (!outputName) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"outputName cannot be nil");
    return nil;
  }
  if (![self validateSignatureKey:signatureKey error:error]) {
    return nil;
  }
  litert::Expected<litert::RankedTensorType> tensorTypeResult =
      _cppCompiledModel->GetOutputTensorType(LRTCPPStringFromNSString(signatureKey),
                                             LRTCPPStringFromNSString(outputName));
  return CreateObjCTensorTypeFromCPPResult(tensorTypeResult, error);
}

- (LRTTensorBufferRequirements *)inputBufferRequirementsAtIndex:(NSUInteger)inputIndex
                                                          error:(NSError **)error {
  return [self inputBufferRequirementsAtIndex:inputIndex signatureIndex:0 error:error];
}

- (LRTTensorBufferRequirements *)inputBufferRequirementsAtIndex:(NSUInteger)inputIndex
                                                 signatureIndex:(NSUInteger)signatureIndex
                                                          error:(NSError **)error {
  if (!_cppCompiledModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Compiled model is not initialized");
    return nil;
  }

  litert::Expected<litert::TensorBufferRequirements> requirementsResult =
      _cppCompiledModel->GetInputBufferRequirements(signatureIndex, inputIndex);
  return CreateObjCBufferRequirementsFromCPPResult(requirementsResult, error);
}

- (LRTTensorBufferRequirements *)inputBufferRequirementsForName:(NSString *)inputName
                                                   signatureKey:(NSString *)signatureKey
                                                          error:(NSError **)error {
  if (!inputName) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"inputName cannot be nil");
    return nil;
  }
  if (![self validateSignatureKey:signatureKey error:error]) {
    return nil;
  }

  litert::Expected<litert::TensorBufferRequirements> requirementsResult =
      _cppCompiledModel->GetInputBufferRequirements(LRTCPPStringFromNSString(signatureKey),
                                                    LRTCPPStringFromNSString(inputName));
  return CreateObjCBufferRequirementsFromCPPResult(requirementsResult, error);
}

- (LRTTensorBufferRequirements *)outputBufferRequirementsAtIndex:(NSUInteger)outputIndex
                                                           error:(NSError **)error {
  return [self outputBufferRequirementsAtIndex:outputIndex signatureIndex:0 error:error];
}

- (LRTTensorBufferRequirements *)outputBufferRequirementsAtIndex:(NSUInteger)outputIndex
                                                  signatureIndex:(NSUInteger)signatureIndex
                                                           error:(NSError **)error {
  if (!_cppCompiledModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Compiled model is not initialized");
    return nil;
  }

  litert::Expected<litert::TensorBufferRequirements> requirementsResult =
      _cppCompiledModel->GetOutputBufferRequirements(signatureIndex, outputIndex);
  return CreateObjCBufferRequirementsFromCPPResult(requirementsResult, error);
}

- (LRTTensorBufferRequirements *)outputBufferRequirementsForName:(NSString *)outputName
                                                    signatureKey:(NSString *)signatureKey
                                                           error:(NSError **)error {
  if (!outputName) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"outputName cannot be nil");
    return nil;
  }
  if (![self validateSignatureKey:signatureKey error:error]) {
    return nil;
  }

  litert::Expected<litert::TensorBufferRequirements> requirementsResult =
      _cppCompiledModel->GetOutputBufferRequirements(LRTCPPStringFromNSString(signatureKey),
                                                     LRTCPPStringFromNSString(outputName));
  return CreateObjCBufferRequirementsFromCPPResult(requirementsResult, error);
}

- (NSArray<LRTTensorBuffer *> *)createInputTensorBuffersWithError:(NSError **)error {
  return [self createInputTensorBuffersForSignatureIndex:0 error:error];
}

- (NSArray<LRTTensorBuffer *> *)createInputTensorBuffersForSignatureIndex:(NSUInteger)signatureIndex
                                                                    error:(NSError **)error {
  if (!_cppCompiledModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Compiled model is not initialized");
    return nil;
  }

  litert::Expected<std::vector<litert::TensorBuffer>> buffersResult =
      _cppCompiledModel->CreateInputBuffers(signatureIndex);
  return CreateObjCTensorBuffersFromCPPResult(buffersResult, error);
}

- (NSArray<LRTTensorBuffer *> *)createInputTensorBuffersForSignatureKey:(NSString *)signatureKey
                                                                  error:(NSError **)error {
  if (![self validateSignatureKey:signatureKey error:error]) {
    return nil;
  }

  litert::Expected<std::vector<litert::TensorBuffer>> buffersResult =
      _cppCompiledModel->CreateInputBuffers(LRTCPPStringFromNSString(signatureKey));
  return CreateObjCTensorBuffersFromCPPResult(buffersResult, error);
}

- (NSArray<LRTTensorBuffer *> *)createOutputTensorBuffersWithError:(NSError **)error {
  return [self createOutputTensorBuffersForSignatureIndex:0 error:error];
}

- (NSArray<LRTTensorBuffer *> *)createOutputTensorBuffersForSignatureIndex:
                                    (NSUInteger)signatureIndex
                                                                     error:(NSError **)error {
  if (!_cppCompiledModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Compiled model is not initialized");
    return nil;
  }

  litert::Expected<std::vector<litert::TensorBuffer>> buffersResult =
      _cppCompiledModel->CreateOutputBuffers(signatureIndex);
  return CreateObjCTensorBuffersFromCPPResult(buffersResult, error);
}

- (NSArray<LRTTensorBuffer *> *)createOutputTensorBuffersForSignatureKey:(NSString *)signatureKey
                                                                   error:(NSError **)error {
  if (![self validateSignatureKey:signatureKey error:error]) {
    return nil;
  }

  litert::Expected<std::vector<litert::TensorBuffer>> buffersResult =
      _cppCompiledModel->CreateOutputBuffers(LRTCPPStringFromNSString(signatureKey));
  return CreateObjCTensorBuffersFromCPPResult(buffersResult, error);
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
  if (!DuplicateObjCTensorBuffersToCPP(inputs, @"input", inputCppBuffers, error)) {
    return NO;
  }

  std::vector<litert::TensorBuffer> outputCppBuffers;
  if (!DuplicateObjCTensorBuffersToCPP(outputs, @"output", outputCppBuffers, error)) {
    return NO;
  }

  litert::Expected<void> runResult =
      _cppCompiledModel->Run(signatureIndex, inputCppBuffers, outputCppBuffers);
  if (!runResult.HasValue()) {
    LRTSetErrorFromCPPError(error, runResult.Error());
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

  if (dimensions.count == 0) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Dimensions array cannot be empty");
    return NO;
  }

  std::vector<int> cppDimensions;
  cppDimensions.reserve(dimensions.count);
  for (NSNumber *dimension in dimensions) {
    if (dimension.intValue <= 0) {
      LRTSetError(error, LRTErrorCodeInvalidArgument, @"Dimension values must be positive");
      return NO;
    }
    cppDimensions.push_back(dimension.intValue);
  }

  litert::Expected<void> resizeResult =
      _cppCompiledModel->ResizeInputTensor(signatureIndex, inputIndex, cppDimensions);

  if (!resizeResult.HasValue()) {
    LRTSetErrorFromCPPError(error, resizeResult.Error());
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

#pragma mark - Private

/**
 * Validates that @c _cppCompiledModel is initialized and @c signatureIndex is within bounds.
 *
 * @param signatureIndex The 0-based signature index to validate.
 * @param error Out-parameter populated on failure.
 * @return @c YES if @c signatureIndex is valid, @c NO otherwise.
 */
- (BOOL)validateSignatureIndex:(NSUInteger)signatureIndex error:(NSError **)error {
  if (!_cppCompiledModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Compiled model is not initialized");
    return NO;
  }
  if (signatureIndex >= _cppCompiledModel->GetNumSignatures()) {
    LRTSetError(error, LRTErrorCodeInvalidArgument,
                [NSString stringWithFormat:@"Signature index %lu out of bounds",
                                           (unsigned long)signatureIndex]);
    return NO;
  }
  return YES;
}

/**
 * Validates that @c _cppCompiledModel is initialized and @c signatureKey is non-empty.
 *
 * @param signatureKey The signature key to validate.
 * @param error Out-parameter populated on failure.
 * @return @c YES if @c signatureKey is valid, @c NO otherwise.
 */
- (BOOL)validateSignatureKey:(NSString *)signatureKey error:(NSError **)error {
  if (signatureKey.length == 0) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"signatureKey cannot be empty");
    return NO;
  }
  if (!_cppCompiledModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Compiled model is not initialized");
    return NO;
  }
  return YES;
}
/**
 * Returns the index of the signature named @c signatureKey, or nil if the model does not have
 * such a signature.
 *
 * @param signatureKey The name/key of the signature in the model.
 * @param error Out-parameter populated on failure.
 * @return The signature index boxed in an @c NSNumber, or @c nil on failure.
 */
- (NSNumber *)signatureIndexForSignatureKey:(NSString *)signatureKey error:(NSError **)error {
  if (![self validateSignatureKey:signatureKey error:error]) {
    return nil;
  }

  litert::Expected<size_t> signatureIndexResult =
      _cppCompiledModel->GetSignatureIndex(LRTCPPStringFromNSString(signatureKey));
  if (!signatureIndexResult.HasValue()) {
    LRTSetErrorFromCPPError(error, signatureIndexResult.Error());
    return nil;
  }
  return @(signatureIndexResult.Value());
}

@end


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

#import "third_party/odml/litert/litert/objc/apis/LRTModel.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#import "third_party/odml/litert/litert/objc/apis/LRTError.h"
#import "third_party/odml/litert/litert/objc/sources/LRTError+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTEnvironment+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTModel+Internal.h"

@implementation LRTModel {
  std::unique_ptr<litert::Model> _cppModel;
  /** Model bytes the model reads from, or nil when it was loaded from a file. */
  NSData *_modelData;
}

+ (instancetype)modelWithModelFilePath:(NSString *)modelFilePath
                           environment:(LRTEnvironment *)environment
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

  litert::Expected<litert::Model> createResult =
      litert::Model::CreateFromFile(*cppEnvironment, LRTCPPStringFromNSString(modelFilePath));

  if (!createResult.HasValue()) {
    LRTSetErrorFromCPPError(error, createResult.Error());
    return nil;
  }

  auto cppModel = std::make_unique<litert::Model>(std::move(createResult.Value()));
  return [[LRTModel alloc] initInternalWithCPPModel:std::move(cppModel)
                                        environment:environment
                                          modelData:nil];
}

+ (instancetype)modelWithModelData:(NSData *)modelData
                       environment:(LRTEnvironment *)environment
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

  // LiteRT does not copy the model bytes, it reads them for as long as the model is alive. Hold
  // on to an immutable copy so that callers may release their buffer, or pass an NSMutableData
  // and keep mutating it, without corrupting the model.
  NSData *ownedModelData = [modelData copy];
  litert::BufferRef<uint8_t> bufferRef(static_cast<const uint8_t *>(ownedModelData.bytes),
                                       ownedModelData.length);
  auto createResult = litert::Model::CreateFromBuffer(*cppEnvironment, bufferRef);

  if (!createResult.HasValue()) {
    LRTSetErrorFromCPPError(error, createResult.Error());
    return nil;
  }

  auto cppModel = std::make_unique<litert::Model>(std::move(createResult.Value()));
  return [[LRTModel alloc] initInternalWithCPPModel:std::move(cppModel)
                                        environment:environment
                                          modelData:ownedModelData];
}

- (instancetype)initInternalWithCPPModel:(std::unique_ptr<litert::Model>)cppModel
                             environment:(LRTEnvironment *)environment
                               modelData:(NSData *)modelData {
  self = [super init];
  if (self) {
    _cppModel = std::move(cppModel);
    _environment = environment;
    _modelData = [modelData copy];
  }
  return self;
}

#pragma mark - Properties

- (NSArray<NSString *> *)signatureKeys {
  if (!_cppModel) {
    return @[];
  }
  litert::Expected<std::vector<litert::StringView>> keysResult = _cppModel->GetSignatureKeys();
  if (!keysResult.HasValue()) {
    return @[];
  }
  return LRTStringArrayFromCPPStringViews(keysResult.Value());
}

#pragma mark - Public

- (NSArray<NSString *> *)inputNamesForSignatureIndex:(NSUInteger)signatureIndex
                                               error:(NSError **)error {
  if (!_cppModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Model is not initialized");
    return nil;
  }

  if (signatureIndex >= _cppModel->GetNumSignatures()) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Signature index out of bounds");
    return nil;
  }

  litert::Expected<std::vector<litert::StringView>> namesResult =
      _cppModel->GetSignatureInputNames(signatureIndex);
  if (!namesResult.HasValue()) {
    LRTSetErrorFromCPPError(error, namesResult.Error());
    return nil;
  }

  return LRTStringArrayFromCPPStringViews(namesResult.Value());
}

- (NSArray<NSString *> *)inputNamesForSignatureKey:(NSString *)signatureKey
                                             error:(NSError **)error {
  if (![self validateSignatureKey:signatureKey error:error]) {
    return nil;
  }

  litert::Expected<std::vector<litert::StringView>> namesResult =
      _cppModel->GetSignatureInputNames(LRTCPPStringFromNSString(signatureKey));
  if (!namesResult.HasValue()) {
    LRTSetErrorFromCPPError(error, namesResult.Error());
    return nil;
  }

  return LRTStringArrayFromCPPStringViews(namesResult.Value());
}

- (NSArray<NSString *> *)outputNamesForSignatureIndex:(NSUInteger)signatureIndex
                                                error:(NSError **)error {
  if (!_cppModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Model is not initialized");
    return nil;
  }

  if (signatureIndex >= _cppModel->GetNumSignatures()) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Signature index out of bounds");
    return nil;
  }

  litert::Expected<std::vector<litert::StringView>> namesResult =
      _cppModel->GetSignatureOutputNames(signatureIndex);
  if (!namesResult.HasValue()) {
    LRTSetErrorFromCPPError(error, namesResult.Error());
    return nil;
  }

  return LRTStringArrayFromCPPStringViews(namesResult.Value());
}

- (NSArray<NSString *> *)outputNamesForSignatureKey:(NSString *)signatureKey
                                              error:(NSError **)error {
  if (![self validateSignatureKey:signatureKey error:error]) {
    return nil;
  }

  litert::Expected<std::vector<litert::StringView>> namesResult =
      _cppModel->GetSignatureOutputNames(LRTCPPStringFromNSString(signatureKey));
  if (!namesResult.HasValue()) {
    LRTSetErrorFromCPPError(error, namesResult.Error());
    return nil;
  }

  return LRTStringArrayFromCPPStringViews(namesResult.Value());
}

- (NSData *)metadataForKey:(NSString *)metadataKey error:(NSError **)error {
  if (!metadataKey) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"metadataKey cannot be nil");
    return nil;
  }

  if (!_cppModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Model is not initialized");
    return nil;
  }

  litert::Expected<litert::Span<const uint8_t>> metadataResult =
      _cppModel->Metadata(LRTCPPStringFromNSString(metadataKey));
  if (!metadataResult.HasValue()) {
    LRTSetErrorFromCPPError(error, metadataResult.Error());
    return nil;
  }

  return [NSData dataWithBytes:metadataResult.Value().data() length:metadataResult.Value().size()];
}

#pragma mark - LRTModel (Internal)

- (litert::Model *)cppModel {
  return _cppModel.get();
}

#pragma mark - Private

- (BOOL)validateSignatureKey:(NSString *)signatureKey error:(NSError **)error {
  if (signatureKey.length == 0) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"signatureKey cannot be empty");
    return NO;
  }
  if (!_cppModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Model is not initialized");
    return NO;
  }
  return YES;
}

@end


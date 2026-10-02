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

NS_ASSUME_NONNULL_BEGIN

/**
 * Converts C++ string views, which need not be NUL-terminated, into an array of NSStrings.
 *
 * Callers index the result by tensor or signature index, so a name that is not valid UTF-8
 * becomes an empty string rather than being dropped, which would shift every later index.
 */
static NSArray<NSString *> *ConvertStringViewsToObjCArray(
    const std::vector<litert::StringView> &stringViews) {
  NSMutableArray<NSString *> *array = [NSMutableArray arrayWithCapacity:stringViews.size()];
  for (const auto &stringView : stringViews) {
    NSString *string = [[NSString alloc] initWithBytes:stringView.data()
                                                length:stringView.size()
                                              encoding:NSUTF8StringEncoding];
    [array addObject:string ?: @""];
  }
  return array;
}

@implementation LRTModel {
  std::unique_ptr<litert::Model> _cppModel;
  /** Model bytes the model reads from, or nil when it was loaded from a file. */
  NSData *_Nullable _modelData;
}

+ (nullable instancetype)modelWithModelFilePath:(NSString *)modelFilePath
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
      litert::Model::CreateFromFile(*cppEnvironment, std::string(modelFilePath.UTF8String));

  if (!createResult.HasValue()) {
    LRTSetErrorFromCppError(error, createResult.Error());
    return nil;
  }

  auto cppModel = std::make_unique<litert::Model>(std::move(createResult.Value()));
  return [[LRTModel alloc] initInternalWithCppModel:std::move(cppModel)
                                        environment:environment
                                          modelData:nil];
}

+ (nullable instancetype)modelWithModelData:(NSData *)modelData
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
    LRTSetErrorFromCppError(error, createResult.Error());
    return nil;
  }

  auto cppModel = std::make_unique<litert::Model>(std::move(createResult.Value()));
  return [[LRTModel alloc] initInternalWithCppModel:std::move(cppModel)
                                        environment:environment
                                          modelData:ownedModelData];
}

- (instancetype)initInternalWithCppModel:(std::unique_ptr<litert::Model>)cppModel
                             environment:(LRTEnvironment *)environment
                               modelData:(nullable NSData *)modelData {
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
  return ConvertStringViewsToObjCArray(keysResult.Value());
}

#pragma mark - Public

- (nullable NSArray<NSString *> *)inputNamesForSignatureIndex:(NSUInteger)signatureIndex
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
    LRTSetErrorFromCppError(error, namesResult.Error());
    return nil;
  }

  return ConvertStringViewsToObjCArray(namesResult.Value());
}

- (nullable NSArray<NSString *> *)inputNamesForSignatureKey:(NSString *)signatureKey
                                                      error:(NSError **)error {
  if (!signatureKey) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"signatureKey cannot be nil");
    return nil;
  }

  if (!_cppModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Model is not initialized");
    return nil;
  }

  litert::Expected<std::vector<litert::StringView>> namesResult =
      _cppModel->GetSignatureInputNames(signatureKey.UTF8String);
  if (!namesResult.HasValue()) {
    LRTSetErrorFromCppError(error, namesResult.Error());
    return nil;
  }

  return ConvertStringViewsToObjCArray(namesResult.Value());
}

- (nullable NSArray<NSString *> *)outputNamesForSignatureIndex:(NSUInteger)signatureIndex
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
    LRTSetErrorFromCppError(error, namesResult.Error());
    return nil;
  }

  return ConvertStringViewsToObjCArray(namesResult.Value());
}

- (nullable NSArray<NSString *> *)outputNamesForSignatureKey:(NSString *)signatureKey
                                                       error:(NSError **)error {
  if (!signatureKey) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"signatureKey cannot be nil");
    return nil;
  }

  if (!_cppModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Model is not initialized");
    return nil;
  }

  litert::Expected<std::vector<litert::StringView>> namesResult =
      _cppModel->GetSignatureOutputNames(signatureKey.UTF8String);
  if (!namesResult.HasValue()) {
    LRTSetErrorFromCppError(error, namesResult.Error());
    return nil;
  }

  return ConvertStringViewsToObjCArray(namesResult.Value());
}

- (nullable NSData *)metadataForKey:(NSString *)metadataKey error:(NSError **)error {
  if (!metadataKey) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"metadataKey cannot be nil");
    return nil;
  }

  if (!_cppModel) {
    LRTSetError(error, LRTErrorCodeRuntimeFailure, @"Model is not initialized");
    return nil;
  }

  litert::Expected<litert::Span<const uint8_t>> metadataResult =
      _cppModel->Metadata(metadataKey.UTF8String);
  if (!metadataResult.HasValue()) {
    LRTSetErrorFromCppError(error, metadataResult.Error());
    return nil;
  }

  return [NSData dataWithBytes:metadataResult.Value().data() length:metadataResult.Value().size()];
}

#pragma mark - LRTModel (Internal)

- (nullable litert::Model *)cppModel {
  return _cppModel.get();
}

@end

NS_ASSUME_NONNULL_END

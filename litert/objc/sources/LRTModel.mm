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

static NSArray<NSString *> *ConvertStringViewsToObjCArray(
    const std::vector<litert::StringView> &stringViews) {
  NSMutableArray<NSString *> *array = [NSMutableArray arrayWithCapacity:stringViews.size()];
  for (const auto &sv : stringViews) {
    NSString *str = [[NSString alloc] initWithBytes:sv.data()
                                             length:sv.size()
                                           encoding:NSUTF8StringEncoding];
    if (str != nil) {
      [array addObject:str];
    }
  }
  return [array copy];
}

@implementation LRTModel {
  std::unique_ptr<litert::Model> _cppModel;
  NSData *_Nullable _modelData;
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

- (nullable litert::Model *)cppModel {
  return _cppModel.get();
}

+ (nullable instancetype)modelWithModelFilePath:(NSString *)modelFilePath
                                    environment:(LRTEnvironment *)environment
                                          error:(NSError **)error {
  if (!modelFilePath) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"modelFilePath cannot be nil");
    }
    return nil;
  }

  if (![environment cppEnvironment]) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Valid LRTEnvironment required");
    }
    return nil;
  }

  litert::Expected<litert::Model> createResult = litert::Model::CreateFromFile(
      *[environment cppEnvironment], std::string(modelFilePath.UTF8String));

  if (!createResult.HasValue()) {
    if (error) {
      *error = CreateLRTError(static_cast<NSInteger>(createResult.Error().Status()),
                              @(createResult.Error().Message().c_str()));
    }
    return nil;
  }

  auto cppPtr = std::make_unique<litert::Model>(std::move(createResult.Value()));
  return [[LRTModel alloc] initInternalWithCppModel:std::move(cppPtr)
                                        environment:environment
                                          modelData:nil];
}

+ (nullable instancetype)modelWithModelData:(NSData *)modelData
                                environment:(LRTEnvironment *)environment
                                      error:(NSError **)error {
  if (!modelData || modelData.length == 0) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"modelData cannot be empty");
    }
    return nil;
  }

  if (!environment || ![environment cppEnvironment]) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Valid LRTEnvironment required");
    }
    return nil;
  }

  litert::BufferRef<uint8_t> bufferRef(static_cast<const uint8_t *>(modelData.bytes),
                                       modelData.length);
  auto createResult = litert::Model::CreateFromBuffer(*[environment cppEnvironment], bufferRef);

  if (!createResult.HasValue()) {
    if (error) {
      *error = CreateLRTError(static_cast<NSInteger>(createResult.Error().Status()),
                              @(createResult.Error().Message().c_str()));
    }
    return nil;
  }

  auto cppPtr = std::make_unique<litert::Model>(std::move(createResult.Value()));
  return [[LRTModel alloc] initInternalWithCppModel:std::move(cppPtr)
                                        environment:environment
                                          modelData:modelData];
}

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

- (nullable NSArray<NSString *> *)inputNamesForSignatureIndex:(NSUInteger)signatureIndex
                                                        error:(NSError **)error {
  if (!_cppModel) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeRuntimeFailure, @"Model is not initialized");
    }
    return nil;
  }

  if (signatureIndex >= _cppModel->GetNumSignatures()) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Signature index out of bounds");
    }
    return nil;
  }

  litert::Expected<std::vector<litert::StringView>> namesResult =
      _cppModel->GetSignatureInputNames(signatureIndex);
  if (!namesResult.HasValue()) {
    if (error) {
      *error = CreateLRTError(static_cast<NSInteger>(namesResult.Error().Status()),
                              @(namesResult.Error().Message().c_str()));
    }
    return nil;
  }

  return ConvertStringViewsToObjCArray(namesResult.Value());
}

- (nullable NSArray<NSString *> *)inputNamesForSignatureKey:(NSString *)signatureKey
                                                      error:(NSError **)error {
  if (!signatureKey) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"signatureKey cannot be nil");
    }
    return nil;
  }

  if (!_cppModel) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeRuntimeFailure, @"Model is not initialized");
    }
    return nil;
  }

  litert::Expected<std::vector<litert::StringView>> namesResult =
      _cppModel->GetSignatureInputNames(signatureKey.UTF8String);
  if (!namesResult.HasValue()) {
    if (error) {
      *error = CreateLRTError(static_cast<NSInteger>(namesResult.Error().Status()),
                              @(namesResult.Error().Message().c_str()));
    }
    return nil;
  }

  return ConvertStringViewsToObjCArray(namesResult.Value());
}

- (nullable NSArray<NSString *> *)outputNamesForSignatureIndex:(NSUInteger)signatureIndex
                                                         error:(NSError **)error {
  if (!_cppModel) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeRuntimeFailure, @"Model is not initialized");
    }
    return nil;
  }

  if (signatureIndex >= _cppModel->GetNumSignatures()) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Signature index out of bounds");
    }
    return nil;
  }

  litert::Expected<std::vector<litert::StringView>> namesResult =
      _cppModel->GetSignatureOutputNames(signatureIndex);
  if (!namesResult.HasValue()) {
    if (error) {
      *error = CreateLRTError(static_cast<NSInteger>(namesResult.Error().Status()),
                              @(namesResult.Error().Message().c_str()));
    }
    return nil;
  }

  return ConvertStringViewsToObjCArray(namesResult.Value());
}

- (nullable NSArray<NSString *> *)outputNamesForSignatureKey:(NSString *)signatureKey
                                                       error:(NSError **)error {
  if (!signatureKey) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"signatureKey cannot be nil");
    }
    return nil;
  }

  if (!_cppModel) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeRuntimeFailure, @"Model is not initialized");
    }
    return nil;
  }

  litert::Expected<std::vector<litert::StringView>> namesResult =
      _cppModel->GetSignatureOutputNames(signatureKey.UTF8String);
  if (!namesResult.HasValue()) {
    if (error) {
      *error = CreateLRTError(static_cast<NSInteger>(namesResult.Error().Status()),
                              @(namesResult.Error().Message().c_str()));
    }
    return nil;
  }

  return ConvertStringViewsToObjCArray(namesResult.Value());
}

- (nullable NSData *)metadataForKey:(NSString *)metadataKey error:(NSError **)error {
  if (!metadataKey) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"metadataKey cannot be nil");
    }
    return nil;
  }

  if (!_cppModel) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeRuntimeFailure, @"Model is not initialized");
    }
    return nil;
  }

  litert::Expected<litert::Span<const uint8_t>> metaResult =
      _cppModel->Metadata(metadataKey.UTF8String);
  if (!metaResult.HasValue()) {
    if (error) {
      *error = CreateLRTError(static_cast<NSInteger>(metaResult.Error().Status()),
                              @(metaResult.Error().Message().c_str()));
    }
    return nil;
  }

  return [NSData dataWithBytes:metaResult.Value().data() length:metaResult.Value().size()];
}

@end

NS_ASSUME_NONNULL_END

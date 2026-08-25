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

#import <Foundation/Foundation.h>

#import "third_party/odml/litert/litert/objc/apis/LRTEnvironment.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * A LiteRT model object for inspecting signatures, tensors, and metadata.
 *
 * Wraps @c litert::Model and represents an uncompiled model flatbuffer.
 */
@interface LRTModel : NSObject

- (instancetype)init NS_UNAVAILABLE;

/**
 * Loads a LiteRT model from a file path.
 *
 * @param modelFilePath The file path to the model file (.tflite).
 * @param environment The environment instance to use for loading the model.
 * @param error Out-parameter populated on failure.
 * @return A new @c LRTModel instance, or @c nil on failure.
 */
+ (nullable instancetype)modelWithModelFilePath:(NSString *)modelFilePath
                                    environment:(LRTEnvironment *)environment
                                          error:(NSError **)error;

/**
 * Loads a LiteRT model from in-memory model data.
 *
 * @note The caller must ensure that @c modelData remains valid and immutable for the
 * lifetime of the @c LRTModel instance.
 *
 * @param modelData The model file content as @c NSData.
 * @param environment The environment instance to use for loading the model.
 * @param error Out-parameter populated on failure.
 * @return A new @c LRTModel instance, or @c nil on failure.
 */
+ (nullable instancetype)modelWithModelData:(NSData *)modelData
                                environment:(LRTEnvironment *)environment
                                      error:(NSError **)error;

/** Environment associated with this model. */
@property(nonatomic, readonly) LRTEnvironment *environment;

/** The signature keys defined in the model. */
@property(nonatomic, readonly, copy) NSArray<NSString *> *signatureKeys;

/**
 * Returns the list of input tensor names for a given signature index.
 *
 * @param signatureIndex The index of the signature.
 * @param error Out-parameter populated on failure.
 * @return Array of input tensor names, or @c nil on failure.
 */
- (nullable NSArray<NSString *> *)inputNamesForSignatureIndex:(NSUInteger)signatureIndex
                                                        error:(NSError **)error;

/**
 * Returns the list of input tensor names for a given signature key.
 *
 * @param signatureKey The key of the signature.
 * @param error Out-parameter populated on failure.
 * @return Array of input tensor names, or @c nil on failure.
 */
- (nullable NSArray<NSString *> *)inputNamesForSignatureKey:(NSString *)signatureKey
                                                      error:(NSError **)error;

/**
 * Returns the list of output tensor names for a given signature index.
 *
 * @param signatureIndex The index of the signature.
 * @param error Out-parameter populated on failure.
 * @return Array of output tensor names, or @c nil on failure.
 */
- (nullable NSArray<NSString *> *)outputNamesForSignatureIndex:(NSUInteger)signatureIndex
                                                         error:(NSError **)error;

/**
 * Returns the list of output tensor names for a given signature key.
 *
 * @param signatureKey The key of the signature.
 * @param error Out-parameter populated on failure.
 * @return Array of output tensor names, or @c nil on failure.
 */
- (nullable NSArray<NSString *> *)outputNamesForSignatureKey:(NSString *)signatureKey
                                                       error:(NSError **)error;

/**
 * Retrieves the metadata buffer associated with the given key.
 *
 * @param metadataKey The key of the metadata buffer in the model.
 * @param error Out-parameter populated on failure.
 * @return The metadata as @c NSData, or @c nil if not found or on failure.
 */
- (nullable NSData *)metadataForKey:(NSString *)metadataKey error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END

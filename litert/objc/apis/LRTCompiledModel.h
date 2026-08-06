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
#import "third_party/odml/litert/litert/objc/apis/LRTOptions.h"
#import "third_party/odml/litert/litert/objc/apis/LRTTensorBuffer.h"

NS_ASSUME_NONNULL_BEGIN

/** A compiled LiteRT model ready for inference execution. */
@interface LRTCompiledModel : NSObject

- (instancetype)init NS_UNAVAILABLE;

/**
 * Creates and compiles a LiteRT model from a file path.
 *
 * The model is loaded into memory, and the caller takes ownership of the
 * returned @c LRTCompiledModel object. The provided @c options are used for model
 * compilation (e.g. to select the accelerator).
 *
 * @note The provided @c environment must outlive the compiled model and any
 * executions running on it.
 *
 * @param modelFilePath Absolute file path to the model file.
 * @param environment LiteRT environment instance. Must outlive the returned compiled model.
 * @param options Optional compilation options.
 * @param error Out-parameter populated on failure.
 * @return A compiled model instance, or @c nil on failure.
 */
+ (nullable instancetype)compiledModelWithModelFilePath:(NSString *)modelFilePath
                                            environment:(LRTEnvironment *)environment
                                                options:(nullable LRTOptions *)options
                                                  error:(NSError **)error;

/**
 * Creates and compiles a LiteRT model from in-memory model data.
 *
 * @param modelData In-memory model byte data.
 * @param environment LiteRT environment instance.
 * @param options Optional compilation options.
 * @param error Out-parameter populated on failure.
 * @return A compiled model instance, or @c nil on failure.
 */
+ (nullable instancetype)compiledModelWithModelData:(NSData *)modelData
                                        environment:(LRTEnvironment *)environment
                                            options:(nullable LRTOptions *)options
                                              error:(NSError **)error;

/**
 * Returns the default signature key for LiteRT models.
 *
 * @return Default signature key string.
 */
+ (NSString *)defaultSignatureKey;

/** Environment used to build this compiled model. */
@property(nonatomic, readonly) LRTEnvironment *environment;

/** Options used during model compilation. */
@property(nonatomic, readonly, nullable) LRTOptions *options;

/**
 * Creates input tensor buffers according to the model's default signature requirements.
 *
 * It uses the model's buffer requirements and tensor types to allocate appropriate
 * @c LRTTensorBuffer instances.
 *
 * @param error Out-parameter populated on failure.
 * @return Array of newly allocated input @c LRTTensorBuffer instances, or @c nil on failure.
 */
- (nullable NSArray<LRTTensorBuffer *> *)createInputTensorBuffersWithError:(NSError **)error;

/**
 * Creates input tensor buffers according to the specified signature index.
 *
 * @param signatureIndex The index of the signature in the model.
 * @param error Out-parameter populated on failure.
 * @return Array of newly allocated input @c LRTTensorBuffer instances, or @c nil on failure.
 */
- (nullable NSArray<LRTTensorBuffer *> *)
    createInputTensorBuffersForSignatureIndex:(NSUInteger)signatureIndex
                                        error:(NSError **)error;

/**
 * Creates input tensor buffers according to the specified signature key.
 *
 * @param signatureKey The name/key of the signature in the model.
 * @param error Out-parameter populated on failure.
 * @return Array of newly allocated input @c LRTTensorBuffer instances, or @c nil on failure.
 */
- (nullable NSArray<LRTTensorBuffer *> *)createInputTensorBuffersForSignatureKey:
                                             (NSString *)signatureKey
                                                                           error:(NSError **)error;

/**
 * Creates output tensor buffers according to the model's default signature requirements.
 *
 * It uses the model's buffer requirements and tensor types to allocate appropriate
 * @c LRTTensorBuffer instances.
 *
 * @param error Out-parameter populated on failure.
 * @return Array of newly allocated output @c LRTTensorBuffer instances, or @c nil on failure.
 */
- (nullable NSArray<LRTTensorBuffer *> *)createOutputTensorBuffersWithError:(NSError **)error;

/**
 * Creates output tensor buffers according to the specified signature index.
 *
 * @param signatureIndex The index of the signature in the model.
 * @param error Out-parameter populated on failure.
 * @return Array of newly allocated output @c LRTTensorBuffer instances, or @c nil on failure.
 */
- (nullable NSArray<LRTTensorBuffer *> *)
    createOutputTensorBuffersForSignatureIndex:(NSUInteger)signatureIndex
                                         error:(NSError **)error;

/**
 * Creates output tensor buffers according to the specified signature key.
 *
 * @param signatureKey The name/key of the signature in the model.
 * @param error Out-parameter populated on failure.
 * @return Array of newly allocated output @c LRTTensorBuffer instances, or @c nil on failure.
 */
- (nullable NSArray<LRTTensorBuffer *> *)createOutputTensorBuffersForSignatureKey:
                                             (NSString *)signatureKey
                                                                            error:(NSError **)error;

/**
 * Runs model inference synchronously for the default signature.
 *
 * @param inputs Array of input tensor buffers matching model signature.
 * @param outputs Array of output tensor buffers matching model signature.
 * @param error Out-parameter populated on failure.
 * @return @c YES on success, @c NO on failure.
 */
- (BOOL)runWithInputs:(NSArray<LRTTensorBuffer *> *)inputs
              outputs:(NSArray<LRTTensorBuffer *> *)outputs
                error:(NSError **)error;

/**
 * Runs model inference synchronously for a specified signature index.
 *
 * @param inputs Array of input tensor buffers matching model signature.
 * @param outputs Array of output tensor buffers matching model signature.
 * @param signatureIndex The index of the signature in the model.
 * @param error Out-parameter populated on failure.
 * @return @c YES on success, @c NO on failure.
 */
- (BOOL)runWithInputs:(NSArray<LRTTensorBuffer *> *)inputs
              outputs:(NSArray<LRTTensorBuffer *> *)outputs
       signatureIndex:(NSUInteger)signatureIndex
                error:(NSError **)error;

/**
 * Runs model inference synchronously for a specified signature key.
 *
 * @param inputs Array of input tensor buffers matching model signature.
 * @param outputs Array of output tensor buffers matching model signature.
 * @param signatureKey The name/key of the signature in the model.
 * @param error Out-parameter populated on failure.
 * @return @c YES on success, @c NO on failure.
 */
- (BOOL)runWithInputs:(NSArray<LRTTensorBuffer *> *)inputs
              outputs:(NSArray<LRTTensorBuffer *> *)outputs
         signatureKey:(NSString *)signatureKey
                error:(NSError **)error;

/**
 * Resizes the specified input tensor of the default signature to support dynamic shapes.
 *
 * After calling this, you must re-create the input and output tensor buffers
 * to match the new shapes.
 *
 * @param inputIndex The index of the input tensor.
 * @param dimensions The new dimensions for the input tensor. An array of NSNumber representing the
 * sizes of each dimension.
 * @param error Out-parameter populated on failure.
 * @return @c YES on success, @c NO on failure.
 */
- (BOOL)resizeInputTensorAtIndex:(NSUInteger)inputIndex
                   newDimensions:(NSArray<NSNumber *> *)dimensions
                           error:(NSError **)error;

/**
 * Resizes the specified input tensor for a given signature index to support dynamic shapes.
 *
 * After calling this, you must re-create the input and output tensor buffers
 * to match the new shapes.
 *
 * @param inputIndex The index of the input tensor.
 * @param signatureIndex The index of the signature containing the input tensor.
 * @param dimensions The new dimensions for the input tensor. An array of NSNumber representing the
 * sizes of each dimension.
 * @param error Out-parameter populated on failure.
 * @return @c YES on success, @c NO on failure.
 */
- (BOOL)resizeInputTensorAtIndex:(NSUInteger)inputIndex
                  signatureIndex:(NSUInteger)signatureIndex
                   newDimensions:(NSArray<NSNumber *> *)dimensions
                           error:(NSError **)error;

/**
 * Resizes the specified input tensor for a given signature key to support dynamic shapes.
 *
 * After calling this, you must re-create the input and output tensor buffers
 * to match the new shapes.
 *
 * @param inputIndex The index of the input tensor.
 * @param signatureKey The name/key of the signature containing the input tensor.
 * @param dimensions The new dimensions for the input tensor. An array of NSNumber representing the
 * sizes of each dimension.
 * @param error Out-parameter populated on failure.
 * @return @c YES on success, @c NO on failure.
 */
- (BOOL)resizeInputTensorAtIndex:(NSUInteger)inputIndex
                    signatureKey:(NSString *)signatureKey
                   newDimensions:(NSArray<NSNumber *> *)dimensions
                           error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END

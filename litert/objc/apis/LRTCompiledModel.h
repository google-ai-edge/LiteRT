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

@end

NS_ASSUME_NONNULL_END

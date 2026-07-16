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
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

/** Options used to configure a LiteRT execution environment. */
@interface LRTEnvironmentOptions : NSObject

/** Metal device handle (`id<MTLDevice>`) for GPU environment initialization. */
@property(nonatomic, nullable) id<MTLDevice> metalDevice;

/** Metal command queue handle (`id<MTLCommandQueue>`) for GPU environment initialization. */
@property(nonatomic, nullable) id<MTLCommandQueue> metalCommandQueue;

@end

/** High-level environment context holding LiteRT runtime state. */
@interface LRTEnvironment : NSObject

- (instancetype)init NS_UNAVAILABLE;

/**
 * Creates and returns a new LiteRT environment instance.
 *
 * @param options Configuration options for the environment, or nil for default options.
 * @param error Out-parameter populated on failure.
 * @return A configured LRTEnvironment instance, or nil on failure.
 */
+ (nullable instancetype)environmentWithOptions:(nullable LRTEnvironmentOptions *)options
                                          error:(NSError **)error;

/** The Metal device associated with this environment, if available. */
@property(nonatomic, readonly, nullable) id<MTLDevice> metalDevice;

/** The Metal command queue associated with this environment, if available. */
@property(nonatomic, readonly, nullable) id<MTLCommandQueue> metalCommandQueue;

@end

NS_ASSUME_NONNULL_END

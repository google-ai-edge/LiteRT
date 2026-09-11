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

NS_ASSUME_NONNULL_BEGIN

/** Hardware accelerator choices for model compilation. */
typedef NS_OPTIONS(NSUInteger, LRTHardwareAccelerators) {
  LRTHardwareAcceleratorNone = 0,
  LRTHardwareAcceleratorCPU = 1 << 0,
  LRTHardwareAcceleratorGPU = 1 << 1,
  LRTHardwareAcceleratorNPU = 1 << 2,
};

/** Options for compiling a LiteRT model. */
@interface LRTOptions : NSObject

/**
 * Initializes options with specified hardware accelerators bitmask.
 *
 * @param hardwareAccelerators Enabled hardware accelerators bitmask.
 */
- (instancetype)initWithHardwareAccelerators:(LRTHardwareAccelerators)hardwareAccelerators
    NS_DESIGNATED_INITIALIZER;

/** Hardware accelerators bitmask enabled for compilation. */
@property(nonatomic, assign, readonly) LRTHardwareAccelerators hardwareAccelerators;

/**
 * Whether to use Metal argument buffers for kernel argument encoding.
 *
 * An argument buffer groups multiple resources (buffers, textures, samplers) into a single
 * buffer passed to Metal Shading Language (MSL) compute shaders. When compiling large models
 * or LLMs with many tensor inputs/outputs, the number of individual buffer binding slots can
 * exceed Metal's hardware binding table limits. Enabling argument buffers indirects resource
 * indexing, avoiding buffer slot exhaustion and reducing CPU-side command encoding overhead
 * on supported hardware (Tier 2 Metal GPUs, Apple Silicon M-series and A-series).
 */
@property(nonatomic, assign) BOOL usesMetalArgumentBuffers;

/**
 * Whether to use MTLResidencySet to manage resource residency on the Metal backend.
 *
 * On iOS 18.0+ / macOS 15.0+, Metal residency sets allow explicit tracking and commit of
 * allocations (such as large model weight buffers and KV cache) into working GPU memory.
 * By keeping model allocations resident, this prevents the OS virtual memory system from
 * paging out or evicting large model weights during background memory pressure, eliminating
 * page faults and hitching during subsequent inference runs.
 */
@property(nonatomic, assign) BOOL enablesMetalResidencySet;

@end

NS_ASSUME_NONNULL_END

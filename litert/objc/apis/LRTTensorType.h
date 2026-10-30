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

// LINT.IfChange(element_types)
/**
 * Element data types supported by LiteRT tensors (matching @c LiteRtElementType in
 * litert_model_types.h).
 */
typedef NS_ENUM(NSInteger, LRTElementType) {
  LRTElementTypeNone = 0,
  LRTElementTypeUnknown = LRTElementTypeNone,
  LRTElementTypeFloat32 = 1,
  LRTElementTypeInt32 = 2,
  LRTElementTypeUInt8 = 3,
  LRTElementTypeInt64 = 4,
  LRTElementTypeString = 5,
  LRTElementTypeBool = 6,
  LRTElementTypeInt16 = 7,
  LRTElementTypeComplex64 = 8,
  LRTElementTypeInt8 = 9,
  LRTElementTypeFloat16 = 10,
  LRTElementTypeFloat64 = 11,
  LRTElementTypeComplex128 = 12,
  LRTElementTypeUInt64 = 13,
  LRTElementTypeResource = 14,
  LRTElementTypeVariant = 15,
  LRTElementTypeUInt32 = 16,
  LRTElementTypeUInt16 = 17,
  LRTElementTypeInt4 = 18,
  LRTElementTypeBFloat16 = 19,
  LRTElementTypeInt2 = 20,
  LRTElementTypeUInt4 = 21,
  LRTElementTypeFloat8E4M3FN = 22,
  LRTElementTypeFloat8E5M2 = 23,
};
// LINT.ThenChange(../../c/litert_model_types.h)

/**
 * Describes the element data type and shape dimensions of a ranked LiteRT tensor.
 *
 * Wraps the value semantics of @c litert::RankedTensorType.
 */
@interface LRTTensorType : NSObject <NSCopying>

/** Scalar element data type of the tensor. */
@property(nonatomic, readonly) LRTElementType elementType;

/**
 * Tensor shape dimensions boxed as @c NSNumber.
 *
 * Static dimensions are positive integers; dynamic (unranked/unspecified) dimensions are
 * represented by @c -1.
 */
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *dimensions;

/** Number of dimensions (rank) of the tensor. */
@property(nonatomic, readonly) NSUInteger rank;

/** Whether all dimensions of the tensor are statically known (strictly positive). */
@property(nonatomic, readonly) BOOL hasStaticShape;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

/**
 * Initializes a tensor type with the specified element type and dimensions.
 *
 * @param elementType Scalar element type.
 * @param dimensions Tensor shape dimensions boxed as @c NSNumber.
 * @return An initialized @c LRTTensorType instance.
 */
- (instancetype)initWithElementType:(LRTElementType)elementType
                         dimensions:(NSArray<NSNumber *> *)dimensions NS_DESIGNATED_INITIALIZER;

/**
 * Returns whether the receiver is equal to another @c LRTTensorType instance.
 *
 * @param otherTensorType The tensor type to compare with the receiver.
 * @return @c YES if both instances have the same element type and dimensions, @c NO otherwise.
 */
- (BOOL)isEqualToTensorType:(LRTTensorType *)otherTensorType;

@end

NS_ASSUME_NONNULL_END

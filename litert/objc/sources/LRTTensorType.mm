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

#import "third_party/odml/litert/litert/objc/apis/LRTTensorType.h"

#include <cstdint>
#include <memory>

#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#import "third_party/odml/litert/litert/objc/apis/LRTError.h"
#import "third_party/odml/litert/litert/objc/sources/LRTError+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTTensorType+Internal.h"

@implementation LRTTensorType

- (instancetype)initWithElementType:(LRTElementType)elementType
                         dimensions:(NSArray<NSNumber *> *)dimensions {
  self = [super init];
  if (self) {
    _elementType = elementType;
    _dimensions = [dimensions copy];
  }
  return self;
}

- (BOOL)isEqual:(id)object {
  if (self == object) {
    return YES;
  }
  if (![object isKindOfClass:[LRTTensorType class]]) {
    return NO;
  }
  return [self isEqualToTensorType:(LRTTensorType *)object];
}

- (NSUInteger)hash {
  const NSUInteger kHashPrime = 31;
  NSUInteger result = static_cast<NSUInteger>(_elementType);
  for (NSNumber *dimension in _dimensions) {
    result = result * kHashPrime + dimension.hash;
  }
  return result;
}

- (NSString *)description {
  return [NSString stringWithFormat:@"<%@: %p, elementType=%ld, dimensions=%@>",
                                    NSStringFromClass([self class]), self, (long)_elementType,
                                    _dimensions];
}

#pragma mark - NSCopying

- (id)copyWithZone:(NSZone *)zone {
  // LRTTensorType is deeply immutable, so returning self is safe.
  return self;
}

#pragma mark - Properties

- (NSUInteger)rank {
  return _dimensions.count;
}

- (BOOL)hasStaticShape {
  for (NSNumber *dimension in _dimensions) {
    if (dimension.integerValue <= 0) {
      return NO;
    }
  }
  return YES;
}

#pragma mark - Public

- (BOOL)isEqualToTensorType:(LRTTensorType *)otherTensorType {
  if (!otherTensorType) {
    return NO;
  }
  return _elementType == otherTensorType.elementType &&
         [_dimensions isEqualToArray:otherTensorType.dimensions];
}

#pragma mark - Internal

+ (instancetype)tensorTypeWithCPPTensorType:(const litert::RankedTensorType &)cppTensorType {
  litert::Span<const litert::Layout::Dim> cppDimensions = cppTensorType.Layout().Dimensions();
  NSMutableArray<NSNumber *> *dimensions = [NSMutableArray arrayWithCapacity:cppDimensions.size()];
  for (litert::Layout::Dim dimension : cppDimensions) {
    [dimensions addObject:@(dimension)];
  }
  LRTElementType elementType = static_cast<LRTElementType>(cppTensorType.ElementType());
  return [[self alloc] initWithElementType:elementType dimensions:dimensions];
}

- (std::unique_ptr<litert::RankedTensorType>)cppTensorTypeWithError:(NSError **)error {
  if (_elementType == LRTElementTypeNone) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Valid LRTElementType required");
    return nullptr;
  }
  if (_dimensions == nil) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Dimensions array cannot be nil");
    return nullptr;
  }

  litert::Dimensions cppDimensions;
  cppDimensions.reserve(_dimensions.count);
  for (NSNumber *dimension in _dimensions) {
    if (dimension.intValue < 0) {
      LRTSetError(error, LRTErrorCodeInvalidArgument, @"Dimension values must be non-negative");
      return nullptr;
    }
    cppDimensions.push_back(dimension.intValue);
  }
  return std::make_unique<litert::RankedTensorType>(static_cast<litert::ElementType>(_elementType),
                                                    litert::Layout(cppDimensions));
}

@end

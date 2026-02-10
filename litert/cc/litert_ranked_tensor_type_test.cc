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

#include "litert/cc/litert_ranked_tensor_type.h"

#include <cstdint>
#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_layout.h"

namespace litert {
namespace {

TEST(RankedTensorTypeTest, ConstructorAndAccessors) {
  RankedTensorType type(ElementType::Float32, Layout(Dimensions{1, 2, 3}));
  EXPECT_EQ(type.ElementType(), ElementType::Float32);
  EXPECT_EQ(type.Layout(), Layout(Dimensions{1, 2, 3}));
}

TEST(RankedTensorTypeTest, FromLiteRtRankedTensorType) {
  LiteRtRankedTensorType c_type;
  c_type.element_type = kLiteRtElementTypeFloat32;
  // Layout is an opaque handle, so we can't easily construct a valid one here
  // without digging into implementation details or using a helper.
  // However, RankedTensorType constructor takes const LiteRtRankedTensorType&.
  // Since Layout wraps LiteRtLayout, and we don't have a way to create a raw
  // LiteRtLayout easily in a test without more deps, let's verify element type.
  // Actually, Layout has a constructor from shape.
  // Let's rely on the C++ constructor mostly.

  // If we really want to test the C constructor, we need a valid LiteRtLayout.
  // For now, let's test the round trip if possible or just the C++ side.

  RankedTensorType type(ElementType::Int32, Layout(Dimensions{4, 5}));
  LiteRtRankedTensorType c_struct = static_cast<LiteRtRankedTensorType>(type);
  EXPECT_EQ(c_struct.element_type, kLiteRtElementTypeInt32);

  RankedTensorType type_from_c(c_struct);
  EXPECT_EQ(type_from_c.ElementType(), ElementType::Int32);
  EXPECT_EQ(type_from_c.Layout(), type.Layout());
}

TEST(RankedTensorTypeTest, Equality) {
  RankedTensorType type1(ElementType::Float32, Layout(Dimensions{1, 2}));
  RankedTensorType type2(ElementType::Float32, Layout(Dimensions{1, 2}));
  RankedTensorType type3(ElementType::Int32, Layout(Dimensions{1, 2}));
  RankedTensorType type4(ElementType::Float32, Layout(Dimensions{1, 3}));

  EXPECT_EQ(type1, type2);
  EXPECT_NE(type1, type3);
  EXPECT_NE(type1, type4);
}

TEST(RankedTensorTypeTest, SetElementType) {
  RankedTensorType type(ElementType::Float32, Layout(Dimensions{1, 2}));
  type.SetElementType(ElementType::Int32);
  EXPECT_EQ(type.ElementType(), ElementType::Int32);
}

TEST(RankedTensorTypeTest, Bytes) {
  RankedTensorType type(ElementType::Float32, Layout(Dimensions{2, 3}));
  // Float32 is 4 bytes. 2 * 3 = 6 elements. 6 * 4 = 24 bytes.
  auto bytes = type.Bytes();
  ASSERT_TRUE(bytes.HasValue());
  EXPECT_EQ(*bytes, 24);
}

TEST(RankedTensorTypeTest, BytesInvalidType) {
  RankedTensorType type(ElementType::None, Layout(Dimensions{10}));
  EXPECT_FALSE(type.Bytes().HasValue());
}

TEST(RankedTensorTypeTest, MakeRankedTensorTypeWithInitializerList) {
  auto type = MakeRankedTensorType<float>({1, 2, 3});
  EXPECT_EQ(type.ElementType(), ElementType::Float32);
  EXPECT_EQ(type.Layout(), Layout(Dimensions{1, 2, 3}));
}

TEST(RankedTensorTypeTest, MakeRankedTensorTypeWithContainer) {
  std::vector<int64_t> shape = {4, 5};
  auto type = MakeRankedTensorType<int32_t>(shape);
  EXPECT_EQ(type.ElementType(), ElementType::Int32);
  EXPECT_EQ(type.Layout(), Layout(Dimensions{4, 5}));
}

}  // namespace
}  // namespace litert

// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/utils/flexbuffer_helpers.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers

namespace qnn {
namespace {

// --- InferShape ---

TEST(InferShapeTest, Scalar) {
  flexbuffers::Builder fbb;
  fbb.Int(42);
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  const auto shape = InferShape(root);
  ASSERT_TRUE(shape.has_value());
  EXPECT_EQ(*shape, std::vector<uint32_t>{});
}

TEST(InferShapeTest, EmptyVector) {
  flexbuffers::Builder fbb;
  fbb.Vector([&fbb]() {});
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  const auto shape = InferShape(root);
  ASSERT_TRUE(shape.has_value());
  EXPECT_EQ(*shape, std::vector<uint32_t>{0});
}

TEST(InferShapeTest, OneD) {
  flexbuffers::Builder fbb;
  fbb.Vector([&fbb]() {
    fbb.Int(1);
    fbb.Int(2);
    fbb.Int(3);
  });
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  const auto shape = InferShape(root);
  ASSERT_TRUE(shape.has_value());
  EXPECT_EQ(*shape, std::vector<uint32_t>{3});
}

TEST(InferShapeTest, TwoDRectangular) {
  flexbuffers::Builder fbb;
  fbb.Vector([&fbb]() {
    fbb.Vector([&fbb]() {
      fbb.Int(1);
      fbb.Int(2);
    });
    fbb.Vector([&fbb]() {
      fbb.Int(3);
      fbb.Int(4);
    });
    fbb.Vector([&fbb]() {
      fbb.Int(5);
      fbb.Int(6);
    });
  });
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  const auto shape = InferShape(root);
  ASSERT_TRUE(shape.has_value());
  EXPECT_EQ(*shape, (std::vector<uint32_t>{3, 2}));
}

TEST(InferShapeTest, Ragged) {
  flexbuffers::Builder fbb;
  fbb.Vector([&fbb]() {
    fbb.Vector([&fbb]() {
      fbb.Int(1);
      fbb.Int(2);
    });
    fbb.Vector([&fbb]() { fbb.Int(3); });
  });
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  const auto shape = InferShape(root);
  EXPECT_FALSE(shape.has_value());
}

// --- GetUniformScalarType ---

TEST(GetUniformScalarTypeTest, ScalarBool) {
  flexbuffers::Builder fbb;
  fbb.Bool(true);
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  EXPECT_EQ(GetUniformScalarType(root), FlexbufferScalarType::kBool);
}

TEST(GetUniformScalarTypeTest, ScalarInt) {
  flexbuffers::Builder fbb;
  fbb.Int(-1);
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  EXPECT_EQ(GetUniformScalarType(root), FlexbufferScalarType::kInt);
}

TEST(GetUniformScalarTypeTest, ScalarUint) {
  flexbuffers::Builder fbb;
  fbb.UInt(42u);
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  EXPECT_EQ(GetUniformScalarType(root), FlexbufferScalarType::kUint);
}

TEST(GetUniformScalarTypeTest, ScalarFloat) {
  flexbuffers::Builder fbb;
  fbb.Float(1.5f);
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  EXPECT_EQ(GetUniformScalarType(root), FlexbufferScalarType::kFloat);
}

TEST(GetUniformScalarTypeTest, OneDInt) {
  flexbuffers::Builder fbb;
  fbb.Vector([&fbb]() {
    fbb.Int(1);
    fbb.Int(2);
    fbb.Int(3);
  });
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  EXPECT_EQ(GetUniformScalarType(root), FlexbufferScalarType::kInt);
}

TEST(GetUniformScalarTypeTest, TwoDFloat) {
  flexbuffers::Builder fbb;
  fbb.Vector([&fbb]() {
    fbb.Vector([&fbb]() {
      fbb.Float(1.0f);
      fbb.Float(2.0f);
    });
    fbb.Vector([&fbb]() {
      fbb.Float(3.0f);
      fbb.Float(4.0f);
    });
  });
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  EXPECT_EQ(GetUniformScalarType(root), FlexbufferScalarType::kFloat);
}

TEST(GetUniformScalarTypeTest, EmptyVectorUnsupported) {
  flexbuffers::Builder fbb;
  fbb.Vector([&fbb]() {});
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  EXPECT_EQ(GetUniformScalarType(root), FlexbufferScalarType::kUnsupported);
}

TEST(GetUniformScalarTypeTest, MixedTypeVectorUnsupported) {
  flexbuffers::Builder fbb;
  fbb.Vector([&fbb]() {
    fbb.Int(1);
    fbb.Float(2.0f);
  });
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  EXPECT_EQ(GetUniformScalarType(root), FlexbufferScalarType::kUnsupported);
}

TEST(GetUniformScalarTypeTest, NestedEmptyVectorUnsupported) {
  flexbuffers::Builder fbb;
  fbb.Vector([&fbb]() {
    fbb.Vector([&fbb]() { fbb.Int(1); });
    fbb.Vector([&fbb]() {});
  });
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  EXPECT_EQ(GetUniformScalarType(root), FlexbufferScalarType::kUnsupported);
}

TEST(GetUniformScalarTypeTest, StringUnsupported) {
  flexbuffers::Builder fbb;
  fbb.String("hello");
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  EXPECT_EQ(GetUniformScalarType(root), FlexbufferScalarType::kUnsupported);
}

// --- FillBuffer<T> ---

TEST(FillBufferTest, ScalarInt) {
  flexbuffers::Builder fbb;
  fbb.Int(42);
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  std::vector<int32_t> data;
  EXPECT_TRUE(FillBuffer<int32_t>(root, data));
  EXPECT_EQ(data, std::vector<int32_t>{42});
}

TEST(FillBufferTest, OneDInt) {
  flexbuffers::Builder fbb;
  fbb.Vector([&fbb]() {
    fbb.Int(1);
    fbb.Int(2);
    fbb.Int(3);
  });
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  std::vector<int32_t> data;
  EXPECT_TRUE(FillBuffer<int32_t>(root, data));
  EXPECT_EQ(data, (std::vector<int32_t>{1, 2, 3}));
}

TEST(FillBufferTest, TwoDRowMajor) {
  flexbuffers::Builder fbb;
  fbb.Vector([&fbb]() {
    fbb.Vector([&fbb]() {
      fbb.Int(1);
      fbb.Int(2);
    });
    fbb.Vector([&fbb]() {
      fbb.Int(3);
      fbb.Int(4);
    });
  });
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  std::vector<int32_t> data;
  EXPECT_TRUE(FillBuffer<int32_t>(root, data));
  EXPECT_EQ(data, (std::vector<int32_t>{1, 2, 3, 4}));
}

TEST(FillBufferTest, EmptyVector) {
  flexbuffers::Builder fbb;
  fbb.Vector([&fbb]() {});
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  std::vector<int32_t> data;
  EXPECT_FALSE(FillBuffer<int32_t>(root, data));
  EXPECT_TRUE(data.empty());
}

TEST(FillBufferTest, TypeMismatch) {
  flexbuffers::Builder fbb;
  fbb.Float(1.5f);
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  std::vector<int32_t> data;
  EXPECT_FALSE(FillBuffer<int32_t>(root, data));
  EXPECT_TRUE(data.empty());
}

TEST(FillBufferTest, MixedTypeVectorRejected) {
  flexbuffers::Builder fbb;
  fbb.Vector([&fbb]() {
    fbb.Int(1);
    fbb.Float(2.0f);
  });
  fbb.Finish();
  const auto root = flexbuffers::GetRoot(fbb.GetBuffer());
  std::vector<int32_t> data;
  EXPECT_FALSE(FillBuffer<int32_t>(root, data));
}

}  // namespace
}  // namespace qnn

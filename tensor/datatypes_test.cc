/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensor/datatypes.h"

#include <array>
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensor/utils/matchers.h"

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::StrEq;

namespace litert::tensor {
namespace {

TEST(DatatypesTest, ToString) {
  EXPECT_THAT(ToString(Type::kUnknown), StrEq("Unknown"));
  EXPECT_THAT(ToString(Type::kBOOL), StrEq("BOOL"));
  EXPECT_THAT(ToString(Type::kI2), StrEq("I2"));
  EXPECT_THAT(ToString(Type::kI4), StrEq("I4"));
  EXPECT_THAT(ToString(Type::kI8), StrEq("I8"));
  EXPECT_THAT(ToString(Type::kI16), StrEq("I16"));
  EXPECT_THAT(ToString(Type::kI32), StrEq("I32"));
  EXPECT_THAT(ToString(Type::kI64), StrEq("I64"));
  EXPECT_THAT(ToString(Type::kU4), StrEq("U4"));
  EXPECT_THAT(ToString(Type::kU8), StrEq("U8"));
  EXPECT_THAT(ToString(Type::kU16), StrEq("U16"));
  EXPECT_THAT(ToString(Type::kU32), StrEq("U32"));
  EXPECT_THAT(ToString(Type::kU64), StrEq("U64"));
  EXPECT_THAT(ToString(Type::kFP16), StrEq("FP16"));
  EXPECT_THAT(ToString(Type::kFP32), StrEq("FP32"));
  EXPECT_THAT(ToString(Type::kFP64), StrEq("FP64"));
  EXPECT_THAT(ToString(Type::kBF16), StrEq("BF16"));
}

TEST(DatatypesTest, BufferSize) {
  EXPECT_THAT(BufferSize(Type::kUnknown, 15), Eq(0));
  EXPECT_THAT(BufferSize(Type::kBOOL, 15), Eq(15));
  EXPECT_THAT(BufferSize(Type::kI2, 15), Eq(4));
  EXPECT_THAT(BufferSize(Type::kI4, 15), Eq(8));
  EXPECT_THAT(BufferSize(Type::kI8, 15), Eq(15 * 1));
  EXPECT_THAT(BufferSize(Type::kI16, 15), Eq(15 * 2));
  EXPECT_THAT(BufferSize(Type::kI32, 15), Eq(15 * 4));
  EXPECT_THAT(BufferSize(Type::kI64, 15), Eq(15 * 8));
  EXPECT_THAT(BufferSize(Type::kU4, 15), Eq(8));
  EXPECT_THAT(BufferSize(Type::kU8, 15), Eq(15 * 1));
  EXPECT_THAT(BufferSize(Type::kU16, 15), Eq(15 * 2));
  EXPECT_THAT(BufferSize(Type::kU32, 15), Eq(15 * 4));
  EXPECT_THAT(BufferSize(Type::kU64, 15), Eq(15 * 8));
  EXPECT_THAT(BufferSize(Type::kFP16, 15), Eq(15 * 2));
  EXPECT_THAT(BufferSize(Type::kFP32, 15), Eq(15 * 4));
  EXPECT_THAT(BufferSize(Type::kFP64, 15), Eq(15 * 8));
  EXPECT_THAT(BufferSize(Type::kBF16, 15), Eq(15 * 2));
}

TEST(ConvertTest, FP16ToBF16) {
  const bf16_t bf16_val = 23.;
  const fp16_t fp16_val = 23.;
  EXPECT_THAT(ConvertTo<Type::kFP16>(bf16_val).val, Eq(fp16_val.val));
  EXPECT_THAT(ConvertTo<Type::kBF16>(fp16_val).val, Eq(bf16_val.val));
}

TEST(ConvertTest, FP16ToFP32) {
  const fp16_t fp16_val = 23.;
  const float fp32_val = 23.;
  EXPECT_THAT(ConvertTo<Type::kFP16>(fp32_val).val, Eq(fp16_val.val));
  EXPECT_THAT(ConvertTo<Type::kFP32>(fp16_val), Eq(fp32_val));
}

TEST(ConvertTest, FP16ToFP16Compiles) {
  const fp16_t fp16_val = 23.;
  EXPECT_THAT(ConvertTo<Type::kFP16>(fp16_val).val, Eq(fp16_val.val));
}

TEST(ConvertTest, BF16ToBF16Compiles) {
  const bf16_t bf16_val = 23.;
  EXPECT_THAT(ConvertTo<Type::kBF16>(bf16_val).val, Eq(bf16_val.val));
}

TEST(RangeConvertTest, FP32ToBF16) {
  const std::array<float, 3> src = {1.0f, 2.5f, -3.0f};
  std::array<bf16_t, 3> dest;
  EXPECT_THAT(Convert(src, dest), IsOk());
  EXPECT_THAT(dest, ElementsAre(bf16_t(1.0f), bf16_t(2.5f), bf16_t(-3.0f)));
}

TEST(RangeConvertTest, I4ToFP32) {
  std::array<int4_t, 2> src{int4_t{-5, 3}, {7, -2}};
  std::array<float, 4> dest;
  EXPECT_THAT(Convert(src, dest), IsOk());
  EXPECT_THAT(dest, ElementsAre(-5.0f, 3.0f, 7.0f, -2.0f));
}

TEST(RangeConvertTest, I32ToI4) {
  std::array<int32_t, 4> src = {-5, 3, 7, -2};
  std::array<int4_t, 2> dest;
  EXPECT_THAT(Convert(src, dest), IsOk());
  EXPECT_THAT(dest, ElementsAre(int4_t{-5, 3}, int4_t{7, -2}));
}

TEST(RangeConvertTest, I2ToBF16) {
  std::array<int2_t, 1> src{{{-1, 0, 1, -2}}};
  std::array<bf16_t, 4> dest;
  EXPECT_THAT(Convert(src, dest), IsOk());
  EXPECT_THAT(dest, ElementsAre(bf16_t(-1), bf16_t(0), bf16_t(1), bf16_t(-2)));
}

TEST(RangeConvertTest, I4ToI2) {
  std::array<int4_t, 3> src{int4_t{-1, 0}, {1, -2}, {1}};
  // We zero-initialize the dest because we aren't converting an element count
  // that is divisible by the packed datatypes element count. If we don't the
  // test will randomly fail due to uninitialized memory.
  std::array<int2_t, 2> dest{};
  EXPECT_THAT(Convert(src, dest), IsOk());
  EXPECT_THAT(dest, ElementsAre(int2_t{-1, 0, 1, -2}, int2_t{1, 0, 0, 0}));
}

TEST(RangeConvertTest, I2ToI4) {
  std::array<int2_t, 2> src{int2_t{-1, 0, 1, -2}, {0, -2, 1, -1}};
  std::array<int4_t, 4> dest;
  EXPECT_THAT(Convert(src, dest), IsOk());
  EXPECT_THAT(dest, ElementsAre(int4_t{-1, 0}, int4_t{1, -2}, int4_t{0, -2},
                                int4_t{1, -1}));
}

}  // namespace
}  // namespace litert::tensor

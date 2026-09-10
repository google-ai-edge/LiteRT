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
#include "third_party/odml/litert/tensor/datatypes.h"

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

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

}  // namespace
}  // namespace litert::tensor

// Copyright 2024 Google LLC.
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

#include "litert/cc/litert_any.h"

#include <cstdint>
#include <variant>

#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "litert/c/litert_any.h"
#include "litert/test/matchers.h"

TEST(Any, ConversionNone) {
  auto variant = litert::ToStdAny(LiteRtAny{/*.type=*/kLiteRtAnyTypeNone});
  EXPECT_TRUE(std::holds_alternative<std::monostate>(variant));

  LITERT_ASSERT_OK_AND_ASSIGN(auto any,
                              litert::ToLiteRtAny(litert::LiteRtVariant{}));
  ASSERT_EQ(any.type, kLiteRtAnyTypeNone);
}

TEST(Any, ConversionBool) {
  auto variant_true = litert::ToStdAny(
      LiteRtAny{/*.type=*/kLiteRtAnyTypeBool, {/*.bool_value=*/true}});
  ASSERT_TRUE(std::holds_alternative<bool>(variant_true));
  ASSERT_EQ(std::get<bool>(variant_true), true);

  auto variant_false = litert::ToStdAny(
      LiteRtAny{/*.type=*/kLiteRtAnyTypeBool, {/*.bool_value=*/false}});
  ASSERT_TRUE(std::holds_alternative<bool>(variant_false));
  ASSERT_EQ(std::get<bool>(variant_false), false);

  LITERT_ASSERT_OK_AND_ASSIGN(auto any_true,
                              litert::ToLiteRtAny(litert::LiteRtVariant(true)));
  ASSERT_EQ(any_true.type, kLiteRtAnyTypeBool);
  ASSERT_EQ(any_true.bool_value, true);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto any_false, litert::ToLiteRtAny(litert::LiteRtVariant(false)));
  ASSERT_EQ(any_false.type, kLiteRtAnyTypeBool);
  ASSERT_EQ(any_false.bool_value, false);
}

TEST(Any, ConversionInt) {
  LiteRtAny litert_any;
  litert_any.type = kLiteRtAnyTypeInt;
  litert_any.int_value = 1234;
  auto variant = litert::ToStdAny(litert_any);
  ASSERT_TRUE(std::holds_alternative<int64_t>(variant));
  ASSERT_EQ(std::get<int64_t>(variant), 1234);

  LITERT_ASSERT_OK_AND_ASSIGN(
      const auto any_int8,
      litert::ToLiteRtAny(litert::LiteRtVariant(static_cast<int8_t>(12))));
  ASSERT_EQ(any_int8.type, kLiteRtAnyTypeInt);
  ASSERT_EQ(any_int8.int_value, 12);

  LITERT_ASSERT_OK_AND_ASSIGN(
      const auto any_int16,
      litert::ToLiteRtAny(litert::LiteRtVariant(static_cast<int16_t>(1234))));
  ASSERT_EQ(any_int16.type, kLiteRtAnyTypeInt);
  ASSERT_EQ(any_int16.int_value, 1234);

  LITERT_ASSERT_OK_AND_ASSIGN(
      const auto any_int32,
      litert::ToLiteRtAny(litert::LiteRtVariant(static_cast<int32_t>(1234))));
  ASSERT_EQ(any_int32.type, kLiteRtAnyTypeInt);
  ASSERT_EQ(any_int32.int_value, 1234);

  LITERT_ASSERT_OK_AND_ASSIGN(
      const auto any_int64,
      litert::ToLiteRtAny(litert::LiteRtVariant(static_cast<int64_t>(1234))));
  ASSERT_EQ(any_int64.type, kLiteRtAnyTypeInt);
  ASSERT_EQ(any_int64.int_value, 1234);
}

TEST(Any, ConversionReal) {
  LiteRtAny litert_any;
  litert_any.type = kLiteRtAnyTypeReal;
  litert_any.real_value = 123.4;
  auto variant = litert::ToStdAny(litert_any);
  ASSERT_TRUE(std::holds_alternative<double>(variant));
  ASSERT_EQ(std::get<double>(variant), 123.4);

  LITERT_ASSERT_OK_AND_ASSIGN(
      const auto any_float,
      litert::ToLiteRtAny(litert::LiteRtVariant(static_cast<float>(1.2))));
  ASSERT_EQ(any_float.type, kLiteRtAnyTypeReal);
  EXPECT_NEAR(any_float.real_value, 1.2, 1e-7);

  LITERT_ASSERT_OK_AND_ASSIGN(
      const auto any_double,
      litert::ToLiteRtAny(litert::LiteRtVariant(static_cast<double>(1.2))));
  ASSERT_EQ(any_double.type, kLiteRtAnyTypeReal);
  EXPECT_NEAR(any_double.real_value, 1.2, 1e-7);
}

TEST(Any, ConversionString) {
  constexpr const char* kTestString = "test";
  LiteRtAny litert_any;
  litert_any.type = kLiteRtAnyTypeString;
  litert_any.str_value = kTestString;
  auto variant = litert::ToStdAny(litert_any);
  ASSERT_TRUE(std::holds_alternative<const char*>(variant));
  ASSERT_EQ(std::get<const char*>(variant), kTestString);

  LITERT_ASSERT_OK_AND_ASSIGN(
      const auto any_char_p,
      litert::ToLiteRtAny(litert::LiteRtVariant("test")));
  ASSERT_EQ(any_char_p.type, kLiteRtAnyTypeString);
  EXPECT_STREQ(any_char_p.str_value, "test");
}

TEST(Any, ConversionPtr) {
  const void* kTestPtr = reinterpret_cast<const void*>(1234);
  LiteRtAny litert_any;
  litert_any.type = kLiteRtAnyTypeVoidPtr;
  litert_any.ptr_value = kTestPtr;
  auto variant = litert::ToStdAny(litert_any);
  // Note: ToStdAny returns void* not const void*, so we check for void*
  ASSERT_TRUE(std::holds_alternative<void*>(variant));
  ASSERT_EQ(std::get<void*>(variant), const_cast<void*>(kTestPtr));

  LITERT_ASSERT_OK_AND_ASSIGN(
      const auto any_ptr, litert::ToLiteRtAny(litert::LiteRtVariant(kTestPtr)));
  ASSERT_EQ(any_ptr.type, kLiteRtAnyTypeVoidPtr);
  EXPECT_EQ(any_ptr.ptr_value, kTestPtr);
}

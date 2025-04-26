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

#include <any>
#include <cstdint>

#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "litert/c/litert_any.h"
#include "litert/test/matchers.h"

TEST(Any, ConversionNone) {
  EXPECT_FALSE(
      litert::ToStdAny(LiteRtAny{/*.type=*/kLiteRtAnyTypeNone}).has_value());

  LITERT_ASSERT_OK_AND_ASSIGN(auto any, litert::ToLiteRtAny(std::any()));
  ASSERT_EQ(any.type, kLiteRtAnyTypeNone);
}

TEST(Any, ConversionBool) {
  ASSERT_EQ(std::any_cast<bool>(litert::ToStdAny(LiteRtAny{
                /*.type=*/kLiteRtAnyTypeBool, {/*.bool_value=*/true}})),
            true);
  ASSERT_EQ(std::any_cast<bool>(litert::ToStdAny(LiteRtAny{
                /*.type=*/kLiteRtAnyTypeBool, {/*.bool_value=*/false}})),
            false);

  LITERT_ASSERT_OK_AND_ASSIGN(auto any_true,
                              litert::ToLiteRtAny(std::any(true)));
  ASSERT_EQ(any_true.type, kLiteRtAnyTypeBool);
  ASSERT_EQ(any_true.bool_value, true);
  LITERT_ASSERT_OK_AND_ASSIGN(auto any_false,
                              litert::ToLiteRtAny(std::any(false)));
  ASSERT_EQ(any_false.type, kLiteRtAnyTypeBool);
  ASSERT_EQ(any_false.bool_value, false);
}

TEST(Any, ConversionInt) {
  LiteRtAny litert_any;
  litert_any.type = kLiteRtAnyTypeInt;
  litert_any.int_value = 1234;
  ASSERT_EQ(std::any_cast<int64_t>(litert::ToStdAny(litert_any)), 1234);

  LITERT_ASSERT_OK_AND_ASSIGN(
      const auto any_int8,
      litert::ToLiteRtAny(std::any(static_cast<int8_t>(12))));
  ASSERT_EQ(any_int8.type, kLiteRtAnyTypeInt);
  ASSERT_EQ(any_int8.int_value, 12);

  LITERT_ASSERT_OK_AND_ASSIGN(
      const auto any_int16,
      litert::ToLiteRtAny(std::any(static_cast<int16_t>(1234))));
  ASSERT_EQ(any_int16.type, kLiteRtAnyTypeInt);
  ASSERT_EQ(any_int16.int_value, 1234);

  LITERT_ASSERT_OK_AND_ASSIGN(
      const auto any_int32,
      litert::ToLiteRtAny(std::any(static_cast<int16_t>(1234))));
  ASSERT_EQ(any_int32.type, kLiteRtAnyTypeInt);
  ASSERT_EQ(any_int32.int_value, 1234);

  LITERT_ASSERT_OK_AND_ASSIGN(
      const auto any_int64,
      litert::ToLiteRtAny(std::any(static_cast<int16_t>(1234))));
  ASSERT_EQ(any_int64.type, kLiteRtAnyTypeInt);
  ASSERT_EQ(any_int64.int_value, 1234);
}

TEST(Any, ConversionReal) {
  LiteRtAny litert_any;
  litert_any.type = kLiteRtAnyTypeReal;
  litert_any.real_value = 123.4;
  ASSERT_EQ(std::any_cast<double>(litert::ToStdAny(litert_any)), 123.4);

  LITERT_ASSERT_OK_AND_ASSIGN(
      const auto any_float,
      litert::ToLiteRtAny(std::any(static_cast<float>(1.2))));
  ASSERT_EQ(any_float.type, kLiteRtAnyTypeReal);
  EXPECT_NEAR(any_float.real_value, 1.2, 1e-7);

  LITERT_ASSERT_OK_AND_ASSIGN(
      const auto any_double,
      litert::ToLiteRtAny(std::any(static_cast<double>(1.2))));
  ASSERT_EQ(any_double.type, kLiteRtAnyTypeReal);
  EXPECT_NEAR(any_double.real_value, 1.2, 1e-7);
}

TEST(Any, ConversionString) {
  constexpr const char* kTestString = "test";
  LiteRtAny litert_any;
  litert_any.type = kLiteRtAnyTypeString;
  litert_any.str_value = kTestString;
  ASSERT_EQ(std::any_cast<const char*>(litert::ToStdAny(litert_any)),
            kTestString);

  LITERT_ASSERT_OK_AND_ASSIGN(const auto any_char_p,
                              litert::ToLiteRtAny(std::any("test")));
  ASSERT_EQ(any_char_p.type, kLiteRtAnyTypeString);
  EXPECT_STREQ(any_char_p.str_value, "test");
}

TEST(Any, ConversionPtr) {
  const void* kTestPtr = reinterpret_cast<const void*>(1234);
  LiteRtAny litert_any;
  litert_any.type = kLiteRtAnyTypeVoidPtr;
  litert_any.ptr_value = kTestPtr;
  ASSERT_EQ(std::any_cast<const void*>(litert::ToStdAny(litert_any)), kTestPtr);

  LITERT_ASSERT_OK_AND_ASSIGN(const auto any_ptr,
                              litert::ToLiteRtAny(std::any(kTestPtr)));
  ASSERT_EQ(any_ptr.type, kLiteRtAnyTypeVoidPtr);
  EXPECT_EQ(any_ptr.ptr_value, kTestPtr);
}

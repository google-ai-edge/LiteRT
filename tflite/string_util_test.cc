/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tflite/string_util.h"

#include <stdint.h>

#include <cstddef>
#include <string>

#include <gtest/gtest.h>
#include "tflite/core/c/c_api_types.h"
#include "tflite/core/interpreter.h"
#include "tflite/string_type.h"

namespace tflite {

TEST(StringUtil, TestStringUtil) {
  Interpreter interpreter;
  interpreter.AddTensors(3);

  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;

  TfLiteTensor* t1 = interpreter.tensor(1);
  t1->type = kTfLiteString;
  t1->allocation_type = kTfLiteDynamic;

  // String tensor with one string of length 3
  union {
    char raw_bytes[15];
    struct {
      int32_t num_strs;
      int32_t offsets[2];
      char str_data[3];
    } tensor_data;
  } data;
  data.tensor_data = {1, {12, 15}, {'X', 'Y', 'Z'}};

  TfLiteQuantization quant;
  quant.type = kTfLiteNoQuantization;
  quant.params = nullptr;
  interpreter.SetTensorParametersReadOnly(
      2, kTfLiteString, "", {1}, quant, data.raw_bytes, sizeof(data.raw_bytes));
  TfLiteTensor* t2 = interpreter.tensor(2);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  char s0[] = "ABC";
  string s1 = "DEFG";
  char s2[] = "";

  // Write strings to tensors
  DynamicBuffer buf0;
  ASSERT_EQ(buf0.AddString(s0, 3), kTfLiteOk);
  DynamicBuffer buf1;
  ASSERT_EQ(buf1.AddString(s1.data(), s1.length()), kTfLiteOk);
  ASSERT_EQ(buf0.AddString(s2, 0), kTfLiteOk);

  auto new_shape = TfLiteIntArrayCreate(2);
  new_shape->data[0] = 2;
  new_shape->data[1] = 1;
  buf0.WriteToTensor(t0, new_shape);
  buf1.WriteToTensorAsVector(t1);

  // Check tensor shapes.
  EXPECT_EQ(t0->dims->size, 2);
  EXPECT_EQ(t0->dims->data[0], 2);
  EXPECT_EQ(t0->dims->data[1], 1);

  EXPECT_EQ(t1->dims->size, 1);
  EXPECT_EQ(t1->dims->data[0], 1);

  // Read strings from tensors.
  ASSERT_EQ(GetStringCount(t0), 2);
  StringRef str_ref;
  str_ref = GetString(t0, 0);
  ASSERT_EQ(string(str_ref.str, str_ref.len), "ABC");
  str_ref = GetString(t0, 1);
  ASSERT_EQ(string(str_ref.str, str_ref.len), "");
  ASSERT_EQ(t0->bytes, 19);

  ASSERT_EQ(GetStringCount(t1), 1);
  str_ref = GetString(t1, 0);
  ASSERT_EQ(string(str_ref.str, str_ref.len), "DEFG");
  ASSERT_EQ(t1->bytes, 16);

  ASSERT_EQ(GetStringCount(t2), 1);
  str_ref = GetString(t2, 0);
  ASSERT_EQ(string(str_ref.str, str_ref.len), "XYZ");
  ASSERT_EQ(t2->bytes, 15);
}

TEST(StringUtil, AddStringOverflow32Length) {
  // Set max size to a small number so we can efficiently test for
  // overflows.
  const size_t max_size = 100;
  DynamicBuffer buf{max_size};
  std::string big_string(max_size + 1, 'A');
  ASSERT_EQ(buf.AddString({big_string.data(), big_string.length()}),
            kTfLiteError);
}

TEST(StringUtil, AddStringToFullBufferOverflow32Length) {
  const size_t max_size = 100;
  DynamicBuffer buf{max_size};
  std::string big_string((max_size / 2) + 1, 'A');
  ASSERT_EQ(buf.AddString({big_string.data(), big_string.length()}), kTfLiteOk);
  EXPECT_EQ(buf.AddString({big_string.data(), big_string.length()}),
            kTfLiteError);
}

TEST(StringUtil, TruncatesCharDataToLen) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;

  DynamicBuffer buf;
  char fake_big[] = "ABCADASDA";
  ASSERT_EQ(buf.AddString({fake_big, 3}), kTfLiteOk);
  buf.WriteToTensorAsVector(t0);

  StringRef added_string = GetString(t0, 0);
  EXPECT_EQ(added_string.len, 3);
  EXPECT_EQ(string(added_string.str, 3), "ABC");
}

TEST(StringUtil, TestAddJoinedStringCharSeparator) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;

  char s0[] = "";
  char s1[] = "ABC";
  char s2[] = "DEFG";
  char s3[] = "";
  char s4[] = "XYZ";

  DynamicBuffer buf;
  buf.AddJoinedString({{s0, 0}, {s1, 3}, {s2, 4}, {s3, 0}, {s4, 3}}, ' ');
  buf.WriteToTensorAsVector(t0);

  ASSERT_EQ(GetStringCount(t0), 1);
  StringRef str_ref;
  str_ref = GetString(t0, 0);
  ASSERT_EQ(string(str_ref.str, str_ref.len), " ABC DEFG  XYZ");
  ASSERT_EQ(t0->bytes, 26);
}

TEST(StringUtil, TestAddJoinedStringStringRefSeparator) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;

  char s[] = " - ";
  char s0[] = "";
  char s1[] = "ABC";
  char s2[] = "DEFG";
  char s3[] = "";
  char s4[] = "XYZ";

  DynamicBuffer buf;
  buf.AddJoinedString({{s0, 0}, {s1, 3}, {s2, 4}, {s3, 0}, {s4, 3}}, {s, 3});
  buf.WriteToTensorAsVector(t0);

  ASSERT_EQ(GetStringCount(t0), 1);
  StringRef str_ref;
  str_ref = GetString(t0, 0);
  ASSERT_EQ(string(str_ref.str, str_ref.len), " - ABC - DEFG -  - XYZ");
  ASSERT_EQ(t0->bytes, 34);
}

TEST(StringUtil, TestEmptyList) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;
  DynamicBuffer buf;
  buf.WriteToTensorAsVector(t0);

  ASSERT_EQ(GetStringCount(t0), 0);
  ASSERT_EQ(t0->bytes, 8);
}

TEST(StringUtil, TestShapes) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;
  t0->dims = TfLiteIntArrayCreate(2);
  t0->dims->data[0] = 2;
  t0->dims->data[1] = 1;

  // Not setting a new shape: number of strings must match
  DynamicBuffer buf;
  buf.AddString("ABC", 3);
  buf.AddString("X", 1);
  buf.WriteToTensor(t0, nullptr);

  ASSERT_EQ(t0->dims->size, 2);
  EXPECT_EQ(t0->dims->data[0], 2);
  EXPECT_EQ(t0->dims->data[1], 1);

  auto new_shape = TfLiteIntArrayCreate(2);
  new_shape->data[0] = 1;
  new_shape->data[1] = 2;

  buf.WriteToTensor(t0, new_shape);

  ASSERT_EQ(t0->dims->size, 2);
  EXPECT_EQ(t0->dims->data[0], 1);
  EXPECT_EQ(t0->dims->data[1], 2);
}

TEST(StringUtil, EmptyStringWithEmptyBuffer) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;

  DynamicBuffer buf;
  std::string empty_string;
  ASSERT_EQ(buf.AddString(empty_string.data(), empty_string.length()),
            kTfLiteOk);
  buf.WriteToTensorAsVector(t0);

  StringRef added_string = GetString(t0, 0);
  EXPECT_EQ(added_string.len, 0);
}

TEST(StringUtil, GetStringOutOfRangeAndNonMonotonicOffsets) {
  int32_t raw_buf[6] = {3, 20, 24, 20, 24, 0x44434241};
  StringRef valid_first = GetString(reinterpret_cast<const char*>(raw_buf), 0);
  EXPECT_EQ(valid_first.len, 4u);

  StringRef non_monotonic =
      GetString(reinterpret_cast<const char*>(raw_buf), 1);
  EXPECT_EQ(non_monotonic.str, nullptr);
  EXPECT_EQ(non_monotonic.len, 0u);

  StringRef out_of_range = GetString(reinterpret_cast<const char*>(raw_buf), 3);
  EXPECT_EQ(out_of_range.str, nullptr);
  EXPECT_EQ(out_of_range.len, 0u);

  // Truncated TfLiteTensor buffer is rejected by byte-bounded GetString.
  TfLiteTensor truncated_tensor{};
  truncated_tensor.type = kTfLiteString;
  truncated_tensor.data.raw = reinterpret_cast<char*>(raw_buf);
  truncated_tensor.bytes = 12;  // < (3 + 2) * sizeof(int32_t) = 20
  EXPECT_EQ(GetStringCount(&truncated_tensor), 0);
  StringRef truncated_ref = GetString(&truncated_tensor, 0);
  EXPECT_EQ(truncated_ref.str, nullptr);
  EXPECT_EQ(truncated_ref.len, 0u);

  // Large num_strings (0x3FFFFFFF, where (uint32_t(num_strings) + 2) * 4 wraps
  // 32-bit size_t to 4) is rejected on both 32-bit and 64-bit targets.
  int32_t wrap32_buf[4] = {0x3FFFFFFF, 8, 12, 0x44434241};
  TfLiteTensor wrap32_tensor{};
  wrap32_tensor.type = kTfLiteString;
  wrap32_tensor.data.raw = reinterpret_cast<char*>(wrap32_buf);
  wrap32_tensor.bytes = sizeof(wrap32_buf);
  EXPECT_EQ(GetStringCount(&wrap32_tensor), 0);
  StringRef wrap32_ref = GetString(&wrap32_tensor, 0);
  EXPECT_EQ(wrap32_ref.str, nullptr);
  EXPECT_EQ(wrap32_ref.len, 0u);
  StringRef wrap32_raw_ref =
      GetString(reinterpret_cast<const char*>(wrap32_buf), 0);
  EXPECT_EQ(wrap32_raw_ref.str, nullptr);
  EXPECT_EQ(wrap32_raw_ref.len, 0u);
}

TEST(StringUtil, AddJoinedStringSingleAndZeroLength) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;

  // 1-string join succeeds even when unused separator exceeds max_size.
  DynamicBuffer single_buf(/*max_length=*/10);
  const char item[] = "hello";
  const char huge_sep[] = "0123456789ABCDEF";
  single_buf.AddJoinedString({{item, 5}}, {huge_sep, 16});
  single_buf.WriteToTensorAsVector(t0);
  ASSERT_EQ(GetStringCount(t0), 1);
  StringRef s0 = GetString(t0, 0);
  EXPECT_EQ(string(s0.str, s0.len), "hello");

  // Joining empty strings on an empty DynamicBuffer appends an empty record.
  DynamicBuffer empty_join_buf;
  empty_join_buf.AddJoinedString({{nullptr, 0}, {nullptr, 0}}, {nullptr, 0});
  empty_join_buf.WriteToTensorAsVector(t0);
  ASSERT_EQ(GetStringCount(t0), 1);
  StringRef empty_s = GetString(t0, 0);
  EXPECT_EQ(empty_s.len, 0u);
}

}  // namespace tflite

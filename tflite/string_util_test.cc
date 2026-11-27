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
#include <cstring>
#include <memory>
#include <string>
#include <vector>

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

namespace {

// A string tensor whose buffer is allocated with exactly the number of bytes
// the tensor reports in `bytes`. A read past `bytes` is then a heap overflow
// under AddressSanitizer, so a test of a bounds check does not rely on the
// bytes after the buffer happening to be harmless.
class ExactStringTensor {
 public:
  explicit ExactStringTensor(const std::vector<int32_t>& header_words,
                             const std::string& payload = "")
      : size_(header_words.size() * sizeof(int32_t) + payload.size()),
        buffer_(new char[size_]) {
    if (!header_words.empty()) {
      memcpy(buffer_.get(), header_words.data(),
             header_words.size() * sizeof(int32_t));
    }
    if (!payload.empty()) {
      memcpy(buffer_.get() + header_words.size() * sizeof(int32_t),
             payload.data(), payload.size());
    }
    tensor_.type = kTfLiteString;
    tensor_.data.raw = buffer_.get();
    tensor_.bytes = size_;
  }

  const TfLiteTensor* tensor() const { return &tensor_; }

 private:
  size_t size_;
  std::unique_ptr<char[]> buffer_;
  TfLiteTensor tensor_{};
};

std::string Str(const StringRef& ref) { return std::string(ref.str, ref.len); }

void ExpectRejected(const StringRef& ref) {
  EXPECT_EQ(ref.str, nullptr);
  EXPECT_EQ(ref.len, 0u);
}

// Writes `buf` into a fresh string tensor and returns the number of strings
// it holds, so a test can tell whether an Add call appended a record.
int RecordCount(DynamicBuffer& buf) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t = interpreter.tensor(0);
  t->type = kTfLiteString;
  t->allocation_type = kTfLiteDynamic;
  buf.WriteToTensorAsVector(t);
  return GetStringCount(t);
}

}  // namespace

TEST(StringUtil, GetStringCountAcceptsValidBuffers) {
  // "AB" and "C": a 16 byte header (count and three offsets), 3 payload bytes.
  ExactStringTensor two({2, 16, 18, 19}, "ABC");
  EXPECT_EQ(GetStringCount(two.tensor()), 2);
  // The smallest valid buffer holds no strings: count and one offset.
  ExactStringTensor none({0, 8});
  EXPECT_EQ(GetStringCount(none.tensor()), 0);
  // A buffer exactly as large as its header is accepted.
  ExactStringTensor one_empty({1, 12, 12});
  EXPECT_EQ(GetStringCount(one_empty.tensor()), 1);
}

TEST(StringUtil, GetStringCountRejectsInvalidTensors) {
  EXPECT_EQ(GetStringCount(static_cast<const TfLiteTensor*>(nullptr)), 0);

  TfLiteTensor no_data{};
  no_data.type = kTfLiteString;
  no_data.bytes = 19;
  EXPECT_EQ(GetStringCount(&no_data), 0);

  // Fewer than four bytes: there is no count to read.
  ExactStringTensor too_short(std::vector<int32_t>{},
                              std::string("\x05\x00\x00", 3));
  EXPECT_EQ(GetStringCount(too_short.tensor()), 0);

  // Negative count.
  ExactStringTensor negative({-1, 8});
  EXPECT_EQ(GetStringCount(negative.tensor()), 0);

  // Three strings need a 20 byte header but only 12 bytes are present.
  ExactStringTensor truncated({3, 20, 24});
  EXPECT_EQ(GetStringCount(truncated.tensor()), 0);

  // One byte short of the header of a one string buffer.
  ExactStringTensor one_short(std::vector<int32_t>{1, 12},
                              std::string("\x0c\x00\x00", 3));
  EXPECT_EQ(GetStringCount(one_short.tensor()), 0);

  // (count + 2) * 4 wraps a 32 bit size_t to 4, which a naive check accepts.
  ExactStringTensor wrapping({0x3FFFFFFF, 8, 12, 0x44434241});
  EXPECT_EQ(GetStringCount(wrapping.tensor()), 0);
}

TEST(StringUtil, GetStringReturnsValidStrings) {
  ExactStringTensor two({2, 16, 18, 19}, "ABC");
  EXPECT_EQ(Str(GetString(two.tensor(), 0)), "AB");
  EXPECT_EQ(Str(GetString(two.tensor(), 1)), "C");
  // A string ending exactly at the end of the buffer is accepted, as is an
  // empty string starting exactly at the end of the header.
  ExactStringTensor empty({1, 12, 12});
  StringRef ref = GetString(empty.tensor(), 0);
  EXPECT_NE(ref.str, nullptr);
  EXPECT_EQ(ref.len, 0u);
  // The legacy raw pointer overload is unchanged and agrees on valid input.
  StringRef raw = GetString(two.tensor()->data.raw, 1);
  EXPECT_EQ(Str(raw), "C");
}

TEST(StringUtil, GetStringRejectsNullTensorAndNullBuffer) {
  ExpectRejected(GetString(static_cast<const TfLiteTensor*>(nullptr), 0));
  TfLiteTensor no_data{};
  no_data.type = kTfLiteString;
  no_data.bytes = 19;
  ExpectRejected(GetString(&no_data, 0));
}

TEST(StringUtil, GetStringRejectsBadIndexAndCount) {
  ExactStringTensor two({2, 16, 18, 19}, "ABC");
  ExpectRejected(GetString(two.tensor(), -1));
  ExpectRejected(GetString(two.tensor(), 2));
  ExpectRejected(GetString(two.tensor(), 1 << 30));

  // Zero strings and a negative count leave no string to return.
  ExactStringTensor none({0, 8});
  ExpectRejected(GetString(none.tensor(), 0));
  ExactStringTensor negative({-1, 8, 8});
  ExpectRejected(GetString(negative.tensor(), 0));
}

TEST(StringUtil, GetStringRejectsTruncatedBuffers) {
  // Fewer than four bytes: the count itself is out of bounds.
  ExactStringTensor too_short(std::vector<int32_t>{},
                              std::string("\x05\x00\x00", 3));
  ExpectRejected(GetString(too_short.tensor(), 0));

  // The count promises a 20 byte header, the buffer is 12 bytes.
  ExactStringTensor truncated({3, 20, 24});
  ExpectRejected(GetString(truncated.tensor(), 0));
  ExpectRejected(GetString(truncated.tensor(), 2));

  // (count + 2) * 4 wraps a 32 bit size_t to 4.
  ExactStringTensor wrapping({0x3FFFFFFF, 8, 12, 0x44434241});
  ExpectRejected(GetString(wrapping.tensor(), 0));
}

TEST(StringUtil, GetStringRejectsInvalidOffsets) {
  // Negative start offset.
  ExactStringTensor negative_start({2, -4, 20, 20});
  ExpectRejected(GetString(negative_start.tensor(), 0));

  // Start offset inside the header.
  ExactStringTensor inside_header({2, 4, 16, 16});
  ExpectRejected(GetString(inside_header.tensor(), 0));

  // Decreasing offsets: string 1 would have a negative length.
  ExactStringTensor decreasing({3, 20, 24, 20, 24}, "ABCD");
  EXPECT_EQ(Str(GetString(decreasing.tensor(), 0)), "ABCD");
  ExpectRejected(GetString(decreasing.tensor(), 1));

  // End offset past the end of the buffer: string 1 ends at 40, buffer is 18.
  ExactStringTensor past_end({2, 16, 18, 40}, "AB");
  EXPECT_EQ(Str(GetString(past_end.tensor(), 0)), "AB");
  ExpectRejected(GetString(past_end.tensor(), 1));
}

TEST(StringUtil, AddJoinedStringJoinsValidInput) {
  const char a[] = "a";
  const char bc[] = "bc";
  const char d[] = "d";
  const char sep[] = ", ";
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;

  DynamicBuffer with_separator;
  with_separator.AddJoinedString({{a, 1}, {bc, 2}, {d, 1}}, {sep, 2});
  with_separator.WriteToTensorAsVector(t0);
  ASSERT_EQ(GetStringCount(t0), 1);
  EXPECT_EQ(Str(GetString(t0, 0)), "a, bc, d");
}

TEST(StringUtil, AddJoinedStringAcceptsNullPointersWithZeroLength) {
  const char a[] = "a";
  const char bc[] = "bc";
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;

  // An empty separator does not need a pointer.
  DynamicBuffer without_separator;
  without_separator.AddJoinedString({{a, 1}, {bc, 2}}, {nullptr, 0});
  without_separator.WriteToTensorAsVector(t0);
  ASSERT_EQ(GetStringCount(t0), 1);
  EXPECT_EQ(Str(GetString(t0, 0)), "abc");

  // Empty elements need no pointer either, and the char overload still works.
  DynamicBuffer empty_elements;
  empty_elements.AddJoinedString({{nullptr, 0}, {a, 1}, {nullptr, 0}}, '-');
  empty_elements.WriteToTensorAsVector(t0);
  ASSERT_EQ(GetStringCount(t0), 1);
  EXPECT_EQ(Str(GetString(t0, 0)), "-a-");
}

TEST(StringUtil, AddJoinedStringRejectsNullPointersWithNonzeroLength) {
  const char a[] = "a";
  const char b[] = "b";
  const char sep[] = ", ";

  // Separator is used between two strings but has no storage.
  DynamicBuffer null_separator;
  null_separator.AddJoinedString({{a, 1}, {b, 1}}, {nullptr, 2});
  EXPECT_EQ(RecordCount(null_separator), 0);

  // The first, a middle and the last element each have no storage.
  DynamicBuffer null_first;
  null_first.AddJoinedString({{nullptr, 3}, {b, 1}}, {sep, 2});
  EXPECT_EQ(RecordCount(null_first), 0);
  DynamicBuffer null_middle;
  null_middle.AddJoinedString({{a, 1}, {nullptr, 3}, {b, 1}}, {sep, 2});
  EXPECT_EQ(RecordCount(null_middle), 0);
  DynamicBuffer null_last;
  null_last.AddJoinedString({{a, 1}, {nullptr, 3}}, {sep, 2});
  EXPECT_EQ(RecordCount(null_last), 0);
  DynamicBuffer null_only;
  null_only.AddJoinedString({{nullptr, 3}}, {sep, 2});
  EXPECT_EQ(RecordCount(null_only), 0);

  // A rejected join leaves earlier records intact and later joins working.
  DynamicBuffer recovers;
  recovers.AddJoinedString({{a, 1}, {nullptr, 3}}, {sep, 2});
  recovers.AddJoinedString({{a, 1}, {b, 1}}, {sep, 2});
  EXPECT_EQ(RecordCount(recovers), 1);

  // A separator that is never used (one string) is not dereferenced.
  DynamicBuffer unused_separator;
  unused_separator.AddJoinedString({{a, 1}}, {nullptr, 2});
  EXPECT_EQ(RecordCount(unused_separator), 1);
}

TEST(StringUtil, AddJoinedStringHandlesEmptyList) {
  DynamicBuffer buf;
  buf.AddJoinedString({}, ' ');
  EXPECT_EQ(RecordCount(buf), 1);
}

TEST(StringUtil, AddJoinedStringEnforcesMaxLength) {
  const char a[] = "a";
  const char eleven[] = "0123456789A";
  const char six[] = "012345";
  const char sep4[] = "----";
  const char sep5[] = "-----";
  const char sep11[] = "-----------";

  // A separator longer than the limit is rejected when it would be used...
  DynamicBuffer long_separator(/*max_length=*/10);
  long_separator.AddJoinedString({{a, 1}, {a, 1}}, {sep11, 11});
  EXPECT_EQ(RecordCount(long_separator), 0);
  // ...and ignored when there is only one string.
  DynamicBuffer unused_long_separator(/*max_length=*/10);
  unused_long_separator.AddJoinedString({{a, 1}}, {sep11, 11});
  EXPECT_EQ(RecordCount(unused_long_separator), 1);

  // Separators alone exceed the limit: 3 separators of 4 bytes, limit 10.
  DynamicBuffer many_separators(/*max_length=*/10);
  many_separators.AddJoinedString({{a, 1}, {a, 1}, {a, 1}, {a, 1}}, {sep4, 4});
  EXPECT_EQ(RecordCount(many_separators), 0);

  // A single element longer than the limit.
  DynamicBuffer long_element(/*max_length=*/10);
  long_element.AddJoinedString({{eleven, 11}}, {nullptr, 0});
  EXPECT_EQ(RecordCount(long_element), 0);

  // Each element fits but the running total does not: 6 + 6 > 10.
  DynamicBuffer long_total(/*max_length=*/10);
  long_total.AddJoinedString({{six, 6}, {six, 6}}, {nullptr, 0});
  EXPECT_EQ(RecordCount(long_total), 0);

  // Total of 9 bytes fits below the limit of 10, a total of 10 does not (the
  // limit is exclusive, as in AddString).
  DynamicBuffer nine(/*max_length=*/10);
  nine.AddJoinedString({{six, 4}, {six, 4}}, {sep5, 1});
  EXPECT_EQ(RecordCount(nine), 1);
  DynamicBuffer ten(/*max_length=*/10);
  ten.AddJoinedString({{six, 5}, {six, 4}}, {sep5, 1});
  EXPECT_EQ(RecordCount(ten), 0);

  // Existing content counts against the limit: 6 bytes stored, join of 3
  // bytes is accepted (9 < 10), join of 4 bytes is not (10 >= 10).
  DynamicBuffer existing(/*max_length=*/10);
  ASSERT_EQ(existing.AddString(six, 6), kTfLiteOk);
  existing.AddJoinedString({{a, 1}, {a, 1}}, {sep5, 1});
  EXPECT_EQ(RecordCount(existing), 2);
  DynamicBuffer existing_full(/*max_length=*/10);
  ASSERT_EQ(existing_full.AddString(six, 6), kTfLiteOk);
  existing_full.AddJoinedString({{a, 1}, {a, 1}}, {sep5, 2});
  EXPECT_EQ(RecordCount(existing_full), 1);
}

}  // namespace tflite

// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"

namespace qnn {
namespace {

bool IsPrefix(std::string_view prefix, std::string_view full) {
  return prefix == full.substr(0, prefix.size());
}

bool CheckLoggoing(const std::string log_path, ::qnn::LogLevel log_level) {
  std::ifstream fin(log_path);
  std::string msg;
  while (std::getline(fin, msg)) {
    // Log severity: DEBUG > VERBOSE > INFO > WARN > ERROR
    switch (log_level) {
      case ::qnn::LogLevel::kOff:
        if (IsPrefix("ERROR:", msg)) return false;
        [[fallthrough]];
      case ::qnn::LogLevel::kError:
        if (IsPrefix("WARNING:", msg)) return false;
        [[fallthrough]];
      case ::qnn::LogLevel::kWarn:
        if (IsPrefix("INFO:", msg)) return false;
        [[fallthrough]];
      case ::qnn::LogLevel::kInfo:
        if (IsPrefix("VERBOSE:", msg)) return false;
        [[fallthrough]];
      case ::qnn::LogLevel::kVerbose:
        if (IsPrefix("DEBUG:", msg)) return false;
        [[fallthrough]];
      default:
        break;
    }
  }
  return true;
}

}  // namespace

class LiteRtLog : public ::testing::TestWithParam<::qnn::LogLevel> {};
INSTANTIATE_TEST_SUITE_P(
    , LiteRtLog,
    ::testing::Values(::qnn::LogLevel::kOff, ::qnn::LogLevel::kError,
                      ::qnn::LogLevel::kWarn, ::qnn::LogLevel::kInfo,
                      ::qnn::LogLevel::kVerbose, ::qnn::LogLevel::kDebug));

TEST_P(LiteRtLog, SanityTest) {
  // Create temp file for log
  std::filesystem::path temp_path =
      std::filesystem::temp_directory_path() / "temp.log";
  std::ofstream fout(temp_path);
  ASSERT_TRUE(fout.is_open());

  // Set log file pointer
  FILE* file_ptr = fopen(temp_path.c_str(), "w");
  ASSERT_NE(file_ptr, nullptr);
  qnn::QNNLogger::SetLogFilePointer(file_ptr);

  // Set log_level and print message to file
  auto log_level = GetParam();
  qnn::QNNLogger::SetLogLevel(log_level);
  QNN_LOG_VERBOSE("This is a verbose message.");
  QNN_LOG_INFO("This is an info message.");
  QNN_LOG_WARNING("This is a warning message.");
  QNN_LOG_ERROR("This is an error message.");
  QNN_LOG_DEBUG("This is a debug message.");
  qnn::QNNLogger::SetLogFilePointer(stderr);
  fclose(file_ptr);

  // Check logging messages are as expected
  ASSERT_EQ(CheckLoggoing(temp_path.string(), log_level), true);

  // Delete the temporary log file
  std::filesystem::remove(temp_path);
}

TEST(MiscTest, TestAlwaysFalse) {
  ASSERT_FALSE(::qnn::always_false<bool>);
  ASSERT_FALSE(::qnn::always_false<signed char>);
  ASSERT_FALSE(::qnn::always_false<unsigned char>);
  ASSERT_FALSE(::qnn::always_false<short int>);
  ASSERT_FALSE(::qnn::always_false<unsigned short int>);
  ASSERT_FALSE(::qnn::always_false<int>);
  ASSERT_FALSE(::qnn::always_false<unsigned int>);
  ASSERT_FALSE(::qnn::always_false<long int>);
  ASSERT_FALSE(::qnn::always_false<unsigned long int>);
  ASSERT_FALSE(::qnn::always_false<long long int>);
  ASSERT_FALSE(::qnn::always_false<unsigned long long int>);
  ASSERT_FALSE(::qnn::always_false<float>);
  ASSERT_FALSE(::qnn::always_false<double>);
  ASSERT_FALSE(::qnn::always_false<long double>);
}

TEST(MiscTests, Quantize) {
  float val = 1;
  float scale = 0.1;
  int32_t zero_point = 1;
  auto q_val = Quantize<std::int8_t>(val, scale, zero_point);
  EXPECT_EQ(q_val, 11);
}

TEST(MiscTests, Dequantize) {
  std::int8_t q_val = 11;
  float scale = 0.1;
  int32_t zero_point = 1;
  auto val = Dequantize(q_val, scale, zero_point);
  EXPECT_FLOAT_EQ(val, 1);
}

TEST(MiscTests, ConvertDataFromInt16toUInt16) {
  constexpr std::array<std::int16_t, 4> int16_data = {0, 1, 2, 3};
  std::vector<std::uint16_t> uint16_data;
  ConvertDataFromInt16toUInt16(int16_data, uint16_data);
  EXPECT_EQ(uint16_data[0], 32768);
  EXPECT_EQ(uint16_data[1], 32769);
  EXPECT_EQ(uint16_data[2], 32770);
  EXPECT_EQ(uint16_data[3], 32771);
}

TEST(MiscTests, ConvertDataFromUInt16toInt16) {
  constexpr std::array<std::uint16_t, 4> uint16_data = {32768, 32769, 32770,
                                                        32771};
  std::vector<std::int16_t> int16_data;
  ConvertDataFromUInt16toInt16(uint16_data, int16_data);
  EXPECT_EQ(int16_data[0], 0);
  EXPECT_EQ(int16_data[1], 1);
  EXPECT_EQ(int16_data[2], 2);
  EXPECT_EQ(int16_data[3], 3);
}

TEST(MiscTests, IsStrEndsWith) {
  EXPECT_TRUE(IsStrEndsWith("example_dump", "_dump"));
  EXPECT_FALSE(IsStrEndsWith("example_dump", "_dumps"));
  EXPECT_TRUE(IsStrEndsWith("test_dump", "_dump"));
  EXPECT_FALSE(IsStrEndsWith("test", "_dump"));
  EXPECT_TRUE(IsStrEndsWith("_dump", "_dump"));
  EXPECT_FALSE(IsStrEndsWith("dump", "_dump"));
}

}  // namespace qnn

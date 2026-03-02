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
#include "litert/vendors/qualcomm/core/backends/htp_backend.h"
#include "litert/vendors/qualcomm/core/backends/ir_backend.h"
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
  ASSERT_TRUE(file_ptr);
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

TEST(MiscTests, LoadHtpBackendApiWithInvalidPathTest) {
  DLHandle handle = ::qnn::CreateDLHandle("/invalid/path/to/libQnn.so");
  auto api = ::qnn::ResolveQnnApi(
      handle.get(), ::qnn::HtpBackend::GetExpectedBackendVersion());

  ASSERT_FALSE(api);
}

TEST(MiscTests, DISABLED_LoadHtpBackendApiTest) {
  DLHandle handle = ::qnn::CreateDLHandle(::qnn::HtpBackend::GetLibraryName());
  auto api = ::qnn::ResolveQnnApi(
      handle.get(), ::qnn::HtpBackend::GetExpectedBackendVersion());

  ASSERT_TRUE(api);
}

TEST(MiscTests, UnpackIntData_Int2) {
  // Single byte: 0xE4 (binary: 11 10 01 00), LSB-first unpacking.
  // bits 0-1: 00 -> 0
  // bits 2-3: 01 -> 1
  // bits 4-5: 10 -> -2
  // bits 6-7: 11 -> -1
  {
    const uint8_t src[] = {0xE4};
    auto dst = UnpackIntData(src, 1, kQuantBitWidth2);
    ASSERT_EQ(dst.size(), 4);
    EXPECT_EQ(dst[0], 0);
    EXPECT_EQ(dst[1], 1);
    EXPECT_EQ(dst[2], -2);
    EXPECT_EQ(dst[3], -1);
  }

  // All zeros: 0x00 -> {0, 0, 0, 0}
  {
    const uint8_t src[] = {0x00};
    auto dst = UnpackIntData(src, 1, kQuantBitWidth2);
    ASSERT_EQ(dst.size(), 4);
    EXPECT_EQ(dst[0], 0);
    EXPECT_EQ(dst[1], 0);
    EXPECT_EQ(dst[2], 0);
    EXPECT_EQ(dst[3], 0);
  }

  // All ones: 0xFF (binary: 11 11 11 11) -> {-1, -1, -1, -1}
  {
    const uint8_t src[] = {0xFF};
    auto dst = UnpackIntData(src, 1, kQuantBitWidth2);
    ASSERT_EQ(dst.size(), 4);
    EXPECT_EQ(dst[0], -1);
    EXPECT_EQ(dst[1], -1);
    EXPECT_EQ(dst[2], -1);
    EXPECT_EQ(dst[3], -1);
  }

  // Multiple bytes: {0xE4, 0x1B}
  // 0x1B (binary: 00 01 10 11):
  //   bits 0-1: 11 -> -1
  //   bits 2-3: 10 -> -2
  //   bits 4-5: 01 ->  1
  //   bits 6-7: 00 ->  0
  {
    const uint8_t src[] = {0xE4, 0x1B};
    auto dst = UnpackIntData(src, 2, kQuantBitWidth2);
    ASSERT_EQ(dst.size(), 8);
    EXPECT_EQ(dst[0], 0);
    EXPECT_EQ(dst[1], 1);
    EXPECT_EQ(dst[2], -2);
    EXPECT_EQ(dst[3], -1);
    EXPECT_EQ(dst[4], -1);
    EXPECT_EQ(dst[5], -2);
    EXPECT_EQ(dst[6], 1);
    EXPECT_EQ(dst[7], 0);
  }
}

TEST(MiscTests, UnpackIntData_Int4) {
  // Single byte: 0xE4 (binary: 1110 0100), LSB-first unpacking.
  // lower nibble: 0100 = 4
  // upper nibble: 1110 = -2 (sign extended)
  {
    const uint8_t src[] = {0xE4};
    auto dst = UnpackIntData(src, 1, kQuantBitWidth4);
    ASSERT_EQ(dst.size(), 2);
    EXPECT_EQ(dst[0], 4);
    EXPECT_EQ(dst[1], -2);
  }

  // All zeros: 0x00 -> {0, 0}
  {
    const uint8_t src[] = {0x00};
    auto dst = UnpackIntData(src, 1, kQuantBitWidth4);
    ASSERT_EQ(dst.size(), 2);
    EXPECT_EQ(dst[0], 0);
    EXPECT_EQ(dst[1], 0);
  }

  // All ones: 0xFF (binary: 1111 1111) -> {-1, -1}
  {
    const uint8_t src[] = {0xFF};
    auto dst = UnpackIntData(src, 1, kQuantBitWidth4);
    ASSERT_EQ(dst.size(), 2);
    EXPECT_EQ(dst[0], -1);
    EXPECT_EQ(dst[1], -1);
  }

  // Min/max: 0x87 (binary: 1000 0111)
  // lower nibble: 0111 = 7 (max positive int4)
  // upper nibble: 1000 = -8 (min negative int4)
  {
    const uint8_t src[] = {0x87};
    auto dst = UnpackIntData(src, 1, kQuantBitWidth4);
    ASSERT_EQ(dst.size(), 2);
    EXPECT_EQ(dst[0], 7);
    EXPECT_EQ(dst[1], -8);
  }

  // Multiple bytes: {0xE4, 0x87}
  {
    const uint8_t src[] = {0xE4, 0x87};
    auto dst = UnpackIntData(src, 2, kQuantBitWidth4);
    ASSERT_EQ(dst.size(), 4);
    EXPECT_EQ(dst[0], 4);
    EXPECT_EQ(dst[1], -2);
    EXPECT_EQ(dst[2], 7);
    EXPECT_EQ(dst[3], -8);
  }
}

}  // namespace qnn

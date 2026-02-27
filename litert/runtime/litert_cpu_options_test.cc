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

#include "litert/runtime/litert_cpu_options.h"

#include <string>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"

namespace litert {
namespace internal {
namespace {

TEST(LiteRtCpuOptionsTest, ParseNumThreads) {
  LiteRtCpuOptionsT options = {};
  std::string toml = "num_threads = 4\n";
  EXPECT_EQ(ParseLiteRtCpuOptions(toml.data(), toml.size(), &options),
            kLiteRtStatusOk);
  EXPECT_EQ(options.xnn.num_threads, 4);
}

TEST(LiteRtCpuOptionsTest, ParseFlags) {
  LiteRtCpuOptionsT options = {};
  std::string toml = "flags = 123\n";
  EXPECT_EQ(ParseLiteRtCpuOptions(toml.data(), toml.size(), &options),
            kLiteRtStatusOk);
  EXPECT_EQ(options.xnn.flags, 123);
}

TEST(LiteRtCpuOptionsTest, ParseWeightCacheFilePath) {
  LiteRtCpuOptionsT options = {};
  std::string toml = "weight_cache_file_path = \"/path/to/cache\"\n";
  EXPECT_EQ(ParseLiteRtCpuOptions(toml.data(), toml.size(), &options),
            kLiteRtStatusOk);
  EXPECT_STREQ(options.xnn.weight_cache_file_path, "/path/to/cache");
  EXPECT_EQ(options.weight_cache_file_path_buffer, "/path/to/cache");
}

TEST(LiteRtCpuOptionsTest, ParseWeightCacheFileDescriptor) {
  LiteRtCpuOptionsT options = {};
  std::string toml = "weight_cache_file_descriptor = 42\n";
  EXPECT_EQ(ParseLiteRtCpuOptions(toml.data(), toml.size(), &options),
            kLiteRtStatusOk);
  EXPECT_EQ(options.xnn.weight_cache_file_descriptor, 42);
}

TEST(LiteRtCpuOptionsTest, ParseMultipleOptions) {
  LiteRtCpuOptionsT options = {};
  std::string toml =
      "num_threads = 8\n"
      "flags = 456\n"
      "weight_cache_file_path = \"/some/path\"\n"
      "weight_cache_file_descriptor = 10\n";
  EXPECT_EQ(ParseLiteRtCpuOptions(toml.data(), toml.size(), &options),
            kLiteRtStatusOk);
  EXPECT_EQ(options.xnn.num_threads, 8);
  EXPECT_EQ(options.xnn.flags, 456);
  EXPECT_STREQ(options.xnn.weight_cache_file_path, "/some/path");
  EXPECT_EQ(options.weight_cache_file_path_buffer, "/some/path");
  EXPECT_EQ(options.xnn.weight_cache_file_descriptor, 10);
}

TEST(LiteRtCpuOptionsTest, InvalidArgs) {
  LiteRtCpuOptionsT options = {};
  std::string toml = "num_threads = 4\n";
  EXPECT_EQ(ParseLiteRtCpuOptions(nullptr, toml.size(), &options),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(ParseLiteRtCpuOptions(toml.data(), 0, &options),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(ParseLiteRtCpuOptions(toml.data(), toml.size(), nullptr),
            kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
}  // namespace internal
}  // namespace litert

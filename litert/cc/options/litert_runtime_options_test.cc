// Copyright 2025 Google LLC.
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

#include "litert/cc/options/litert_runtime_options.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/options/litert_runtime_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {

using ::testing::litert::IsOk;

TEST(RuntimeOptionsTest, CreateWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, RuntimeOptions::Create());
}

TEST(RuntimeOptionsTest, EnableProfilingWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, RuntimeOptions::Create());
  EXPECT_THAT(options.SetEnableProfiling(true), IsOk());
  LITERT_ASSERT_OK_AND_ASSIGN(bool enabled, options.GetEnableProfiling());
  EXPECT_TRUE(enabled);
}

TEST(RuntimeOptionsTest, ErrorReporterModeWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, RuntimeOptions::Create());
  EXPECT_THAT(options.SetErrorReporterMode(kLiteRtErrorReporterModeStderr),
              IsOk());
  LITERT_ASSERT_OK_AND_ASSIGN(auto mode, options.GetErrorReporterMode());
  EXPECT_EQ(mode, kLiteRtErrorReporterModeStderr);
}

TEST(RuntimeOptionsTest, CompressQuantizationZeroPointsWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, RuntimeOptions::Create());
  EXPECT_THAT(options.SetCompressQuantizationZeroPoints(true), IsOk());
  LITERT_ASSERT_OK_AND_ASSIGN(bool enabled,
                              options.GetCompressQuantizationZeroPoints());
  EXPECT_TRUE(enabled);
}

TEST(RuntimeOptionsTest, CreateOpaqueOptionsWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, RuntimeOptions::Create());
  EXPECT_THAT(options.SetEnableProfiling(true), IsOk());
  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  ASSERT_EQ(LrtGetOpaqueRuntimeOptionsData(options.Get(), &identifier, &payload,
                                           &payload_deleter),
            kLiteRtStatusOk);
  LiteRtOpaqueOptions opaque_opts = nullptr;
  ASSERT_EQ(LiteRtCreateOpaqueOptions(identifier, payload, payload_deleter,
                                      &opaque_opts),
            kLiteRtStatusOk);
  litert::OpaqueOptions cpp_opaque_opts =
      litert::OpaqueOptions::WrapCObject(opaque_opts, litert::OwnHandle::kYes);
}

}  // namespace
}  // namespace litert

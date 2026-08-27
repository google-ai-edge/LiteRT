/* Copyright 2026 Google LLC.

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

#include "tensor/examples/utils/perfetto_session.h"

#include <filesystem>  // NOLINT
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensor/utils/matchers.h"
#include "perfetto/tracing/track_event.h"  // from @perfetto

namespace litert::tensor {
namespace {

using testing::IsNull;
using testing::Not;

TEST(PerfettoSessionTest, CreateAndStop) {
  std::string trace_path = testing::TempDir() + "/test_trace.perfetto";
  {
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PerfettoSession> session,
                                    PerfettoSession::Create(trace_path));
    EXPECT_THAT(session, Not(IsNull()));
    TRACE_EVENT(kTensorApiCategory, "TestEvent");
    EXPECT_THAT(session->StopAndSave(), IsOk());
  }
  EXPECT_TRUE(std::filesystem::exists(trace_path));
  EXPECT_GT(std::filesystem::file_size(trace_path), 0);
  std::filesystem::remove(trace_path);
}

TEST(PerfettoSessionTest, EmptyOutputPath) {
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PerfettoSession> session,
                                  PerfettoSession::Create(""));
  EXPECT_THAT(session, Not(IsNull()));
  TRACE_EVENT(kTensorApiCategory, "TestEventEmptyPath");
  EXPECT_THAT(session->StopAndSave(), IsOk());
}

}  // namespace
}  // namespace litert::tensor

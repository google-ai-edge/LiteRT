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

#include "litert/ats/executor.h"

#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/matchers.h"
#include "litert/test/simple_buffer.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {
namespace {

using ::testing::ElementsAreArray;

#ifndef _TEST_NPU
TEST(CpuCompiledModelExecutorTest, CreateAndRunModel) {
  std::vector<int32_t> cst_data = {1};
  TensorDetails lhs = {{2, 2}, kLiteRtElementTypeInt32, "lhs"};
  TensorDetails rhs = {{},
                       kLiteRtElementTypeInt32,
                       "cst",
                       MakeBufferRef(cst_data.cbegin(), cst_data.cend())};
  TensorDetails output = {{2, 2}, kLiteRtElementTypeInt32, "output"};

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model, SingleOpModel<kLiteRtOpCodeTflAdd>(
                      {std::move(lhs), std::move(rhs)}, {std::move(output)},
                      tflite::ActivationFunctionType_NONE, false));

  LITERT_ASSERT_OK_AND_ASSIGN(auto executor,
                              CpuCompiledModelExecutor::Create(*model));
  std::vector<SimpleBuffer> inputs;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input, SimpleBuffer::Create<int32_t>({2, 2}, {1, 1, 1, 1}));
  inputs.push_back(std::move(input));
  LITERT_ASSERT_OK_AND_ASSIGN(auto outputs, executor.Run(inputs));
  EXPECT_THAT(outputs.front().Span<int32_t>(), ElementsAreArray({2, 2, 2, 2}));
}

#else

TEST(NpuCompiledModelExecutorTest, CreateAndRunModel) {
  std::vector<int32_t> cst_data = {1};
  TensorDetails lhs = {{2, 2}, kLiteRtElementTypeInt32, "lhs"};
  TensorDetails rhs = {{},
                       kLiteRtElementTypeInt32,
                       "cst",
                       MakeBufferRef(cst_data.cbegin(), cst_data.cend())};
  TensorDetails output = {{2, 2}, kLiteRtElementTypeInt32, "output"};

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model, SingleOpModel<kLiteRtOpCodeTflAdd>(
                      {std::move(lhs), std::move(rhs)}, {std::move(output)},
                      tflite::ActivationFunctionType_NONE, false));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto executor, NpuCompiledModelExecutor::Create(*model, "/data/local/tmp",
                                                      "/data/local/tmp"));

  std::vector<SimpleBuffer> inputs;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input, SimpleBuffer::Create<int32_t>({2, 2}, {1, 1, 1, 1}));
  inputs.push_back(std::move(input));

  LITERT_ASSERT_OK_AND_ASSIGN(auto outputs, executor.Run(inputs));

  EXPECT_THAT(outputs.front().Span<int32_t>(), ElementsAreArray({2, 2, 2, 2}));
}

#endif

}  // namespace
}  // namespace litert::testing

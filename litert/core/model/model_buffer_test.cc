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

#include "litert/core/model/model_buffer.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_op_code.h"
#include "litert/core/dispatch_op_schema.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/test/common.h"
#include "litert/test/testdata/simple_model_test_vectors.h"
#include "tflite/converter/allocation.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#if !defined(LITERT_NO_BUILTIN_OPS)
#include "tflite/kernels/register.h"
#else
#include "tflite/mutable_op_resolver.h"
#endif  // LITERT_NO_BUILTIN_OPS
#include "tflite/model_builder.h"
#include "tflite/stderr_reporter.h"

namespace litert::internal {
namespace {

static constexpr absl::string_view kNpuFile = kGoogleTensorModelFileName;
static constexpr absl::string_view kTfliteFile = "simple_model_npu.tflite";
static constexpr absl::string_view kCascadedTfliteFile =
    "simple_cascade_model_npu.tflite";

TEST(GetModelBufWithByteCode, CreateInterpreter) {
  auto model_with_byte_code =
      GetModelBufWithByteCode(testing::GetTestFilePath(kTfliteFile),
                              testing::GetTestFilePath(kNpuFile));
  ASSERT_TRUE(model_with_byte_code);

  auto alloc = std::make_unique<tflite::MemoryAllocation>(
      model_with_byte_code->Data(), model_with_byte_code->Size(),
      tflite::DefaultErrorReporter());

  auto fb_model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(alloc->base()), alloc->bytes());
  ASSERT_NE(fb_model, nullptr);

#if !defined(LITERT_NO_BUILTIN_OPS)
  tflite::ops::builtin::BuiltinOpResolver resolver;
#else
  tflite::MutableOpResolver resolver;
#endif
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*fb_model, resolver)(&interpreter);
  EXPECT_NE(interpreter, nullptr);
}

TEST(GetModelBufWithByteCode, CheckAppended) {
  auto model_with_byte_code =
      GetModelBufWithByteCode(testing::GetTestFilePath(kTfliteFile),
                              testing::GetTestFilePath(kNpuFile));
  ASSERT_TRUE(model_with_byte_code);

  auto model = LoadModelFromBuffer(*model_with_byte_code);
  ASSERT_TRUE(model);

  auto* op = model->get()->Subgraphs().front()->Ops().front();
  ASSERT_EQ(op->OpCode(), kLiteRtOpCodeTflCustom);
  auto dispatch_opts = GetDispatchOpOptions(op->CustomOptions());
  EXPECT_EQ(dispatch_opts.name, "");
  EXPECT_LE(dispatch_opts.bytecode_offset + dispatch_opts.bytecode_size,
            model_with_byte_code->Size());
}

TEST(GetModelBufWithByteCode, CreateInterpreterWithMultpleNpuNodes) {
  absl::flat_hash_map<std::string, std::string> custom_code_to_npu_file = {
      {"DISPATCH_OP_1", testing::GetTestFilePath(kNpuFile)},
      {"DISPATCH_OP_2", testing::GetTestFilePath(kNpuFile)},
      {"DISPATCH_OP_3", testing::GetTestFilePath(kNpuFile)},
  };

  auto model_with_byte_code = GetModelBufWithByteCode(
      testing::GetTestFilePath(kCascadedTfliteFile), custom_code_to_npu_file);
  ASSERT_TRUE(model_with_byte_code);

  auto alloc = std::make_unique<tflite::MemoryAllocation>(
      model_with_byte_code->Data(), model_with_byte_code->Size(),
      tflite::DefaultErrorReporter());

  auto fb_model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(alloc->base()), alloc->bytes());
  ASSERT_NE(fb_model, nullptr);

#ifndef LITERT_NO_BUILTIN_OPS
  tflite::ops::builtin::BuiltinOpResolver resolver;
#else
  tflite::MutableOpResolver resolver;
#endif
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*fb_model, resolver)(&interpreter);
  EXPECT_NE(interpreter, nullptr);
}

TEST(GetModelBufWithByteCode, CheckAppendedWithMultipleNpuOps) {
  absl::flat_hash_map<std::string, std::string> custom_code_to_npu_file = {
      {"DISPATCH_OP_1", testing::GetTestFilePath(kNpuFile)},
      {"DISPATCH_OP_2", testing::GetTestFilePath(kNpuFile)},
      {"DISPATCH_OP_3", testing::GetTestFilePath(kNpuFile)},
  };

  auto model_with_byte_code = GetModelBufWithByteCode(
      testing::GetTestFilePath(kCascadedTfliteFile), custom_code_to_npu_file);
  ASSERT_TRUE(model_with_byte_code);

  auto model = LoadModelFromBuffer(*model_with_byte_code);
  ASSERT_TRUE(model);

  for (auto& op : model->get()->Subgraphs().front()->Ops()) {
    ASSERT_EQ(op->OpCode(), kLiteRtOpCodeTflCustom);
    auto dispatch_opts = GetDispatchOpOptions(op->CustomOptions());
    EXPECT_EQ(dispatch_opts.name, "");
    EXPECT_LE(dispatch_opts.bytecode_offset + dispatch_opts.bytecode_size,
              model_with_byte_code->Size());
  }
}

}  // namespace
}  // namespace litert::internal

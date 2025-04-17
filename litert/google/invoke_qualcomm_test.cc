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

#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_logging.h"
#include "litert/google/invoke_qualcomm_util.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/test/common.h"
#include "tflite/interpreter.h"
#include "tflite/model_builder.h"
#include "tflite/profiling/time.h"

// Tool for running an arbitrary tflite w/ npu bytecode model through
// dispatch delegate.

ABSL_FLAG(std::string, model, "", "Model resulting from 'apply plugin'.");
ABSL_FLAG(std::string, err, "--", "Where to send error logs.");

namespace litert::tools {
namespace {

using ::tflite::FlatBufferModel;
using ::tflite::Interpreter;

TEST(InvokeModel, Run) {
  const std::string model_path = absl::GetFlag(FLAGS_model);
  const std::string err = absl::GetFlag(FLAGS_err);
  auto [env, runtime, display, dispatch_delegate] =
      SetupInvocation(model_path, err);
  auto& rt = *runtime;
  auto& disp = *display;
  litert::internal::ExternalLiteRtBufferContext buffer_context;
  rt.Interpreter().SetExternalContext(kTfLiteLiteRtBufferContext,
                                      &buffer_context);

  auto invoke_scope = disp.StartS("Invoking model with npu dispatch");

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  ASSERT_EQ(rt.Interpreter().ModifyGraphWithDelegate(dispatch_delegate.get()),
            kTfLiteOk);
  ASSERT_EQ(rt.Interpreter().AllocateTensors(), kTfLiteOk);
  uint64_t start = tflite::profiling::time::NowMicros();
  ASSERT_EQ(rt.Interpreter().Invoke(), kTfLiteOk);
  uint64_t end = tflite::profiling::time::NowMicros();
  LITERT_LOG(LITERT_INFO, "Invoke took %lu microseconds", end - start);
}

}  // namespace

}  // namespace litert::tools

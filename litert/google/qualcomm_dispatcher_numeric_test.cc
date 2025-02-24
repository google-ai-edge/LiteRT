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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "testing/base/public/gunit.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/log/absl_check.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/odml/litert/litert/cc/litert_expected.h"
#include "third_party/odml/litert/litert/core/util/flatbuffer_tools.h"
#include "third_party/odml/litert/litert/google/invoke_qualcomm_util.h"
#include "third_party/odml/litert/litert/runtime/external_litert_buffer_context.h"
#include "third_party/odml/litert/litert/test/common.h"
#include "third_party/tensorflow/lite/interpreter.h"
#include "third_party/tensorflow/lite/kernels/kernel_util.h"
#include "third_party/tensorflow/lite/model_builder.h"

// Tool for running an arbitrary tflite w/ npu bytecode model through
// dispatch delegate.

ABSL_FLAG(std::string, cpu_model, "", "CPU model.");
ABSL_FLAG(std::string, npu_model, "", "NPU model.");
ABSL_FLAG(std::string, err, "--", "Where to send error logs.");

namespace litert::tools {
namespace {

using ::litert::internal::FlatbufferWrapper;
using ::litert::testing::TflRuntime;
using ::tflite::FlatBufferModel;
using ::tflite::Interpreter;

void FillInputTensor(tflite::Interpreter& interpreter, float scale) {
  for (size_t k = 0; k < interpreter.inputs().size(); ++k) {
    TfLiteTensor* tensor_ptr = interpreter.tensor(interpreter.inputs()[k]);
    const auto tensor_elements_count = tflite::NumElements(tensor_ptr);
    if (tensor_ptr->type == kTfLiteFloat32) {
      float* p = interpreter.typed_input_tensor<float>(k);
      for (int i = 0; i < tensor_elements_count; ++i) {
        p[i] = std::sin(i * scale + k) + 0.0001;
      }
    }
    if (tensor_ptr->type == kTfLiteInt32) {
      int* p = interpreter.typed_input_tensor<int>(k);
      for (int i = 0; i < tensor_elements_count; ++i) {
        p[i] = i % 32 + 1 + k;
      }
    }
    if (tensor_ptr->type == kTfLiteInt8) {
      int8_t* p = interpreter.typed_input_tensor<int8_t>(k);
      for (int i = 0; i < tensor_elements_count; ++i) {
        p[i] = i % 256 - 128 + 1 + k;
      }
    }
    if (tensor_ptr->type == kTfLiteUInt8) {
      uint8_t* p = interpreter.typed_input_tensor<uint8_t>(k);
      for (int i = 0; i < tensor_elements_count; ++i) {
        p[i] = i % 256 + 1 + k;
      }
    }
  }
}

void PrintCpuNpuResultsDiff(tflite::Interpreter* cpu, tflite::Interpreter* npu,
                            float eps) {
  for (size_t i = 0; i < cpu->outputs().size(); ++i) {
    std::vector<std::pair<float, int>> all_diffs;
    TfLiteTensor* tensor_ptr = cpu->tensor(cpu->outputs()[i]);
    const auto tensor_elements_count = tflite::NumElements(tensor_ptr);

    std::cout << "Output " << tensor_ptr->name << ":" << std::endl;

    const int kMaxPrint = 20;
    int printed = 0;
    int total_different = 0;
    double MSE = 0;
    float mean_diff = 0;

    auto get_val = [&](tflite::Interpreter* interp, int output_index,
                       int element_index, float& val) {
      if (tensor_ptr->type == kTfLiteFloat32) {
        const float* out = interp->typed_output_tensor<float>(output_index);
        val = out[element_index];
      }
      if (tensor_ptr->type == kTfLiteInt32) {
        const int* out = interp->typed_output_tensor<int>(output_index);
        val = out[element_index];
      }
      if (tensor_ptr->type == kTfLiteInt8) {
        const int8_t* out = interp->typed_output_tensor<int8_t>(output_index);
        val = out[element_index];
      }
      if (tensor_ptr->type == kTfLiteUInt8) {
        const uint8_t* out = interp->typed_output_tensor<uint8_t>(output_index);
        val = out[element_index];
      }
    };

    for (int k = 0; k < tensor_elements_count; ++k) {
      float cpu_val = 0.0f;
      float gpu_val = 0.0f;
      get_val(cpu, i, k, cpu_val);
      get_val(npu, i, k, gpu_val);
      const float abs_diff = fabs(cpu_val - gpu_val);
      const double diff_square = (cpu_val - gpu_val) * (cpu_val - gpu_val);
      MSE += diff_square;
      mean_diff += abs_diff;

      all_diffs.push_back(std::make_pair(abs_diff, k));
      if (abs_diff > eps) {
        total_different++;
        if (printed < kMaxPrint) {
          std::cout << "Element #" << k << ": CPU value - " << cpu_val
                    << ", npu value - " << gpu_val << ", abs diff - "
                    << abs_diff << std::endl;
          printed++;
        }
        if (printed == kMaxPrint) {
          std::cout << "Printed " << kMaxPrint
                    << " different elements, threshhold - " << eps
                    << ", next different elements skipped" << std::endl;
          printed++;
        }
      }
    }
    std::sort(all_diffs.begin(), all_diffs.end());
    std::sort(all_diffs.begin(), all_diffs.end(),
              [](auto& left, auto& right) { return left.first < right.first; });
    std::cout << "Max diff: " << all_diffs.back().first << std::endl;
    std::cout << "Min diff: " << all_diffs.front().first << std::endl;

    for (int ii = 0; ii < kMaxPrint && ii < all_diffs.size(); ++ii) {
      float cpu_val = 0.0f;
      float gpu_val = 0.0f;
      get_val(cpu, i, all_diffs[all_diffs.size() - ii - 1].second, cpu_val);
      get_val(npu, i, all_diffs[all_diffs.size() - ii - 1].second, gpu_val);
      std::cout << "Top " << ii
                << " diff: " << all_diffs[all_diffs.size() - ii - 1].first
                << " @ element #: "
                << all_diffs[all_diffs.size() - ii - 1].second
                << ", CPU val: " << cpu_val << " , NPU val: " << gpu_val
                << std::endl;
    }

    std::cout << "Mean diff: " << mean_diff / all_diffs.size() << std::endl;
    std::cout << "MSE: " << MSE / tensor_elements_count << std::endl;
    std::cout << "Total " << total_different << " out of "
              << tensor_elements_count
              << " are different elements, for output #" << i
              << ", threshhold - " << eps << std::endl;
  }
}

TEST(InvokeModel, Veirify) {
  const std::string model_path = absl::GetFlag(FLAGS_npu_model);
  const std::string err = absl::GetFlag(FLAGS_err);
  // Setup npu interpreter.
  auto [env, runtime, display, dispatch_delegate] =
      SetupInvocation(model_path, err);
  auto& npu_rt = *runtime;
  auto& disp = *display;
  litert::internal::ExternalLiteRtBufferContext npu_buffer_context;
  npu_rt.Interpreter().SetExternalContext(kTfLiteLiteRtBufferContext,
                                          &npu_buffer_context);

  disp.StartS("Set up cpu interpreter");
  // Setup cpu interpreter.
  const std::string cpu_model_path = absl::GetFlag(FLAGS_cpu_model);
  auto cpu_runtime = TflRuntime::CreateFromFlatBuffer(
      *FlatbufferWrapper::CreateFromTflFile(cpu_model_path));
  ABSL_CHECK(cpu_runtime) << "Could not setup cpu runtime";
  auto& cpu_rt = **cpu_runtime;
  litert::internal::ExternalLiteRtBufferContext cpu_buffer_context;
  cpu_rt.Interpreter().SetExternalContext(kTfLiteLiteRtBufferContext,
                                          &cpu_buffer_context);

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif
  auto cpu_scope = disp.StartS("Invoking model with CPU runtime");  // NOLINT
  ASSERT_EQ(cpu_rt.Interpreter().AllocateTensors(), kTfLiteOk);
  FillInputTensor(cpu_rt.Interpreter(), 0.12345);
  ASSERT_EQ(cpu_rt.Interpreter().Invoke(), kTfLiteOk);

  auto npu_scope = disp.StartS("Invoking model with NPU runtime");  // NOLINT
  ASSERT_EQ(
      npu_rt.Interpreter().ModifyGraphWithDelegate(dispatch_delegate.get()),
      kTfLiteOk);
  ASSERT_EQ(npu_rt.Interpreter().AllocateTensors(), kTfLiteOk);
  FillInputTensor(npu_rt.Interpreter(), 0.12345);
  ASSERT_EQ(npu_rt.Interpreter().Invoke(), kTfLiteOk);

  PrintCpuNpuResultsDiff(&cpu_rt.Interpreter(), &npu_rt.Interpreter(), 1e-4f);
}
}  // namespace

}  // namespace litert::tools

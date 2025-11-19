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

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_custom_op_kernel.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_custom_op_test_vectors.h"

using testing::FloatNear;
using testing::Pointwise;

namespace litert {
namespace {

class MyCustomOpKernel : public CustomOpKernel {
 public:
  const std::string& OpName() const override { return kOpName; }

  int OpVersion() const override { return 1; };

  Expected<void> Init(const void* init_data, size_t init_data_size) override {
    return {};
  }

  Expected<void> GetOutputLayouts(
      const std::vector<Layout>& input_layouts,
      std::vector<Layout>& output_layouts) override {
    if (!(input_layouts.size() == 2 && output_layouts.size() == 1)) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Invalid number of arguments");
    }
    output_layouts[0] = input_layouts[0];
    return {};
  }

  Expected<void> Run(const std::vector<TensorBuffer>& inputs,
                     std::vector<TensorBuffer>& outputs) override {
    LITERT_ASSIGN_OR_RETURN(auto tensor_type, outputs[0].TensorType());
    LITERT_ASSIGN_OR_RETURN(size_t num_elements,
                            tensor_type.Layout().NumElements());
    LITERT_ASSIGN_OR_RETURN(auto input0_lock_and_addr,
                            TensorBufferScopedLock::Create<float>(
                                inputs[0], TensorBuffer::LockMode::kRead));
    LITERT_ASSIGN_OR_RETURN(auto input1_lock_and_addr,
                            TensorBufferScopedLock::Create<float>(
                                inputs[1], TensorBuffer::LockMode::kRead));
    LITERT_ASSIGN_OR_RETURN(auto output_lock_and_addr,
                            TensorBufferScopedLock::Create<float>(
                                outputs[0], TensorBuffer::LockMode::kWrite));

    const float* input0 = input0_lock_and_addr.second;
    const float* input1 = input1_lock_and_addr.second;
    float* output = output_lock_and_addr.second;

    for (auto i = 0; i < num_elements; ++i) {
      output[i] = input0[i] + input1[i];
    }

    return {};
  }

  Expected<void> Destroy() override {
    // Nothing to do.
    return {};
  }

 private:
  const std::string kOpName = "MyCustomOp";
};

TEST(CompiledModelTest, CustomOp) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  LITERT_ASSERT_OK_AND_ASSIGN(Options options, Options::Create());
  options.SetHardwareAccelerators(HwAccelerators::kCpu);

  MyCustomOpKernel my_custom_op_kernel;
  ASSERT_TRUE(options.AddCustomOpKernel(my_custom_op_kernel));

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, testing::GetTestFilePath(kModelFileName),
                            options));

  // Create and fill input and output buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> input_buffers,
                              compiled_model.CreateInputBuffers());

  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> output_buffers,
                              compiled_model.CreateOutputBuffers());

  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model with input and output buffers.
  compiled_model.Run(input_buffers, output_buffers);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

}  // namespace
}  // namespace litert

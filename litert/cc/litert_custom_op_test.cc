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

#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_custom_op_kernel.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_string_util.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_custom_op_test_vectors.h"

using ::testing::ElementsAreArray;
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
      return Unexpected(Status::kErrorInvalidArgument,
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

class NormalizeKernel : public CustomOpKernel {
 public:
  const std::string& OpName() const override { return kOpName; }
  int OpVersion() const override { return 1; }

  Expected<void> Init(const void* init_data, size_t init_data_size) override {
    return {};
  }

  Expected<void> GetOutputLayouts(
      const std::vector<Layout>& input_layouts,
      std::vector<Layout>& output_layouts) override {
    if (!(input_layouts.size() == 1 && output_layouts.size() == 1)) {
      return Unexpected(Status::kErrorInvalidArgument,
                        "Invalid number of arguments");
    }
    output_layouts[0] = input_layouts[0];
    return {};
  }

  Expected<void> Run(const std::vector<TensorBuffer>& inputs,
                     std::vector<TensorBuffer>& outputs) override {
    LITERT_ASSIGN_OR_RETURN(std::vector<std::string> input_strs,
                            litert::util::GetStringsFromTensorBuffer(
                                const_cast<TensorBuffer&>(inputs[0])));

    std::vector<std::string> output_strs;
    output_strs.reserve(input_strs.size());
    for (const auto& str : input_strs) {
      std::string lower_str = str;
      absl::c_transform(lower_str, lower_str.begin(),
                        [](unsigned char c) { return std::tolower(c); });
      output_strs.push_back(std::move(lower_str));
    }

    std::vector<uint8_t> serialized =
        litert::util::SerializeStrings(output_strs);
    LITERT_RETURN_IF_ERROR(
        outputs[0].Write<uint8_t>(absl::MakeConstSpan(serialized)));
    return {};
  }

  Expected<void> Destroy() override { return {}; }

 private:
  const std::string kOpName = "Normalize";
};

TEST(CompiledModelTest, DynamicStringCustomOps) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  LITERT_ASSERT_OK_AND_ASSIGN(Options options, Options::Create());
  options.SetHardwareAccelerators(HwAccelerators::kCpu);

  NormalizeKernel normalize_kernel;
  ASSERT_TRUE(options.AddCustomOpKernel(normalize_kernel));

  std::string model_path =
      testing::GetTestFilePath("dynamic_string_custom_ops.tflite");

  LITERT_ASSERT_OK_AND_ASSIGN(CompiledModel compiled_model,
                              CompiledModel::Create(env, model_path, options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names,
                              compiled_model.GetSignatureInputNames());
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names,
                              compiled_model.GetSignatureOutputNames());

  std::vector<std::string> input_strings = {"Google", "LiteRT", "Awesome"};

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer input_buffer,
      litert::util::CreateTensorBufferFromStrings(env, input_strings));

  std::vector<std::string> expected_outputs = {"google", "litert", "awesome"};

  size_t kOutputBufferSize = 200;
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer output_buffer,
      TensorBuffer::CreateManaged(
          env, TensorBufferType::kHostMemory,
          RankedTensorType(ElementType::TfString, Layout(Dimensions({3}))),
          kOutputBufferSize));

  absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
  input_map[input_names[0]] = std::move(input_buffer);

  absl::flat_hash_map<absl::string_view, TensorBuffer> output_map;
  output_map[output_names[0]] = std::move(output_buffer);

  LITERT_ASSERT_OK(compiled_model.Run(input_map, output_map));

  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<std::string> actual_outputs,
      litert::util::GetStringsFromTensorBuffer(output_map[output_names[0]]));

  EXPECT_THAT(actual_outputs, ElementsAreArray(expected_outputs));
}

}  // namespace
}  // namespace litert

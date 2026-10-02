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

#include "litert/cc/kernels/audio_frontend/irfft_kernel.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "tflite/schema/schema_generated.h"
#include "third_party/tflite_micro/tensorflow/lite/micro/audio_frontend/testdata/fft_test_data.h"

namespace litert {
namespace audio_frontend {
namespace {

using ::testing::ElementsAreArray;
using ::testing::FloatNear;
using ::testing::Pointwise;

template <typename T>
class IrfftKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Default parameters for the kernel.
    fft_length_ = 64;
    input_dims_ = {1, fft_length_ + 2};
    element_type_ = ElementType::Float32;
  }

  void InitKernel() {
    Build();
    ASSERT_TRUE(kernel_->Init(init_data_.data(), init_data_.size()));

    Layout input_layout(Dimensions(input_dims_.begin(), input_dims_.end()));
    std::vector<Layout> output_layouts = {Layout()};
    std::vector<Layout> input_layouts = {input_layout};
    ASSERT_TRUE(kernel_->GetOutputLayouts(input_layouts, output_layouts));
    output_layout_ = output_layouts[0];
  }

  tflite::TensorType ConvertElementTypeToTensorType(ElementType element_type) {
    switch (element_type) {
      case ElementType::Float32:
        return tflite::TensorType_FLOAT32;
      case ElementType::Int16:
        return tflite::TensorType_INT16;
      case ElementType::Int32:
        return tflite::TensorType_INT32;
      default:
        return tflite::TensorType_MIN;
    }
  }

  void Build() {
    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("T",
              static_cast<int>(ConvertElementTypeToTensorType(element_type_)));
      fbb.Int("fft_length", fft_length_);
    });
    fbb.Finish();
    init_data_ = fbb.GetBuffer();

    if (kernel_) {
      delete kernel_;
    }
    kernel_ = new IrfftKernel();
  }

  void TearDown() override {
    if (kernel_) {
      kernel_->Destroy();
      delete kernel_;
      kernel_ = nullptr;
    }
  }

  Expected<std::vector<T>> Invoke(const std::vector<T>& input_data) {
    LITERT_ASSIGN_OR_RETURN(size_t output_num_elements,
                            output_layout_.NumElements());

    Layout input_layout(Dimensions(input_dims_.begin(), input_dims_.end()));
    LITERT_ASSIGN_OR_RETURN(size_t input_num_elements,
                            input_layout.NumElements());

    litert::RankedTensorType input_tensor_type(element_type_,
                                               std::move(input_layout));
    auto input_buffer = TensorBuffer::CreateManagedHostMemory(
        input_tensor_type, input_num_elements * sizeof(T));
    if (!input_buffer) {
      return input_buffer.Error();
    }
    LITERT_RETURN_IF_ERROR(
        input_buffer->Write(absl::MakeConstSpan(input_data)));

    // Allocate aligned host memory for the output.
    size_t output_buffer_size = output_num_elements * sizeof(T);
    void* host_mem_addr = aligned_alloc(64, output_buffer_size);
    if (!host_mem_addr) {
      return Unexpected(kLiteRtStatusErrorMemoryAllocationFailure,
                        "Failed to allocate aligned memory");
    }
    absl::Cleanup free_mem = [host_mem_addr] { free(host_mem_addr); };

    litert::RankedTensorType output_tensor_type(element_type_,
                                                std::move(output_layout_));
    auto output_buffer = TensorBuffer::CreateFromHostMemory(
        output_tensor_type, host_mem_addr, output_buffer_size);
    if (!output_buffer) {
      return output_buffer.Error();
    }

    std::vector<TensorBuffer> inputs;
    inputs.push_back(std::move(*input_buffer));
    std::vector<TensorBuffer> outputs;
    outputs.push_back(std::move(*output_buffer));

    LITERT_RETURN_IF_ERROR(kernel_->Run(inputs, outputs));

    // Copy data from the aligned buffer to the result vector.
    std::vector<T> output_data_buffer(output_num_elements);
    std::memcpy(output_data_buffer.data(), host_mem_addr, output_buffer_size);
    return output_data_buffer;
  }

  IrfftKernel* kernel_ = nullptr;
  std::vector<uint8_t> init_data_;
  int fft_length_;
  std::vector<int> input_dims_;
  ElementType element_type_;
  Layout output_layout_;
};

class IrfftKernelTestFloat32 : public IrfftKernelTest<float> {
 protected:
  void SetUp() override {
    IrfftKernelTest<float>::SetUp();
    element_type_ = ElementType::Float32;
  }
};

class IrfftKernelTestInt16 : public IrfftKernelTest<int16_t> {
 protected:
  void SetUp() override {
    IrfftKernelTest<int16_t>::SetUp();
    element_type_ = ElementType::Int16;
  }
};

class IrfftKernelTestInt32 : public IrfftKernelTest<int32_t> {
 protected:
  void SetUp() override {
    IrfftKernelTest<int32_t>::SetUp();
    element_type_ = ElementType::Int32;
  }
};

TEST_F(IrfftKernelTestInt16, IrfftTestLength64Int16) {
  fft_length_ = 64;
  input_dims_ = {1, 66};
  InitKernel();

  auto result = Invoke({256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
                        256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
                        256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
                        256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
                        256, 0, 256, 0, 256, 0, 256, 0, 256, 0});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result, ElementsAreArray(
                           {256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST_F(IrfftKernelTestInt16, IrfftTestLength64Int16OuterDims4) {
  fft_length_ = 64;
  input_dims_ = {2, 2, 66};
  InitKernel();

  auto result = Invoke(
      {256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
       256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
       256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
       256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
       256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
       256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
       256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
       256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
       256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
       256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
       256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
       256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
       256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
       256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
       256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(
      *result,
      ElementsAreArray(
          {256, 0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0,
           0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0,
           0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0,
           0,   0, 0, 0, 256, 0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0,
           0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0,
           0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0,
           0,   0, 0, 0, 0,   0, 0, 0, 256, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0,
           0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0,
           0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0,
           0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 256, 0, 0, 0, 0, 0, 0, 0,
           0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0,
           0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0,
           0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0}));
}

TEST_F(IrfftKernelTestInt32, IrfftTestLength64Int32) {
  fft_length_ = 64;
  input_dims_ = {1, 66};
  InitKernel();

  auto result = Invoke({256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
                        256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
                        256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
                        256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
                        256, 0, 256, 0, 256, 0, 256, 0, 256, 0});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result, ElementsAreArray(
                           {256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST_F(IrfftKernelTestFloat32, IrfftTestLength64Float) {
  fft_length_ = 64;
  input_dims_ = {1, 66};
  InitKernel();

  auto result = Invoke({256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
                        256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
                        256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
                        256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
                        256, 0, 256, 0, 256, 0, 256, 0, 256, 0});
  ASSERT_TRUE(result) << result.Error().Message();
  std::vector<float> expected_output(64, 0.0f);
  expected_output[0] = 256.0f;
  EXPECT_THAT(*result, Pointwise(FloatNear(1e-5), expected_output));
}

TEST_F(IrfftKernelTestInt16, IrfftTestLength512Int16) {
  fft_length_ = 512;
  input_dims_ = {1, 514};
  InitKernel();

  auto result =
      Invoke(std::vector<int16_t>(tflite::kIrfftInt16Length512Input,
                                  tflite::kIrfftInt16Length512Input + 514));
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result,
              ElementsAreArray(tflite::kIrfftInt16Length512Golden, 512));
}

TEST_F(IrfftKernelTestInt32, IrfftTestLength512Int32) {
  fft_length_ = 512;
  input_dims_ = {1, 514};
  InitKernel();

  auto result =
      Invoke(std::vector<int32_t>(tflite::kIrfftInt32Length512Input,
                                  tflite::kIrfftInt32Length512Input + 514));
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result,
              ElementsAreArray(tflite::kIrfftInt32Length512Golden, 512));
}

TEST_F(IrfftKernelTestFloat32, IrfftTestLength512Float) {
  fft_length_ = 512;
  input_dims_ = {1, 514};
  InitKernel();

  auto result =
      Invoke(std::vector<float>(tflite::kIrfftFloatLength512Input,
                                tflite::kIrfftFloatLength512Input + 514));
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(
      *result,
      Pointwise(FloatNear(1e-5),
                std::vector<float>(tflite::kIrfftFloatLength512Golden,
                                   tflite::kIrfftFloatLength512Golden + 512)));
}

TEST_F(IrfftKernelTestFloat32, InitFailsWithNullInitData) {
  IrfftKernel kernel;
  EXPECT_FALSE(kernel.Init(nullptr, 0));
  kernel.Destroy();
}

TEST_F(IrfftKernelTestFloat32, InitFailsWithEmptyInitData) {
  IrfftKernel kernel;
  std::vector<uint8_t> empty_data;
  EXPECT_FALSE(kernel.Init(empty_data.data(), empty_data.size()));
  kernel.Destroy();
}

TEST_F(IrfftKernelTestFloat32, GetOutputLayoutsFailsWithWrongInputSize) {
  fft_length_ = 64;
  input_dims_ = {1, 66};
  InitKernel();

  Layout input_layout(Dimensions(input_dims_.begin(), input_dims_.end()));
  std::vector<Layout> input_layouts = {input_layout, input_layout};
  std::vector<Layout> output_layouts = {Layout()};
  auto result = kernel_->GetOutputLayouts(input_layouts, output_layouts);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST_F(IrfftKernelTestFloat32, GetOutputLayoutsFailsWithWrongOutputSize) {
  fft_length_ = 64;
  input_dims_ = {1, 66};
  InitKernel();

  Layout input_layout(Dimensions(input_dims_.begin(), input_dims_.end()));
  std::vector<Layout> input_layouts = {input_layout};
  std::vector<Layout> output_layouts = {Layout(), Layout()};
  auto result = kernel_->GetOutputLayouts(input_layouts, output_layouts);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST_F(IrfftKernelTestFloat32, GetOutputLayoutsFailsWithEmptyInputDimensions) {
  fft_length_ = 64;
  element_type_ = ElementType::Float32;
  Build();
  ASSERT_TRUE(kernel_->Init(init_data_.data(), init_data_.size()));

  input_dims_ = {};
  Layout input_layout(Dimensions(input_dims_.begin(), input_dims_.end()));
  std::vector<Layout> input_layouts = {input_layout};
  std::vector<Layout> output_layouts = {Layout()};
  auto result = kernel_->GetOutputLayouts(input_layouts, output_layouts);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST_F(IrfftKernelTestFloat32, GetOutputLayoutsFailsWithWrongLastDimension) {
  fft_length_ = 64;
  element_type_ = ElementType::Float32;
  Build();
  ASSERT_TRUE(kernel_->Init(init_data_.data(), init_data_.size()));

  input_dims_ = {1, 65};  // Should be 66 (fft_length / 2 + 1)
  Layout input_layout(Dimensions(input_dims_.begin(), input_dims_.end()));
  std::vector<Layout> input_layouts = {input_layout};
  std::vector<Layout> output_layouts = {Layout()};
  auto result = kernel_->GetOutputLayouts(input_layouts, output_layouts);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(result.Error().Message(),
            "IrfftKernel: Input tensor last dimension must be fft_length + 2.");
}

TEST_F(IrfftKernelTestFloat32, RunFailsWithWrongInputSize) {
  fft_length_ = 64;
  input_dims_ = {1, 66};
  InitKernel();

  Layout input_layout(Dimensions(input_dims_.begin(), input_dims_.end()));
  litert::RankedTensorType input_tensor_type(element_type_,
                                             std::move(input_layout));
  auto input_buffer = TensorBuffer::CreateManagedHostMemory(input_tensor_type,
                                                            66 * sizeof(float));

  std::vector<TensorBuffer> inputs;
  inputs.push_back(std::move(*input_buffer));
  inputs.push_back(std::move(*input_buffer));  // Add a second input

  std::vector<TensorBuffer> outputs;
  auto output_buffer = TensorBuffer::CreateManagedHostMemory(
      litert::RankedTensorType(element_type_, std::move(output_layout_)),
      64 * sizeof(float));
  ASSERT_TRUE(output_buffer);
  outputs.push_back(std::move(*output_buffer));

  auto result = kernel_->Run(inputs, outputs);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST_F(IrfftKernelTestFloat32, RunFailsWithWrongOutputSize) {
  fft_length_ = 64;
  input_dims_ = {1, 66};
  InitKernel();

  Layout input_layout(Dimensions(input_dims_.begin(), input_dims_.end()));
  litert::RankedTensorType input_tensor_type(element_type_,
                                             std::move(input_layout));
  auto input_buffer = TensorBuffer::CreateManagedHostMemory(input_tensor_type,
                                                            66 * sizeof(float));

  std::vector<TensorBuffer> inputs;
  inputs.push_back(std::move(*input_buffer));

  std::vector<TensorBuffer> outputs;
  auto output_buffer1 = TensorBuffer::CreateManagedHostMemory(
      litert::RankedTensorType(element_type_, std::move(output_layout_)),
      64 * sizeof(float));
  ASSERT_TRUE(output_buffer1);
  outputs.push_back(std::move(*output_buffer1));

  auto output_buffer2 = TensorBuffer::CreateManagedHostMemory(
      litert::RankedTensorType(element_type_, std::move(output_layout_)),
      64 * sizeof(float));
  ASSERT_TRUE(output_buffer2);
  outputs.push_back(std::move(*output_buffer2));  // Add a second output

  auto result = kernel_->Run(inputs, outputs);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST_F(IrfftKernelTestFloat32, DoubleDestroyIsSafe) {
  fft_length_ = 64;
  input_dims_ = {1, 66};
  InitKernel();
  kernel_->Destroy();
  // Calling destroy a second time should be safe if state_ is nulled.
  kernel_->Destroy();
}

TEST_F(IrfftKernelTestFloat32, ReInitAfterDestroy) {
  fft_length_ = 64;
  input_dims_ = {1, 66};
  InitKernel();
  kernel_->Destroy();
  // Re-initializing after destroy should be possible.
  InitKernel();
  // Call destroy again to clean up the re-initialized kernel.
  kernel_->Destroy();
}

TEST_F(IrfftKernelTestFloat32, RunFailsWithInputTensorTypeMismatch) {
  fft_length_ = 64;
  input_dims_ = {1, 66};
  // Init kernel with Float32.
  InitKernel();

  // Create input buffer with Int16, causing a type mismatch.
  Layout input_layout(Dimensions(input_dims_.begin(), input_dims_.end()));
  litert::RankedTensorType input_tensor_type(ElementType::Int16,
                                             std::move(input_layout));
  auto input_buffer = TensorBuffer::CreateManagedHostMemory(
      input_tensor_type, 66 * sizeof(int16_t));
  ASSERT_TRUE(input_buffer);

  // Create output buffer with Float32 (matching kernel).
  Layout output_layout(Dimensions({1, 64}));
  litert::RankedTensorType output_tensor_type(ElementType::Float32,
                                              std::move(output_layout));
  auto output_buffer = TensorBuffer::CreateManagedHostMemory(
      output_tensor_type, 64 * sizeof(float));
  ASSERT_TRUE(output_buffer);

  std::vector<TensorBuffer> inputs;
  inputs.push_back(std::move(*input_buffer));
  std::vector<TensorBuffer> outputs;
  outputs.push_back(std::move(*output_buffer));

  auto result = kernel_->Run(inputs, outputs);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(result.Error().Message(), "IrfftKernel: Tensor type mismatch.");
}

TEST_F(IrfftKernelTestFloat32, RunFailsWithOutputTensorTypeMismatch) {
  fft_length_ = 64;
  input_dims_ = {1, 66};
  // Init kernel with Float32.
  InitKernel();

  // Create input buffer with Float32 (matching kernel).
  Layout input_layout(Dimensions(input_dims_.begin(), input_dims_.end()));
  litert::RankedTensorType input_tensor_type(ElementType::Float32,
                                             std::move(input_layout));
  auto input_buffer = TensorBuffer::CreateManagedHostMemory(input_tensor_type,
                                                            66 * sizeof(float));
  ASSERT_TRUE(input_buffer);

  // Create output buffer with Int16, causing a type mismatch.
  Layout output_layout(Dimensions({1, 64}));
  litert::RankedTensorType output_tensor_type(ElementType::Int16,
                                              std::move(output_layout));
  auto output_buffer = TensorBuffer::CreateManagedHostMemory(
      output_tensor_type, 64 * sizeof(int16_t));
  ASSERT_TRUE(output_buffer);

  std::vector<TensorBuffer> inputs;
  inputs.push_back(std::move(*input_buffer));
  std::vector<TensorBuffer> outputs;
  outputs.push_back(std::move(*output_buffer));

  auto result = kernel_->Run(inputs, outputs);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(result.Error().Message(), "IrfftKernel: Tensor type mismatch.");
}

}  // namespace
}  // namespace audio_frontend
}  // namespace litert

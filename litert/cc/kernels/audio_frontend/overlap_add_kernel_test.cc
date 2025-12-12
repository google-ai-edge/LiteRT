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

#include "litert/cc/kernels/audio_frontend/overlap_add_kernel.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "tflite/schema/schema_generated.h"

namespace litert {
namespace audio_frontend {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::Pointwise;

template <typename T>
class OverlapAddKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Default parameters for the kernel.
    frame_step_ = 1;
    input_dims_ = {1, 3};
    element_type_ = ElementType::Float32;
  }

  void InitKernel() {
    Build();
    ASSERT_TRUE(kernel_->Init(init_data_.data(), init_data_.size()));

    Layout input_layout(Dimensions(input_dims_.begin(), input_dims_.end()));
    std::vector<Layout> output_layouts = {Layout()};
    std::vector<Layout> input_layouts = {input_layout};
    ASSERT_TRUE(kernel_->GetOutputLayouts(input_layouts, output_layouts));
  }

  tflite::TensorType ConvertElementTypeToTensorType(ElementType element_type) {
    switch (element_type) {
      case ElementType::Float32:
        return tflite::TensorType_FLOAT32;
      case ElementType::Int16:
        return tflite::TensorType_INT16;
      default:
        return tflite::TensorType_MIN;
    }
  }

  void Build() {
    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("T",
              static_cast<int>(ConvertElementTypeToTensorType(element_type_)));
      fbb.Int("frame_step", frame_step_);
    });
    fbb.Finish();
    init_data_ = fbb.GetBuffer();

    kernel_ = new OverlapAddKernel();
  }

  void TearDown() override {
    if (kernel_) {
      kernel_->Destroy();
      delete kernel_;
      kernel_ = nullptr;
    }
  }

  Expected<std::vector<T>> Invoke(const std::vector<T>& input_data) {
    // The output number of elements is the product of all input dimensions
    // except the last one.
    size_t output_num_elements = 1;
    if (!input_dims_.empty()) {
      for (size_t i = 0; i < input_dims_.size() - 1; ++i) {
        output_num_elements *= input_dims_[i];
      }
    }
    Layout output_layout(Dimensions({static_cast<int>(output_num_elements)}));
    Layout input_layout(Dimensions(input_dims_.begin(), input_dims_.end()));

    LITERT_ASSIGN_OR_RETURN(size_t input_num_elements,
                            input_layout.NumElements());

    litert::RankedTensorType input_tensor_type(element_type_,
                                               std::move(input_layout));
    auto input_buffer = TensorBuffer::CreateManagedHostMemory(
        input_tensor_type, input_num_elements * sizeof(T));
    if (!input_buffer) {
      ABSL_LOG(ERROR) << "Failed to create input TensorBuffer: "
                      << input_buffer.Error().Message();
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
    // Ensure the allocated memory is freed.
    absl::Cleanup free_mem = [host_mem_addr] { free(host_mem_addr); };

    litert::RankedTensorType output_tensor_type(element_type_,
                                                std::move(output_layout));
    auto output_buffer = TensorBuffer::CreateFromHostMemory(
        output_tensor_type, host_mem_addr, output_buffer_size);
    if (!output_buffer) {
      ABSL_LOG(ERROR)
          << "Failed to create output TensorBuffer from host memory: "
          << output_buffer.Error().Message();
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

  OverlapAddKernel* kernel_ = nullptr;
  std::vector<uint8_t> init_data_;
  int frame_step_;
  std::vector<int> input_dims_;
  ElementType element_type_;
};

class OverlapAddKernelTestFloat32 : public OverlapAddKernelTest<float> {
 protected:
  void SetUp() override {
    OverlapAddKernelTest<float>::SetUp();
    element_type_ = ElementType::Float32;
  }
};

class OverlapAddKernelTestInt16 : public OverlapAddKernelTest<int16_t> {
 protected:
  void SetUp() override {
    OverlapAddKernelTest<int16_t>::SetUp();
    element_type_ = ElementType::Int16;
  }
};

TEST_F(OverlapAddKernelTestFloat32, OverlapAddTestFloat32) {
  input_dims_ = {1, 3};
  frame_step_ = 1;
  InitKernel();

  auto result = Invoke({0.1, -3.2, 1.7});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result, Pointwise(FloatEq(), {0.1}));

  result = Invoke({0.5, -0.7, 1.3});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result, Pointwise(FloatEq(), {-2.7}));

  result = Invoke({2.2, 6.7, 5.3});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result, Pointwise(FloatEq(), {3.2}));

  result = Invoke({-4.2, 1.2, 9.1});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result, Pointwise(FloatEq(), {3.8}));
}

TEST_F(OverlapAddKernelTestInt16, OverlapAddTestInt16) {
  input_dims_ = {1, 3};
  frame_step_ = 1;
  InitKernel();

  auto result = Invoke({1, -32, 17});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result, ElementsAre(1));

  result = Invoke({5, -7, 13});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result, ElementsAre(-27));

  result = Invoke({22, 67, 53});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result, ElementsAre(32));

  result = Invoke({-42, 12, 91});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result, ElementsAre(38));
}

TEST_F(OverlapAddKernelTestFloat32, OverlapAddNFrames2Float32) {
  input_dims_ = {2, 3};
  frame_step_ = 1;
  InitKernel();

  auto result = Invoke({0.1, -3.2, 1.7, 0.5, -0.7, 1.3});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result, Pointwise(FloatEq(), {0.1, -2.7}));

  result = Invoke({2.2, 6.7, 5.3, -4.2, 1.2, 9.1});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result, Pointwise(FloatEq(), {3.2, 3.8}));
}

TEST_F(OverlapAddKernelTestInt16, OverlapAddNFrames2Int16) {
  input_dims_ = {2, 3};
  frame_step_ = 1;
  InitKernel();

  auto result = Invoke({1, -32, 17, 5, -7, 13});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result, ElementsAre(1, -27));

  result = Invoke({22, 67, 53, -42, 12, 91});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result, ElementsAre(32, 38));
}

TEST_F(OverlapAddKernelTestFloat32, OverlapAddNFrames4Float32) {
  input_dims_ = {4, 3};
  frame_step_ = 1;
  InitKernel();

  auto result =
      Invoke({0.1, -3.2, 1.7, 0.5, -0.7, 1.3, 2.2, 6.7, 5.3, -4.2, 1.2, 9.1});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result, Pointwise(FloatEq(), {0.1, -2.7, 3.2, 3.8}));
}

TEST_F(OverlapAddKernelTestInt16, OverlapAddNFrames4Int16) {
  input_dims_ = {4, 3};
  frame_step_ = 1;
  InitKernel();

  auto result = Invoke({1, -32, 17, 5, -7, 13, 22, 67, 53, -42, 12, 91});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result, ElementsAre(1, -27, 32, 38));
}

TEST_F(OverlapAddKernelTestFloat32, OverlapAddNFrames4OuterDims4Float32) {
  input_dims_ = {2, 2, 4, 3};
  frame_step_ = 1;
  InitKernel();

  auto result =
      Invoke({0.1, -3.2, 1.7, 0.5, -0.7, 1.3, 2.2, 6.7, 5.3, -4.2, 1.2, 9.1,
              0.1, -3.2, 1.7, 0.5, -0.7, 1.3, 2.2, 6.7, 5.3, -4.2, 1.2, 9.1,
              0.1, -3.2, 1.7, 0.5, -0.7, 1.3, 2.2, 6.7, 5.3, -4.2, 1.2, 9.1,
              0.1, -3.2, 1.7, 0.5, -0.7, 1.3, 2.2, 6.7, 5.3, -4.2, 1.2, 9.1});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result,
              Pointwise(FloatEq(), {0.1, -2.7, 3.2, 3.8, 0.1, -2.7, 3.2, 3.8,
                                    0.1, -2.7, 3.2, 3.8, 0.1, -2.7, 3.2, 3.8}));
}

TEST_F(OverlapAddKernelTestInt16, OverlapAddNFrames4OuterDims4Int16) {
  input_dims_ = {2, 2, 4, 3};
  frame_step_ = 1;
  InitKernel();

  auto result = Invoke({1, -32, 17, 5, -7, 13, 22, 67, 53, -42, 12, 91,
                        1, -32, 17, 5, -7, 13, 22, 67, 53, -42, 12, 91,
                        1, -32, 17, 5, -7, 13, 22, 67, 53, -42, 12, 91,
                        1, -32, 17, 5, -7, 13, 22, 67, 53, -42, 12, 91});
  ASSERT_TRUE(result) << result.Error().Message();
  EXPECT_THAT(*result, ElementsAre(1, -27, 32, 38, 1, -27, 32, 38, 1, -27, 32,
                                   38, 1, -27, 32, 38));
}

TEST_F(OverlapAddKernelTestFloat32, InitFailsWithNullInitData) {
  OverlapAddKernel kernel;
  EXPECT_FALSE(kernel.Init(nullptr, 0));
  EXPECT_EQ(kernel.Init(nullptr, 0).Error().Status(),
            kLiteRtStatusErrorInvalidArgument);
  kernel.Destroy();
}

TEST_F(OverlapAddKernelTestFloat32, InitFailsWithEmptyInitData) {
  OverlapAddKernel kernel;
  std::vector<uint8_t> empty_data;
  EXPECT_FALSE(kernel.Init(empty_data.data(), empty_data.size()));
  EXPECT_EQ(kernel.Init(empty_data.data(), empty_data.size()).Error().Status(),
            kLiteRtStatusErrorInvalidArgument);
  kernel.Destroy();
}

TEST_F(OverlapAddKernelTestFloat32, GetOutputLayoutsFailsWithOneDimension) {
  frame_step_ = 1;
  element_type_ = ElementType::Float32;
  Build();
  ASSERT_TRUE(kernel_->Init(init_data_.data(), init_data_.size()));

  input_dims_ = {1};  // Single dimension
  Layout input_layout(Dimensions(input_dims_.begin(), input_dims_.end()));
  std::vector<Layout> input_layouts = {input_layout};
  std::vector<Layout> output_layouts = {Layout()};

  auto result = kernel_->GetOutputLayouts(input_layouts, output_layouts);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
  EXPECT_THAT(
      result.Error().Message(),
      testing::HasSubstr(
          "OverlapAddKernel: Input tensor must have at least 2 dimensions."));
}

TEST_F(OverlapAddKernelTestFloat32, GetOutputLayoutsFailsWithZeroDimensions) {
  frame_step_ = 1;
  element_type_ = ElementType::Float32;
  Build();
  ASSERT_TRUE(kernel_->Init(init_data_.data(), init_data_.size()));

  input_dims_ = {};  // Zero dimensions
  Layout input_layout(Dimensions(input_dims_.begin(), input_dims_.end()));
  std::vector<Layout> input_layouts = {input_layout};
  std::vector<Layout> output_layouts = {Layout()};

  auto result = kernel_->GetOutputLayouts(input_layouts, output_layouts);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
  EXPECT_THAT(
      result.Error().Message(),
      testing::HasSubstr(
          "OverlapAddKernel: Input tensor must have at least 2 dimensions."));
}

TEST_F(OverlapAddKernelTestFloat32, MultipleDestroyCalls) {
  input_dims_ = {1, 3};
  frame_step_ = 1;
  InitKernel();

  // First call to Destroy()
  EXPECT_TRUE(kernel_->Destroy());

  // Second call to Destroy() - should not crash.
  EXPECT_TRUE(kernel_->Destroy());
}

TEST_F(OverlapAddKernelTestInt16, MultipleDestroyCalls) {
  input_dims_ = {1, 3};
  frame_step_ = 1;
  InitKernel();

  // First call to Destroy()
  EXPECT_TRUE(kernel_->Destroy());

  // Second call to Destroy() - should not crash.
  EXPECT_TRUE(kernel_->Destroy());
}

TEST_F(OverlapAddKernelTestFloat32, GetOutputLayoutsCorrectDimensions) {
  input_dims_ = {4, 5};
  frame_step_ = 3;
  Build();
  ASSERT_TRUE(kernel_->Init(init_data_.data(), init_data_.size()));

  Layout input_layout(Dimensions(input_dims_.begin(), input_dims_.end()));
  std::vector<Layout> output_layouts = {Layout()};
  std::vector<Layout> input_layouts = {input_layout};
  ASSERT_TRUE(kernel_->GetOutputLayouts(input_layouts, output_layouts));

  ASSERT_EQ(output_layouts.size(), 1);
  const auto& output_layout = output_layouts[0];
  EXPECT_THAT(output_layout.Dimensions(), ElementsAre(12));
}

}  // namespace
}  // namespace audio_frontend
}  // namespace litert

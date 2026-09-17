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

#include "tensor/runners/model_chain.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/test/common.h"
#include "litert/test/testdata/simple_model_test_vectors.h"
#include "tensor/runners/litert/litert_buffer.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::NotNull;
using BufferMap = FunctionalModelStage::BufferMap;

TEST(BoundaryLayoutNegotiatorTest, HarmonizeCompatibleDescriptors) {
  HardwareBufferDescriptor prod_desc;
  prod_desc.shape = {1, 128};
  prod_desc.element_type = litert::ElementType::Float32;
  prod_desc.size_bytes = 128 * sizeof(float);
  prod_desc.alignment = 64;
  prod_desc.gpu_writable = true;

  HardwareBufferDescriptor cons_desc;
  cons_desc.shape = {1, 128};
  cons_desc.element_type = litert::ElementType::Float32;
  cons_desc.size_bytes = 128 * sizeof(float);
  cons_desc.alignment = 128;
  cons_desc.npu_accessible = true;

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto harmonized,
      BoundaryLayoutNegotiator::HarmonizeStageBoundary(prod_desc, cons_desc));
  EXPECT_THAT(harmonized.alignment, Eq(128));
  EXPECT_TRUE(harmonized.gpu_writable);
  EXPECT_TRUE(harmonized.npu_accessible);
  EXPECT_THAT(harmonized.element_type, Eq(litert::ElementType::Float32));
  EXPECT_THAT(harmonized.shape, Eq(std::vector<int32_t>({1, 128})));
}

TEST(BoundaryLayoutNegotiatorTest, FailsOnTypeMismatch) {
  HardwareBufferDescriptor prod_desc;
  prod_desc.shape = {4};
  prod_desc.element_type = litert::ElementType::Float32;

  HardwareBufferDescriptor cons_desc;
  cons_desc.shape = {4};
  cons_desc.element_type = litert::ElementType::Int32;

  auto harmonized_or =
      BoundaryLayoutNegotiator::HarmonizeStageBoundary(prod_desc, cons_desc);
  EXPECT_FALSE(harmonized_or.ok());
  EXPECT_THAT(harmonized_or.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
}

TEST(BoundaryLayoutNegotiatorTest, FailsOnElementCountMismatch) {
  HardwareBufferDescriptor prod_desc;
  prod_desc.shape = {2, 4};
  prod_desc.element_type = litert::ElementType::Float32;

  HardwareBufferDescriptor cons_desc;
  cons_desc.shape = {3, 4};
  cons_desc.element_type = litert::ElementType::Float32;

  auto harmonized_or =
      BoundaryLayoutNegotiator::HarmonizeStageBoundary(prod_desc, cons_desc);
  EXPECT_FALSE(harmonized_or.ok());
  EXPECT_THAT(harmonized_or.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
}

TEST(BoundaryLayoutNegotiatorTest, AllowsSingletonDimensionDifferences) {
  HardwareBufferDescriptor prod_desc;
  prod_desc.shape = {1, 128, 1};
  prod_desc.element_type = litert::ElementType::Float32;

  HardwareBufferDescriptor cons_desc;
  cons_desc.shape = {128};
  cons_desc.element_type = litert::ElementType::Float32;

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto harmonized,
      BoundaryLayoutNegotiator::HarmonizeStageBoundary(prod_desc, cons_desc));
  EXPECT_THAT(harmonized.shape, Eq(std::vector<int32_t>({1, 128, 1})));
}

TEST(BoundaryLayoutNegotiatorTest, FailsOnTransposedNonSingletonDimensions) {
  HardwareBufferDescriptor prod_desc;
  prod_desc.shape = {2, 4};
  prod_desc.element_type = litert::ElementType::Float32;

  HardwareBufferDescriptor cons_desc;
  cons_desc.shape = {4, 2};
  cons_desc.element_type = litert::ElementType::Float32;

  auto harmonized_or =
      BoundaryLayoutNegotiator::HarmonizeStageBoundary(prod_desc, cons_desc);
  EXPECT_FALSE(harmonized_or.ok());
  EXPECT_THAT(harmonized_or.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
}

TEST(BoundaryLayoutNegotiatorTest, NegotiatesWebGpuBufferType) {
  HardwareBufferDescriptor prod_desc;
  prod_desc.shape = {1, 64};
  prod_desc.element_type = litert::ElementType::Float32;
  prod_desc.buffer_type = litert::TensorBufferType::kHostMemory;

  HardwareBufferDescriptor cons_desc;
  cons_desc.shape = {1, 64};
  cons_desc.element_type = litert::ElementType::Float32;
  cons_desc.buffer_type = litert::TensorBufferType::kWebGpuBuffer;

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto harmonized,
      BoundaryLayoutNegotiator::HarmonizeStageBoundary(prod_desc, cons_desc));
  EXPECT_THAT(harmonized.buffer_type,
              Eq(litert::TensorBufferType::kWebGpuBuffer));
}

TEST(BoundaryLayoutNegotiatorTest, HarmonizesAlignmentToStrictest) {
  HardwareBufferDescriptor prod_desc;
  prod_desc.shape = {4};
  prod_desc.element_type = litert::ElementType::Float32;
  prod_desc.alignment = 16;

  HardwareBufferDescriptor cons_a_desc = prod_desc;
  cons_a_desc.alignment = 64;

  HardwareBufferDescriptor cons_b_desc = prod_desc;
  cons_b_desc.alignment = 128;

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto harm1,
      BoundaryLayoutNegotiator::HarmonizeStageBoundary(prod_desc, cons_a_desc));
  EXPECT_THAT(harm1.alignment, Eq(64));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto harm2,
      BoundaryLayoutNegotiator::HarmonizeStageBoundary(harm1, cons_b_desc));
  EXPECT_THAT(harm2.alignment, Eq(128));
}

TEST(BoundaryLayoutNegotiatorTest, FailsOnUnderallocatedSizeBytes) {
  HardwareBufferDescriptor prod_desc;
  prod_desc.shape = {2, 2};
  prod_desc.element_type = litert::ElementType::Float32;
  prod_desc.size_bytes = 4;  // Needs 16 bytes for 4 float32 elements.

  HardwareBufferDescriptor cons_desc;
  cons_desc.shape = {2, 2};
  cons_desc.element_type = litert::ElementType::Float32;

  auto harmonized_or =
      BoundaryLayoutNegotiator::HarmonizeStageBoundary(prod_desc, cons_desc);
  EXPECT_FALSE(harmonized_or.ok());
  EXPECT_THAT(harmonized_or.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(harmonized_or.status().message(),
              HasSubstr("smaller than required packed bytes"));
}

TEST(ModelChainTest, EndToEndLinearPipelineZeroCopyFlow) {
  // Stage 1: Scale by 2.0
  HardwareBufferDescriptor stage1_in_desc;
  stage1_in_desc.shape = {4};
  stage1_in_desc.element_type = litert::ElementType::Float32;
  stage1_in_desc.size_bytes = 4 * sizeof(float);

  HardwareBufferDescriptor stage1_out_desc = stage1_in_desc;

  auto stage1 = std::make_shared<FunctionalModelStage>(
      "Stage1_Scale",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"in", stage1_in_desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"out", stage1_out_desc}},
      [](const BufferMap& inputs, const BufferMap& outputs) {
        auto in_span = inputs.at("in")->Lock();
        const auto* in_data = reinterpret_cast<const float*>(in_span.data());
        auto out_span = outputs.at("out")->LockMutable();
        auto* out_data = reinterpret_cast<float*>(out_span.data());
        size_t count = in_span.size() / sizeof(float);
        for (size_t i = 0; i < count; ++i) {
          out_data[i] = in_data[i] * 2.0f;
        }
        return absl::OkStatus();
      });

  // Stage 2: Add 5.0
  HardwareBufferDescriptor stage2_in_desc = stage1_out_desc;
  HardwareBufferDescriptor stage2_out_desc = stage1_out_desc;

  auto stage2 = std::make_shared<FunctionalModelStage>(
      "Stage2_Add",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"in", stage2_in_desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"out", stage2_out_desc}},
      [](const BufferMap& inputs, const BufferMap& outputs) {
        auto in_span = inputs.at("in")->Lock();
        const auto* in_data = reinterpret_cast<const float*>(in_span.data());
        auto out_span = outputs.at("out")->LockMutable();
        auto* out_data = reinterpret_cast<float*>(out_span.data());
        size_t count = in_span.size() / sizeof(float);
        for (size_t i = 0; i < count; ++i) {
          out_data[i] = in_data[i] + 5.0f;
        }
        return absl::OkStatus();
      });

  // Stage 3: Square (x * x)
  HardwareBufferDescriptor stage3_in_desc = stage2_out_desc;
  HardwareBufferDescriptor stage3_out_desc = stage2_out_desc;

  auto stage3 = std::make_shared<FunctionalModelStage>(
      "Stage3_Square",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"in", stage3_in_desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"out", stage3_out_desc}},
      [](const BufferMap& inputs, const BufferMap& outputs) {
        auto in_span = inputs.at("in")->Lock();
        const auto* in_data = reinterpret_cast<const float*>(in_span.data());
        auto out_span = outputs.at("out")->LockMutable();
        auto* out_data = reinterpret_cast<float*>(out_span.data());
        size_t count = in_span.size() / sizeof(float);
        for (size_t i = 0; i < count; ++i) {
          out_data[i] = in_data[i] * in_data[i];
        }
        return absl::OkStatus();
      });

  // Build ModelChain with auto-inferred linear connections
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto chain, ModelChain::Builder()
                                                  .AddStage(stage1)
                                                  .AddStage(stage2)
                                                  .AddStage(stage3)
                                                  .Build());

  // Verify Zero-Copy Shared Memory Guarantee:
  // Stage 1 output buffer must be the exact same shared buffer as Stage 2
  // input.
  auto stage1_out_buf = stage1->GetOutputBuffer("out");
  auto stage2_in_buf = stage2->GetInputBuffer("in");
  ASSERT_THAT(stage1_out_buf, NotNull());
  ASSERT_THAT(stage2_in_buf, NotNull());
  EXPECT_THAT(stage1_out_buf.get(), Eq(stage2_in_buf.get()));
  EXPECT_EQ(stage1_out_buf->tensor_buffer().Get(),
            stage2_in_buf->tensor_buffer().Get());

  // Stage 2 output buffer must be the exact same shared buffer as Stage 3
  // input.
  auto stage2_out_buf = stage2->GetOutputBuffer("out");
  auto stage3_in_buf = stage3->GetInputBuffer("in");
  ASSERT_THAT(stage2_out_buf, NotNull());
  ASSERT_THAT(stage3_in_buf, NotNull());
  EXPECT_THAT(stage2_out_buf.get(), Eq(stage3_in_buf.get()));
  EXPECT_EQ(stage2_out_buf->tensor_buffer().Get(),
            stage3_in_buf->tensor_buffer().Get());

  EXPECT_THAT(chain.GetIntermediateBuffers().size(), Eq(2));

  // Bind entry input: [1.0, 2.0, 3.0, 4.0]
  auto env_or = litert::Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::make_shared<litert::Environment>(std::move(*env_or));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto input_buf,
      LitertBuffer::CreateManagedHost(env, {4}, litert::ElementType::Float32,
                                      4 * sizeof(float)));
  {
    auto span = input_buf->LockMutable();
    auto* span_data = reinterpret_cast<float*>(span.data());
    span_data[0] = 1.0f;
    span_data[1] = 2.0f;
    span_data[2] = 3.0f;
    span_data[3] = 4.0f;
  }
  ASSERT_TRUE(chain.SetInputBuffer("in", input_buf).ok());

  // Execute the entire chain
  ASSERT_TRUE(chain.Execute().ok());

  // Verify Terminal Output:
  // Input: [1.0, 2.0, 3.0, 4.0]
  // After Stage 1 (x 2.0):  [2.0, 4.0, 6.0, 8.0]
  // After Stage 2 (+ 5.0):  [7.0, 9.0, 11.0, 13.0]
  // After Stage 3 (x^2):    [49.0, 81.0, 121.0, 169.0]
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto terminal_buf,
                                  chain.GetOutputBuffer("out"));

  auto out_span = terminal_buf->Lock();
  const auto* out_data = reinterpret_cast<const float*>(out_span.data());
  ASSERT_THAT(out_span.size(), Eq(4 * sizeof(float)));
  EXPECT_FLOAT_EQ(out_data[0], 49.0f);
  EXPECT_FLOAT_EQ(out_data[1], 81.0f);
  EXPECT_FLOAT_EQ(out_data[2], 121.0f);
  EXPECT_FLOAT_EQ(out_data[3], 169.0f);
}

TEST(ModelChainTest, BranchingAndMergingPipeline) {
  // Topology:
  // Root -> BranchA -> Merge
  //      -> BranchB /
  HardwareBufferDescriptor desc;
  desc.shape = {2};
  desc.element_type = litert::ElementType::Float32;
  desc.size_bytes = 2 * sizeof(float);

  auto root = std::make_shared<FunctionalModelStage>(
      "Root",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{{"in", desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{{"out", desc}},
      [](const BufferMap& inputs, const BufferMap& outputs) {
        auto in = inputs.at("in")->Lock();
        const auto* in_data = reinterpret_cast<const float*>(in.data());
        auto out = outputs.at("out")->LockMutable();
        auto* out_data = reinterpret_cast<float*>(out.data());
        out_data[0] = in_data[0] + 10.0f;
        out_data[1] = in_data[1] + 10.0f;
        return absl::OkStatus();
      });

  auto branch_a = std::make_shared<FunctionalModelStage>(
      "BranchA",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{{"in", desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{{"out", desc}},
      [](const BufferMap& inputs, const BufferMap& outputs) {
        auto in = inputs.at("in")->Lock();
        const auto* in_data = reinterpret_cast<const float*>(in.data());
        auto out = outputs.at("out")->LockMutable();
        auto* out_data = reinterpret_cast<float*>(out.data());
        out_data[0] = in_data[0] * 2.0f;
        out_data[1] = in_data[1] * 2.0f;
        return absl::OkStatus();
      });

  auto branch_b = std::make_shared<FunctionalModelStage>(
      "BranchB",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{{"in", desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{{"out", desc}},
      [](const BufferMap& inputs, const BufferMap& outputs) {
        auto in = inputs.at("in")->Lock();
        const auto* in_data = reinterpret_cast<const float*>(in.data());
        auto out = outputs.at("out")->LockMutable();
        auto* out_data = reinterpret_cast<float*>(out.data());
        out_data[0] = in_data[0] + 100.0f;
        out_data[1] = in_data[1] + 100.0f;
        return absl::OkStatus();
      });

  auto merge = std::make_shared<FunctionalModelStage>(
      "Merge",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{{"a", desc},
                                                                 {"b", desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"final", desc}},
      [](const BufferMap& inputs, const BufferMap& outputs) {
        auto in_a = inputs.at("a")->Lock();
        const auto* a_data = reinterpret_cast<const float*>(in_a.data());
        auto in_b = inputs.at("b")->Lock();
        const auto* b_data = reinterpret_cast<const float*>(in_b.data());
        auto out = outputs.at("final")->LockMutable();
        auto* out_data = reinterpret_cast<float*>(out.data());
        out_data[0] = a_data[0] + b_data[0];
        out_data[1] = a_data[1] + b_data[1];
        return absl::OkStatus();
      });

  // Explicit connections
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto chain, ModelChain::Builder()
                      .AddStage(root)
                      .AddStage(branch_a)
                      .AddStage(branch_b)
                      .AddStage(merge)
                      .Connect("Root", "out", "BranchA", "in")
                      .Connect("Root", "out", "BranchB", "in")
                      .Connect("BranchA", "out", "Merge", "a")
                      .Connect("BranchB", "out", "Merge", "b")
                      .Build());

  // Input: [2.0, 3.0]
  auto env_or = litert::Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::make_shared<litert::Environment>(std::move(*env_or));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto in_buf,
      LitertBuffer::CreateManagedHost(env, {2}, litert::ElementType::Float32,
                                      2 * sizeof(float)));
  {
    auto span = in_buf->LockMutable();
    auto* span_data = reinterpret_cast<float*>(span.data());
    span_data[0] = 2.0f;
    span_data[1] = 3.0f;
  }
  ASSERT_TRUE(chain.SetInputBuffer("Root", "in", in_buf).ok());

  ASSERT_TRUE(chain.Execute().ok());

  // Expected:
  // Root out: [12.0, 13.0]
  // BranchA out (x 2): [24.0, 26.0]
  // BranchB out (+ 100): [112.0, 113.0]
  // Merge out (A + B): [136.0, 139.0]
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto final_buf,
                                  chain.GetOutputBuffer("Merge", "final"));

  auto final_span = final_buf->Lock();
  const auto* final_data = reinterpret_cast<const float*>(final_span.data());
  EXPECT_FLOAT_EQ(final_data[0], 136.0f);
  EXPECT_FLOAT_EQ(final_data[1], 139.0f);
}

TEST(ModelChainValidationTest, DetectsDuplicateStageNames) {
  HardwareBufferDescriptor desc;
  desc.shape = {1};
  desc.element_type = litert::ElementType::Float32;

  auto stage1 = std::make_shared<FunctionalModelStage>(
      "MyStage",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{{"in", desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{{"out", desc}},
      [](const BufferMap&, const BufferMap&) { return absl::OkStatus(); });

  auto stage2 = std::make_shared<FunctionalModelStage>(
      "MyStage",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{{"in", desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{{"out", desc}},
      [](const BufferMap&, const BufferMap&) { return absl::OkStatus(); });

  auto chain_or =
      ModelChain::Builder().AddStage(stage1).AddStage(stage2).Build();
  EXPECT_FALSE(chain_or.ok());
  EXPECT_THAT(chain_or.status().code(), Eq(absl::StatusCode::kAlreadyExists));
}

TEST(ModelChainValidationTest, DetectsCyclicDependency) {
  HardwareBufferDescriptor desc;
  desc.shape = {1};
  desc.element_type = litert::ElementType::Float32;

  auto stage_a = std::make_shared<FunctionalModelStage>(
      "A",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{{"in", desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{{"out", desc}},
      [](const BufferMap&, const BufferMap&) { return absl::OkStatus(); });

  auto stage_b = std::make_shared<FunctionalModelStage>(
      "B",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{{"in", desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{{"out", desc}},
      [](const BufferMap&, const BufferMap&) { return absl::OkStatus(); });

  auto chain_or = ModelChain::Builder()
                      .AddStage(stage_a)
                      .AddStage(stage_b)
                      .Connect("A", "out", "B", "in")
                      .Connect("B", "out", "A", "in")
                      .Build();
  EXPECT_FALSE(chain_or.ok());
  EXPECT_THAT(chain_or.status().code(), Eq(absl::StatusCode::kInvalidArgument));
}

TEST(ModelChainExecutionTest, FanOutWithDifferentAlignmentRequirements) {
  // Producer: 16-byte alignment
  HardwareBufferDescriptor prod_out_desc;
  prod_out_desc.shape = {4};
  prod_out_desc.element_type = litert::ElementType::Float32;
  prod_out_desc.alignment = 16;

  // Consumer A: 64-byte alignment
  HardwareBufferDescriptor cons_a_in_desc;
  cons_a_in_desc.shape = {4};
  cons_a_in_desc.element_type = litert::ElementType::Float32;
  cons_a_in_desc.alignment = 64;
  HardwareBufferDescriptor cons_a_out_desc = cons_a_in_desc;

  // Consumer B: 128-byte alignment
  HardwareBufferDescriptor cons_b_in_desc;
  cons_b_in_desc.shape = {4};
  cons_b_in_desc.element_type = litert::ElementType::Float32;
  cons_b_in_desc.alignment = 128;
  HardwareBufferDescriptor cons_b_out_desc = cons_b_in_desc;

  auto producer = std::make_shared<FunctionalModelStage>(
      "Producer",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"out", prod_out_desc}},
      [](const BufferMap&, const BufferMap& outputs) {
        auto out_span = outputs.at("out")->LockMutable();
        auto* data = reinterpret_cast<float*>(out_span.data());
        for (int i = 0; i < 4; ++i) {
          data[i] = static_cast<float>(i + 1);
        }
        return absl::OkStatus();
      });

  auto consumer_a = std::make_shared<FunctionalModelStage>(
      "ConsumerA",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"in", cons_a_in_desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"out", cons_a_out_desc}},
      [](const BufferMap& inputs, const BufferMap& outputs) {
        auto in_span = inputs.at("in")->Lock();
        auto out_span = outputs.at("out")->LockMutable();
        const auto* in_data = reinterpret_cast<const float*>(in_span.data());
        auto* out_data = reinterpret_cast<float*>(out_span.data());
        for (int i = 0; i < 4; ++i) {
          out_data[i] = in_data[i] * 2.0f;
        }
        return absl::OkStatus();
      });

  auto consumer_b = std::make_shared<FunctionalModelStage>(
      "ConsumerB",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"in", cons_b_in_desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"out", cons_b_out_desc}},
      [](const BufferMap& inputs, const BufferMap& outputs) {
        auto in_span = inputs.at("in")->Lock();
        auto out_span = outputs.at("out")->LockMutable();
        const auto* in_data = reinterpret_cast<const float*>(in_span.data());
        auto* out_data = reinterpret_cast<float*>(out_span.data());
        for (int i = 0; i < 4; ++i) {
          out_data[i] = in_data[i] * 10.0f;
        }
        return absl::OkStatus();
      });

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto chain, ModelChain::Builder()
                      .AddStage(producer)
                      .AddStage(consumer_a)
                      .AddStage(consumer_b)
                      .Connect("Producer", "out", "ConsumerA", "in")
                      .Connect("Producer", "out", "ConsumerB", "in")
                      .Build());

  // Verify Zero-Copy Shared Memory Guarantee across fan-out consumers
  auto prod_buf = producer->GetOutputBuffer("out");
  auto cons_a_buf = consumer_a->GetInputBuffer("in");
  auto cons_b_buf = consumer_b->GetInputBuffer("in");

  ASSERT_THAT(prod_buf, NotNull());
  ASSERT_THAT(cons_a_buf, NotNull());
  ASSERT_THAT(cons_b_buf, NotNull());
  EXPECT_EQ(prod_buf.get(), cons_a_buf.get());
  EXPECT_EQ(prod_buf.get(), cons_b_buf.get());

  // Verify that the negotiated buffer satisfies the strictest alignment
  // (128 bytes).
  {
    auto span = prod_buf->Lock();
    uintptr_t addr = reinterpret_cast<uintptr_t>(span.data());
    EXPECT_EQ(addr % 128, 0);
  }

  // Execute and verify results
  ASSERT_TRUE(chain.Execute().ok());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto out_a,
                                  chain.GetOutputBuffer("ConsumerA", "out"));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto out_b,
                                  chain.GetOutputBuffer("ConsumerB", "out"));

  auto span_a = out_a->Lock();
  const auto* data_a = reinterpret_cast<const float*>(span_a.data());
  EXPECT_FLOAT_EQ(data_a[0], 2.0f);
  EXPECT_FLOAT_EQ(data_a[3], 8.0f);

  auto span_b = out_b->Lock();
  const auto* data_b = reinterpret_cast<const float*>(span_b.data());
  EXPECT_FLOAT_EQ(data_b[0], 10.0f);
  EXPECT_FLOAT_EQ(data_b[3], 40.0f);
}

TEST(ModelChainValidationTest, DetectsAmbiguousInputAcrossMultipleStages) {
  HardwareBufferDescriptor desc;
  desc.shape = {1};
  desc.element_type = litert::ElementType::Float32;

  auto stage_a = std::make_shared<FunctionalModelStage>(
      "StageA",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"data", desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"out_a", desc}},
      [](const BufferMap&, const BufferMap&) { return absl::OkStatus(); });

  auto stage_b = std::make_shared<FunctionalModelStage>(
      "StageB",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"data", desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"out_b", desc}},
      [](const BufferMap&, const BufferMap&) { return absl::OkStatus(); });

  auto stage_merge = std::make_shared<FunctionalModelStage>(
      "Merge",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"in_a", desc}, {"in_b", desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"final", desc}},
      [](const BufferMap&, const BufferMap&) { return absl::OkStatus(); });

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto chain, ModelChain::Builder()
                      .AddStage(stage_a)
                      .AddStage(stage_b)
                      .AddStage(stage_merge)
                      .Connect("StageA", "out_a", "Merge", "in_a")
                      .Connect("StageB", "out_b", "Merge", "in_b")
                      .Build());

  auto env_or = litert::Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::make_shared<litert::Environment>(std::move(*env_or));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto buf,
      LitertBuffer::CreateManagedHost(env, {1}, litert::ElementType::Float32,
                                      sizeof(float)));

  auto status = chain.SetInputBuffer("data", buf);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.code(), Eq(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("StageA"));
  EXPECT_THAT(status.message(), HasSubstr("StageB"));
}

TEST(ModelChainValidationTest, DetectsAmbiguousOutputAcrossMultipleStages) {
  HardwareBufferDescriptor desc;
  desc.shape = {1};
  desc.element_type = litert::ElementType::Float32;

  auto stage_root = std::make_shared<FunctionalModelStage>(
      "Root",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"in", desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"out_a", desc}, {"out_b", desc}},
      [](const BufferMap&, const BufferMap&) { return absl::OkStatus(); });

  auto stage_a = std::make_shared<FunctionalModelStage>(
      "StageA",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"in_a", desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"result", desc}},
      [](const BufferMap&, const BufferMap&) { return absl::OkStatus(); });

  auto stage_b = std::make_shared<FunctionalModelStage>(
      "StageB",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"in_b", desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"result", desc}},
      [](const BufferMap&, const BufferMap&) { return absl::OkStatus(); });

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto chain, ModelChain::Builder()
                      .AddStage(stage_root)
                      .AddStage(stage_a)
                      .AddStage(stage_b)
                      .Connect("Root", "out_a", "StageA", "in_a")
                      .Connect("Root", "out_b", "StageB", "in_b")
                      .Build());

  auto out_or = chain.GetOutputBuffer("result");
  EXPECT_FALSE(out_or.ok());
  EXPECT_THAT(out_or.status().code(), Eq(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(out_or.status().message(), HasSubstr("StageA"));
  EXPECT_THAT(out_or.status().message(), HasSubstr("StageB"));
}

TEST(CompiledModelStageTest, IntrospectsAndExecutesModelInModelChain) {
  auto env_or = litert::Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::make_shared<litert::Environment>(std::move(*env_or));

  auto options_or = litert::Options::Create();
  ASSERT_TRUE(options_or.HasValue());
  options_or->SetHardwareAccelerators(litert::HwAccelerators::kCpu);

  std::string model_path = testing::GetTestFilePath(kModelFileName);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto stage,
      CompiledModelStage::Create(env, "CompiledSimpleModel", model_path,
                                 std::move(*options_or)));

  // Verify introspected signature names
  auto input_names = stage->InputNames();
  auto output_names = stage->OutputNames();
  ASSERT_THAT(input_names.size(), Eq(2));
  ASSERT_THAT(output_names.size(), Eq(1));

  // Verify introspected descriptors
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto in0_desc,
                                  stage->GetInputDescriptor(input_names[0]));
  EXPECT_THAT(in0_desc.element_type, Eq(litert::ElementType::Float32));
  EXPECT_THAT(in0_desc.shape, Eq(std::vector<int32_t>({2})));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto in1_desc,
                                  stage->GetInputDescriptor(input_names[1]));
  EXPECT_THAT(in1_desc.element_type, Eq(litert::ElementType::Float32));
  EXPECT_THAT(in1_desc.shape, Eq(std::vector<int32_t>({2})));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto out_desc,
                                  stage->GetOutputDescriptor(output_names[0]));
  EXPECT_THAT(out_desc.element_type, Eq(litert::ElementType::Float32));
  EXPECT_THAT(out_desc.shape, Eq(std::vector<int32_t>({2})));

  // Build a ModelChain with a downstream consumer stage that multiplies
  // output by 3.
  HardwareBufferDescriptor post_out_desc = out_desc;
  auto post_stage = std::make_shared<FunctionalModelStage>(
      "PostMultiply",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"in", out_desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{
          {"out", post_out_desc}},
      [](const BufferMap& inputs, const BufferMap& outputs) {
        auto in_span = inputs.at("in")->Lock();
        const auto* in_data = reinterpret_cast<const float*>(in_span.data());
        auto out_span = outputs.at("out")->LockMutable();
        auto* out_data = reinterpret_cast<float*>(out_span.data());
        for (int i = 0; i < 2; ++i) {
          out_data[i] = in_data[i] * 3.0f;
        }
        return absl::OkStatus();
      });

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto chain, ModelChain::Builder()
                      .AddStage(stage)
                      .AddStage(post_stage)
                      .Connect("CompiledSimpleModel", output_names[0],
                               "PostMultiply", "in")
                      .Build());

  // Allocate entry input buffers: [1.0, 2.0] and [10.0, 20.0]
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto in_buf0,
      LitertBuffer::CreateManagedHost(env, {2}, litert::ElementType::Float32,
                                      2 * sizeof(float)));
  {
    auto span = in_buf0->LockMutable();
    auto* d = reinterpret_cast<float*>(span.data());
    d[0] = 1.0f;
    d[1] = 2.0f;
  }

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto in_buf1,
      LitertBuffer::CreateManagedHost(env, {2}, litert::ElementType::Float32,
                                      2 * sizeof(float)));
  {
    auto span = in_buf1->LockMutable();
    auto* d = reinterpret_cast<float*>(span.data());
    d[0] = 10.0f;
    d[1] = 20.0f;
  }

  ASSERT_TRUE(
      chain.SetInputBuffer("CompiledSimpleModel", input_names[0], in_buf0)
          .ok());
  ASSERT_TRUE(
      chain.SetInputBuffer("CompiledSimpleModel", input_names[1], in_buf1)
          .ok());

  // Execute the chain
  ASSERT_TRUE(chain.Execute().ok());

  // Verify terminal output:
  // SimpleModel adds input0 [1, 2] and input1 [10, 20] -> [11, 22]
  // PostMultiply multiplies by 3 -> [33, 66]
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto out_buf, chain.GetOutputBuffer("out"));
  auto out_span = out_buf->Lock();
  const auto* out_data = reinterpret_cast<const float*>(out_span.data());
  EXPECT_FLOAT_EQ(out_data[0], 33.0f);
  EXPECT_FLOAT_EQ(out_data[1], 66.0f);
}

}  // namespace
}  // namespace litert::tensor

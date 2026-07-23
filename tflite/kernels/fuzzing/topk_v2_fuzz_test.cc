/* Copyright 2026 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <limits>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"
#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"
#include "tflite/core/kernels/builtin_op_kernels.h"
#include "tflite/kernels/fuzzing/fuzzing_util.h"
#include "tflite/kernels/fuzzing/one_op_fuzz_model.h"
#include "tflite/schema/schema_generated.h"

namespace tflite {
namespace ops {
namespace builtin {

TfLiteRegistration* Register_TOPK_V2();

}  // namespace builtin
}  // namespace ops

namespace {

using fuzzing::RunResult;

struct TopKV2Case {
  std::vector<int32_t> input_shape;
  std::vector<uint8_t> input_data;
  int32_t k;
  TensorType input_type;
  TensorType k_type;
  TensorType output_index_type;
  bool dynamic_k;
  bool invoke;
};

constexpr size_t kMaxInputElements = 512;
constexpr size_t kMaxLiveAllocationBytes = 64 * 1024 * 1024;

bool IsSupportedInputType(TensorType type) {
  return type == TensorType_FLOAT32 || type == TensorType_UINT8 ||
         type == TensorType_INT8 || type == TensorType_INT16 ||
         type == TensorType_INT32 || type == TensorType_INT64;
}

RunResult RunTopKV2Case(const TopKV2Case& test_case) {
  if (!IsSupportedInputType(test_case.input_type) ||
      (test_case.k_type != TensorType_INT32 &&
       test_case.k_type != TensorType_INT16 &&
       test_case.k_type != TensorType_INT64) ||
      (test_case.output_index_type != TensorType_INT32 &&
       test_case.output_index_type != TensorType_INT16)) {
    return RunResult::kRejected;
  }

  size_t input_elements = 0;
  if (!fuzzing::CheckedShapeElementCount(test_case.input_shape,
                                         &input_elements) ||
      input_elements > kMaxInputElements) {
    return RunResult::kRejected;
  }

  std::vector<uint8_t> input_bytes =
      fuzzing::MakeValues(test_case.input_type, input_elements, 31);
  fuzzing::OverlayBytes(test_case.input_data, &input_bytes);
  std::vector<uint8_t> k_bytes = fuzzing::MakeIntegerValues(
      test_case.k_type, std::vector<int64_t>{test_case.k});

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      fuzzing::CreateAlignedBuffer(&builder, std::vector<uint8_t>{})};
  if (!test_case.dynamic_k) {
    buffers.push_back(fuzzing::CreateAlignedBuffer(&builder, k_bytes));
  }

  const auto input_shape = builder.CreateVector(test_case.input_shape);
  const auto k_shape = builder.CreateVector(std::vector<int32_t>{1});
  const auto input_tensor =
      CreateTensor(builder, input_shape, test_case.input_type);
  const auto k_tensor = CreateTensor(builder, k_shape, test_case.k_type,
                                     test_case.dynamic_k ? 0 : 1);
  const auto output_values_tensor =
      CreateTensor(builder, input_shape, test_case.input_type);
  const auto output_indices_tensor =
      CreateTensor(builder, input_shape, test_case.output_index_type);

  fuzzing::OneOpModelSpec model_spec;
  model_spec.description = "topk_v2_fuzz";
  model_spec.builtin_operator = BuiltinOperator_TOPK_V2;
  model_spec.version = 1;
  model_spec.builtin_options_type = BuiltinOptions_TopKV2Options;
  model_spec.tensors = {input_tensor, k_tensor, output_values_tensor,
                        output_indices_tensor};
  model_spec.buffers = std::move(buffers);
  model_spec.model_inputs = test_case.dynamic_k ? std::vector<int32_t>{0, 1}
                                                : std::vector<int32_t>{0};
  model_spec.model_outputs = {2, 3};
  model_spec.op_inputs = {0, 1};
  model_spec.op_outputs = {2, 3};

  fuzzing::OneOpRunSpec run_spec;
  run_spec.registration = ops::builtin::Register_TOPK_V2();
  run_spec.min_version = 1;
  run_spec.max_version = 1;
  run_spec.max_live_allocation_bytes = kMaxLiveAllocationBytes;
  run_spec.invoke = test_case.invoke;
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/0, test_case.input_shape, std::move(input_bytes)});
  if (test_case.dynamic_k) {
    run_spec.runtime_tensors.push_back(
        {/*tensor_index=*/1, std::vector<int32_t>{1}, std::move(k_bytes)});
  }

  return fuzzing::BuildAndRunOneOpModel(&builder, model_spec, run_spec);
}

auto TopKV2CaseDomain() {
  return fuzztest::StructOf<TopKV2Case>(
      fuzztest::VectorOf(fuzztest::InRange<int32_t>(0, 8))
          .WithMinSize(0)
          .WithMaxSize(6),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      fuzztest::OneOf(
          fuzztest::InRange<int32_t>(-4, 16),
          fuzztest::Just<int32_t>(std::numeric_limits<int32_t>::max()),
          fuzztest::Just<int32_t>(std::numeric_limits<int32_t>::min())),
      fuzztest::ElementOf<TensorType>(
          {TensorType_FLOAT32, TensorType_UINT8, TensorType_INT8,
           TensorType_INT16, TensorType_INT32, TensorType_INT64}),
      fuzztest::ElementOf<TensorType>(
          {TensorType_INT32, TensorType_INT16, TensorType_INT64}),
      fuzztest::ElementOf<TensorType>({TensorType_INT32, TensorType_INT16}),
      fuzztest::Arbitrary<bool>(), fuzztest::Arbitrary<bool>());
}

void TopKV2NeverCrashes(const TopKV2Case& test_case) {
  EXPECT_NE(RunTopKV2Case(test_case), RunResult::kHarnessFailure);
}

TEST(TopKV2FuzzTest, Rank64Smoke) {
  TopKV2Case test_case;
  test_case.input_shape.assign(64, 1);
  test_case.input_shape.back() = 3;
  test_case.k = 2;
  test_case.input_type = TensorType_FLOAT32;
  test_case.k_type = TensorType_INT32;
  test_case.output_index_type = TensorType_INT32;
  test_case.dynamic_k = true;
  test_case.invoke = true;
  EXPECT_EQ(RunTopKV2Case(test_case), RunResult::kSuccess);
}

FUZZ_TEST(TopKV2FuzzTest, TopKV2NeverCrashes)
    .WithDomains(TopKV2CaseDomain());

}  // namespace
}  // namespace tflite

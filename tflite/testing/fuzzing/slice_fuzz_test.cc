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

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"
#include "fuzztest/fuzztest.h"
#include "tflite/core/kernels/builtin_op_kernels.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/testing/fuzzing/fuzzing_util.h"
#include "tflite/testing/fuzzing/one_op_fuzz_model.h"

#if defined(TFLITE_SLICE_FUZZ_ENABLE_XNNPACK)
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#endif

namespace tflite {
namespace ops {
namespace builtin {

TfLiteRegistration* Register_SLICE();
TfLiteRegistration* Register_SLICE_REF();

}  // namespace builtin
}  // namespace ops

namespace {

using fuzzing::RunResult;

enum class IndexSpecKind { kConstant, kDynamic };
enum class KernelVariant { kReference, kGenericOptimized };
enum class ExecutionMode { kBuiltin, kXnnpack };

struct SliceCase {
  std::vector<int32_t> input_shape;
  std::vector<int64_t> begin;
  std::vector<int64_t> size;
  std::vector<uint8_t> input_data;
  TensorType input_type;
  TensorType index_type;
  IndexSpecKind index_spec_kind;
  bool invoke;
};

constexpr size_t kMaxInputElements = 512;
constexpr size_t kMaxLiveAllocationBytes = 64 * 1024 * 1024;

bool IsSupportedInputType(TensorType type) {
  switch (type) {
    case TensorType_FLOAT32:
    case TensorType_FLOAT16:
    case TensorType_BFLOAT16:
    case TensorType_UINT8:
    case TensorType_UINT32:
    case TensorType_INT8:
    case TensorType_INT4:
    case TensorType_INT16:
    case TensorType_INT32:
    case TensorType_INT64:
    case TensorType_BOOL:
      return true;
    default:
      return false;
  }
}

TfLiteRegistration* SliceRegistration(KernelVariant kernel_variant) {
  return kernel_variant == KernelVariant::kReference
             ? ops::builtin::Register_SLICE_REF()
             : ops::builtin::Register_SLICE();
}

TfLiteStatus ApplyXnnpackDelegate(Interpreter* interpreter) {
#if defined(TFLITE_SLICE_FUZZ_ENABLE_XNNPACK)
  TfLiteXNNPackDelegateOptions options = TfLiteXNNPackDelegateOptionsDefault();
  options.num_threads = 1;
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate(
      TfLiteXNNPackDelegateCreate(&options), TfLiteXNNPackDelegateDelete);
  if (delegate == nullptr) return kTfLiteError;
  return interpreter->ModifyGraphWithDelegate(std::move(delegate));
#else
  (void)interpreter;
  return kTfLiteError;
#endif
}

bool HasDelegateNode(const Interpreter& interpreter) {
  for (const int node_index : interpreter.execution_plan()) {
    const auto* node_and_registration =
        interpreter.node_and_registration(node_index);
    if (node_and_registration != nullptr &&
        node_and_registration->second.builtin_code ==
            BuiltinOperator_DELEGATE) {
      return true;
    }
  }
  return false;
}

RunResult RunSliceCase(const SliceCase& test_case, KernelVariant kernel_variant,
                       ExecutionMode execution_mode) {
  if (!IsSupportedInputType(test_case.input_type) ||
      (test_case.index_type != TensorType_INT32 &&
       test_case.index_type != TensorType_INT64) ||
      test_case.begin.size() != test_case.input_shape.size() ||
      test_case.size.size() != test_case.input_shape.size()) {
    return RunResult::kRejected;
  }

  size_t input_elements = 0;
  if (!fuzzing::CheckedShapeElementCount(test_case.input_shape,
                                         &input_elements) ||
      input_elements > kMaxInputElements) {
    return RunResult::kRejected;
  }

  std::vector<uint8_t> input_bytes =
      fuzzing::MakeValues(test_case.input_type, input_elements, 53);
  fuzzing::OverlayBytes(test_case.input_data, &input_bytes);
  fuzzing::ApplyCentralTensorInputInvariants(test_case.input_type,
                                             &input_bytes);
  const std::vector<uint8_t> begin_bytes =
      fuzzing::MakeIntegerValues(test_case.index_type, test_case.begin);
  const std::vector<uint8_t> size_bytes =
      fuzzing::MakeIntegerValues(test_case.index_type, test_case.size);

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      fuzzing::CreateAlignedBuffer(&builder, std::vector<uint8_t>{})};
  const bool dynamic_indices =
      test_case.index_spec_kind == IndexSpecKind::kDynamic;
  uint32_t begin_buffer = 0;
  uint32_t size_buffer = 0;
  if (!dynamic_indices) {
    begin_buffer = buffers.size();
    buffers.push_back(fuzzing::CreateAlignedBuffer(&builder, begin_bytes));
    size_buffer = buffers.size();
    buffers.push_back(fuzzing::CreateAlignedBuffer(&builder, size_bytes));
  }

  const auto input_shape = builder.CreateVector(test_case.input_shape);
  const auto index_shape = builder.CreateVector(
      std::vector<int32_t>{static_cast<int32_t>(test_case.input_shape.size())});
  const auto empty_output_shape = builder.CreateVector(std::vector<int32_t>{});
  flatbuffers::Offset<QuantizationParameters> quantization = 0;
  if (test_case.input_type == TensorType_INT8 ||
      test_case.input_type == TensorType_UINT8) {
    quantization = CreateQuantizationParameters(
        builder, 0, 0, builder.CreateVector<float>({0.25f}),
        builder.CreateVector<int64_t>({0}));
  }

  const auto input_tensor =
      CreateTensor(builder, input_shape, test_case.input_type, /*buffer=*/0,
                   /*name=*/0, quantization);
  const auto begin_tensor =
      CreateTensor(builder, index_shape, test_case.index_type, begin_buffer);
  const auto size_tensor =
      CreateTensor(builder, index_shape, test_case.index_type, size_buffer);
  const auto output_tensor =
      CreateTensor(builder, empty_output_shape, test_case.input_type,
                   /*buffer=*/0, /*name=*/0, quantization);

  fuzzing::OneOpModelSpec model_spec;
  model_spec.description = "slice_fuzz";
  model_spec.builtin_operator = BuiltinOperator_SLICE;
  model_spec.version = 1;
  model_spec.builtin_options_type = BuiltinOptions_SliceOptions;
  model_spec.builtin_options = CreateSliceOptions(builder).Union();
  model_spec.tensors = {input_tensor, begin_tensor, size_tensor, output_tensor};
  model_spec.buffers = std::move(buffers);
  model_spec.model_inputs =
      dynamic_indices ? std::vector<int32_t>{0, 1, 2} : std::vector<int32_t>{0};
  model_spec.model_outputs = {3};
  model_spec.op_inputs = {0, 1, 2};
  model_spec.op_outputs = {3};

  fuzzing::OneOpRunSpec run_spec;
  run_spec.registration = SliceRegistration(kernel_variant);
  run_spec.min_version = 1;
  run_spec.max_version = 1;
  run_spec.max_live_allocation_bytes = kMaxLiveAllocationBytes;
  run_spec.invoke = test_case.invoke;
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/0, test_case.input_shape, std::move(input_bytes)});
  if (dynamic_indices) {
    const std::vector<int32_t> runtime_index_shape = {
        static_cast<int32_t>(test_case.input_shape.size())};
    run_spec.runtime_tensors.push_back(
        {/*tensor_index=*/1, runtime_index_shape, begin_bytes});
    run_spec.runtime_tensors.push_back(
        {/*tensor_index=*/2, runtime_index_shape, size_bytes});
  }
  if (execution_mode == ExecutionMode::kXnnpack) {
    run_spec.pre_allocate = [](Interpreter* interpreter) {
      return ApplyXnnpackDelegate(interpreter) == kTfLiteOk &&
                     HasDelegateNode(*interpreter)
                 ? RunResult::kSuccess
                 : RunResult::kRejected;
    };
  }

  return fuzzing::BuildAndRunOneOpModel(&builder, model_spec, run_spec);
}

auto SliceInputShapeDomain(size_t max_rank) {
  return fuzztest::VectorOf(fuzztest::InRange<int32_t>(1, 2))
      .WithMinSize(1)
      .WithMaxSize(max_rank);
}

std::vector<int64_t> MakeBegins(const std::vector<int32_t>& input_shape,
                                const std::vector<uint8_t>& begin_seeds) {
  std::vector<int64_t> begin(input_shape.size());
  for (size_t i = 0; i < input_shape.size(); ++i) {
    const uint8_t seed =
        begin_seeds.empty() ? 0 : begin_seeds[i % begin_seeds.size()];
    begin[i] = seed % input_shape[i];
  }
  return begin;
}

std::vector<int64_t> MakeValidSizes(const std::vector<int32_t>& input_shape,
                                    const std::vector<int64_t>& begin,
                                    const std::vector<uint8_t>& size_seeds,
                                    bool allow_special_sizes) {
  std::vector<int64_t> size(input_shape.size());
  for (size_t i = 0; i < input_shape.size(); ++i) {
    const uint8_t seed =
        size_seeds.empty() ? 0 : size_seeds[i % size_seeds.size()];
    const int64_t remaining = input_shape[i] - begin[i];
    if (allow_special_sizes && seed % 3 == 0) {
      size[i] = -1;
    } else if (allow_special_sizes && seed % 3 == 1) {
      size[i] = 0;
    } else {
      size[i] = 1 + seed % remaining;
    }
  }
  return size;
}

auto ValidSliceCaseDomain() {
  return fuzztest::Map(
      [](std::vector<int32_t> input_shape, std::vector<uint8_t> begin_seeds,
         std::vector<uint8_t> size_seeds, std::vector<uint8_t> input_data,
         TensorType input_type, TensorType index_type,
         IndexSpecKind index_spec_kind) {
        std::vector<int64_t> begin = MakeBegins(input_shape, begin_seeds);
        std::vector<int64_t> size = MakeValidSizes(
            input_shape, begin, size_seeds, /*allow_special_sizes=*/true);
        return SliceCase{
            std::move(input_shape), std::move(begin), std::move(size),
            std::move(input_data),  input_type,       index_type,
            index_spec_kind,        /*invoke=*/true};
      },
      SliceInputShapeDomain(/*max_rank=*/8),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(8),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(8),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      fuzztest::ElementOf<TensorType>(
          {TensorType_FLOAT32, TensorType_FLOAT16, TensorType_BFLOAT16,
           TensorType_UINT8, TensorType_UINT32, TensorType_INT8,
           TensorType_INT4, TensorType_INT16, TensorType_INT32,
           TensorType_INT64, TensorType_BOOL}),
      fuzztest::ElementOf<TensorType>({TensorType_INT32, TensorType_INT64}),
      fuzztest::ElementOf<IndexSpecKind>(
          {IndexSpecKind::kConstant, IndexSpecKind::kDynamic}));
}

auto MalformedSliceSizeCaseDomain() {
  return fuzztest::Map(
      [](std::vector<int32_t> input_shape, std::vector<uint8_t> begin_seeds,
         uint8_t bad_axis, TensorType input_type, TensorType index_type,
         IndexSpecKind index_spec_kind) {
        std::vector<int64_t> begin = MakeBegins(input_shape, begin_seeds);
        std::vector<int64_t> size(input_shape.size(), 1);
        size[bad_axis % size.size()] = -2;
        return SliceCase{
            std::move(input_shape), std::move(begin), std::move(size),
            /*input_data=*/{},      input_type,       index_type,
            index_spec_kind,        /*invoke=*/true};
      },
      SliceInputShapeDomain(/*max_rank=*/8),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(8),
      fuzztest::Arbitrary<uint8_t>(),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_UINT8,
                                       TensorType_INT32, TensorType_BOOL}),
      fuzztest::ElementOf<TensorType>({TensorType_INT32, TensorType_INT64}),
      fuzztest::ElementOf<IndexSpecKind>(
          {IndexSpecKind::kConstant, IndexSpecKind::kDynamic}));
}

auto MalformedSliceBeginCaseDomain() {
  return fuzztest::Map(
      [](std::vector<int32_t> input_shape, std::vector<uint8_t> begin_seeds,
         uint8_t bad_axis, bool size_to_end, TensorType input_type,
         TensorType index_type, IndexSpecKind index_spec_kind) {
        std::vector<int64_t> begin = MakeBegins(input_shape, begin_seeds);
        std::vector<int64_t> size(input_shape.size(), 1);
        const size_t axis = bad_axis % begin.size();
        begin[axis] = -1;
        if (size_to_end) size[axis] = -1;
        return SliceCase{
            std::move(input_shape), std::move(begin), std::move(size),
            /*input_data=*/{},      input_type,       index_type,
            index_spec_kind,        /*invoke=*/true};
      },
      SliceInputShapeDomain(/*max_rank=*/8),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(8),
      fuzztest::Arbitrary<uint8_t>(), fuzztest::Arbitrary<bool>(),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_UINT8,
                                       TensorType_INT32, TensorType_BOOL}),
      fuzztest::ElementOf<TensorType>({TensorType_INT32, TensorType_INT64}),
      fuzztest::ElementOf<IndexSpecKind>(
          {IndexSpecKind::kConstant, IndexSpecKind::kDynamic}));
}

#if defined(TFLITE_SLICE_FUZZ_ENABLE_XNNPACK)
auto XnnpackSliceCaseDomain() {
  return fuzztest::Map(
      [](std::vector<int32_t> input_shape, std::vector<uint8_t> begin_seeds,
         std::vector<uint8_t> size_seeds, std::vector<uint8_t> input_data,
         TensorType input_type, TensorType index_type) {
        std::vector<int64_t> begin = MakeBegins(input_shape, begin_seeds);
        std::vector<int64_t> size = MakeValidSizes(
            input_shape, begin, size_seeds, /*allow_special_sizes=*/false);
        return SliceCase{
            std::move(input_shape),   std::move(begin), std::move(size),
            std::move(input_data),    input_type,       index_type,
            IndexSpecKind::kConstant,
            /*invoke=*/true};
      },
      SliceInputShapeDomain(/*max_rank=*/6),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(6),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(6),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      fuzztest::ElementOf<TensorType>(
          {TensorType_FLOAT32, TensorType_UINT8, TensorType_INT8}),
      fuzztest::ElementOf<TensorType>({TensorType_INT32, TensorType_INT64}));
}
#endif

void SliceExecutesValidCases(const SliceCase& test_case) {
  SCOPED_TRACE(::testing::Message()
               << "shape=" << ::testing::PrintToString(test_case.input_shape)
               << ", begin=" << ::testing::PrintToString(test_case.begin)
               << ", size=" << ::testing::PrintToString(test_case.size)
               << ", input_type=" << static_cast<int>(test_case.input_type)
               << ", index_type=" << static_cast<int>(test_case.index_type)
               << ", dynamic="
               << (test_case.index_spec_kind == IndexSpecKind::kDynamic));
  ASSERT_EQ(RunSliceCase(test_case, KernelVariant::kGenericOptimized,
                         ExecutionMode::kBuiltin),
            RunResult::kSuccess);
}

void SliceReferenceExecutesValidCases(const SliceCase& test_case) {
  ASSERT_EQ(RunSliceCase(test_case, KernelVariant::kReference,
                         ExecutionMode::kBuiltin),
            RunResult::kSuccess);
}

void SliceRejectsInvalidSizes(const SliceCase& test_case) {
  ASSERT_EQ(RunSliceCase(test_case, KernelVariant::kGenericOptimized,
                         ExecutionMode::kBuiltin),
            RunResult::kRejected);
}

void SliceReferenceRejectsInvalidSizes(const SliceCase& test_case) {
  ASSERT_EQ(RunSliceCase(test_case, KernelVariant::kReference,
                         ExecutionMode::kBuiltin),
            RunResult::kRejected);
}

void SliceRejectsInvalidBegins(const SliceCase& test_case) {
  ASSERT_EQ(RunSliceCase(test_case, KernelVariant::kGenericOptimized,
                         ExecutionMode::kBuiltin),
            RunResult::kRejected);
}

void SliceReferenceRejectsInvalidBegins(const SliceCase& test_case) {
  ASSERT_EQ(RunSliceCase(test_case, KernelVariant::kReference,
                         ExecutionMode::kBuiltin),
            RunResult::kRejected);
}

TEST(SliceFuzzTest, RankSixReferencePathSmoke) {
  const SliceCase test_case{/*input_shape=*/{1, 1, 1, 1, 2, 2},
                            /*begin=*/{0, 0, 0, 0, 1, 0},
                            /*size=*/{1, 1, 1, 1, 1, 2},
                            /*input_data=*/{},
                            TensorType_INT32,
                            TensorType_INT64,
                            IndexSpecKind::kDynamic,
                            /*invoke=*/true};
  EXPECT_EQ(RunSliceCase(test_case, KernelVariant::kGenericOptimized,
                         ExecutionMode::kBuiltin),
            RunResult::kSuccess);
  EXPECT_EQ(RunSliceCase(test_case, KernelVariant::kReference,
                         ExecutionMode::kBuiltin),
            RunResult::kSuccess);
}

FUZZ_TEST(SliceFuzzTest, SliceExecutesValidCases)
    .WithDomains(ValidSliceCaseDomain());
FUZZ_TEST(SliceFuzzTest, SliceReferenceExecutesValidCases)
    .WithDomains(ValidSliceCaseDomain());
FUZZ_TEST(SliceFuzzTest, SliceRejectsInvalidSizes)
    .WithDomains(MalformedSliceSizeCaseDomain());
FUZZ_TEST(SliceFuzzTest, SliceReferenceRejectsInvalidSizes)
    .WithDomains(MalformedSliceSizeCaseDomain());
FUZZ_TEST(SliceFuzzTest, SliceRejectsInvalidBegins)
    .WithDomains(MalformedSliceBeginCaseDomain());
FUZZ_TEST(SliceFuzzTest, SliceReferenceRejectsInvalidBegins)
    .WithDomains(MalformedSliceBeginCaseDomain());

#if defined(TFLITE_SLICE_FUZZ_ENABLE_XNNPACK)
TEST(SliceFuzzTest, SliceXnnpackRankSixSmokeDelegates) {
  EXPECT_EQ(
      RunSliceCase({/*input_shape=*/{1, 1, 1, 1, 2, 2},
                    /*begin=*/{0, 0, 0, 0, 1, 0},
                    /*size=*/{1, 1, 1, 1, 1, 2},
                    /*input_data=*/{}, TensorType_FLOAT32, TensorType_INT64,
                    IndexSpecKind::kConstant,
                    /*invoke=*/true},
                   KernelVariant::kGenericOptimized, ExecutionMode::kXnnpack),
      RunResult::kSuccess);
}

void SliceXnnpackExecutesValidCases(const SliceCase& test_case) {
  SCOPED_TRACE(::testing::Message()
               << "shape=" << ::testing::PrintToString(test_case.input_shape)
               << ", begin=" << ::testing::PrintToString(test_case.begin)
               << ", size=" << ::testing::PrintToString(test_case.size)
               << ", input_type=" << static_cast<int>(test_case.input_type)
               << ", index_type=" << static_cast<int>(test_case.index_type));
  ASSERT_EQ(RunSliceCase(test_case, KernelVariant::kGenericOptimized,
                         ExecutionMode::kXnnpack),
            RunResult::kSuccess);
}

FUZZ_TEST(SliceFuzzTest, SliceXnnpackExecutesValidCases)
    .WithDomains(XnnpackSliceCaseDomain());
#endif

}  // namespace
}  // namespace tflite

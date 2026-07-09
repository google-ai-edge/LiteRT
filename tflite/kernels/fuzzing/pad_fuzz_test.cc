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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"
#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"
#include "tflite/core/interpreter.h"
#include "tflite/core/kernels/builtin_op_kernels.h"
#include "tflite/kernels/fuzzing/fuzzing_util.h"
#include "tflite/kernels/fuzzing/one_op_fuzz_model.h"
#include "tflite/schema/schema_generated.h"
#if defined(TFLITE_PAD_FUZZ_ENABLE_XNNPACK)
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#endif

namespace tflite {
namespace ops {
namespace builtin {

TfLiteRegistration* Register_PAD_REF();
TfLiteRegistration* Register_PAD_GENERIC_OPT();
TfLiteRegistration* Register_PADV2_REF();
TfLiteRegistration* Register_PADV2_GENERIC_OPT();

}  // namespace builtin
}  // namespace ops

namespace {

using fuzzing::RunResult;

enum class PadKind { kPad, kPadV2 };
enum class KernelVariant { kReference, kGenericOptimized };
enum class ExecutionMode { kBuiltin, kXnnpack };
enum class PaddingShapeKind { kValid, kWrongRows, kWrongColumns, kRankOne };

struct PadCase {
  std::vector<int32_t> input_shape;
  std::vector<int64_t> padding_values;
  // Byte overlays add arbitrary bit patterns to otherwise structured values.
  // They are bounded independently of tensor size to keep each iteration fast.
  std::vector<uint8_t> input_data;
  std::vector<uint8_t> padding_data;
  std::vector<uint8_t> value_data;
  PadKind pad_kind;
  PaddingShapeKind padding_shape_kind;
  TensorType input_type;
  TensorType padding_type;
  bool dynamic_paddings;
  bool quantized_int8;
  bool non_scalar_pad_value;
  bool invoke;
};

struct ProductStressSpec {
  int32_t rank;
  std::vector<int32_t> input_dims;
  std::vector<int64_t> target_output_dims;
  std::vector<uint8_t> padding_split;
  PadKind pad_kind;
  TensorType input_type;
  TensorType padding_type;
  bool dynamic_paddings;
  bool quantized_int8;
};

// Keep individual allocations small so malformed cases exercise validation
// paths without turning a fuzz iteration into an accidental stress test.
constexpr size_t kMaxInputElements = 256;
constexpr size_t kMaxFuzzerLiveAllocationBytes = 64 * 1024 * 1024;
constexpr int32_t kMaxProductStressRank = 8;

size_t ElementCount(const std::vector<int32_t>& shape) {
  size_t result = 1;
  for (int32_t dimension : shape) {
    if (dimension < 0 || dimension > kMaxInputElements) {
      return kMaxInputElements + 1;
    }
    if (dimension != 0 &&
        result > kMaxInputElements / static_cast<size_t>(dimension)) {
      return kMaxInputElements + 1;
    }
    result *= static_cast<size_t>(dimension);
  }
  return result;
}

bool PadOutputIsWithinFuzzerBudget(const PadCase& test_case) {
  if (test_case.padding_shape_kind != PaddingShapeKind::kValid ||
      test_case.input_shape.empty() ||
      test_case.padding_values.size() != test_case.input_shape.size() * 2) {
    return false;
  }
  const size_t element_size = fuzzing::TypeSize(test_case.input_type);
  if (element_size == 0) {
    return false;
  }

  size_t output_elements = 1;
  for (size_t i = 0; i < test_case.input_shape.size(); ++i) {
    const int64_t input_dim = test_case.input_shape[i];
    const int64_t left_padding = test_case.padding_values[i * 2];
    const int64_t right_padding = test_case.padding_values[i * 2 + 1];
    if (input_dim < 0 || left_padding < 0 || right_padding < 0) {
      return false;
    }
    if (left_padding > std::numeric_limits<int64_t>::max() - input_dim ||
        right_padding >
            std::numeric_limits<int64_t>::max() - input_dim - left_padding) {
      return false;
    }
    const int64_t output_dim = input_dim + left_padding + right_padding;
    if (output_dim < 0 || output_dim > std::numeric_limits<int32_t>::max()) {
      return false;
    }
    if (!fuzzing::CheckedMultiply(output_elements,
                                  static_cast<size_t>(output_dim),
                                  &output_elements)) {
      return false;
    }
  }

  size_t output_bytes = 0;
  if (!fuzzing::CheckedMultiply(output_elements, element_size,
                                &output_bytes)) {
    return false;
  }
  return output_bytes <= kMaxFuzzerLiveAllocationBytes;
}

std::vector<int32_t> PaddingShape(const PadCase& test_case, size_t rank,
                                  size_t* rows, size_t* columns) {
  *rows = rank;
  *columns = 2;
  // Most cases use [rank, 2]. The other shapes model malformed model files
  // and are expected to be rejected by Prepare or Invoke.
  switch (test_case.padding_shape_kind) {
    case PaddingShapeKind::kWrongRows:
      *rows = rank + 1;
      break;
    case PaddingShapeKind::kWrongColumns:
      *columns = 1 + (test_case.input_shape.size() % 3);
      if (*columns == 2) *columns = 3;
      break;
    case PaddingShapeKind::kRankOne:
      return {static_cast<int32_t>(rank * 2)};
    case PaddingShapeKind::kValid:
      break;
  }
  return {static_cast<int32_t>(*rows), static_cast<int32_t>(*columns)};
}

PadCase MakeProductStressPadCase(const ProductStressSpec& spec) {
  PadCase test_case;
  const int32_t rank =
      std::clamp(spec.rank, int32_t{1}, kMaxProductStressRank);
  test_case.input_shape.reserve(rank);
  test_case.padding_values.reserve(static_cast<size_t>(rank) * 2);

  for (int32_t i = 0; i < rank; ++i) {
    const int32_t input_dim =
        std::max<int32_t>(1, spec.input_dims[static_cast<size_t>(i)]);
    int64_t target_dim = spec.target_output_dims[static_cast<size_t>(i)];
    if (target_dim < input_dim) {
      target_dim = input_dim;
    }
    const int64_t total_padding = target_dim - input_dim;
    const uint8_t split = spec.padding_split[static_cast<size_t>(i)];
    const int64_t left_padding =
        total_padding == 0 ? 0 : (total_padding * split) / UINT8_MAX;
    const int64_t right_padding = total_padding - left_padding;

    test_case.input_shape.push_back(input_dim);
    test_case.padding_values.push_back(left_padding);
    test_case.padding_values.push_back(right_padding);
  }

  test_case.input_data = {};
  test_case.padding_data = {};
  test_case.value_data = {};
  test_case.pad_kind = spec.pad_kind;
  test_case.padding_shape_kind = PaddingShapeKind::kValid;
  test_case.input_type = spec.input_type;
  test_case.padding_type = spec.padding_type;
  test_case.dynamic_paddings = spec.dynamic_paddings;
  test_case.quantized_int8 = spec.quantized_int8;
  test_case.non_scalar_pad_value = false;
  test_case.invoke = true;
  return test_case;
}

TfLiteRegistration* GetPadRegistration(PadKind pad_kind,
                                        KernelVariant kernel_variant) {
  if (pad_kind == PadKind::kPad) {
    return kernel_variant == KernelVariant::kReference
               ? ops::builtin::Register_PAD_REF()
               : ops::builtin::Register_PAD_GENERIC_OPT();
  }
  return kernel_variant == KernelVariant::kReference
             ? ops::builtin::Register_PADV2_REF()
             : ops::builtin::Register_PADV2_GENERIC_OPT();
}

TfLiteStatus ApplyXnnpackDelegate(Interpreter* interpreter) {
#if defined(TFLITE_PAD_FUZZ_ENABLE_XNNPACK)
  TfLiteXNNPackDelegateOptions options =
      TfLiteXNNPackDelegateOptionsDefault();
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
  for (int node_index : interpreter.execution_plan()) {
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

RunResult RunPadCase(const PadCase& test_case, KernelVariant kernel_variant,
                     ExecutionMode execution_mode) {
  if (ElementCount(test_case.input_shape) > kMaxInputElements ||
      fuzzing::TypeSize(test_case.input_type) == 0 ||
      fuzzing::TypeSize(test_case.padding_type) == 0) {
    return RunResult::kRejected;
  }

  const size_t input_elements = ElementCount(test_case.input_shape);
  const size_t rank = test_case.input_shape.size();
  size_t padding_rows = 0;
  size_t padding_columns = 0;
  const std::vector<int32_t> padding_shape =
      PaddingShape(test_case, rank, &padding_rows, &padding_columns);
  const std::vector<int64_t> padding_values = fuzzing::MaterializeValues(
      test_case.padding_values, padding_rows * padding_columns);
  std::vector<uint8_t> input_bytes =
      fuzzing::MakeValues(test_case.input_type, input_elements, 1);
  fuzzing::OverlayBytes(test_case.input_data, &input_bytes);
  fuzzing::ApplyCentralTensorInputInvariants(test_case.input_type,
                                             &input_bytes);
  std::vector<uint8_t> padding_bytes =
      fuzzing::MakeIntegerValues(test_case.padding_type, padding_values);
  fuzzing::OverlayBytes(test_case.padding_data, &padding_bytes);
  fuzzing::ApplyCentralTensorInputInvariants(test_case.padding_type,
                                             &padding_bytes);

  // Build a minimal one-node FlatBuffer model instead of calling the kernel's
  // internal Eval function. This preserves the model-parser, allocator, and
  // Prepare paths that an untrusted model would exercise.
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      fuzzing::CreateAlignedBuffer(&builder, std::vector<uint8_t>{})};

  // A dynamic paddings tensor has no constant buffer and is populated after
  // AllocateTensors, matching normal runtime use.
  if (!test_case.dynamic_paddings) {
    buffers.push_back(fuzzing::CreateAlignedBuffer(&builder, padding_bytes));
  }

  uint32_t padding_buffer = test_case.dynamic_paddings ? 0 : 1;
  const auto input_shape = builder.CreateVector(test_case.input_shape);
  const auto padding_shape_offset = builder.CreateVector(padding_shape);
  const auto output_shape = builder.CreateVector(test_case.input_shape);

  const bool quantized =
      test_case.quantized_int8 && test_case.input_type == TensorType_INT8;
  flatbuffers::Offset<QuantizationParameters> quantization = 0;
  if (quantized) {
    quantization = CreateQuantizationParameters(
        builder, 0, 0, builder.CreateVector<float>({0.25f}),
        builder.CreateVector<int64_t>({0}));
  }

  const auto input_tensor = CreateTensor(
      builder, input_shape, test_case.input_type, 0, 0, quantization);
  const auto padding_tensor = CreateTensor(
      builder, padding_shape_offset, test_case.padding_type, padding_buffer);

  std::vector<flatbuffers::Offset<Tensor>> tensors = {input_tensor,
                                                      padding_tensor};
  std::vector<int32_t> inputs = {0, 1};
  if (test_case.pad_kind == PadKind::kPadV2) {
    const std::vector<int32_t> value_shape = {
        test_case.non_scalar_pad_value ? 2 : 1};
    std::vector<uint8_t> value_bytes = fuzzing::MakeValues(
        test_case.input_type, value_shape[0],
        test_case.padding_values.empty() ? 0 : test_case.padding_values[0]);
    fuzzing::OverlayBytes(test_case.value_data, &value_bytes);
    fuzzing::ApplyCentralTensorInputInvariants(test_case.input_type,
                                               &value_bytes);
    const uint32_t value_buffer = buffers.size();
    buffers.push_back(fuzzing::CreateAlignedBuffer(&builder, value_bytes));
    const auto value_tensor =
        CreateTensor(builder, builder.CreateVector(value_shape),
                     test_case.input_type, value_buffer, 0, quantization);
    tensors.push_back(value_tensor);
    inputs.push_back(2);
  }

  const int output_index = tensors.size();
  tensors.push_back(CreateTensor(builder, output_shape, test_case.input_type, 0,
                                 0, quantization));
  const std::vector<int32_t> outputs = {output_index};
  const BuiltinOperator op_code = test_case.pad_kind == PadKind::kPad
                                      ? BuiltinOperator_PAD
                                      : BuiltinOperator_PADV2;
  const auto options = test_case.pad_kind == PadKind::kPad
                           ? CreatePadOptions(builder).Union()
                           : CreatePadV2Options(builder).Union();
  const auto builtin_options = test_case.pad_kind == PadKind::kPad
                                   ? BuiltinOptions_PadOptions
                                   : BuiltinOptions_PadV2Options;
  fuzzing::OneOpModelSpec model_spec;
  model_spec.description = "pad_fuzz";
  model_spec.builtin_operator = op_code;
  model_spec.builtin_options_type = builtin_options;
  model_spec.builtin_options = options;
  model_spec.tensors = std::move(tensors);
  model_spec.buffers = std::move(buffers);
  model_spec.model_inputs = inputs;
  model_spec.model_outputs = outputs;
  model_spec.op_inputs = std::move(inputs);
  model_spec.op_outputs = outputs;

  fuzzing::OneOpRunSpec run_spec;
  run_spec.registration =
      GetPadRegistration(test_case.pad_kind, kernel_variant);
  run_spec.max_live_allocation_bytes = kMaxFuzzerLiveAllocationBytes;
  run_spec.invoke = test_case.invoke;
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/0, test_case.input_shape, std::move(input_bytes)});
  if (test_case.dynamic_paddings) {
    run_spec.runtime_tensors.push_back(
        {/*tensor_index=*/1, padding_shape, std::move(padding_bytes)});
  }
  if (execution_mode == ExecutionMode::kXnnpack) {
    run_spec.post_allocate = [](Interpreter* interpreter) {
      return ApplyXnnpackDelegate(interpreter) == kTfLiteOk &&
                     HasDelegateNode(*interpreter)
                 ? RunResult::kSuccess
                 : RunResult::kRejected;
    };
  }
  return fuzzing::BuildAndRunOneOpModel(&builder, model_spec, run_spec);
}

auto PaddingValueDomain() {
  // Bias generation toward boundary values: these are the values most likely
  // to expose integer narrowing, addition, and output-size overflow bugs.
  return fuzztest::OneOf(
      fuzztest::InRange<int64_t>(-4, 4), fuzztest::Just<int64_t>(INT32_MAX),
      fuzztest::Just<int64_t>(INT32_MIN),
      fuzztest::Just<int64_t>(static_cast<int64_t>(INT32_MAX) + 1),
      fuzztest::Just<int64_t>(static_cast<int64_t>(INT32_MIN) - 1));
}

auto PadCaseDomain() {
  // The product domain combines ordinary models with deliberately malformed
  // tensor metadata, unsupported types, and unusual invocation states.
  return fuzztest::StructOf<PadCase>(
      fuzztest::VectorOf(fuzztest::InRange<int32_t>(0, 2))
          .WithMinSize(0)
          .WithMaxSize(8),
      fuzztest::VectorOf(PaddingValueDomain()).WithMinSize(1).WithMaxSize(8),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>())
          .WithMaxSize(64),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>())
          .WithMaxSize(64),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>())
          .WithMaxSize(64),
      fuzztest::ElementOf<PadKind>({PadKind::kPad, PadKind::kPadV2}),
      fuzztest::ElementOf<PaddingShapeKind>(
          {PaddingShapeKind::kValid, PaddingShapeKind::kWrongRows,
           PaddingShapeKind::kWrongColumns, PaddingShapeKind::kRankOne}),
      fuzztest::ElementOf<TensorType>(
          {TensorType_FLOAT32, TensorType_INT8, TensorType_INT32}),
      fuzztest::ElementOf<TensorType>({TensorType_INT8, TensorType_INT16,
                                       TensorType_INT32, TensorType_INT64,
                                       TensorType_BOOL, TensorType_FLOAT32}),
      fuzztest::Arbitrary<bool>(), fuzztest::Arbitrary<bool>(),
      fuzztest::Arbitrary<bool>(), fuzztest::Arbitrary<bool>());
}

auto ProductStressTargetDimensionDomain() {
  // These values stress both near-boundary products and wrap-to-small cases:
  // 46341^2 exceeds INT32_MAX, 65536^2 is 2^32, and repeated 65536 factors
  // overflow 64-bit size_t products.
  return fuzztest::OneOf(
      fuzztest::InRange<int64_t>(1, 4), fuzztest::Just<int64_t>(32767),
      fuzztest::Just<int64_t>(32768), fuzztest::Just<int64_t>(46339),
      fuzztest::Just<int64_t>(46340), fuzztest::Just<int64_t>(46341),
      fuzztest::Just<int64_t>(65535), fuzztest::Just<int64_t>(65536),
      fuzztest::Just<int64_t>(static_cast<int64_t>(INT32_MAX)),
      fuzztest::Just<int64_t>(static_cast<int64_t>(INT32_MAX) + 1));
}

auto ProductStressSpecDomain() {
  return fuzztest::StructOf<ProductStressSpec>(
      fuzztest::InRange<int32_t>(1, kMaxProductStressRank),
      fuzztest::VectorOf(fuzztest::InRange<int32_t>(1, 2))
          .WithMinSize(kMaxProductStressRank)
          .WithMaxSize(kMaxProductStressRank),
      fuzztest::VectorOf(ProductStressTargetDimensionDomain())
          .WithMinSize(kMaxProductStressRank)
          .WithMaxSize(kMaxProductStressRank),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>())
          .WithMinSize(kMaxProductStressRank)
          .WithMaxSize(kMaxProductStressRank),
      fuzztest::ElementOf<PadKind>({PadKind::kPad, PadKind::kPadV2}),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_INT8,
                                       TensorType_INT32, TensorType_BOOL}),
      fuzztest::ElementOf<TensorType>({TensorType_INT32, TensorType_INT64}),
      fuzztest::Arbitrary<bool>(), fuzztest::Arbitrary<bool>());
}

auto XnnpackProductStressSpecDomain() {
  // Keep valid in-budget cases inside XNNPACK's delegated PAD surface while
  // still generating output products that should be rejected before delegation.
  return fuzztest::StructOf<ProductStressSpec>(
      fuzztest::InRange<int32_t>(1, 6),
      fuzztest::VectorOf(fuzztest::InRange<int32_t>(1, 2))
          .WithMinSize(kMaxProductStressRank)
          .WithMaxSize(kMaxProductStressRank),
      fuzztest::VectorOf(ProductStressTargetDimensionDomain())
          .WithMinSize(kMaxProductStressRank)
          .WithMaxSize(kMaxProductStressRank),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>())
          .WithMinSize(kMaxProductStressRank)
          .WithMaxSize(kMaxProductStressRank),
      fuzztest::Just(PadKind::kPad),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_INT8}),
      fuzztest::Just(TensorType_INT32), fuzztest::Just(false),
      fuzztest::Just(true));
}

auto XnnpackPadCaseDomain() {
  // The XNNPACK PAD delegate accepts a narrower surface than the builtin
  // kernel: static INT32 paddings, rank >= 1, and PAD rather than PADV2. This
  // domain keeps enough structure to reach delegate code frequently while the
  // byte overlay still mutates tensor contents.
  return fuzztest::StructOf<PadCase>(
      fuzztest::VectorOf(fuzztest::InRange<int32_t>(1, 4))
          .WithMinSize(1)
          .WithMaxSize(6),
      fuzztest::VectorOf(fuzztest::InRange<int64_t>(0, 3))
          .WithMinSize(1)
          .WithMaxSize(12),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      fuzztest::Just(std::vector<uint8_t>{}),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      fuzztest::Just(PadKind::kPad), fuzztest::Just(PaddingShapeKind::kValid),
      fuzztest::ElementOf<TensorType>({TensorType_FLOAT32, TensorType_INT8}),
      fuzztest::Just(TensorType_INT32), fuzztest::Just(false),
      fuzztest::Just(true), fuzztest::Just(false),
      fuzztest::Arbitrary<bool>());
}

void PadNeverCrashes(const PadCase& test_case) {
  // Rejection is expected for invalid models. A harness failure means the
  // fuzzer itself constructed an invalid test setup, so treat that as a bug.
  EXPECT_NE(RunPadCase(test_case, KernelVariant::kGenericOptimized,
                       ExecutionMode::kBuiltin),
            RunResult::kHarnessFailure);
}

void PadReferenceNeverCrashes(const PadCase& test_case) {
  EXPECT_NE(RunPadCase(test_case, KernelVariant::kReference,
                       ExecutionMode::kBuiltin),
            RunResult::kHarnessFailure);
}

void PadRejectsProductStressOverflow(const ProductStressSpec& spec) {
  const PadCase test_case = MakeProductStressPadCase(spec);
  const bool must_reject = !PadOutputIsWithinFuzzerBudget(test_case);
  const RunResult result =
      RunPadCase(test_case, KernelVariant::kGenericOptimized,
                 ExecutionMode::kBuiltin);
  EXPECT_NE(result, RunResult::kHarnessFailure);
  if (must_reject) {
    EXPECT_EQ(result, RunResult::kRejected);
  }
}

void PadReferenceRejectsProductStressOverflow(const ProductStressSpec& spec) {
  const PadCase test_case = MakeProductStressPadCase(spec);
  const bool must_reject = !PadOutputIsWithinFuzzerBudget(test_case);
  const RunResult result =
      RunPadCase(test_case, KernelVariant::kReference, ExecutionMode::kBuiltin);
  EXPECT_NE(result, RunResult::kHarnessFailure);
  if (must_reject) {
    EXPECT_EQ(result, RunResult::kRejected);
  }
}

FUZZ_TEST(PadFuzzTest, PadNeverCrashes).WithDomains(PadCaseDomain());
FUZZ_TEST(PadFuzzTest, PadReferenceNeverCrashes).WithDomains(PadCaseDomain());
FUZZ_TEST(PadFuzzTest, PadRejectsProductStressOverflow)
    .WithDomains(ProductStressSpecDomain());
FUZZ_TEST(PadFuzzTest, PadReferenceRejectsProductStressOverflow)
    .WithDomains(ProductStressSpecDomain());

#if defined(TFLITE_PAD_FUZZ_ENABLE_XNNPACK)
TEST(PadFuzzTest, PadXnnpackSmokeDelegates) {
  PadCase test_case;
  test_case.input_shape = {2, 2};
  test_case.padding_values = {1, 0, 0, 1};
  test_case.input_data = {};
  test_case.padding_data = {};
  test_case.value_data = {};
  test_case.pad_kind = PadKind::kPad;
  test_case.padding_shape_kind = PaddingShapeKind::kValid;
  test_case.input_type = TensorType_FLOAT32;
  test_case.padding_type = TensorType_INT32;
  test_case.dynamic_paddings = false;
  test_case.quantized_int8 = false;
  test_case.non_scalar_pad_value = false;
  test_case.invoke = true;

  EXPECT_EQ(RunPadCase(test_case, KernelVariant::kGenericOptimized,
                       ExecutionMode::kXnnpack),
            RunResult::kSuccess);
}

void PadXnnpackNeverCrashes(const PadCase& test_case) {
  EXPECT_NE(RunPadCase(test_case, KernelVariant::kGenericOptimized,
                       ExecutionMode::kXnnpack),
            RunResult::kHarnessFailure);
}

void PadXnnpackRejectsProductStressOverflow(const ProductStressSpec& spec) {
  const PadCase test_case = MakeProductStressPadCase(spec);
  const bool must_reject = !PadOutputIsWithinFuzzerBudget(test_case);
  const RunResult result =
      RunPadCase(test_case, KernelVariant::kGenericOptimized,
                 ExecutionMode::kXnnpack);
  EXPECT_NE(result, RunResult::kHarnessFailure);
  if (must_reject) {
    EXPECT_EQ(result, RunResult::kRejected);
  }
}

FUZZ_TEST(PadFuzzTest, PadXnnpackNeverCrashes)
    .WithDomains(XnnpackPadCaseDomain());
FUZZ_TEST(PadFuzzTest, PadXnnpackRejectsProductStressOverflow)
    .WithDomains(XnnpackProductStressSpecDomain());
#endif

}  // namespace
}  // namespace tflite

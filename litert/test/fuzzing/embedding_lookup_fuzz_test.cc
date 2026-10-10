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
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"
#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "litert/test/fuzzing/fuzzing_util.h"
#include "litert/test/fuzzing/one_op_fuzz_model.h"
#include "tflite/c/common.h"
#include "tflite/kernels/builtin_op_kernels.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/util.h"

namespace tflite {
namespace {

using fuzzing::RunResult;

constexpr size_t kMaxElements = 1024;
constexpr size_t kMaxLiveAllocationBytes = 64 * 1024 * 1024;

struct SimpleLookupCase {
  std::vector<int32_t> lookup_shape;
  std::vector<int32_t> value_shape;
  std::vector<int32_t> lookup_indices;
  std::vector<uint8_t> value_bytes;
  TensorType lookup_type = TensorType_INT32;
  TensorType value_type = TensorType_FLOAT32;
  bool invoke = true;
};

struct HybridLookupCase {
  std::vector<int32_t> lookup_shape;
  std::vector<int32_t> value_shape;
  std::vector<int32_t> lookup_indices;
  std::vector<uint8_t> value_bytes;
  std::vector<float> scales;
  std::vector<int64_t> zero_points;
  int32_t quantized_dimension = 0;
  TensorType value_type = TensorType_INT8;
  bool invoke = true;
};

struct BlockwiseLookupCase {
  std::vector<int32_t> lookup_shape;
  std::vector<int32_t> value_shape;
  std::vector<int32_t> scale_shape;
  std::vector<int32_t> lookup_indices;
  std::vector<uint8_t> value_bytes;
  std::vector<uint16_t> scale_fp16_values;
  int32_t blocksize = 32;
  int32_t scale_tensor_index = 3;
  TensorType value_type = TensorType_INT4;
  TensorType scale_type = TensorType_FLOAT16;
  bool invoke = true;
};

std::vector<int32_t> ComputeEmbeddingOutputShape(
    const std::vector<int32_t>& lookup_shape,
    const std::vector<int32_t>& value_shape) {
  if (lookup_shape.empty() || value_shape.size() < 2) {
    return {1};
  }
  std::vector<int32_t> output_shape;
  output_shape.reserve(value_shape.size());
  output_shape.push_back(std::max<int32_t>(0, lookup_shape[0]));
  for (size_t i = 1; i < value_shape.size(); ++i) {
    output_shape.push_back(std::max<int32_t>(0, value_shape[i]));
  }
  return output_shape;
}

std::vector<uint8_t> MakeInt32TensorBytes(const std::vector<int32_t>& indices,
                                          size_t count) {
  std::vector<int64_t> materialized(count, 0);
  if (!indices.empty()) {
    for (size_t i = 0; i < count; ++i) {
      materialized[i] = indices[i % indices.size()];
    }
  }
  return fuzzing::MakeIntegerValues(TensorType_INT32, materialized);
}

RunResult RunSimpleLookupCase(const SimpleLookupCase& test_case) {
  size_t lookup_elements = 0;
  size_t value_elements = 0;
  if (!fuzzing::CheckedShapeElementCount(test_case.lookup_shape,
                                         &lookup_elements) ||
      !fuzzing::CheckedShapeElementCount(test_case.value_shape,
                                         &value_elements) ||
      lookup_elements > kMaxElements || value_elements > kMaxElements) {
    return RunResult::kRejected;
  }

  std::vector<uint8_t> lookup_bytes;
  if (test_case.lookup_type == TensorType_INT32) {
    lookup_bytes =
        MakeInt32TensorBytes(test_case.lookup_indices, lookup_elements);
  } else {
    lookup_bytes = fuzzing::MakeValues(test_case.lookup_type, lookup_elements,
                                       /*seed=*/11);
  }
  std::vector<uint8_t> value_bytes =
      fuzzing::MakeValues(test_case.value_type, value_elements, /*seed=*/19);
  fuzzing::OverlayBytes(test_case.value_bytes, &value_bytes);

  flatbuffers::FlatBufferBuilder builder;
  const std::vector<int32_t> out_shape_vec = ComputeEmbeddingOutputShape(
      test_case.lookup_shape, test_case.value_shape);
  const auto lookup_tensor =
      CreateTensor(builder, builder.CreateVector(test_case.lookup_shape),
                   test_case.lookup_type);
  const auto value_tensor =
      CreateTensor(builder, builder.CreateVector(test_case.value_shape),
                   test_case.value_type);
  const auto output_tensor = CreateTensor(
      builder, builder.CreateVector(out_shape_vec), test_case.value_type);

  fuzzing::OneOpModelSpec model_spec;
  model_spec.description = "embedding_lookup_simple_fuzz";
  model_spec.builtin_operator = BuiltinOperator_EMBEDDING_LOOKUP;
  model_spec.version = 4;
  model_spec.tensors = {lookup_tensor, value_tensor, output_tensor};
  model_spec.buffers = {
      fuzzing::CreateAlignedBuffer(&builder, std::vector<uint8_t>{})};
  model_spec.model_inputs = {0, 1};
  model_spec.model_outputs = {2};
  model_spec.op_inputs = {0, 1};
  model_spec.op_outputs = {2};

  fuzzing::OneOpRunSpec run_spec;
  run_spec.registration = ops::builtin::Register_EMBEDDING_LOOKUP();
  run_spec.min_version = 1;
  run_spec.max_version = 4;
  run_spec.max_live_allocation_bytes = kMaxLiveAllocationBytes;
  run_spec.invoke = test_case.invoke;
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/0, test_case.lookup_shape, std::move(lookup_bytes)});
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/1, test_case.value_shape, std::move(value_bytes)});

  return fuzzing::BuildAndRunOneOpModel(&builder, model_spec, run_spec);
}

RunResult RunHybridLookupCase(const HybridLookupCase& test_case) {
  size_t lookup_elements = 0;
  size_t value_elements = 0;
  if (!fuzzing::CheckedShapeElementCount(test_case.lookup_shape,
                                         &lookup_elements) ||
      !fuzzing::CheckedShapeElementCount(test_case.value_shape,
                                         &value_elements) ||
      lookup_elements > kMaxElements || value_elements > kMaxElements) {
    return RunResult::kRejected;
  }

  std::vector<uint8_t> lookup_bytes =
      MakeInt32TensorBytes(test_case.lookup_indices, lookup_elements);
  std::vector<uint8_t> value_bytes =
      fuzzing::MakeValues(test_case.value_type, value_elements, /*seed=*/29);
  fuzzing::OverlayBytes(test_case.value_bytes, &value_bytes);

  flatbuffers::FlatBufferBuilder builder;
  const auto quant_params = CreateQuantizationParameters(
      builder, /*min=*/0, /*max=*/0,
      test_case.scales.empty() ? 0 : builder.CreateVector(test_case.scales),
      test_case.zero_points.empty()
          ? 0
          : builder.CreateVector(test_case.zero_points),
      QuantizationDetails_NONE, /*details=*/0, test_case.quantized_dimension);

  const std::vector<int32_t> out_shape_vec = ComputeEmbeddingOutputShape(
      test_case.lookup_shape, test_case.value_shape);
  const auto lookup_tensor = CreateTensor(
      builder, builder.CreateVector(test_case.lookup_shape), TensorType_INT32);
  const auto value_tensor = CreateTensor(
      builder, builder.CreateVector(test_case.value_shape),
      test_case.value_type, /*buffer=*/0, /*name=*/0, quant_params);
  const auto output_tensor = CreateTensor(
      builder, builder.CreateVector(out_shape_vec), TensorType_FLOAT32);

  fuzzing::OneOpModelSpec model_spec;
  model_spec.description = "embedding_lookup_hybrid_fuzz";
  model_spec.builtin_operator = BuiltinOperator_EMBEDDING_LOOKUP;
  model_spec.version = 4;
  model_spec.tensors = {lookup_tensor, value_tensor, output_tensor};
  model_spec.buffers = {
      fuzzing::CreateAlignedBuffer(&builder, std::vector<uint8_t>{})};
  model_spec.model_inputs = {0, 1};
  model_spec.model_outputs = {2};
  model_spec.op_inputs = {0, 1};
  model_spec.op_outputs = {2};

  fuzzing::OneOpRunSpec run_spec;
  run_spec.registration = ops::builtin::Register_EMBEDDING_LOOKUP();
  run_spec.min_version = 1;
  run_spec.max_version = 4;
  run_spec.max_live_allocation_bytes = kMaxLiveAllocationBytes;
  run_spec.invoke = test_case.invoke;
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/0, test_case.lookup_shape, std::move(lookup_bytes)});
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/1, test_case.value_shape, std::move(value_bytes)});

  return fuzzing::BuildAndRunOneOpModel(&builder, model_spec, run_spec);
}

RunResult RunBlockwiseLookupCase(const BlockwiseLookupCase& test_case) {
  size_t lookup_elements = 0;
  size_t value_elements = 0;
  size_t scale_elements = 0;
  if (!fuzzing::CheckedShapeElementCount(test_case.lookup_shape,
                                         &lookup_elements) ||
      !fuzzing::CheckedShapeElementCount(test_case.value_shape,
                                         &value_elements) ||
      !fuzzing::CheckedShapeElementCount(test_case.scale_shape,
                                         &scale_elements) ||
      lookup_elements > kMaxElements || value_elements > kMaxElements ||
      scale_elements > kMaxElements) {
    return RunResult::kRejected;
  }

  std::vector<uint8_t> lookup_bytes =
      MakeInt32TensorBytes(test_case.lookup_indices, lookup_elements);
  std::vector<uint8_t> value_bytes =
      fuzzing::MakeValues(test_case.value_type, value_elements, /*seed=*/37);
  fuzzing::OverlayBytes(test_case.value_bytes, &value_bytes);

  std::vector<uint8_t> scale_bytes =
      fuzzing::MakeValues(test_case.scale_type, scale_elements, /*seed=*/41);
  if (!test_case.scale_fp16_values.empty() &&
      test_case.scale_type == TensorType_FLOAT16) {
    scale_bytes.resize(scale_elements * sizeof(uint16_t));
    for (size_t i = 0; i < scale_elements; ++i) {
      const uint16_t val =
          test_case.scale_fp16_values[i % test_case.scale_fp16_values.size()];
      scale_bytes[2 * i] = static_cast<uint8_t>(val & 0xFF);
      scale_bytes[2 * i + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
    }
  }

  flatbuffers::FlatBufferBuilder builder;
  auto blockwise_quant =
      CreateBlockwiseQuantization(builder, test_case.scale_tensor_index,
                                  /*zero_points=*/0, test_case.blocksize);
  const auto quant_params = CreateQuantizationParameters(
      builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
      QuantizationDetails_BlockwiseQuantization, blockwise_quant.Union(),
      /*quantized_dimension=*/1);

  const std::vector<int32_t> out_shape_vec = ComputeEmbeddingOutputShape(
      test_case.lookup_shape, test_case.value_shape);
  const auto lookup_tensor = CreateTensor(
      builder, builder.CreateVector(test_case.lookup_shape), TensorType_INT32);
  const auto value_tensor = CreateTensor(
      builder, builder.CreateVector(test_case.value_shape),
      test_case.value_type, /*buffer=*/0, /*name=*/0, quant_params);
  const auto output_tensor = CreateTensor(
      builder, builder.CreateVector(out_shape_vec), TensorType_FLOAT32);
  const auto scale_tensor =
      CreateTensor(builder, builder.CreateVector(test_case.scale_shape),
                   test_case.scale_type, /*buffer=*/1);

  fuzzing::OneOpModelSpec model_spec;
  model_spec.description = "embedding_lookup_blockwise_fuzz";
  model_spec.builtin_operator = BuiltinOperator_EMBEDDING_LOOKUP;
  model_spec.version = 4;
  model_spec.tensors = {lookup_tensor, value_tensor, output_tensor,
                        scale_tensor};
  model_spec.buffers = {
      fuzzing::CreateAlignedBuffer(&builder, std::vector<uint8_t>{}),
      fuzzing::CreateAlignedBuffer(&builder, scale_bytes)};
  model_spec.model_inputs = {0, 1};
  model_spec.model_outputs = {2};
  model_spec.op_inputs = {0, 1};
  model_spec.op_outputs = {2};

  fuzzing::OneOpRunSpec run_spec;
  run_spec.registration = ops::builtin::Register_EMBEDDING_LOOKUP();
  run_spec.min_version = 1;
  run_spec.max_version = 4;
  run_spec.max_live_allocation_bytes = kMaxLiveAllocationBytes;
  run_spec.invoke = test_case.invoke;
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/0, test_case.lookup_shape, std::move(lookup_bytes)});
  run_spec.runtime_tensors.push_back(
      {/*tensor_index=*/1, test_case.value_shape, std::move(value_bytes)});

  return fuzzing::BuildAndRunOneOpModel(&builder, model_spec, run_spec);
}

auto ValidSimpleLookupCaseDomain() {
  return fuzztest::Map(
      [](int32_t lookup_count, std::vector<int32_t> tail_dims, int32_t rows,
         std::vector<uint32_t> raw_indices, std::vector<uint8_t> value_bytes,
         TensorType value_type) {
        SimpleLookupCase test_case;
        test_case.lookup_shape = {lookup_count};
        test_case.value_shape.push_back(rows);
        test_case.value_shape.insert(test_case.value_shape.end(),
                                     tail_dims.begin(), tail_dims.end());
        test_case.lookup_indices.resize(lookup_count, 0);
        for (int32_t i = 0; i < lookup_count; ++i) {
          const uint32_t seed =
              raw_indices.empty() ? 0 : raw_indices[i % raw_indices.size()];
          test_case.lookup_indices[i] = static_cast<int32_t>(seed % rows);
        }
        test_case.value_bytes = std::move(value_bytes);
        test_case.value_type = value_type;
        test_case.invoke = true;
        return test_case;
      },
      fuzztest::InRange<int32_t>(0, 8),
      fuzztest::VectorOf(fuzztest::InRange<int32_t>(0, 4))
          .WithMinSize(1)
          .WithMaxSize(3),
      fuzztest::InRange<int32_t>(1, 8),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint32_t>()).WithMaxSize(8),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      fuzztest::ElementOf({TensorType_FLOAT32, TensorType_INT32,
                           TensorType_INT8, TensorType_UINT8}));
}

auto ValidHybridLookupCaseDomain() {
  return fuzztest::Map(
      [](int32_t lookup_count, std::vector<int32_t> tail_dims, int32_t rows,
         std::vector<uint32_t> raw_indices, std::vector<uint8_t> value_bytes,
         TensorType value_type, bool per_axis) {
        HybridLookupCase test_case;
        test_case.lookup_shape = {lookup_count};
        test_case.value_shape.push_back(rows);
        test_case.value_shape.insert(test_case.value_shape.end(),
                                     tail_dims.begin(), tail_dims.end());
        test_case.lookup_indices.resize(lookup_count, 0);
        for (int32_t i = 0; i < lookup_count; ++i) {
          const uint32_t seed =
              raw_indices.empty() ? 0 : raw_indices[i % raw_indices.size()];
          test_case.lookup_indices[i] = static_cast<int32_t>(seed % rows);
        }
        const int32_t num_scales = per_axis ? rows : 1;
        test_case.scales.assign(num_scales, 0.25f);
        test_case.zero_points.assign(num_scales, 0);
        test_case.quantized_dimension = 0;
        test_case.value_bytes = std::move(value_bytes);
        test_case.value_type = value_type;
        test_case.invoke = true;
        return test_case;
      },
      fuzztest::InRange<int32_t>(0, 8),
      fuzztest::VectorOf(fuzztest::InRange<int32_t>(1, 4))
          .WithMinSize(1)
          .WithMaxSize(3),
      fuzztest::InRange<int32_t>(1, 8),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint32_t>()).WithMaxSize(8),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      fuzztest::ElementOf({TensorType_INT8, TensorType_INT4, TensorType_INT2}),
      fuzztest::Arbitrary<bool>());
}

auto ValidBlockwiseLookupCaseDomain() {
  return fuzztest::Map(
      [](int32_t lookup_count, int32_t rows, int32_t blocksize,
         int32_t num_blocks, std::vector<uint32_t> raw_indices,
         std::vector<uint8_t> value_bytes, TensorType value_type) {
        BlockwiseLookupCase test_case;
        test_case.lookup_shape = {lookup_count};
        test_case.value_shape = {rows, blocksize * num_blocks};
        test_case.scale_shape = {rows, num_blocks};
        test_case.blocksize = blocksize;
        test_case.scale_tensor_index = 3;
        test_case.lookup_indices.resize(lookup_count, 0);
        for (int32_t i = 0; i < lookup_count; ++i) {
          const uint32_t seed =
              raw_indices.empty() ? 0 : raw_indices[i % raw_indices.size()];
          test_case.lookup_indices[i] = static_cast<int32_t>(seed % rows);
        }
        test_case.scale_fp16_values.assign(
            static_cast<size_t>(rows * num_blocks), 0x3C00);
        test_case.value_bytes = std::move(value_bytes);
        test_case.value_type = value_type;
        test_case.scale_type = TensorType_FLOAT16;
        test_case.invoke = true;
        return test_case;
      },
      fuzztest::InRange<int32_t>(0, 8), fuzztest::InRange<int32_t>(1, 4),
      fuzztest::ElementOf({32, 64}), fuzztest::InRange<int32_t>(1, 3),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint32_t>()).WithMaxSize(8),
      fuzztest::VectorOf(fuzztest::Arbitrary<uint8_t>()).WithMaxSize(64),
      fuzztest::ElementOf({TensorType_INT4, TensorType_INT2}));
}

void EmbeddingLookupSimpleExecutesValidCases(
    const SimpleLookupCase& test_case) {
  EXPECT_EQ(RunSimpleLookupCase(test_case), RunResult::kSuccess);
}
FUZZ_TEST(EmbeddingLookupFuzz, EmbeddingLookupSimpleExecutesValidCases)
    .WithDomains(ValidSimpleLookupCaseDomain());

void EmbeddingLookupHybridExecutesValidCases(
    const HybridLookupCase& test_case) {
  EXPECT_EQ(RunHybridLookupCase(test_case), RunResult::kSuccess);
}
FUZZ_TEST(EmbeddingLookupFuzz, EmbeddingLookupHybridExecutesValidCases)
    .WithDomains(ValidHybridLookupCaseDomain());

void EmbeddingLookupBlockwiseExecutesValidCases(
    const BlockwiseLookupCase& test_case) {
  EXPECT_EQ(RunBlockwiseLookupCase(test_case), RunResult::kSuccess);
}
FUZZ_TEST(EmbeddingLookupFuzz, EmbeddingLookupBlockwiseExecutesValidCases)
    .WithDomains(ValidBlockwiseLookupCaseDomain());

void EmbeddingLookupRejectsOutOfBoundsIndex(int32_t rows, int32_t bad_offset,
                                            bool negative_index, int32_t mode) {
  const int32_t oob_index =
      negative_index ? -bad_offset : rows + (bad_offset - 1);
  if (mode == 0) {
    SimpleLookupCase test_case;
    test_case.lookup_shape = {2};
    test_case.value_shape = {rows, 4};
    test_case.lookup_indices = {0, oob_index};
    test_case.value_type = TensorType_FLOAT32;
    test_case.invoke = true;
    EXPECT_EQ(RunSimpleLookupCase(test_case), RunResult::kRejected);
  } else if (mode == 1) {
    HybridLookupCase test_case;
    test_case.lookup_shape = {2};
    test_case.value_shape = {rows, 4};
    test_case.lookup_indices = {0, oob_index};
    test_case.scales = {0.5f};
    test_case.zero_points = {0};
    test_case.value_type = TensorType_INT4;
    test_case.invoke = true;
    EXPECT_EQ(RunHybridLookupCase(test_case), RunResult::kRejected);
  } else {
    BlockwiseLookupCase test_case;
    test_case.lookup_shape = {2};
    test_case.value_shape = {rows, 32};
    test_case.scale_shape = {rows, 1};
    test_case.blocksize = 32;
    test_case.scale_tensor_index = 3;
    test_case.lookup_indices = {0, oob_index};
    test_case.scale_fp16_values.assign(rows, 0x3C00);
    test_case.value_type = TensorType_INT4;
    test_case.invoke = true;
    EXPECT_EQ(RunBlockwiseLookupCase(test_case), RunResult::kRejected);
  }
}
FUZZ_TEST(EmbeddingLookupFuzz, EmbeddingLookupRejectsOutOfBoundsIndex)
    .WithDomains(fuzztest::InRange<int32_t>(1, 8),
                 fuzztest::InRange<int32_t>(1, 100),
                 fuzztest::Arbitrary<bool>(), fuzztest::InRange<int32_t>(0, 2));

void EmbeddingLookupRejectsInvalidRanksAndQuantization(int32_t defect_kind) {
  switch (defect_kind) {
    case 0: {
      // lookup rank 2 instead of rank 1.
      SimpleLookupCase test_case;
      test_case.lookup_shape = {2, 1};
      test_case.value_shape = {4, 4};
      test_case.lookup_indices = {0, 1};
      EXPECT_EQ(RunSimpleLookupCase(test_case), RunResult::kRejected);
      break;
    }
    case 1: {
      // value rank 1 instead of rank >= 2.
      SimpleLookupCase test_case;
      test_case.lookup_shape = {2};
      test_case.value_shape = {4};
      test_case.lookup_indices = {0, 1};
      EXPECT_EQ(RunSimpleLookupCase(test_case), RunResult::kRejected);
      break;
    }
    case 2: {
      // lookup type FLOAT32 instead of INT32.
      SimpleLookupCase test_case;
      test_case.lookup_shape = {2};
      test_case.value_shape = {4, 4};
      test_case.lookup_type = TensorType_FLOAT32;
      EXPECT_EQ(RunSimpleLookupCase(test_case), RunResult::kRejected);
      break;
    }
    case 3: {
      // blocksize == 0.
      BlockwiseLookupCase test_case;
      test_case.lookup_shape = {1};
      test_case.value_shape = {2, 32};
      test_case.scale_shape = {2, 1};
      test_case.blocksize = 0;
      test_case.lookup_indices = {0};
      EXPECT_EQ(RunBlockwiseLookupCase(test_case), RunResult::kRejected);
      break;
    }
    case 4: {
      // blocksize not divisible by values_per_byte (3 for INT4).
      BlockwiseLookupCase test_case;
      test_case.lookup_shape = {1};
      test_case.value_shape = {2, 30};
      test_case.scale_shape = {2, 10};
      test_case.blocksize = 3;
      test_case.lookup_indices = {0};
      EXPECT_EQ(RunBlockwiseLookupCase(test_case), RunResult::kRejected);
      break;
    }
    case 5: {
      // column_size not divisible by blocksize.
      BlockwiseLookupCase test_case;
      test_case.lookup_shape = {1};
      test_case.value_shape = {2, 34};
      test_case.scale_shape = {2, 1};
      test_case.blocksize = 32;
      test_case.lookup_indices = {0};
      EXPECT_EQ(RunBlockwiseLookupCase(test_case), RunResult::kRejected);
      break;
    }
    case 6: {
      // out-of-range scale_tensor_index.
      BlockwiseLookupCase test_case;
      test_case.lookup_shape = {1};
      test_case.value_shape = {2, 32};
      test_case.scale_shape = {2, 1};
      test_case.blocksize = 32;
      test_case.scale_tensor_index = 99;
      test_case.lookup_indices = {0};
      EXPECT_EQ(RunBlockwiseLookupCase(test_case), RunResult::kRejected);
      break;
    }
    case 7: {
      // wrong scale tensor element type (FLOAT32 instead of FLOAT16).
      BlockwiseLookupCase test_case;
      test_case.lookup_shape = {1};
      test_case.value_shape = {2, 32};
      test_case.scale_shape = {2, 1};
      test_case.blocksize = 32;
      test_case.scale_type = TensorType_FLOAT32;
      test_case.lookup_indices = {0};
      EXPECT_EQ(RunBlockwiseLookupCase(test_case), RunResult::kRejected);
      break;
    }
    case 8: {
      // undersized scale tensor ([1, 1] instead of [2, 1]).
      BlockwiseLookupCase test_case;
      test_case.lookup_shape = {1};
      test_case.value_shape = {2, 32};
      test_case.scale_shape = {1, 1};
      test_case.blocksize = 32;
      test_case.lookup_indices = {1};
      EXPECT_EQ(RunBlockwiseLookupCase(test_case), RunResult::kRejected);
      break;
    }
    default: {
      // empty value rows [0, 4] with non-empty lookup [1].
      SimpleLookupCase test_case;
      test_case.lookup_shape = {1};
      test_case.value_shape = {0, 4};
      test_case.lookup_indices = {0};
      EXPECT_EQ(RunSimpleLookupCase(test_case), RunResult::kRejected);
      break;
    }
  }
}
FUZZ_TEST(EmbeddingLookupFuzz,
          EmbeddingLookupRejectsInvalidRanksAndQuantization)
    .WithDomains(fuzztest::InRange<int32_t>(0, 9));

void SilentReportError(TfLiteContext*, const char*, ...) {}

void BytesRequiredSubByteFuzz(TfLiteType type, std::vector<int> dims) {
  TfLiteContext context{};
  context.ReportError = SilentReportError;
  size_t bytes = 123;
  const TfLiteStatus status =
      BytesRequired(type, dims.data(), dims.size(), &bytes, &context);
  bool all_positive = true;
  for (int d : dims) {
    if (d <= 0) {
      all_positive = false;
      break;
    }
  }
  if (status == kTfLiteOk && all_positive) {
    EXPECT_GT(bytes, 0u);
  }
}
FUZZ_TEST(EmbeddingLookupFuzz, BytesRequiredSubByteFuzz)
    .WithDomains(fuzztest::ElementOf({kTfLiteInt4, kTfLiteUInt4, kTfLiteInt2}),
                 fuzztest::VectorOf(fuzztest::Arbitrary<int>()).WithMaxSize(8));

TEST(EmbeddingLookupFuzzSmokeTest, SimpleValidCaseExecutes) {
  SimpleLookupCase test_case;
  test_case.lookup_shape = {2};
  test_case.value_shape = {4, 3};
  test_case.lookup_indices = {1, 3};
  test_case.value_type = TensorType_FLOAT32;
  EXPECT_EQ(RunSimpleLookupCase(test_case), RunResult::kSuccess);
}

TEST(EmbeddingLookupFuzzSmokeTest, HybridInt4AndInt2ValidCasesExecute) {
  for (TensorType type : {TensorType_INT8, TensorType_INT4, TensorType_INT2}) {
    HybridLookupCase test_case;
    test_case.lookup_shape = {2};
    test_case.value_shape = {3, 4};
    test_case.lookup_indices = {0, 2};
    test_case.scales = {0.5f, 0.25f, 1.0f};
    test_case.zero_points = {0, 0, 0};
    test_case.value_bytes = {0xFF, 0x80, 0x7F, 0x01};
    test_case.value_type = type;
    EXPECT_EQ(RunHybridLookupCase(test_case), RunResult::kSuccess);
  }
}

TEST(EmbeddingLookupFuzzSmokeTest, BlockwiseInt4AndInt2ValidCasesExecute) {
  for (TensorType type : {TensorType_INT4, TensorType_INT2}) {
    BlockwiseLookupCase test_case;
    test_case.lookup_shape = {2};
    test_case.value_shape = {3, 32};
    test_case.scale_shape = {3, 1};
    test_case.blocksize = 32;
    test_case.lookup_indices = {0, 2};
    test_case.scale_fp16_values = {0x3C00, 0x3C00, 0x3C00};
    test_case.value_bytes = {0xFF, 0x80, 0x7F, 0x01};
    test_case.value_type = type;
    EXPECT_EQ(RunBlockwiseLookupCase(test_case), RunResult::kSuccess);
  }
}

TEST(EmbeddingLookupFuzzSmokeTest,
     ZeroLengthLookupWithEmptyValueTableExecutes) {
  SimpleLookupCase test_case;
  test_case.lookup_shape = {0};
  test_case.value_shape = {0, 4};
  test_case.lookup_indices = {};
  test_case.value_type = TensorType_FLOAT32;
  EXPECT_EQ(RunSimpleLookupCase(test_case), RunResult::kSuccess);
}

TEST(EmbeddingLookupFuzzSmokeTest,
     OutOfBoundsAndMalformedBlockwiseCasesAreRejected) {
  EmbeddingLookupRejectsOutOfBoundsIndex(/*rows=*/3, /*bad_offset=*/1,
                                         /*negative_index=*/false, /*mode=*/0);
  EmbeddingLookupRejectsOutOfBoundsIndex(/*rows=*/3, /*bad_offset=*/1,
                                         /*negative_index=*/true, /*mode=*/1);
  EmbeddingLookupRejectsOutOfBoundsIndex(/*rows=*/3, /*bad_offset=*/1,
                                         /*negative_index=*/false, /*mode=*/2);
  for (int32_t defect = 0; defect <= 9; ++defect) {
    EmbeddingLookupRejectsInvalidRanksAndQuantization(defect);
  }
}

}  // namespace
}  // namespace tflite

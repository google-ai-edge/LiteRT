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

#include "tensor/examples/gemma4/helpers/quantized_embedding.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::examples::gemma4 {
namespace {

using ::testing::ElementsAre;
using ::testing::Pointwise;

MATCHER_P(FloatNearMatcher, tol, "") {
  return std::abs(std::get<0>(arg) - std::get<1>(arg)) <= tol;
}

TEST(QuantizedEmbeddingTest, Fp32Lookup) {
  // Vocab size = 4, Embedding dim = 4
  const std::vector<float> data = {
      1.0f,  2.0f,  3.0f,  4.0f,   // row 0
      5.0f,  6.0f,  7.0f,  8.0f,   // row 1
      9.0f,  10.0f, 11.0f, 12.0f,  // row 2
      13.0f, 14.0f, 15.0f, 16.0f   // row 3
  };

  TensorHandle tensor({.name = "emb",
                       .type = Type::kFP32,
                       .shape = {4, 4},
                       .buffer = std::make_shared<SpanCpuBuffer>(data)});

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GemmaEmbeddingTable> table,
                                  GemmaEmbeddingTable::Create(tensor));

  EXPECT_EQ(table->VocabSize(), 4);
  EXPECT_EQ(table->EmbeddingDim(), 4);

  std::vector<int32_t> tokens = {2, 0};
  std::vector<float> output(tokens.size() * 4);
  ASSERT_THAT(table->Lookup(tokens, absl::MakeSpan(output)), IsOk());
  EXPECT_THAT(output, Pointwise(FloatNearMatcher(1e-5f),
                                {
                                    9.0f, 10.0f, 11.0f, 12.0f,  // row 2
                                    1.0f, 2.0f, 3.0f, 4.0f      // row 0
                                }));
}

TEST(QuantizedEmbeddingTest, Int8PerChannelLookup) {
  // Vocab size = 2, Embedding dim = 4
  const std::vector<int8_t> data = {
      10, -20, 30, -40,  // row 0
      5,  -10, 15, -20   // row 1
  };
  std::vector<float> scales = {0.1f, 0.5f};
  std::vector<int64_t> zero_points = {0, 0};

  TensorHandle tensor(
      {.name = "emb_i8",
       .type = Type::kI8,
       .shape = {2, 4},
       .buffer = std::make_shared<SpanCpuBuffer>(data),
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           scales, zero_points, /*quantized_dimension=*/0)});

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GemmaEmbeddingTable> table,
                                  GemmaEmbeddingTable::Create(tensor));

  std::vector<int32_t> tokens = {1, 0};
  std::vector<float> output(tokens.size() * 4);
  ASSERT_THAT(table->Lookup(tokens, absl::MakeSpan(output)), IsOk());
  EXPECT_THAT(output, Pointwise(FloatNearMatcher(1e-5f),
                                {
                                    2.5f, -5.0f, 7.5f, -10.0f,  // row 1 * 0.5
                                    1.0f, -2.0f, 3.0f, -4.0f    // row 0 * 0.1
                                }));
}

TEST(QuantizedEmbeddingTest, Int4BlockwisePackedLookup) {
  // Vocab size = 2, Embedding dim = 4 (block size = 2, so 2 blocks per row)
  // Row 0 values: [1, -2, 3, -4]
  //   nibble 0 (low) = 1 (0x1), nibble 1 (high) = -2 (0xE)
  //   nibble 2 (low) = 3 (0x3), nibble 3 (high) = -4 (0xC)
  // Row 1 values: [-8, 7, 0, -1]
  //   nibble 0 (low) = -8 (0x8), nibble 1 (high) = 7 (0x7)
  //   nibble 2 (low) = 0 (0x0), nibble 3 (high) = -1 (0xF)
  const std::vector<uint8_t> packed_bytes = {
      0xE1, 0xC3,  // row 0
      0x78, 0xF0   // row 1
  };

  // 2 blocks per row:
  // row 0 block 0 scale = 0.5, row 0 block 1 scale = 2.0
  // row 1 block 0 scale = 1.0, row 1 block 1 scale = 0.25
  std::vector<float> scales = {0.5f, 2.0f, 1.0f, 0.25f};

  TensorHandle tensor(
      {.name = "emb_i4",
       .type = Type::kI4,
       .shape = {2, 4},
       .buffer = std::make_shared<SpanCpuBuffer>(packed_bytes),
       .quantization = std::make_shared<BlockwiseQuantization>(
           scales, /*zero_points=*/std::vector<int64_t>{0}, /*block_size=*/2,
           /*quantized_dimension=*/0)});

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GemmaEmbeddingTable> table,
                                  GemmaEmbeddingTable::Create(tensor));

  std::vector<int32_t> tokens = {0, 1};
  std::vector<float> output(tokens.size() * 4);
  ASSERT_THAT(table->Lookup(tokens, absl::MakeSpan(output)), IsOk());
  EXPECT_THAT(output, Pointwise(FloatNearMatcher(1e-5f),
                                {// row 0: [1*0.5, -2*0.5, 3*2.0, -4*2.0]
                                 0.5f, -1.0f, 6.0f, -8.0f,
                                 // row 1: [-8*1.0, 7*1.0, 0*0.25, -1*0.25]
                                 -8.0f, 7.0f, 0.0f, -0.25f}));
}

TEST(QuantizedEmbeddingTest, LookupPerLayer) {
  // Vocab size = 2, num_layers = 2, per_layer_dim = 2
  const std::vector<float> data = {
      1.0f, 2.0f, 3.0f, 4.0f,  // row 0: Layer 0 [1, 2], Layer 1 [3, 4]
      5.0f, 6.0f, 7.0f, 8.0f   // row 1: Layer 0 [5, 6], Layer 1 [7, 8]
  };
  TensorHandle tensor({.name = "ple",
                       .type = Type::kFP32,
                       .shape = {2, 4},
                       .buffer = std::make_shared<SpanCpuBuffer>(data)});

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GemmaEmbeddingTable> table,
                                  GemmaEmbeddingTable::Create(tensor));

  std::vector<int32_t> tokens = {1, 0};
  const int seq_len = tokens.size();
  std::vector<std::vector<float>> output_per_layer(
      2, std::vector<float>(seq_len * 2));

  ASSERT_THAT(table->LookupPerLayer(tokens, /*num_layers=*/2,
                                    /*per_layer_dim=*/2,
                                    absl::MakeSpan(output_per_layer)),
              IsOk());

  EXPECT_THAT(
      output_per_layer,
      ElementsAre(
          // Layer 0 for tokens [1, 0]: [5, 6] for token 1, [1, 2] for token 0
          Pointwise(FloatNearMatcher(1e-5f), {5.0f, 6.0f, 1.0f, 2.0f}),
          // Layer 1 for tokens [1, 0]: [7, 8] for token 1, [3, 4] for token 0
          Pointwise(FloatNearMatcher(1e-5f), {7.0f, 8.0f, 3.0f, 4.0f})));
}

TEST(QuantizedEmbeddingTest, LookupSingleTokenFp32) {
  const std::vector<float> data = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
  };
  TensorHandle tensor({.name = "emb",
                       .type = Type::kFP32,
                       .shape = {2, 4},
                       .buffer = std::make_shared<SpanCpuBuffer>(data)});

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GemmaEmbeddingTable> table,
                                  GemmaEmbeddingTable::Create(tensor));

  EXPECT_THAT(table->Lookup(/*token_id=*/1),
              IsOkAndHolds(Pointwise(FloatNearMatcher(1e-5f),
                                     {5.0f, 6.0f, 7.0f, 8.0f})));
}

TEST(QuantizedEmbeddingTest, LookupSingleTokenInt4) {
  const std::vector<uint8_t> packed_bytes = {
      0xE1, 0xC3,  // row 0: [1, -2, 3, -4]
      0x78, 0xF0   // row 1: [-8, 7, 0, -1]
  };
  std::vector<float> scales = {0.5f, 2.0f, 1.0f, 0.25f};

  TensorHandle tensor(
      {.name = "emb_i4",
       .type = Type::kI4,
       .shape = {2, 4},
       .buffer = std::make_shared<SpanCpuBuffer>(packed_bytes),
       .quantization = std::make_shared<BlockwiseQuantization>(
           scales, /*zero_points=*/std::vector<int64_t>{0}, /*block_size=*/2,
           /*quantized_dimension=*/0)});

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GemmaEmbeddingTable> table,
                                  GemmaEmbeddingTable::Create(tensor));
  EXPECT_THAT(table->Lookup(/*token_id=*/0),
              IsOkAndHolds(Pointwise(FloatNearMatcher(1e-5f),
                                     {0.5f, -1.0f, 6.0f, -8.0f})));
}

TEST(QuantizedEmbeddingTest, LookupPerLayerSingleToken) {
  const std::vector<float> data = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
  };
  TensorHandle tensor({.name = "ple",
                       .type = Type::kFP32,
                       .shape = {2, 4},
                       .buffer = std::make_shared<SpanCpuBuffer>(data)});

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GemmaEmbeddingTable> table,
                                  GemmaEmbeddingTable::Create(tensor));

  EXPECT_THAT(table->LookupPerLayer(/*token_id=*/1,
                                    /*num_layers=*/2,
                                    /*per_layer_dim=*/2),
              IsOkAndHolds(ElementsAre(
                  Pointwise(FloatNearMatcher(1e-5f), {5.0f, 6.0f}),
                  Pointwise(FloatNearMatcher(1e-5f), {7.0f, 8.0f}))));
}

TEST(QuantizedEmbeddingTest, Int4BlockwisePhysicalHalvedShapeAutoDeduce) {
  // Physical packed shape = {2, 2} (2 rows, 2 bytes per row = 4 4-bit elements
  // per row) Row 0 values: [1, -2, 3, -4] -> bytes: 0xE1, 0xC3 Row 1 values:
  // [-8, 7, 0, -1] -> bytes: 0x78, 0xF0
  std::vector<uint8_t> packed_bytes = {
      0xE1, 0xC3,  // row 0
      0x78, 0xF0   // row 1
  };
  auto buffer = OwningCpuBuffer::Copy<Type::kU8>(packed_bytes);

  // 2 blocks per row (block_size = 2, logical dim = 4):
  std::vector<float> scales = {0.5f, 2.0f, 1.0f, 0.25f};
  auto quant = std::make_shared<BlockwiseQuantization>(
      scales, std::vector<int64_t>{0}, /*block_size=*/2,
      /*quantized_dimension=*/0);

  // Tensor shape is physical {2, 2}
  TensorHandle tensor({.name = "emb_i4_packed",
                       .type = Type::kI4,
                       .shape = {2, 2},
                       .buffer = buffer,
                       .quantization = quant});

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GemmaEmbeddingTable> table,
                                  GemmaEmbeddingTable::Create(tensor));

  // Logical embedding dimension should be auto-deduced as 4 (2 blocks *
  // block_size 2)
  EXPECT_EQ(table->EmbeddingDim(), 4);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const float> result,
                                  table->Lookup(/*token_id=*/0));
  std::vector<float> expected = {0.5f, -1.0f, 6.0f, -8.0f};
  EXPECT_THAT(result, Pointwise(FloatNearMatcher(1e-5f), expected));
}

TEST(QuantizedEmbeddingTest, Int4BlockwisePhysicalHalvedShapeWithExpectedDim) {
  // Physical packed shape = {2, 2}, explicitly passing expected_emb_dim = 4
  std::vector<uint8_t> packed_bytes = {
      0xE1, 0xC3,  // row 0
      0x78, 0xF0   // row 1
  };
  auto buffer = OwningCpuBuffer::Copy<Type::kU8>(packed_bytes);

  std::vector<float> scales = {0.5f, 2.0f, 1.0f, 0.25f};
  auto quant = std::make_shared<BlockwiseQuantization>(
      scales, std::vector<int64_t>{0}, /*block_size=*/2,
      /*quantized_dimension=*/0);

  TensorHandle tensor({.name = "emb_i4_packed",
                       .type = Type::kI4,
                       .shape = {2, 2},
                       .buffer = buffer,
                       .quantization = quant});

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GemmaEmbeddingTable> table,
                                  GemmaEmbeddingTable::Create(tensor));

  EXPECT_EQ(table->EmbeddingDim(), 4);

  std::vector<int32_t> tokens = {0, 1};
  std::vector<float> output(tokens.size() * 4);
  ASSERT_THAT(table->Lookup(tokens, absl::MakeSpan(output)), IsOk());

  std::vector<float> expected = {0.5f,  -1.0f, 6.0f, -8.0f,
                                 -8.0f, 7.0f,  0.0f, -0.25f};
  EXPECT_THAT(output, Pointwise(FloatNearMatcher(1e-5f), expected));
}

TEST(QuantizedEmbeddingTest, Int8BlockwiseAsymmetricLookup) {
  const std::vector<int8_t> data = {
      10, -20, 30, -40,  // row 0
      5,  -10, 15, -20   // row 1
  };
  std::vector<float> scales = {0.1f, 0.2f, 0.5f, 1.0f};
  std::vector<int64_t> zero_points = {2, -5, 1, -2};

  TensorHandle tensor(
      {.name = "emb_i8_bw_asym",
       .type = Type::kI8,
       .shape = {2, 4},
       .buffer = std::make_shared<SpanCpuBuffer>(data),
       .quantization = std::make_shared<BlockwiseQuantization>(
           scales, zero_points, /*block_size=*/2,
           /*quantized_dimension=*/0)});

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GemmaEmbeddingTable> table,
                                  GemmaEmbeddingTable::Create(tensor));

  std::vector<int32_t> tokens = {1, 0};
  std::vector<float> output(tokens.size() * 4);
  ASSERT_THAT(table->Lookup(tokens, absl::MakeSpan(output)), IsOk());
  EXPECT_THAT(output, Pointwise(FloatNearMatcher(1e-5f),
                                {// row 1: [(5-1)*0.5, (-10-1)*0.5,
                                 // (15-(-2))*1.0, (-20-(-2))*1.0]
                                 2.0f, -5.5f, 17.0f, -18.0f,
                                 // row 0: [(10-2)*0.1, (-20-2)*0.1,
                                 // (30-(-5))*0.2, (-40-(-5))*0.2]
                                 0.8f, -2.2f, 7.0f, -7.0f}));
}

TEST(QuantizedEmbeddingTest, Int4BlockwiseAsymmetricLookup) {
  const std::vector<uint8_t> packed_bytes = {
      0xE1, 0xC3,  // row 0: [1, -2, 3, -4]
      0x78, 0xF0   // row 1: [-8, 7, 0, -1]
  };
  std::vector<float> scales = {0.5f, 2.0f, 1.0f, 0.25f};
  std::vector<int64_t> zero_points = {1, -1, 2, -2};

  TensorHandle tensor(
      {.name = "emb_i4_bw_asym",
       .type = Type::kI4,
       .shape = {2, 4},
       .buffer = std::make_shared<SpanCpuBuffer>(packed_bytes),
       .quantization = std::make_shared<BlockwiseQuantization>(
           scales, zero_points, /*block_size=*/2,
           /*quantized_dimension=*/0)});

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GemmaEmbeddingTable> table,
                                  GemmaEmbeddingTable::Create(tensor));

  std::vector<int32_t> tokens = {0, 1};
  std::vector<float> output(tokens.size() * 4);
  ASSERT_THAT(table->Lookup(tokens, absl::MakeSpan(output)), IsOk());
  EXPECT_THAT(output, Pointwise(FloatNearMatcher(1e-5f),
                                {// row 0: [(1-1)*0.5, (-2-1)*0.5,
                                 // (3-(-1))*2.0, (-4-(-1))*2.0]
                                 0.0f, -1.5f, 8.0f, -6.0f,
                                 // row 1: [(-8-2)*1.0, (7-2)*1.0,
                                 // (0-(-2))*0.25, (-1-(-2))*0.25]
                                 -10.0f, 5.0f, 0.5f, 0.25f}));
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4

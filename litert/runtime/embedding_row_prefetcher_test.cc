// Copyright 2026 Google LLC.
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

#include "litert/runtime/embedding_row_prefetcher.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/detached_buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/core/util/memory_advice.h"
#include "litert/test/matchers.h"
#include "tflite/c/common.h"
#include "tflite/core/subgraph.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::litert::IsError;

TEST(EmbeddingRowPrefetcherTest, ComputePrefetchRangesPureMath) {
  using EmbeddingTable = EmbeddingRowPrefetcher::EmbeddingTable;
  constexpr size_t kPage = 4096;
  // A page-aligned synthetic buffer: 66 pages leave room for alignment slack.
  std::vector<uint8_t> buf(66 * kPage);
  const uintptr_t base = (reinterpret_cast<uintptr_t>(buf.data()) + kPage - 1) &
                         ~uintptr_t{kPage - 1};
  const EmbeddingTable table_a = {reinterpret_cast<const uint8_t*>(base),
                                  /*num_rows=*/1024, /*row_bytes=*/256};

  // Duplicates hinted once, negative and out-of-range ids skipped, rows 0, 5
  // and 17 (pages 0 and 1) coalesce into a single range.
  {
    const int32_t ids[] = {5, 5, -1, 2000, 0, 17};
    EXPECT_THAT(EmbeddingRowPrefetcher::ComputePrefetchRanges(
                    absl::MakeConstSpan(&table_a, 1), ids, kPage),
                ElementsAre(PageRange{base, base + 2 * kPage}));
  }

  // The whole table (256 KiB) is split into 128 KiB pieces.
  {
    std::vector<int32_t> ids(1024);
    for (int32_t i = 0; i < 1024; ++i) ids[i] = i;
    EXPECT_THAT(EmbeddingRowPrefetcher::ComputePrefetchRanges(
                    absl::MakeConstSpan(&table_a, 1), ids, kPage),
                ElementsAre(PageRange{base, base + 131072},
                            PageRange{base + 131072, base + 262144}));
  }

  // Two tables: ranges are sorted and the gap between them is kept.
  {
    const EmbeddingTable table_b = {
        reinterpret_cast<const uint8_t*>(base + 2 * kPage), /*num_rows=*/16,
        /*row_bytes=*/kPage};
    const EmbeddingTable tables[] = {table_a, table_b};
    const int32_t ids[] = {0, 1};
    EXPECT_THAT(
        EmbeddingRowPrefetcher::ComputePrefetchRanges(tables, ids, kPage),
        ElementsAre(PageRange{base, base + kPage},
                    PageRange{base + 2 * kPage, base + 4 * kPage}));
  }

  // Degenerate inputs produce no ranges.
  {
    const int32_t ids[] = {0, 1};
    const int32_t negative_ids[] = {-3, -1};
    const EmbeddingTable empty_table = {reinterpret_cast<const uint8_t*>(base),
                                        /*num_rows=*/0, /*row_bytes=*/256};
    EXPECT_TRUE(EmbeddingRowPrefetcher::ComputePrefetchRanges(
                    absl::MakeConstSpan(&table_a, 1), {}, kPage)
                    .empty());
    EXPECT_TRUE(EmbeddingRowPrefetcher::ComputePrefetchRanges(
                    absl::MakeConstSpan(&table_a, 1), negative_ids, kPage)
                    .empty());
    EXPECT_TRUE(EmbeddingRowPrefetcher::ComputePrefetchRanges(
                    absl::MakeConstSpan(&table_a, 1), ids, /*page_size=*/0)
                    .empty());
    EXPECT_TRUE(EmbeddingRowPrefetcher::ComputePrefetchRanges(
                    absl::MakeConstSpan(&empty_table, 1), ids, kPage)
                    .empty());
  }
}

TEST(EmbeddingRowPrefetcherTest, PrefetchOnEmptyReportsNotFound) {
  EmbeddingRowPrefetcher empty;
  EXPECT_TRUE(empty.empty());
  EXPECT_THAT(empty.Prefetch({0}), IsError(kLiteRtStatusErrorNotFound));
  EXPECT_THAT(empty.Prefetch({}), IsError(kLiteRtStatusErrorNotFound));
}

// The kernel-contract test below pins the prefetcher's row addressing to the
// EMBEDDING_LOOKUP kernel's: every table has kRows rows of kRowBytes bytes, and
// row k is all zero except byte k, which is 0xFF, so every byte position of a
// row is the marker of some row and a wrong row stride or base address shows
// up as the marker in the wrong place.
constexpr int kRows = 32;
constexpr int kRowBytes = 32;
static_assert(kRows == kRowBytes,
              "every byte position of a row must carry the marker of some row");

bool IsFloatType(tflite::TensorType type) {
  return type == tflite::TensorType_FLOAT32 ||
         type == tflite::TensorType_FLOAT16;
}

// Builds an EMBEDDING_LOOKUP model in memory: lookup INT32 [kRows] -> output
// [kRows, cols], with a constant table of `type` and shape [kRows, cols]
// filled as described above. The output has the table's type for the float
// types, so the kernel's Eval takes the bytewise EvalSimple path, and FLOAT32
// otherwise; the integer types get the symmetric scale-1 parameters the
// kernel's hybrid path requires.
flatbuffers::DetachedBuffer BuildEmbeddingLookupModel(tflite::TensorType type,
                                                      int cols) {
  auto model = std::make_unique<tflite::ModelT>();
  model->version = 3;

  auto opcode = std::make_unique<tflite::OperatorCodeT>();
  opcode->deprecated_builtin_code =
      static_cast<int8_t>(tflite::BuiltinOperator_EMBEDDING_LOOKUP);
  opcode->builtin_code = tflite::BuiltinOperator_EMBEDDING_LOOKUP;
  opcode->version = 1;
  model->operator_codes.push_back(std::move(opcode));

  // Buffer 0 is the empty buffer of the non-constant tensors.
  model->buffers.push_back(std::make_unique<tflite::BufferT>());
  auto table_buffer = std::make_unique<tflite::BufferT>();
  table_buffer->data.assign(kRows * kRowBytes, 0);
  for (int k = 0; k < kRows; ++k) table_buffer->data[k * kRowBytes + k] = 0xFF;
  model->buffers.push_back(std::move(table_buffer));

  auto lookup = std::make_unique<tflite::TensorT>();
  lookup->type = tflite::TensorType_INT32;
  lookup->shape = {kRows};
  lookup->buffer = 0;
  lookup->name = "lookup";
  auto table = std::make_unique<tflite::TensorT>();
  table->type = type;
  table->shape = {kRows, cols};
  table->buffer = 1;
  table->name = "table";
  if (!IsFloatType(type)) {
    table->quantization = std::make_unique<tflite::QuantizationParametersT>();
    table->quantization->scale = {1.0f};
    table->quantization->zero_point = {0};
  }
  auto output = std::make_unique<tflite::TensorT>();
  output->type = IsFloatType(type) ? type : tflite::TensorType_FLOAT32;
  output->shape = {kRows, cols};
  output->buffer = 0;
  output->name = "output";

  auto op = std::make_unique<tflite::OperatorT>();
  op->opcode_index = 0;
  op->inputs = {0, 1};
  op->outputs = {2};

  auto subgraph = std::make_unique<tflite::SubGraphT>();
  subgraph->tensors.push_back(std::move(lookup));
  subgraph->tensors.push_back(std::move(table));
  subgraph->tensors.push_back(std::move(output));
  subgraph->inputs = {0};
  subgraph->outputs = {2};
  subgraph->operators.push_back(std::move(op));
  model->subgraphs.push_back(std::move(subgraph));

  flatbuffers::FlatBufferBuilder builder;
  tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, model.get()));
  return builder.Release();
}

// The kernel's reading of the `row_bytes` packed bytes at `row` with scale 1,
// written so that it does not depend on the order in which the kernel emits
// the sub-byte fields of a byte (the fill above gives every byte equal
// fields). Not used for the float types, which the kernel copies bytewise.
std::vector<float> DecodeRow(tflite::TensorType type, const uint8_t* row,
                             size_t row_bytes) {
  std::vector<float> values;
  for (size_t i = 0; i < row_bytes; ++i) {
    const uint8_t b = row[i];
    switch (type) {
      case tflite::TensorType_INT8:
        values.push_back(static_cast<int8_t>(b));
        break;
      case tflite::TensorType_INT4: {
        const int v = b & 0xF;
        values.insert(values.end(), 2, v >= 8 ? v - 16 : v);
        break;
      }
      case tflite::TensorType_INT2: {
        const int v = b & 0x3;
        values.insert(values.end(), 4, v >= 2 ? v - 4 : v);
        break;
      }
      default:
        ADD_FAILURE() << "DecodeRow: unsupported tensor type " << type;
        return values;
    }
  }
  return values;
}

TEST(EmbeddingRowPrefetcherTest, RowAddressesMatchEmbeddingLookupKernel) {
  struct TestCase {
    const char* name;
    tflite::TensorType type;
    int cols;  // Chosen so that every table has kRowBytes bytes per row.
  };
  const TestCase kCases[] = {
      {"float32", tflite::TensorType_FLOAT32, 8},
      {"float16", tflite::TensorType_FLOAT16, 16},
      {"int8", tflite::TensorType_INT8, 32},
      {"int4", tflite::TensorType_INT4, 64},
      {"int2", tflite::TensorType_INT2, 128},
  };
  const size_t page_size = GetPageSize();
  const uintptr_t page_mask = ~(static_cast<uintptr_t>(page_size) - 1);

  for (const TestCase& tc : kCases) {
    SCOPED_TRACE(tc.name);
    const flatbuffers::DetachedBuffer model_buffer =
        BuildEmbeddingLookupModel(tc.type, tc.cols);
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromBuffer(
            reinterpret_cast<const char*>(model_buffer.data()),
            model_buffer.size());
    ASSERT_NE(model, nullptr);
    tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    ASSERT_EQ(builder(&interpreter), kTfLiteOk);
    ASSERT_NE(interpreter, nullptr);
    ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    // 1. Discovery finds the table where the kernel reads it.
    const EmbeddingRowPrefetcher prefetcher =
        EmbeddingRowPrefetcher::Discover(*interpreter->subgraph(0));
    ASSERT_EQ(prefetcher.tables().size(), 1);
    const EmbeddingRowPrefetcher::EmbeddingTable& table =
        prefetcher.tables()[0];
    const TfLiteTensor* value = interpreter->tensor(1);
    EXPECT_EQ(table.data, reinterpret_cast<const uint8_t*>(value->data.raw));
    EXPECT_EQ(table.num_rows, kRows);
    EXPECT_EQ(table.row_bytes, kRowBytes);
    EXPECT_EQ(table.num_rows * table.row_bytes, value->bytes);

    // 2. The kernel copies every row; output row k is its reading of table
    // row k. For the float types the output has the table's type, so the
    // kernel copies the row bytewise (EvalSimple); the integer types are
    // dequantized to FLOAT32 (EvalHybrid).
    int32_t* ids = interpreter->typed_input_tensor<int32_t>(0);
    for (int32_t k = 0; k < kRows; ++k) ids[k] = k;
    ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
    const TfLiteTensor* output = interpreter->output_tensor(0);
    ASSERT_NE(output, nullptr);
    if (IsFloatType(tc.type)) {
      ASSERT_EQ(output->type, value->type);
      ASSERT_EQ(output->bytes, value->bytes);
    } else {
      ASSERT_EQ(output->type, kTfLiteFloat32);
    }

    for (int k = 0; k < kRows; ++k) {
      const uint8_t* row = table.data + k * table.row_bytes;
      // 3. The bytes at the prefetcher's row address decode to what the kernel
      // produced for that row.
      if (IsFloatType(tc.type)) {
        // Raw bytes, never floats: the 0xFF00 marker pattern is a NaN as
        // float16.
        const uint8_t* output_row =
            reinterpret_cast<const uint8_t*>(output->data.raw) +
            k * table.row_bytes;
        EXPECT_EQ(std::memcmp(output_row, row, table.row_bytes), 0)
            << "row " << k;
      } else {
        EXPECT_THAT(absl::MakeConstSpan(output->data.f + k * tc.cols, tc.cols),
                    ElementsAreArray(DecodeRow(tc.type, row, table.row_bytes)))
            << "row " << k;
      }
      // 4. One row of one table is exactly one page-rounded range.
      const uintptr_t addr = reinterpret_cast<uintptr_t>(row);
      const PageRange expected = {
          addr & page_mask,
          (addr + table.row_bytes + page_size - 1) & page_mask};
      EXPECT_THAT(
          EmbeddingRowPrefetcher::ComputePrefetchRanges(
              prefetcher.tables(), {static_cast<int32_t>(k)}, page_size),
          ElementsAre(expected))
          << "row " << k;
      // 5. The hint itself is accepted where the platform has one (the table
      // lives in heap memory, which madvise and PrefetchVirtualMemory accept);
      // Apple refuses it because the heap table is in a writable mapping.
#if defined(__linux__) || defined(LITERT_WINDOWS_OS)
      LITERT_EXPECT_OK(prefetcher.Prefetch({static_cast<int32_t>(k)}));
#else
      EXPECT_THAT(prefetcher.Prefetch({static_cast<int32_t>(k)}),
                  IsError(kLiteRtStatusErrorUnsupported));
#endif
    }
  }
}

}  // namespace
}  // namespace litert::internal

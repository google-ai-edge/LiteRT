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

#ifndef ODML_LITERT_LITERT_RUNTIME_EMBEDDING_ROW_PREFETCHER_H_
#define ODML_LITERT_LITERT_RUNTIME_EMBEDDING_ROW_PREFETCHER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/core/util/memory_advice.h"
#include "tflite/core/subgraph.h"

namespace litert::internal {

// Hints the OS to read ahead the rows of the constant tables that the CPU
// EMBEDDING_LOOKUP kernels of one subgraph read. The only place in LiteRT that
// knows how that kernel addresses its table; see the LINT blocks in the .cc.
class EmbeddingRowPrefetcher {
 public:
  // A constant table read by a CPU EMBEDDING_LOOKUP op, as the kernel
  // addresses it (`tensorflow/lite/kernels/embedding_lookup.cc`: `EvalSimple`
  // reads row `i` at `data + i * row_bytes` with `row_bytes = bytes / dims[0]`;
  // `EvalHybrid` unpacks sub-byte elements at byte offset `(col_size * i) >> 1`
  // for INT4 and `(col_size * i) >> 2` for INT2, which matches `i * (bytes /
  // dims[0])` when the constant buffer is tightly packed (`bytes ==
  // BytesRequired(type, dims)`, `tensorflow/lite/util.cc`), as the converter
  // emits it). Only tables whose rows are a whole number of bytes are collected
  // (`bytes % dims[0] == 0`, true for the shipped Gemma int4/int2 tables);
  // blockwise-quantized scale tensors are not hinted.
  struct EmbeddingTable {
    const uint8_t* data;  // TfLiteTensor::data.raw (kTfLiteMmapRo).
    size_t num_rows;      // dims[0].
    size_t row_bytes;     // bytes / dims[0].
  };

  // Walks the post-delegation execution plan of `subgraph` (as
  // LiteRtCompiledModelT::CheckCpuTensors() does): an EMBEDDING_LOOKUP that a
  // delegate took shows up as a delegate node, so only tables the CPU kernel
  // will actually read are collected, and only those whose rows are a whole
  // number of bytes (true for the shipped Gemma int4/int2 tables);
  // blockwise-quantized scale tensors are not hinted. Call it after
  // AllocateTensors() so the plan is the one Invoke() runs.
  static EmbeddingRowPrefetcher Discover(const tflite::Subgraph& subgraph);

  // True when the subgraph runs no CPU EMBEDDING_LOOKUP whose table rows are
  // whole bytes. A cached empty prefetcher is the negative answer.
  bool empty() const { return tables_.empty(); }

  // The discovered tables (for tests).
  absl::Span<const EmbeddingTable> tables() const { return tables_; }

  // Hints the OS to prefetch the rows `token_ids` of every table. Errors:
  // NotFound (empty()), Unsupported (no OS hint on this platform),
  // RuntimeFailure (the OS rejected a hint). An empty `token_ids` only probes
  // and reports NotFound / Unsupported the same way.
  Expected<void> Prefetch(absl::Span<const int32_t> token_ids) const;

  // Pure helper (exposed for tests): the page-aligned ranges that cover the
  // rows `token_ids` of every table, sorted, coalesced across tables and split
  // into pieces of at most 128 KiB. Ids outside [0, num_rows) are skipped;
  // duplicates are hinted once. `page_size` must be a power of two.
  static std::vector<PageRange> ComputePrefetchRanges(
      absl::Span<const EmbeddingTable> tables,
      absl::Span<const int32_t> token_ids, size_t page_size);

 private:
  std::vector<EmbeddingTable> tables_;
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_EMBEDDING_ROW_PREFETCHER_H_

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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/util/memory_advice.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/core/subgraph.h"

namespace litert::internal {
namespace {

// Largest single prefetch hint, so one call never blocks on a huge readahead.
constexpr size_t kMaxPrefetchRangeBytes = 128 * 1024;

}  // namespace

EmbeddingRowPrefetcher EmbeddingRowPrefetcher::Discover(
    const tflite::Subgraph& subgraph) {
  EmbeddingRowPrefetcher prefetcher;
  std::vector<EmbeddingTable>& tables = prefetcher.tables_;
  const auto& execution_plan = subgraph.execution_plan();
  const auto& nodes_and_registration = subgraph.nodes_and_registration();
  for (int node_index : execution_plan) {
    const TfLiteNode& node = nodes_and_registration[node_index].first;
    const TfLiteRegistration& registration =
        nodes_and_registration[node_index].second;
    // LINT.IfChange(embedding_lookup_table)
    if (registration.builtin_code != kTfLiteBuiltinEmbeddingLookup ||
        node.inputs == nullptr || node.inputs->size < 2) {
      continue;
    }
    // Input 0 is the lookup ids, input 1 the value table
    // (tensorflow/lite/kernels/embedding_lookup.cc).
    const int value_tensor_index = node.inputs->data[1];
    if (value_tensor_index == kTfLiteOptionalTensor) continue;
    const TfLiteTensor* value = subgraph.tensor(value_tensor_index);
    if (value == nullptr || value->allocation_type != kTfLiteMmapRo ||
        value->data.raw == nullptr || value->dims == nullptr ||
        value->dims->size < 2 || value->dims->data[0] <= 0) {
      continue;
    }
    const size_t num_rows = static_cast<size_t>(value->dims->data[0]);
    // Rows must be whole bytes for data + i * row_bytes to be the kernel's
    // address.
    if (value->bytes % num_rows != 0) continue;
    const size_t row_bytes = value->bytes / num_rows;
    if (row_bytes == 0) continue;
    tables.push_back({reinterpret_cast<const uint8_t*>(value->data.raw),
                      num_rows, row_bytes});
    // LINT.ThenChange(//tflite/kernels/embedding_lookup.cc:table_input)
  }
#if defined(__APPLE__)
  // One-time check: the table addresses and the mapping protections do not
  // change after tensor allocation; the caller guarantees it (see
  // LiteRtCompiledModelT::InitializeRuntime()). One writable table disables
  // the hint for the whole subgraph.
  for (const EmbeddingTable& table : prefetcher.tables_) {
    // num_rows * row_bytes is the tensor's byte size (row_bytes is
    // bytes / num_rows with no remainder), so it cannot overflow.
    if (!IsReadOnlyMapping(table.data, table.num_rows * table.row_bytes)) {
      prefetcher.writable_mapping_ = true;
      break;
    }
  }
#endif  // defined(__APPLE__)
  return prefetcher;
}

Expected<void> EmbeddingRowPrefetcher::Prefetch(
    absl::Span<const int32_t> token_ids) const {
  if (tables_.empty()) {
    return Unexpected(
        kLiteRtStatusErrorNotFound,
        "No prefetchable CPU EMBEDDING_LOOKUP table in the subgraph.");
  }
#if defined(__APPLE__)
  if (writable_mapping_) {
    return Unexpected(
        kLiteRtStatusErrorUnsupported,
        "Embedding row prefetch on Apple platforms needs every table in a "
        "read-only mapping (madvise(MADV_WILLNEED) copy-on-writes writable "
        "private pages); a table of this subgraph is mapped writable.");
  }
#endif  // defined(__APPLE__)
  // The empty hint issues nothing and only reports whether this platform has
  // one, so it doubles as the probe.
  if (auto probe = AdviseWillNeed({}); !probe) return probe;
  if (token_ids.empty()) return {};  // Probe only.
  return AdviseWillNeed(
      ComputePrefetchRanges(tables_, token_ids, GetPageSize()));
}

std::vector<PageRange> EmbeddingRowPrefetcher::ComputePrefetchRanges(
    absl::Span<const EmbeddingTable> tables,
    absl::Span<const int32_t> token_ids, size_t page_size) {
  std::vector<PageRange> result;
  if (tables.empty() || token_ids.empty() || page_size == 0 ||
      (page_size & (page_size - 1)) != 0) {
    return result;
  }
  std::vector<int32_t> rows;
  rows.reserve(token_ids.size());
  for (int32_t token : token_ids) {
    if (token >= 0) rows.push_back(token);
  }
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
  if (rows.empty()) return result;

  // The page size is a power of two, so masking with ~(page - 1) rounds down.
  const uintptr_t page = page_size;
  const uintptr_t page_mask = ~(page - 1);
  std::vector<PageRange> ranges;
  ranges.reserve(rows.size() * tables.size());
  for (const EmbeddingTable& table : tables) {
    if (table.data == nullptr || table.num_rows == 0 || table.row_bytes == 0) {
      continue;
    }
    // LINT.IfChange(embedding_lookup_row_address)
    for (int32_t row : rows) {
      if (static_cast<size_t>(row) >= table.num_rows) break;  // rows sorted.
      const uintptr_t begin = reinterpret_cast<uintptr_t>(
          table.data + static_cast<size_t>(row) * table.row_bytes);
      const uintptr_t end = begin + table.row_bytes;
      ranges.push_back({begin & page_mask, (end + page - 1) & page_mask});
    }
    // LINT.ThenChange(
    //     //tflite/kernels/embedding_lookup.cc:row_address|
    //     //tflite/kernels/embedding_lookup.cc:packed_row_address
    // )
  }
  if (ranges.empty()) return result;
  std::sort(
      ranges.begin(), ranges.end(),
      [](const PageRange& a, const PageRange& b) { return a.begin < b.begin; });

  // Coalesce adjacent/overlapping page ranges, then split each run into pieces
  // of at most kMaxPrefetchRangeBytes (page-aligned as long as that constant
  // is a multiple of the page size; otherwise a piece is one page).
  const uintptr_t piece = std::max<uintptr_t>(kMaxPrefetchRangeBytes, page);
  auto flush = [&result, piece](const PageRange& range) {
    for (uintptr_t begin = range.begin; begin < range.end; begin += piece) {
      result.push_back({begin, std::min(range.end, begin + piece)});
    }
  };
  PageRange current = ranges.front();
  for (size_t i = 1; i < ranges.size(); ++i) {
    if (ranges[i].begin <= current.end) {
      current.end = std::max(current.end, ranges[i].end);
    } else {
      flush(current);
      current = ranges[i];
    }
  }
  flush(current);
  return result;
}

}  // namespace litert::internal

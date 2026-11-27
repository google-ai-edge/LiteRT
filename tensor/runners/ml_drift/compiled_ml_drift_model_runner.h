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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_COMPILED_ML_DRIFT_MODEL_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_COMPILED_ML_DRIFT_MODEL_RUNNER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/container/node_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/model_hints.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/profiling_info.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_conversion.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/backends/ml_drift/ml_drift_conversion.h"
#include "tensor/backends/ml_drift/ml_drift_helpers.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/graph_traversal.h"
#include "tensor/runners/ml_drift/compiled_ml_drift_model_runner_util.h"
#include "tensor/runners/ml_drift/ml_drift_dummy_weight_creator.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/parallel.h"

namespace litert::tensor {

namespace ml_drift_runner_internal {

// `sum_i` (per output channel, the sum of the signed quantized values over the
// input channels) of the unpacked INT4 FullyConnected weights `values` (OHWI).
inline std::vector<int32_t> ComputeWeightsSumI(
    const ml_drift::OHWI& weights_ohwi, absl::Span<const int8_t> values) {
  const int out_channels = weights_ohwi.o;
  const size_t row =
      static_cast<size_t>(weights_ohwi.h) * weights_ohwi.w * weights_ohwi.i;
  std::vector<int32_t> sum_i(out_channels, 0);
  for (int o = 0; o < out_channels; ++o) {
    int32_t s = 0;
    const int8_t* r = values.data() + static_cast<size_t>(o) * row;
    for (size_t k = 0; k < row; ++k) s += r[k];
    sum_i[o] = s;
  }
  return sum_i;
}

// INT4 weights in the kernel layout `desc` (`elements_count` elements), and
// their `sum_i` when requested.
struct RearrangedInt4Weights {
  std::vector<uint8_t> data;
  std::optional<std::vector<int32_t>> sum_i;
};

// Cache-tiled, zero-copy rearrangement of packed INT4 OHWI weights into ML
// Drift's UINT4 (+8 shifted, 0x88 padded) I4O4 layouts (`kOSpatialIOGroupI4O4`
// and `k2DYIsSpatialIOAndXIsOGroupI4O4`) when `h == 1`, `w == 1`, `o % 4 == 0`,
// and `i % 4 == 0`. Returns false otherwise so callers fall back to
// `ml_drift::RearrangeWeightsUInt4Packed`.
inline bool TryDirectRearrangePackedInt4AsUint4I4O4(
    const uint8_t* src, size_t src_bytes, const ml_drift::OHWI& ohwi,
    const ml_drift::WeightsDescription& desc, absl::Span<uint8_t> dst,
    std::optional<std::vector<int32_t>>* sum_i_out, size_t max_threads = 16) {
  if (ohwi.h != 1 || ohwi.w != 1 || ohwi.o <= 0 || ohwi.i <= 0 ||
      ohwi.o % 4 != 0 || ohwi.i % 4 != 0) {
    return false;
  }
  const bool is_linear_i4o4 =
      desc.layout == ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  const bool is_2d_i4o4 =
      desc.layout == ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
  if (!is_linear_i4o4 && !is_2d_i4o4) return false;

  const size_t o_dim = static_cast<size_t>(ohwi.o);
  const size_t i_dim = static_cast<size_t>(ohwi.i);
  const size_t row_bytes = i_dim / 2;
  if (src_bytes < o_dim * row_bytes) return false;

  const int dst_group_size_int = desc.GetOutputGroupSize();
  if (dst_group_size_int <= 0) return false;
  const size_t dst_group_size = static_cast<size_t>(dst_group_size_int);
  const size_t dst_slices = o_dim / 4;
  const size_t src_slices = i_dim / 4;
  const size_t dst_groups = (dst_slices + dst_group_size - 1) / dst_group_size;
  if (dst.size() < dst_groups * src_slices * dst_group_size * 8) return false;

  std::vector<int32_t> sum_i;
  const bool want_sum_i = (sum_i_out != nullptr);
  if (want_sum_i) sum_i.assign(o_dim, 0);

  uint8_t* dst_ptr = dst.data();
  const size_t slice_stride =
      is_linear_i4o4 ? dst_group_size : dst_groups * dst_group_size;
  constexpr size_t kDGroupTile = 16;
  constexpr size_t kSTile = 32;  // 64 bytes per source row = 1 cache line.
  const size_t num_dg_tiles =
      (dst_group_size + kDGroupTile - 1) / kDGroupTile;
  const size_t total_dg_tiles = dst_groups * num_dg_tiles;

  auto process_tile_range = [&](size_t t_begin, size_t t_end) {
    auto run = [&](auto want_sum_tag) {
      constexpr bool kWantSum = decltype(want_sum_tag)::value;
      const int64_t bias = 8LL * static_cast<int64_t>(i_dim);
      for (size_t t = t_begin; t < t_end; ++t) {
        const size_t d = t / num_dg_tiles;
        const size_t dg0 = (t % num_dg_tiles) * kDGroupTile;
        const size_t dg_end = std::min(dg0 + kDGroupTile, dst_group_size);
        const size_t d_base_offset =
            is_linear_i4o4 ? d * src_slices * dst_group_size
                           : d * dst_group_size;
        int64_t tile_sums[kDGroupTile][4] = {};
        for (size_t s0 = 0; s0 < src_slices; s0 += kSTile) {
          const size_t s_end = std::min(s0 + kSTile, src_slices);
          alignas(64) uint8_t tile[kSTile][kDGroupTile * 8];
          for (size_t dg = dg0; dg < dg_end; ++dg) {
            const size_t dg_off = (dg - dg0) * 8;
            const size_t d_ch = 4 * (d * dst_group_size + dg);
            if (d_ch >= o_dim) {
              for (size_t s = s0; s < s_end; ++s) {
                std::fill_n(&tile[s - s0][dg_off], 8, uint8_t{0x88});
              }
              continue;
            }
            const uint8_t* r0 = src + (d_ch + 0) * row_bytes;
            const uint8_t* r1 = src + (d_ch + 1) * row_bytes;
            const uint8_t* r2 = src + (d_ch + 2) * row_bytes;
            const uint8_t* r3 = src + (d_ch + 3) * row_bytes;
            int64_t s_acc0 = 0;
            int64_t s_acc1 = 0;
            int64_t s_acc2 = 0;
            int64_t s_acc3 = 0;
            for (size_t s = s0; s < s_end; ++s) {
              const size_t col = s * 2;
              const uint8_t r0_b0 = r0[col];
              const uint8_t r0_b1 = r0[col + 1];
              const uint8_t r1_b0 = r1[col];
              const uint8_t r1_b1 = r1[col + 1];
              const uint8_t r2_b0 = r2[col];
              const uint8_t r2_b1 = r2[col + 1];
              const uint8_t r3_b0 = r3[col];
              const uint8_t r3_b1 = r3[col + 1];
              const uint8_t o0 = static_cast<uint8_t>(((r1_b0 & 0x0Fu) << 4) |
                                                      (r0_b0 & 0x0Fu)) ^
                                 0x88u;
              const uint8_t o1 = static_cast<uint8_t>(((r3_b0 & 0x0Fu) << 4) |
                                                      (r2_b0 & 0x0Fu)) ^
                                 0x88u;
              const uint8_t o2 =
                  static_cast<uint8_t>((r1_b0 & 0xF0u) | (r0_b0 >> 4)) ^ 0x88u;
              const uint8_t o3 =
                  static_cast<uint8_t>((r3_b0 & 0xF0u) | (r2_b0 >> 4)) ^ 0x88u;
              const uint8_t o4 = static_cast<uint8_t>(((r1_b1 & 0x0Fu) << 4) |
                                                      (r0_b1 & 0x0Fu)) ^
                                 0x88u;
              const uint8_t o5 = static_cast<uint8_t>(((r3_b1 & 0x0Fu) << 4) |
                                                      (r2_b1 & 0x0Fu)) ^
                                 0x88u;
              const uint8_t o6 =
                  static_cast<uint8_t>((r1_b1 & 0xF0u) | (r0_b1 >> 4)) ^ 0x88u;
              const uint8_t o7 =
                  static_cast<uint8_t>((r3_b1 & 0xF0u) | (r2_b1 >> 4)) ^ 0x88u;
              uint8_t* tile_out = &tile[s - s0][dg_off];
              tile_out[0] = o0;
              tile_out[1] = o1;
              tile_out[2] = o2;
              tile_out[3] = o3;
              tile_out[4] = o4;
              tile_out[5] = o5;
              tile_out[6] = o6;
              tile_out[7] = o7;
              if constexpr (kWantSum) {
                s_acc0 +=
                    (o0 & 0x0Fu) + (o2 & 0x0Fu) + (o4 & 0x0Fu) + (o6 & 0x0Fu);
                s_acc1 += (o0 >> 4) + (o2 >> 4) + (o4 >> 4) + (o6 >> 4);
                s_acc2 +=
                    (o1 & 0x0Fu) + (o3 & 0x0Fu) + (o5 & 0x0Fu) + (o7 & 0x0Fu);
                s_acc3 += (o1 >> 4) + (o3 >> 4) + (o5 >> 4) + (o7 >> 4);
              }
            }
            if constexpr (kWantSum) {
              const size_t local = dg - dg0;
              tile_sums[local][0] += s_acc0;
              tile_sums[local][1] += s_acc1;
              tile_sums[local][2] += s_acc2;
              tile_sums[local][3] += s_acc3;
            }
          }
          // Flush each slice row contiguously across `dg` to avoid strided
          // cache-line write amplification.
          const size_t row_out_bytes = (dg_end - dg0) * 8;
          for (size_t s = s0; s < s_end; ++s) {
            uint8_t* out =
                dst_ptr + (d_base_offset + s * slice_stride + dg0) * 8;
            std::memcpy(out, &tile[s - s0][0], row_out_bytes);
          }
        }
        if constexpr (kWantSum) {
          for (size_t dg = dg0; dg < dg_end; ++dg) {
            const size_t d_ch = 4 * (d * dst_group_size + dg);
            if (d_ch >= o_dim) continue;
            const size_t local = dg - dg0;
            sum_i[d_ch + 0] = static_cast<int32_t>(tile_sums[local][0] - bias);
            sum_i[d_ch + 1] = static_cast<int32_t>(tile_sums[local][1] - bias);
            sum_i[d_ch + 2] = static_cast<int32_t>(tile_sums[local][2] - bias);
            sum_i[d_ch + 3] = static_cast<int32_t>(tile_sums[local][3] - bias);
          }
        }
      }
    };
    if (want_sum_i) {
      run(std::true_type{});
    } else {
      run(std::false_type{});
    }
  };

  constexpr size_t kParallelMinElements = 512ULL << 10;
  if (max_threads > 1 && total_dg_tiles > 1 &&
      o_dim * i_dim >= kParallelMinElements &&
      !internal::InOuterParallelPrepackWorker()) {
    ParallelFor(total_dg_tiles, /*min_grain=*/1, process_tile_range,
                max_threads);
  } else {
    process_tile_range(0, total_dg_tiles);
  }
  if (want_sum_i) *sum_i_out = std::move(sum_i);
  return true;
}

// Cache-tiled, zero-copy rearrangement of packed INT2 OHWI weights into ML
// Drift's UINT2 (+2 shifted, 0xAA padded) I4O4 layouts (`kOSpatialIOGroupI4O4`
// and `k2DYIsSpatialIOAndXIsOGroupI4O4`) when `h == 1` and `w == 1`. Returns
// false otherwise so callers fall back to `RearrangeWeightsInt8AsUint2`.
inline bool TryDirectRearrangePackedInt2AsUint2I4O4(
    const uint8_t* src, size_t src_bytes, const ml_drift::OHWI& ohwi,
    const ml_drift::WeightsDescription& desc, absl::Span<uint8_t> dst,
    std::optional<std::vector<int32_t>>* sum_i_out = nullptr,
    size_t max_threads = 16) {
  if (ohwi.h != 1 || ohwi.w != 1 || ohwi.o <= 0 || ohwi.i <= 0) {
    return false;
  }
  const bool is_linear_i4o4 =
      desc.layout == ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  const bool is_tex_i4o4 =
      desc.layout == ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
  if (!is_linear_i4o4 && !is_tex_i4o4) return false;

  const size_t o_dim = static_cast<size_t>(ohwi.o);
  const size_t i_dim = static_cast<size_t>(ohwi.i);
  const size_t total_elems = o_dim * i_dim;
  if (src_bytes < (total_elems + 3) / 4) return false;

  const int dst_group_size_int = desc.GetOutputGroupSize();
  if (dst_group_size_int <= 0) return false;
  const size_t dst_group_size = static_cast<size_t>(dst_group_size_int);
  const size_t dst_slices =
      static_cast<size_t>(ml_drift::DivideRoundUp(ohwi.o, 4));
  const size_t src_slices =
      static_cast<size_t>(ml_drift::DivideRoundUp(ohwi.i, 4));
  const size_t dst_groups =
      (dst_slices + dst_group_size - 1) / dst_group_size;
  if (dst.size() < dst_groups * dst_group_size * src_slices * 4) return false;

  std::vector<int32_t> sum_i;
  const bool want_sum_i = (sum_i_out != nullptr);
  if (want_sum_i) sum_i.assign(o_dim, 0);

  uint8_t* dst_ptr = dst.data();
  const size_t slice_stride =
      is_linear_i4o4 ? dst_group_size : dst_groups * dst_group_size;
  const bool aligned_cols = (i_dim & 3u) == 0;
  const size_t row_bytes = i_dim >> 2;
  constexpr size_t kDGroupTile = 16;
  constexpr size_t kSTile = 64;  // 64 bytes per source row on aligned inputs.
  const size_t num_dg_tiles =
      (dst_group_size + kDGroupTile - 1) / kDGroupTile;
  const size_t total_dg_tiles = dst_groups * num_dg_tiles;

  auto get_crumb = [&](size_t linear_idx) -> uint8_t {
    const uint8_t b = src[linear_idx >> 2];
    return (b >> ((linear_idx & 3u) * 2)) & 0x03u;
  };
  auto sum_byte = [](uint8_t b) -> int64_t {
    const uint8_t u = b ^ 0xAAu;
    const uint8_t pair_sum =
        static_cast<uint8_t>((u & 0x33u) + ((u >> 2) & 0x33u));
    return static_cast<int64_t>(pair_sum & 0x0Fu) +
           static_cast<int64_t>(pair_sum >> 4);
  };

  auto process_tile_range = [&](size_t t_begin, size_t t_end) {
    auto run = [&](auto want_sum_tag) {
      constexpr bool kWantSum = decltype(want_sum_tag)::value;
      const int64_t bias = 2LL * static_cast<int64_t>(src_slices) * 4;
      for (size_t t = t_begin; t < t_end; ++t) {
        const size_t d = t / num_dg_tiles;
        const size_t dg0 = (t % num_dg_tiles) * kDGroupTile;
        const size_t dg_end = std::min(dg0 + kDGroupTile, dst_group_size);
        const size_t d_base_offset =
            is_linear_i4o4 ? d * src_slices * dst_group_size
                           : d * dst_group_size;
        int64_t tile_sums[kDGroupTile][4] = {};
        for (size_t s0 = 0; s0 < src_slices; s0 += kSTile) {
          const size_t s_end = std::min(s0 + kSTile, src_slices);
          alignas(64) uint8_t tile[kSTile][kDGroupTile * 4];
          for (size_t dg = dg0; dg < dg_end; ++dg) {
            const size_t dg_off = (dg - dg0) * 4;
            const size_t d_ch = 4 * (d * dst_group_size + dg);
            if (d_ch >= o_dim) {
              for (size_t s = s0; s < s_end; ++s) {
                std::fill_n(&tile[s - s0][dg_off], 4, uint8_t{0xAA});
              }
              continue;
            }
            int64_t s_acc0 = 0;
            int64_t s_acc1 = 0;
            int64_t s_acc2 = 0;
            int64_t s_acc3 = 0;
            if (aligned_cols && d_ch + 4 <= o_dim) {
              const uint8_t* r0 = src + (d_ch + 0) * row_bytes;
              const uint8_t* r1 = src + (d_ch + 1) * row_bytes;
              const uint8_t* r2 = src + (d_ch + 2) * row_bytes;
              const uint8_t* r3 = src + (d_ch + 3) * row_bytes;
              for (size_t s = s0; s < s_end; ++s) {
                const uint8_t b0 = r0[s];
                const uint8_t b1 = r1[s];
                const uint8_t b2 = r2[s];
                const uint8_t b3 = r3[s];
                uint8_t* tile_out = &tile[s - s0][dg_off];
                tile_out[0] = static_cast<uint8_t>(
                    (((b3 & 0x03u) << 6) | ((b2 & 0x03u) << 4) |
                     ((b1 & 0x03u) << 2) | (b0 & 0x03u)) ^
                    0xAAu);
                tile_out[1] = static_cast<uint8_t>(
                    (((b3 & 0x0Cu) << 4) | ((b2 & 0x0Cu) << 2) |
                     (b1 & 0x0Cu) | ((b0 >> 2) & 0x03u)) ^
                    0xAAu);
                tile_out[2] = static_cast<uint8_t>(
                    (((b3 & 0x30u) << 2) | (b2 & 0x30u) |
                     ((b1 >> 2) & 0x0Cu) | ((b0 >> 4) & 0x03u)) ^
                    0xAAu);
                tile_out[3] = static_cast<uint8_t>(
                    ((b3 & 0xC0u) | ((b2 >> 2) & 0x30u) |
                     ((b1 >> 4) & 0x0Cu) | (b0 >> 6)) ^
                    0xAAu);
                if constexpr (kWantSum) {
                  s_acc0 += sum_byte(b0);
                  s_acc1 += sum_byte(b1);
                  s_acc2 += sum_byte(b2);
                  s_acc3 += sum_byte(b3);
                }
              }
            } else {
              for (size_t s = s0; s < s_end; ++s) {
                const size_t s_base = s * 4;
                uint8_t* tile_out = &tile[s - s0][dg_off];
                for (size_t k = 0; k < 4; ++k) {
                  const size_t s_ch = s_base + k;
                  if (s_ch >= i_dim) {
                    tile_out[k] = 0xAAu;
                    if constexpr (kWantSum) {
                      s_acc0 += 2;
                      s_acc1 += 2;
                      s_acc2 += 2;
                      s_acc3 += 2;
                    }
                    continue;
                  }
                  const uint8_t v0 =
                      (d_ch + 0 < o_dim) ? get_crumb((d_ch + 0) * i_dim + s_ch)
                                         : 0u;
                  const uint8_t v1 =
                      (d_ch + 1 < o_dim) ? get_crumb((d_ch + 1) * i_dim + s_ch)
                                         : 0u;
                  const uint8_t v2 =
                      (d_ch + 2 < o_dim) ? get_crumb((d_ch + 2) * i_dim + s_ch)
                                         : 0u;
                  const uint8_t v3 =
                      (d_ch + 3 < o_dim) ? get_crumb((d_ch + 3) * i_dim + s_ch)
                                         : 0u;
                  tile_out[k] = static_cast<uint8_t>(
                      ((v3 << 6) | (v2 << 4) | (v1 << 2) | v0) ^ 0xAAu);
                  if constexpr (kWantSum) {
                    s_acc0 += v0 ^ 0x02u;
                    s_acc1 += v1 ^ 0x02u;
                    s_acc2 += v2 ^ 0x02u;
                    s_acc3 += v3 ^ 0x02u;
                  }
                }
              }
            }
            if constexpr (kWantSum) {
              const size_t local = dg - dg0;
              tile_sums[local][0] += s_acc0;
              tile_sums[local][1] += s_acc1;
              tile_sums[local][2] += s_acc2;
              tile_sums[local][3] += s_acc3;
            }
          }
          const size_t row_out_bytes = (dg_end - dg0) * 4;
          for (size_t s = s0; s < s_end; ++s) {
            uint8_t* out =
                dst_ptr + (d_base_offset + s * slice_stride + dg0) * 4;
            std::memcpy(out, &tile[s - s0][0], row_out_bytes);
          }
        }
        if constexpr (kWantSum) {
          for (size_t dg = dg0; dg < dg_end; ++dg) {
            const size_t d_ch = 4 * (d * dst_group_size + dg);
            if (d_ch >= o_dim) continue;
            const size_t local = dg - dg0;
            if (d_ch + 0 < o_dim) {
              sum_i[d_ch + 0] =
                  static_cast<int32_t>(tile_sums[local][0] - bias);
            }
            if (d_ch + 1 < o_dim) {
              sum_i[d_ch + 1] =
                  static_cast<int32_t>(tile_sums[local][1] - bias);
            }
            if (d_ch + 2 < o_dim) {
              sum_i[d_ch + 2] =
                  static_cast<int32_t>(tile_sums[local][2] - bias);
            }
            if (d_ch + 3 < o_dim) {
              sum_i[d_ch + 3] =
                  static_cast<int32_t>(tile_sums[local][3] - bias);
            }
          }
        }
      }
    };
    if (want_sum_i) {
      run(std::true_type{});
    } else {
      run(std::false_type{});
    }
  };

  constexpr size_t kParallelMinElements = 512ULL << 10;
  if (max_threads > 1 && total_dg_tiles > 1 &&
      total_elems >= kParallelMinElements &&
      !internal::InOuterParallelPrepackWorker()) {
    ParallelFor(total_dg_tiles, /*min_grain=*/1, process_tile_range,
                max_threads);
  } else {
    process_tile_range(0, total_dg_tiles);
  }
  if (want_sum_i) *sum_i_out = std::move(sum_i);
  return true;
}

// Fast zero-copy equivalent of `ml_drift::ScaleOrZeroPointToTensorDesc` +
// `UpdateToSupportedStorageType` for 2-D blockwise scales/zero-points
// (`OHWI(O, 1, 1, num_blocks)` with `num_blocks > 1`). Avoids copying
// `raw_values` and avoids materializing the intermediate `OHIToHOIO4` tensor.
inline ml_drift::TensorDescriptor BuildBlockwiseScaleOrZpDescriptor(
    const ml_drift::GpuInfo& gpu_info, absl::Span<const float> raw_values,
    const ml_drift::OHWI& shape, ml_drift::DataType dst_data_type,
    std::optional<ml_drift::TensorStorageType> default_2d_storage =
        std::nullopt) {
  if (shape.h != 1 || shape.w != 1 || shape.i <= 1 ||
      (dst_data_type != ml_drift::DataType::kFloat16 &&
       dst_data_type != ml_drift::DataType::kFloat32)) {
    ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kFloat32> src;
    src.shape = shape;
    src.spanned_data = absl::MakeSpan(const_cast<float*>(raw_values.data()),
                                      raw_values.size());
    ml_drift::TensorDescriptor td =
        ml_drift::ScaleOrZeroPointToTensorDesc(gpu_info, src, dst_data_type);
    td.UpdateToSupportedStorageType(gpu_info, td.GetBHWDCShape()).IgnoreError();
    return td;
  }
  const int o_ch = shape.o;
  const int num_blocks = shape.i;
  const int o_slices = ml_drift::DivideRoundUp(o_ch, 4);
  const ml_drift::BHWC hwc_shape(1, 1, o_slices, 4 * num_blocks);
  if (!default_2d_storage.has_value()) {
    float probe_vals[8] = {};
    ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kFloat32> probe;
    probe.shape = ml_drift::OHWI(4, 1, 1, 2);
    probe.spanned_data = absl::MakeSpan(probe_vals, 8);
    default_2d_storage =
        ml_drift::ScaleOrZeroPointToTensorDesc(gpu_info, probe, dst_data_type)
            .GetStorageType();
  }
  const ml_drift::TensorStorageType storage_type =
      ml_drift::TensorDescriptor(dst_data_type, *default_2d_storage,
                                 ml_drift::Layout::kHWC)
              .CanCreateTensorWithShape(gpu_info, hwc_shape)
              .ok()
          ? *default_2d_storage
          : ml_drift::TensorStorageType::kBuffer;
  ml_drift::TensorDescriptor td(dst_data_type, storage_type,
                                ml_drift::Layout::kHWC);
  td.SetBHWCShape(hwc_shape);
  td.UpdateToSupportedStorageType(gpu_info, td.GetBHWDCShape()).IgnoreError();

  // For `b == 1 && h == 1 && d == 1` and default physical layouts, both
  // `DSHWBC4` (`kBuffer`/`kImageBuffer`/`kTextureArray`/`kTexture3D`) and
  // `HSWBDC4` (`kTexture2D`) place `(o_slice, block, ch)` at
  // `(block * o_slices + o_slice) * 4 + ch`.
  const size_t total_elems =
      static_cast<size_t>(num_blocks) * static_cast<size_t>(o_slices) * 4;
  std::vector<uint8_t> packed(
      total_elems * static_cast<size_t>(ml_drift::SizeOf(dst_data_type)));
  constexpr int kOTile = 16;
  constexpr int kBTile = 32;
  auto pack_typed = [&](auto* dst) {
    using DstT = std::remove_pointer_t<decltype(dst)>;
    for (int os0 = 0; os0 < o_slices; os0 += kOTile) {
      const int os_end = std::min(os0 + kOTile, o_slices);
      for (int b0 = 0; b0 < num_blocks; b0 += kBTile) {
        const int b_end = std::min(b0 + kBTile, num_blocks);
        for (int os = os0; os < os_end; ++os) {
          const int base_o = os * 4;
          if (base_o + 3 < o_ch) {
            const float* r0 =
                raw_values.data() +
                static_cast<size_t>(base_o + 0) * num_blocks;
            const float* r1 =
                raw_values.data() +
                static_cast<size_t>(base_o + 1) * num_blocks;
            const float* r2 =
                raw_values.data() +
                static_cast<size_t>(base_o + 2) * num_blocks;
            const float* r3 =
                raw_values.data() +
                static_cast<size_t>(base_o + 3) * num_blocks;
            for (int b = b0; b < b_end; ++b) {
              DstT* out =
                  dst + (static_cast<size_t>(b) * o_slices + os) * 4;
              out[0] = DstT(r0[b]);
              out[1] = DstT(r1[b]);
              out[2] = DstT(r2[b]);
              out[3] = DstT(r3[b]);
            }
          } else {
            for (int b = b0; b < b_end; ++b) {
              DstT* out =
                  dst + (static_cast<size_t>(b) * o_slices + os) * 4;
              for (int ch = 0; ch < 4; ++ch) {
                const int o = base_o + ch;
                out[ch] = DstT(
                    o < o_ch
                        ? raw_values[static_cast<size_t>(o) * num_blocks + b]
                        : 0.0f);
              }
            }
          }
        }
      }
    }
  };
  if (dst_data_type == ml_drift::DataType::kFloat16) {
    pack_typed(reinterpret_cast<ml_drift::half*>(packed.data()));
  } else {
    pack_typed(reinterpret_cast<float*>(packed.data()));
  }
  td.SetData(std::move(packed));
  return td;
}

inline bool SupportsDirectI4O4Pack(const ml_drift::WeightsDescription& desc,
                                   const ml_drift::OHWI& ohwi, bool is_int2) {
  if (ohwi.h != 1 || ohwi.w != 1 || ohwi.o <= 0 || ohwi.i <= 0 ||
      desc.GetOutputGroupSize() <= 0) {
    return false;
  }
  if (desc.layout != ml_drift::WeightsLayout::kOSpatialIOGroupI4O4 &&
      desc.layout !=
          ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    return false;
  }
  if (is_int2) {
    return desc.type == ml_drift::DataType::kUint2 ||
           desc.type == ml_drift::DataType::kInt2;
  }
  return (ohwi.o % 4 == 0) && (ohwi.i % 4 == 0) &&
         (desc.type == ml_drift::DataType::kUint4 ||
          desc.type == ml_drift::DataType::kInt4);
}

// Rearranges the dense signed INT4 weights `packed` (two per byte, OHWI
// `ohwi`) into the kernel layout `desc`. The I4O4 layouts rearrange straight
// from the packed values (and accumulate `sum_i` in the same pass), like
// `GpuModelBuilder` does for packed INT4 FullyConnected weights: one copy of
// the packed bytes instead of three full-size INT8 intermediates (~1.9 GB for
// a 262144 x 2048 tied head). Other layouts unpack to INT8 first.
inline absl::StatusOr<RearrangedInt4Weights> RearrangeInt4Weights(
    absl::Span<const int8_t> packed, const ml_drift::OHWI& ohwi,
    const ml_drift::WeightsDescription& desc, int elements_count,
    bool want_sum_i, size_t max_threads = 16) {
  if (desc.type != ml_drift::DataType::kUint4 &&
      desc.type != ml_drift::DataType::kInt4) {
    return absl::UnimplementedError(
        "Only INT4 weight rearrangement is supported in pre-rearrange for "
        "now.");
  }
  const size_t num_elements = ohwi.DimensionsProduct();
  const size_t packed_bytes = (num_elements + 1) / 2;
  if (packed.size() < packed_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "INT4 weights buffer has ", packed.size(), " bytes, expected ",
        packed_bytes, " for ", num_elements, " values."));
  }
  RearrangedInt4Weights out;
  out.data.resize(
      (static_cast<size_t>(elements_count) * ml_drift::SizeInBitsOf(desc.type) +
       7) /
      8);
  if (desc.layout == ml_drift::WeightsLayout::kOSpatialIOGroupI4O4 ||
      desc.layout == ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    const auto* raw_u8 = reinterpret_cast<const uint8_t*>(packed.data());
    if (TryDirectRearrangePackedInt4AsUint4I4O4(
            raw_u8, packed_bytes, ohwi, desc, absl::MakeSpan(out.data),
            want_sum_i ? &out.sum_i : nullptr, max_threads)) {
      return out;
    }
    ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kUint8> bytes;
    bytes.shape = ohwi;
    bytes.spanned_data =
        absl::MakeSpan(const_cast<uint8_t*>(raw_u8), packed_bytes);
    // The converter always writes `sum_i`, 4 output channels at a time.
    std::vector<int32_t> sum_i((ohwi.o + 3) / 4 * 4, 0);
    LRT_TENSOR_RETURN_IF_ERROR(ml_drift::RearrangeWeightsUInt4Packed(
        bytes, desc, absl::MakeSpan(out.data), absl::MakeSpan(sum_i),
        /*pad_value=*/8, /*swap_dims=*/false));
    if (want_sum_i) {
      sum_i.resize(ohwi.o);
      out.sum_i = std::move(sum_i);
    }
    return out;
  }
  ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kInt8> unpacked;
  unpacked.shape = ohwi;
  unpacked.data.assign(num_elements + 64, 0);
  ml_drift::UnpackDenseInt4IntoInt8(packed.data(), num_elements,
                                    unpacked.data.data());
  if (want_sum_i) {
    out.sum_i = ComputeWeightsSumI(
        ohwi, absl::MakeConstSpan(unpacked.data).first(num_elements));
  }
  ml_drift::RearrangeWeightsInt8AsUint4(unpacked, desc,
                                        absl::MakeSpan(out.data), 8, 8u);
  return out;
}

// Rearranges dense signed INT2 weights `packed` (four per byte, OHWI `ohwi`)
// into the kernel layout `desc`. I4O4 layouts rearrange directly from packed
// bytes (and optionally accumulate `sum_i` in the same pass); other layouts
// unpack to INT8 first.
inline absl::StatusOr<RearrangedInt4Weights> RearrangeInt2Weights(
    absl::Span<const uint8_t> packed, const ml_drift::OHWI& ohwi,
    const ml_drift::WeightsDescription& desc, int elements_count,
    bool want_sum_i, size_t max_threads = 16) {
  if (desc.type != ml_drift::DataType::kUint2 &&
      desc.type != ml_drift::DataType::kInt2) {
    return absl::UnimplementedError(
        "Only INT4 and INT2 weight rearrangement is supported in "
        "pre-rearrange.");
  }
  const size_t num_elements = ohwi.DimensionsProduct();
  const size_t packed_bytes = (num_elements + 3) / 4;
  if (packed.size() < packed_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "INT2 weights buffer has ", packed.size(), " bytes, expected ",
        packed_bytes, " for ", num_elements, " values."));
  }
  RearrangedInt4Weights out;
  out.data.resize(
      (static_cast<size_t>(elements_count) * ml_drift::SizeInBitsOf(desc.type) +
       7) /
      8);
  if (TryDirectRearrangePackedInt2AsUint2I4O4(
          packed.data(), packed.size(), ohwi, desc, absl::MakeSpan(out.data),
          want_sum_i ? &out.sum_i : nullptr, max_threads)) {
    return out;
  }
  if (want_sum_i) {
    LRT_TENSOR_ASSIGN_OR_RETURN(out.sum_i,
                                ComputePackedInt2WeightsSumI(packed, ohwi));
  }
  std::vector<int8_t> unpacked(num_elements + 64, 0);
  for (size_t i = 0; i < num_elements; ++i) {
    const size_t byte_idx = i >> 2;
    if (byte_idx < packed.size()) {
      int8_t q = (packed[byte_idx] >> ((i & 3u) * 2)) & 0x03;
      if (q & 0x02) q |= 0xFC;
      unpacked[i] = q;
    }
  }
  ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kInt8> weights;
  weights.shape = ohwi;
  weights.data = std::move(unpacked);
  ml_drift::RearrangeWeightsInt8AsUint2(weights, desc,
                                        absl::MakeSpan(out.data), 2, 2u);
  return out;
}

// Whether `Backend` has `CreateConstTensorFromHostData` (see
// `MlDriftClBackend`).
template <typename Backend, typename = void>
struct HasCreateConstTensorFromHostData : std::false_type {};
template <typename Backend>
struct HasCreateConstTensorFromHostData<
    Backend, std::void_t<decltype(&Backend::CreateConstTensorFromHostData)>>
    : std::true_type {};

}  // namespace ml_drift_runner_internal

template <typename Backend>
class CompiledMlDriftModelRunner {
 public:
  using Environment = typename Backend::Environment;
  using InferenceContext = typename Backend::InferenceContext;
  using BackendTensor = typename Backend::Tensor;

  // Cross-signature cache of packed constant weight descriptors and allocated
  // GPU weight tensors. Passing the same `SharedWeightsCache` across prefill
  // and decode `CompiledMlDriftModelRunner` instances sharing an `Environment`
  // avoids re-packing weights on the CPU and avoids allocating duplicate
  // constant weight buffers/textures on the GPU.
  struct SharedWeightsCache {
    // Guards concurrent `BuildModel()` lookups and uploads when a backend
    // finalizes multiple passes in parallel (e.g. Metal `Warmup()`).
    absl::Mutex mutex;
    absl::flat_hash_map<std::string, ml_drift::TensorDescriptor> descriptors;
    // Optional: the packed bytes of `descriptors` without data, by weight
    // name, in host memory that `owner` keeps alive (e.g. memory-mapped from
    // a persistent weight cache). Uploaded like descriptor data and dropped
    // once uploaded, unless the GPU tensor is created over them (see
    // `zero_copy_host_data`).
    struct HostData {
      absl::Span<const uint8_t> data;
      // Type-erased owner of the memory of `data` (e.g. a memory mapping):
      // `std::shared_ptr<const void>` keeps the deleter of the object it was
      // created from, so the last reference destroys that object (and so
      // unmaps the memory) whatever its type.
      std::shared_ptr<const void> owner;
      // True when `owner` guarantees that the whole OS VM pages enclosing
      // `data` belong to this mapping (e.g. a file `mmap`), allowing backends
      // like Metal's `newBufferWithBytesNoCopy` to wrap page-aligned bounds.
      bool owns_enclosing_pages = false;
    };
    absl::flat_hash_map<std::string, HostData> host_data;
    // Creates the GPU tensors of `host_data` over the host memory itself
    // instead of copying it, where the backend and GPU allow it:
    // `Backend::CreateConstTensorFromHostData` wraps the memory when it can
    // (e.g. OpenCL `CL_MEM_USE_HOST_PTR` buffers, `Backend::WrapHostData`)
    // and copies it otherwise. On GPUs that share memory with the CPU
    // (unified memory, e.g. mobile SoCs), the GPU then reads the weights
    // straight from the (memory-mapped) host pages, without a second copy.
    bool zero_copy_host_data = false;
    // Keeps alive, for each tensor created over `host_data`, the host memory
    // and the backend objects wrapping it (type-erased, see
    // `HostData::owner`). Declared before `tensors`, which must be destroyed
    // first.
    std::vector<std::shared_ptr<const void>> zero_copy_owners;
    absl::flat_hash_map<std::string, std::unique_ptr<BackendTensor>> tensors;
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    // When true, runners drop the host buffers of fused constant weights
    // (`+`-joined names, e.g. from `TryConcatConstantFcWeights`) once this
    // cache holds their packed descriptor, and the packed data of the
    // `descriptors` whose GPU tensors exist (`BuildModel()`). That mutates the
    // graph passed to the runner, so only set it when every graph is compiled
    // with this cache (e.g. a decoder runtime that builds a fresh graph for
    // every pass).
    bool release_fused_host_buffers = false;
    // Optional: called with the packed descriptor (with data) of every weight
    // when its GPU tensor is created, e.g. to persist the packed weights for
    // later runs (which then pre-populate `descriptors`).
    absl::AnyInvocable<void(const std::string& name,
                            const ml_drift::TensorDescriptor& desc)>
        on_upload;
    // Optional: called serially on the compiling thread with each packed
    // constant weight descriptor (with data) as soon as `Prepare()` produces
    // it. When it returns true, the caller persisted the bytes (e.g. to a
    // weight cache that will be mapped into `host_data` before `BuildModel()`),
    // so the runner drops the descriptor's heap bytes immediately and streams
    // INT4 packing in bounded-memory batches instead of holding every packed
    // weight in RAM at once.
    absl::AnyInvocable<bool(const std::string& name,
                            const ml_drift::TensorDescriptor& desc)>
        on_pack;
  };

  // Host wall time of the compilation phases of `Prepare()` and
  // `BuildModel()`. The GPU weight conversion (`SetConvertWeightsOnGpu`,
  // `RunGpuWeightConversions`) runs between `build_gpu_model` and
  // `upload_shared_weights` and is in neither: it is reported in
  // `weight_pack_stats()["gpu_convert"]` and counted in `compilation_time()`.
  struct CompileTiming {
    // `RearrangeAndMapConstantWeights()`: packing constant FullyConnected /
    // EmbeddingLookup weights into kernel layouts on the CPU (see
    // `weight_pack_stats()` for the split by kind).
    absl::Duration rearrange_weights;
    // `AppendGpuModelBuilder()`: lowering the graph ops to ML Drift operations
    // (including the weight packing of ops that pack their own weights).
    absl::Duration lower_ops;
    // `GpuModelBuilder::GetGpuModel()` alone: merging (fusing) the nodes,
    // generating the kernel source code and resolving the kernel arguments
    // into the `GpuModel`. CPU only, no GPU work.
    absl::Duration build_gpu_model;
    // Creating the GPU tensors of the `SharedWeightsCache` weights that are
    // not in the cache yet (host -> GPU copies), and their bytes. The bytes
    // also count the `host_data` that `zero_copy_host_data` wrapped without
    // copying (see `num_zero_copy_tensors`). Only these are uploaded here:
    // without a `SharedWeightsCache` (and for the other constants)
    // `InitFromGpuModel` uploads the constants, which is counted in
    // `init_inference_context`.
    absl::Duration upload_shared_weights;
    size_t upload_shared_weights_bytes = 0;
    // Shared weights whose GPU tensor was created over their host memory
    // (`SharedWeightsCache::zero_copy_host_data`).
    size_t num_zero_copy_tensors = 0;
    // `InitFromGpuModel`: memory allocation, upload of the remaining
    // constants, kernel compilation and tuning.
    absl::Duration init_inference_context;
    size_t num_nodes = 0;
    // Nodes that convert weights to another layout on every run (e.g. when a
    // kernel needs another layout than the packed weights).
    size_t num_nodes_to_convert_weights = 0;
  };

  CompiledMlDriftModelRunner(std::vector<TensorHandle> flat_inputs,
                             std::vector<TensorHandle> flat_outputs,
                             ml_drift::CalculationsPrecision precision =
                                 ml_drift::CalculationsPrecision::kF32,
                             Environment* shared_env = nullptr,
                             SharedWeightsCache* shared_weights = nullptr,
                             bool allow_8bit_convs = false)
      : flat_inputs_(std::move(flat_inputs)),
        flat_outputs_(std::move(flat_outputs)),
        shared_weights_(shared_weights),
        allow_8bit_convs_(allow_8bit_convs || Allow8BitConvsFromEnv()) {
    if (shared_env != nullptr) {
      env_ptr_ = shared_env;
      env_initialized_ = true;
    }
    ABSL_CHECK_OK(EnsureEnvInitialized());

    gpu_model_info_ = {
        .precision = precision,
        .storage_type =
            Backend::GetStorageTypeForBuilder(*env_ptr_, env_initialized_),
    };
    const char* fast_tuning_env = std::getenv("LRT_FAST_TUNING");
    if (fast_tuning_env == nullptr ||
        std::string(fast_tuning_env) != "false") {
      gpu_model_info_.hints.Add(ml_drift::ModelHints::kFastTuning);
    }
    gpu_model_info_.hints.Add(ml_drift::ModelHints::kPreferTextureWeights);
    gpu_model_info_.hints.Add(ml_drift::ModelHints::kEnableHostMappedPointer);
    // `LRT_ALLOW_8BIT_CONVS=1` lets ML Drift run large quantized
    // FullyConnected / convolutions with dynamically INT8-quantized
    // activations and INT8 dot products.
    if (!allow_8bit_convs_) {
      gpu_model_info_.hints.Add(ml_drift::ModelHints::kDisallow8bitConvs);
    }
    gpu_model_info_.hints.Add(ml_drift::ModelHints::kAllowSpecialKernels);
    const char* allow_cb_env = std::getenv("LRT_ALLOW_COMMAND_BUFFER");
    gpu_model_info_.hints.allow_cl_khr_command_buffer =
        (allow_cb_env == nullptr) || (std::string(allow_cb_env) == "true");
    const ml_drift::GpuInfo gpu_info =
        Backend::GetGpuInfoForBuilder(*env_ptr_, env_initialized_);

    ABSL_LOG(INFO) << "Runner constructor: precision="
                   << static_cast<int>(precision)
                   << ", env_initialized=" << env_initialized_
                   << ", storage_type="
                   << static_cast<int>(gpu_model_info_.storage_type);
    ABSL_LOG(INFO) << "GpuInfo: vendor=" << static_cast<int>(gpu_info.vendor)
                   << ", api=" << static_cast<int>(gpu_info.gpu_api)
                   << ", wave_sizes_size="
                   << gpu_info.supported_wave_sizes.size()
                   << ", matmul_ops_size=" << gpu_info.wave_mat_mul_ops.size();
    ABSL_LOG(INFO) << "OpenClInfo: device_name="
                   << gpu_info.opencl_info.device_name << ", extensions_size="
                   << gpu_info.opencl_info.extensions.size();
    ABSL_LOG(INFO) << "Creating model builder";
    model_builder_ = std::make_unique<ml_drift::GpuModelBuilder>(
        gpu_info, gpu_model_info_.hints, gpu_model_info_.precision,
        gpu_model_info_.storage_type);
    conversion_ctx_.gpu_model_builder = model_builder_.get();
    conversion_ctx_.allow_8bit_convs = allow_8bit_convs_;
    conversion_ctx_.hints = gpu_model_info_.hints;
    conversion_ctx_.default_float_type =
        ml_drift::DeduceDataTypeFromPrecision(gpu_model_info_.precision);
    conversion_ctx_.shared_weight_descriptors =
        shared_weights_ != nullptr ? &shared_weights_->descriptors : nullptr;
  }

  // When true and `ml_drift::WeightsManager::IsGpuWeightsPreparationSupported`
  // (OpenCL only for now), constant INT4 / INT8 / INT2 FullyConnected and
  // EmbeddingLookup weights are uploaded as raw OHWI values and rearranged
  // into the kernel layout (and `sum_i`) by a one-shot ML Drift conversion
  // model on the GPU instead of on the CPU. Produces the same bytes as the CPU
  // path. Off by default. Set before `Prepare()`.
  void SetConvertWeightsOnGpu(bool enable) { convert_weights_on_gpu_ = enable; }
  // Tests: converts on the GPU even where ML Drift disables it (e.g. a GPU
  // info faked to select other weight layouts).
  void ForceGpuWeightConversionForTesting() {
    convert_weights_on_gpu_ = true;
    force_gpu_weight_conversion_ = true;
  }
  void SetMaxGpuWeightConversionBatchBytesForTesting(size_t bytes) {
    max_gpu_weight_conversion_batch_bytes_ = bytes;
  }

  void SetSharedWeightsCache(SharedWeightsCache* shared_weights) {
    shared_weights_ = shared_weights;
    conversion_ctx_.shared_weight_descriptors =
        shared_weights_ != nullptr ? &shared_weights_->descriptors : nullptr;
  }

  absl::Status Prepare(
      const absl::flat_hash_set<std::string>& external_weights = {}) {
    if (prepared_) return absl::OkStatus();
    conversion_ctx_.external_weight_names = external_weights;
    conversion_ctx_.persists_shared_weights =
        shared_weights_ != nullptr && shared_weights_->on_upload != nullptr;
    name_to_tensor_map_.clear();
    const absl::Time rearrange_start = absl::Now();
    LRT_TENSOR_RETURN_IF_ERROR(RearrangeAndMapConstantWeights());
    compile_timing_.rearrange_weights = absl::Now() - rearrange_start;
    const absl::Time lower_start = absl::Now();
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto tensor_to_handle,
        AppendGpuModelBuilder(conversion_ctx_, flat_outputs_));
    compile_timing_.lower_ops = absl::Now() - lower_start;

    conversion_ctx_.graph_tensor_to_handle = std::move(tensor_to_handle);

    for (const auto& handle : flat_outputs_) {
      LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info,
                                  graph::GetInfo(handle.GetRaw()));
      if (!info.name.empty()) {
        name_to_tensor_map_[info.name] = handle.GetRaw();
      }
    }
    for (const auto& handle : flat_inputs_) {
      LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info,
                                  graph::GetInfo(handle.GetRaw()));
      if (!info.name.empty()) {
        name_to_tensor_map_[info.name] = handle.GetRaw();
      }
    }
    prepared_ = true;
    return absl::OkStatus();
  }

  absl::Status BuildModel(
      const ml_drift::ExternalTensorsInfo& external_tensors = {}) {
    // `CaptureCompiledHandlesAndReleaseBuilder()` freed the model builder.
    if (build_state_released_) {
      return absl::FailedPreconditionError(
          "BuildModel() has already been called");
    }
    const absl::Time compilation_start = absl::Now();

    LRT_TENSOR_RETURN_IF_ERROR(Prepare());

    graph_input_ids_.clear();
    for (const auto& input_handle : flat_inputs_) {
      if (conversion_ctx_.graph_tensor_to_handle.contains(
              input_handle.GetRaw())) {
        graph_input_ids_.push_back(
            conversion_ctx_.graph_tensor_to_handle.at(input_handle.GetRaw())
                .id);
      }
    }

    graph_output_ids_.clear();
    for (const auto& output_handle : flat_outputs_) {
      if (conversion_ctx_.graph_tensor_to_handle.contains(
              output_handle.GetRaw())) {
        graph_output_ids_.push_back(
            conversion_ctx_.graph_tensor_to_handle.at(output_handle.GetRaw())
                .id);
      }
    }

    const absl::Time gpu_model_start = absl::Now();
    LRT_TENSOR_RETURN_IF_ERROR(conversion_ctx_.gpu_model_builder->GetGpuModel(
        graph_input_ids_, graph_output_ids_, &gpu_model_));
    compile_timing_.build_gpu_model = absl::Now() - gpu_model_start;

    compile_timing_.num_nodes_to_convert_weights = 0;
    for (const auto& node : gpu_model_.nodes) {
      if (absl::StartsWith(node.name, "weights_convert")) {
        ++compile_timing_.num_nodes_to_convert_weights;
      }
    }

    ml_drift::ExternalTensorsInfo combined_external_tensors = external_tensors;
    LRT_TENSOR_RETURN_IF_ERROR(
        RunGpuWeightConversions(&combined_external_tensors));
    // `GpuModelBuilder` is no longer needed once `gpu_model_` is finalized and
    // GPU weight conversions have run; release it before uploading weights and
    // initializing the inference context to reduce peak host memory.
    conversion_ctx_.gpu_model_builder = nullptr;
    model_builder_.reset();

    const absl::Time upload_start = absl::Now();
    if (shared_weights_ != nullptr) {
      absl::MutexLock lock(shared_weights_->mutex);
      for (const auto& [weight_name, handle] :
           conversion_ctx_.weight_name_to_handle) {
        if (conversion_ctx_.external_weight_names.contains(weight_name)) {
          continue;
        }
        auto const_it = gpu_model_.const_tensors.find(handle.id);
        if (const_it == gpu_model_.const_tensors.end()) {
          continue;
        }
        auto tensor_it = shared_weights_->tensors.find(weight_name);
        if (tensor_it == shared_weights_->tensors.end()) {
          const ml_drift::TensorDescriptor* upload_desc = &const_it->second;
          auto shared_desc_it = shared_weights_->descriptors.find(weight_name);
          if (shared_desc_it != shared_weights_->descriptors.end() &&
              !shared_desc_it->second.GetData().empty()) {
            upload_desc = &shared_desc_it->second;
          }
          auto host_it = shared_weights_->host_data.end();
          if (upload_desc->GetData().empty()) {
            host_it = shared_weights_->host_data.find(weight_name);
          }
          if (upload_desc->GetData().empty() &&
              host_it == shared_weights_->host_data.end() &&
              upload_desc->GetMemorySizeInBytes() > 0) {
            // Neither the model nor the cache has data (or `host_data`) for
            // these weights and the cache has no GPU tensor for them. The
            // runners never leave the cache like that (a descriptor loses its
            // data only when its GPU tensor is added, and the weights this
            // runner converts on the GPU are bound above, without publishing
            // dataless descriptors in `Prepare()`), so this only triggers
            // when the caller removed entries of `SharedWeightsCache::tensors`
            // or added `descriptors` without data or `host_data`. Uploading
            // would create an uninitialized tensor.
            return absl::FailedPreconditionError(
                absl::StrCat("No data for the weights '", weight_name, "'"));
          }
          auto gpu_tensor = std::make_unique<BackendTensor>();
          if (host_it != shared_weights_->host_data.end()) {
            LRT_TENSOR_RETURN_IF_ERROR(CreateConstTensorFromHostData(
                *upload_desc, host_it->second, gpu_tensor.get()));
            compile_timing_.upload_shared_weights_bytes +=
                host_it->second.data.size();
            shared_weights_->host_data.erase(host_it);
          } else {
            if (upload_desc->GetData().size() <
                upload_desc->GetMemorySizeInBytes()) {
              return absl::InvalidArgumentError(absl::StrCat(
                  "Weights '", weight_name, "' have ",
                  upload_desc->GetData().size(), " bytes, expected ",
                  upload_desc->GetMemorySizeInBytes()));
            }
            LRT_TENSOR_RETURN_IF_ERROR(Backend::CreateConstTensor(
                env_ptr_, *upload_desc, gpu_tensor.get()));
            compile_timing_.upload_shared_weights_bytes +=
                upload_desc->GetData().size();
            if (shared_weights_->on_upload) {
              shared_weights_->on_upload(weight_name, *upload_desc);
            }
          }
          if (shared_desc_it != shared_weights_->descriptors.end()) {
            shared_desc_it->second =
                CopyDescriptorWithoutData(shared_desc_it->second);
          } else {
            shared_weights_->descriptors.emplace(
                weight_name, CopyDescriptorWithoutData(*upload_desc));
          }
          tensor_it = shared_weights_->tensors
                          .emplace(weight_name, std::move(gpu_tensor))
                          .first;
          ++shared_weights_->cache_misses;
        } else {
          ++shared_weights_->cache_hits;
        }
        gpu_model_.tensors[handle.id] =
            CopyDescriptorWithoutData(const_it->second);
        gpu_model_.const_tensors.erase(const_it);
        combined_external_tensors.immutable_tensors[handle.id] =
            tensor_it->second.get();
      }
      ReleaseUploadedSharedDescriptorData();
    }
    compile_timing_.upload_shared_weights = absl::Now() - upload_start;

    const absl::Time init_start = absl::Now();
    LRT_TENSOR_RETURN_IF_ERROR(Backend::InitInferenceContext(
        &inference_context_, gpu_model_info_, &gpu_model_, env_ptr_,
        combined_external_tensors));

    const absl::Time end = absl::Now();
    compilation_time_ = end - compilation_start;
    compile_timing_.init_inference_context = end - init_start;
    compile_timing_.num_nodes = gpu_model_.nodes.size();
    CaptureCompiledHandlesAndReleaseBuilder();
    return absl::OkStatus();
  }

  // Creates the GPU tensor of shared weights from their `host_data`: over the
  // host memory when `zero_copy_host_data` and the backend allow it, else by
  // copying it.
  absl::Status CreateConstTensorFromHostData(
      const ml_drift::TensorDescriptor& desc,
      const typename SharedWeightsCache::HostData& host,
      BackendTensor* tensor) {
    if constexpr (ml_drift_runner_internal::HasCreateConstTensorFromHostData<
                      Backend>::value) {
      std::shared_ptr<const void> gpu_owner;
      if constexpr (std::is_invocable_v<
                        decltype(&Backend::CreateConstTensorFromHostData),
                        Environment*, const ml_drift::TensorDescriptor&,
                        absl::Span<const uint8_t>, bool, BackendTensor*,
                        std::shared_ptr<const void>*, bool>) {
        LRT_TENSOR_RETURN_IF_ERROR(Backend::CreateConstTensorFromHostData(
            env_ptr_, desc, host.data, shared_weights_->zero_copy_host_data,
            tensor, &gpu_owner, host.owns_enclosing_pages));
      } else {
        LRT_TENSOR_RETURN_IF_ERROR(Backend::CreateConstTensorFromHostData(
            env_ptr_, desc, host.data, shared_weights_->zero_copy_host_data,
            tensor, &gpu_owner));
      }
      if (gpu_owner != nullptr) {
        // Keeps both alive in `zero_copy_owners`. Members are destroyed in
        // reverse declaration order, so the last reference releases the
        // backend objects (`gpu`) before the host memory they wrap (`host`).
        // Each one is a type-erased `shared_ptr` that runs the deleter of the
        // object it was created from (e.g. `clReleaseMemObject`, unmapping).
        struct Owners {
          std::shared_ptr<const void> host;
          std::shared_ptr<const void> gpu;
        };
        shared_weights_->zero_copy_owners.push_back(std::make_shared<Owners>(
            Owners{.host = host.owner, .gpu = std::move(gpu_owner)}));
        ++compile_timing_.num_zero_copy_tensors;
      }
      return absl::OkStatus();
    } else {
      if (host.data.size() < desc.GetMemorySizeInBytes()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Constant tensor data has ", host.data.size(),
                         " bytes, expected ", desc.GetMemorySizeInBytes()));
      }
      ml_drift::TensorDescriptor with_data = CopyDescriptorWithoutData(desc);
      with_data.SetData(
          std::vector<uint8_t>(host.data.begin(), host.data.end()));
      return Backend::CreateConstTensor(env_ptr_, with_data, tensor);
    }
  }

  // Releases graph/conversion metadata held solely for compilation and initial
  // tensor pointer resolution (`GetBackendTensor` / `GetTensorValueId`). When
  // `keep_io_lookup` is false (callers that already cached `BackendTensor*`
  // pointers), drops `flat_inputs_`, `flat_outputs_`, and lookup maps too.
  void ReleaseCompileTemporaries(bool keep_io_lookup = false) {
    conversion_ctx_.gpu_model_builder = nullptr;
    model_builder_.reset();
    gpu_model_ = ml_drift::GpuModel{};
    released_weight_tensors_.clear();
    conversion_ctx_.weight_name_to_handle.clear();
    conversion_ctx_.weight_sum_i.clear();
    if (!keep_io_lookup) {
      name_to_tensor_map_.clear();
      conversion_ctx_.graph_tensor_to_handle.clear();
      flat_inputs_.clear();
      flat_outputs_.clear();
    }
  }
  const CompileTiming& compile_timing() const { return compile_timing_; }
  const absl::flat_hash_map<std::string, MlDriftBuildContext::WeightPackStat>&
  weight_pack_stats() const {
    return conversion_ctx_.weight_pack_stats;
  }

  // One-line summary of `compile_timing()` and `weight_pack_stats()`.
  std::string CompileTimingString() const {
    const CompileTiming& t = compile_timing_;
    std::string s = absl::StrCat(
        "rearrange_weights_ms=",
        absl::ToDoubleMilliseconds(t.rearrange_weights),
        " lower_ops_ms=", absl::ToDoubleMilliseconds(t.lower_ops),
        " build_gpu_model_ms=", absl::ToDoubleMilliseconds(t.build_gpu_model),
        " upload_shared_weights_ms=",
        absl::ToDoubleMilliseconds(t.upload_shared_weights),
        " upload_shared_weights_mb=",
        static_cast<double>(t.upload_shared_weights_bytes) / (1 << 20),
        " num_zero_copy_tensors=", t.num_zero_copy_tensors, " init_context_ms=",
        absl::ToDoubleMilliseconds(t.init_inference_context),
        " build_model_ms=", absl::ToDoubleMilliseconds(compilation_time_),
        " num_nodes=", t.num_nodes,
        " num_nodes_to_convert_weights=", t.num_nodes_to_convert_weights);
    std::vector<std::string> kinds;
    kinds.reserve(conversion_ctx_.weight_pack_stats.size());
    for (const auto& [kind, stat] : conversion_ctx_.weight_pack_stats) {
      kinds.push_back(kind);
    }
    std::sort(kinds.begin(), kinds.end());
    for (const std::string& kind : kinds) {
      const auto& stat = conversion_ctx_.weight_pack_stats.at(kind);
      absl::StrAppend(&s, " pack_", kind,
                      "=(ms=", absl::ToDoubleMilliseconds(stat.time),
                      ",mb=", static_cast<double>(stat.bytes) / (1 << 20),
                      ",n=", stat.count, ")");
    }
    return s;
  }

  InferenceContext& inference_context() { return inference_context_; }
  const InferenceContext& inference_context() const {
    return inference_context_;
  }

  absl::StatusOr<ml_drift::ProfilingInfo> Profile() {
    ml_drift::ProfilingInfo profiling_info;
    LRT_TENSOR_RETURN_IF_ERROR(inference_context_.Profile(
        env_ptr_->profiling_queue(), &profiling_info));
    return profiling_info;
  }

  absl::StatusOr<ml_drift::ValueId> GetWeightValueId(
      const std::string& name) const {
    if (build_state_released_) {
      auto it = weight_name_to_value_id_.find(name);
      if (it == weight_name_to_value_id_.end()) {
        return absl::NotFoundError(
            absl::StrCat("Weight tensor handle not found: ", name));
      }
      return it->second;
    }
    auto it = conversion_ctx_.weight_name_to_handle.find(name);
    if (it == conversion_ctx_.weight_name_to_handle.end()) {
      return absl::NotFoundError(
          absl::StrCat("Weight tensor handle not found: ", name));
    }
    return it->second.id;
  }

  absl::StatusOr<ml_drift::ValueId> GetTensorValueId(
      const std::string& name) const {
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto th,
                                FindTensorHandle(name, "Tensor"));
    return th.id;
  }

  // The descriptor (without data) that the graph gives the input or output
  // tensor `name`, e.g. to check that an external tensor matches it. Valid
  // after `Prepare()`.
  absl::StatusOr<ml_drift::TensorDescriptor> GetTensorDescriptor(
      const std::string& name) const {
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto th,
                                FindTensorHandle(name, "Tensor"));
    return CopyDescriptorWithoutData(th.tensor_desc);
  }

  absl::flat_hash_map<ml_drift::ValueId, BackendTensor*> GetWeightTensors() {
    absl::flat_hash_map<ml_drift::ValueId, BackendTensor*> tensors;
    if (build_state_released_) {
      for (const ml_drift::ValueId id : weight_value_ids_) {
        if (auto* t = Backend::GetTensor(&inference_context_, id);
            t != nullptr) {
          tensors[id] = t;
        }
      }
      return tensors;
    }
    for (const auto& [graph_tensor, handle] :
         conversion_ctx_.graph_tensor_to_handle) {
      auto info = graph::GetInfo(graph_tensor);
      const bool is_weight = (info.ok() && info->buffer != nullptr) ||
                             released_weight_tensors_.contains(graph_tensor);
      if (is_weight) {
        auto* t = Backend::GetTensor(&inference_context_, handle.id);
        if (t != nullptr) {
          tensors[handle.id] = t;
        }
      }
    }
    return tensors;
  }

  absl::Status SetInput(const std::string& name, TensorHandle& arg) {
    auto th_or = FindTensorHandle(name, "Input tensor");
    if (!th_or.ok()) {
      const bool exists_unmapped = build_state_released_
                                       ? known_unmapped_tensors_.contains(name)
                                       : FindTensorByName(name).has_value();
      if (exists_unmapped) {
        return absl::OkStatus();
      }
      return absl::InvalidArgumentError(
          absl::StrCat("Input tensor not found: ", name));
    }

    auto buffer_or = arg.GetBuffer();
    if (!buffer_or.ok())
      return absl::InvalidArgumentError("Tensor has no buffer");

    LRT_TENSOR_RETURN_IF_ERROR(EnsureEnvInitialized());

    const auto& th = *th_or;
    if (Backend::CopyBufferToTensor(&inference_context_, env_ptr_, th.id,
                                    &buffer_or.value())
            .ok()) {
      return absl::OkStatus();
    }

    // Fallback for CPU buffers
    auto locked = buffer_or.value().Lock();
    if (arg.GetType() == Type::kFP32) {
      auto span = std::move(locked).template As<const float>();
      std::vector<float> data(span.begin(), span.end());

      ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kFloat32> t;
      t.id = th.id;
      t.data.resize(th.tensor_desc.GetBHWCShape().DimensionsProduct());
      t.shape = th.tensor_desc.GetBHWCShape();
      t.data.assign(data.begin(), data.end());
      return Backend::SetInputTensorFloat(&inference_context_, env_ptr_, t.id,
                                          t);
    } else if (arg.GetType() == Type::kI32) {
      auto span = std::move(locked).template As<const int>();
      std::vector<int> data(span.begin(), span.end());

      ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kInt32> t;
      t.id = th.id;
      t.data.resize(th.tensor_desc.GetBHWCShape().DimensionsProduct());
      t.shape = th.tensor_desc.GetBHWCShape();
      t.data.assign(data.begin(), data.end());
      return Backend::SetInputTensorInt32(&inference_context_, env_ptr_, t.id,
                                          t);
    }
    return absl::InvalidArgumentError(absl::StrCat(
        "Unsupported tensor type for SetInput fallback: ", arg.GetType()));
  }

  absl::Status Enqueue() {
    LRT_TENSOR_RETURN_IF_ERROR(EnsureEnvInitialized());
    return Backend::AddToQueue(&inference_context_, env_ptr_);
  }

  absl::Status Wait() { return Backend::WaitForCompletion(env_ptr_); }

  absl::Status Run() {
    LRT_TENSOR_RETURN_IF_ERROR(Enqueue());
    return Wait();
  }
  absl::StatusOr<std::unique_ptr<Buffer>> GetInputBuffer(
      const std::string& name) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto th,
                                FindTensorHandle(name, "Input tensor"));
    return Backend::CreateInputBuffer(&inference_context_, env_ptr_, th.id);
  }

  absl::Status GetOutput(const std::string& name, Buffer& buffer) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto th,
                                FindTensorHandle(name, "Output tensor"));
    return Backend::CopyTensorToBuffer(&inference_context_, env_ptr_, th.id,
                                       &buffer);
  }

  absl::StatusOr<std::vector<int32_t>> GetInt32Output(const std::string& name) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto th,
                                FindTensorHandle(name, "Output tensor"));
    LRT_TENSOR_RETURN_IF_ERROR(EnsureEnvInitialized());

    ml_drift::TensorInt32 t;
    t.id = th.id;
    t.data.resize(th.tensor_desc.GetBHWCShape().DimensionsProduct());
    t.shape = th.tensor_desc.GetBHWCShape();

    LRT_TENSOR_RETURN_IF_ERROR(Backend::GetOutputTensorInt32(
        &inference_context_, env_ptr_, th.id, &t));
    return t.data;
  }

  // Reads a float output in logical BHWC order (the raw device buffer uses
  // the backend's padded storage layout, so `GetOutputBuffer` is not a
  // substitute for this).
  absl::StatusOr<std::vector<float>> GetFloatOutput(const std::string& name) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto th,
                                FindTensorHandle(name, "Output tensor"));
    LRT_TENSOR_RETURN_IF_ERROR(EnsureEnvInitialized());

    ml_drift::TensorFloat32 t;
    t.id = th.id;
    t.data.resize(th.tensor_desc.GetBHWCShape().DimensionsProduct());
    t.shape = th.tensor_desc.GetBHWCShape();

    LRT_TENSOR_RETURN_IF_ERROR(Backend::GetOutputTensorFloat(
        &inference_context_, env_ptr_, th.id, &t));
    return t.data;
  }

  absl::StatusOr<std::shared_ptr<Buffer>> GetOutputBuffer(

      const std::string& name) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto th,
                                FindTensorHandle(name, "Output tensor"));
    return Backend::CreateOutputBuffer(&inference_context_, env_ptr_, th.id);
  }

  absl::StatusOr<BackendTensor*> GetBackendTensor(const std::string& name) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto th,
                                FindTensorHandle(name, "Tensor"));
    BackendTensor* tensor = Backend::GetTensor(&inference_context_, th.id);
    if (tensor == nullptr) {
      return absl::InternalError(
          absl::StrCat("Backend tensor is null for: ", name));
    }
    return tensor;
  }

  // Returns the compilation time.
  absl::Duration compilation_time() const { return compilation_time_; }

 protected:
  Environment& env() { return *env_ptr_; }
  const Environment& env() const { return *env_ptr_; }

  absl::Status EnsureEnvInitialized() {
    if (env_initialized_) return absl::OkStatus();
    LRT_TENSOR_RETURN_IF_ERROR(Backend::InitEnvironment(env_ptr_));
    env_initialized_ = true;
    return absl::OkStatus();
  }

  std::optional<graph::Tensor> FindTensorByName(const std::string& name) const {
    auto it = name_to_tensor_map_.find(name);
    if (it != name_to_tensor_map_.end()) {
      return it->second;
    }
    return std::nullopt;
  }

 private:
  static std::string WeightsSumIName(absl::string_view weights_name) {
    return absl::StrCat(weights_name, ".sum_i");
  }

  // Registers `sum_i` (per output channel, the sum of the signed quantized
  // values over the input channels) of per-channel quantized (or 1-block-per-
  // row `BlockwiseQuantization`) FullyConnected weights, with which ML Drift
  // runs long prefill FullyConnected layers as INT8 convolutions on
  // dynamically INT8-quantized activations. Only when 8-bit convolutions are
  // allowed.
  bool WantsWeightsSumI(const graph::TensorInformation& info) const {
    if (!allow_8bit_convs_ || info.quantization == nullptr) return false;
    if (info.quantization->As<const graph::PerChannelAffineQuantization>()
            .ok()) {
      return true;
    }
    if (auto bq = info.quantization->As<const graph::BlockwiseQuantization>();
        bq.ok() && info.shape.size() == 2 && bq->quantized_dimension == 0 &&
        bq->block_size == info.shape[1] && bq->zero_points.IsSymmetric()) {
      return true;
    }
    return false;
  }

  // Inserts `desc` into `shared_weight_descriptors` under `name`, letting
  // `shared_weights_->on_pack` persist and drop its heap bytes immediately
  // when streaming to a weight cache. Returns a dataless copy for the model
  // builder.
  ml_drift::TensorDescriptor StoreSharedWeightDescriptor(
      const std::string& name, ml_drift::TensorDescriptor desc) {
    ml_drift::TensorDescriptor dataless = CopyDescriptorWithoutData(desc);
    if (shared_weights_ != nullptr && shared_weights_->on_pack &&
        !desc.GetData().empty() && shared_weights_->on_pack(name, desc)) {
      desc = CopyDescriptorWithoutData(desc);
    }
    conversion_ctx_.shared_weight_descriptors->emplace(name, std::move(desc));
    return dataless;
  }

  absl::Status AddWeightsSumI(const graph::Tensor& weights_tensor,
                              const graph::TensorInformation& info,
                              std::vector<int32_t> sums) {
    ml_drift::Tensor<ml_drift::Linear, ml_drift::DataType::kInt32> sum_i;
    sum_i.shape = ml_drift::Linear(static_cast<int>(sums.size()));
    sum_i.data = std::move(sums);
    ml_drift::TensorDescriptor desc =
        ml_drift::CreateConstantLinearTensorDescriptor(
            model_builder_->gpu_info(), sum_i);
    ml_drift::GpuModelBuilder::TensorHandle handle;
    if (!info.name.empty() &&
        conversion_ctx_.shared_weight_descriptors != nullptr) {
      handle = model_builder_->AddConstantTensor(StoreSharedWeightDescriptor(
          WeightsSumIName(info.name), std::move(desc)));
    } else {
      handle = model_builder_->AddConstantTensor(std::move(desc));
    }
    conversion_ctx_.weight_sum_i[weights_tensor] = handle;
    if (!info.name.empty()) {
      conversion_ctx_.weight_name_to_handle.insert(
          {WeightsSumIName(info.name), handle});
    }
    return absl::OkStatus();
  }

  // `MaybeAddWeightsSumI` for weights found in the shared weights cache.
  void ReuseSharedWeightsSumI(const graph::Tensor& weights_tensor,
                              const std::string& weights_name) {
    if (!allow_8bit_convs_) return;
    const std::string name = WeightsSumIName(weights_name);
    auto it = conversion_ctx_.shared_weight_descriptors->find(name);
    if (it == conversion_ctx_.shared_weight_descriptors->end()) return;
    auto handle = model_builder_->AddConstantTensor(
        CopyDescriptorWithoutData(it->second));
    conversion_ctx_.weight_sum_i[weights_tensor] = handle;
    conversion_ctx_.weight_name_to_handle.insert({name, handle});
  }

  // Host-side INT4/INT2 weight preparation (rearranging into the kernel's
  // packed layout and optional `sum_i`), computed in parallel ahead of GPU
  // tensor registration.
  struct PreparedInt4Weights {
    bool is_int2 = false;
    ml_drift::WeightsDescription desc;
    ml_drift::OHWI ohwi;
    int elements_count = 0;
    std::vector<uint8_t> rearranged;
    std::optional<std::vector<int32_t>> sum_i;
    absl::Status status;
  };

  struct PackedWeightJob {
    graph::Tensor tensor;
    bool is_int2 = false;
    bool has_fc = false;
    bool fc = false;
    bool is_embedding = false;
    ml_drift::WeightsDescription desc;
    ml_drift::OHWI ohwi;
    int elements_count = 0;
  };

  static size_t GetWeightPrepThreads() {
    size_t max_threads = 16;
    if (const char* e = std::getenv("LRT_WEIGHT_PREP_THREADS"); e != nullptr) {
      int parsed = 0;
      if (absl::SimpleAtoi(e, &parsed)) {
        max_threads = static_cast<size_t>(std::max(1, parsed));
      }
    }
    return max_threads;
  }

  static void RearrangePackedWeightJob(const PackedWeightJob& job,
                                       absl::Span<const int8_t> raw,
                                       size_t max_threads,
                                       PreparedInt4Weights* out) {
    absl::StatusOr<ml_drift_runner_internal::RearrangedInt4Weights> rearranged =
        job.is_int2
            ? ml_drift_runner_internal::RearrangeInt2Weights(
                  absl::MakeConstSpan(
                      reinterpret_cast<const uint8_t*>(raw.data()), raw.size()),
                  job.ohwi, job.desc, job.elements_count, job.fc, max_threads)
            : ml_drift_runner_internal::RearrangeInt4Weights(
                  raw, job.ohwi, job.desc, job.elements_count, job.fc,
                  max_threads);
    if (!rearranged.ok()) {
      out->status = rearranged.status();
      return;
    }
    out->rearranged = std::move(rearranged->data);
    out->sum_i = std::move(rearranged->sum_i);
  }

  static bool ShouldPackSoloParallel(const PackedWeightJob& job) {
    constexpr size_t kSoloParallelMinElements = 1ULL << 20;
    return job.ohwi.DimensionsProduct() >= kSoloParallelMinElements &&
           ml_drift_runner_internal::SupportsDirectI4O4Pack(job.desc, job.ohwi,
                                                            job.is_int2);
  }

  std::vector<PackedWeightJob> CollectPackedWeightJobs(
      absl::Span<const graph::Operation* const> execution_plan) {
    std::vector<PackedWeightJob> jobs;
    absl::flat_hash_map<graph::Tensor, int> job_index;
    auto compute_want_sum_i = [&](const graph::TensorInformation& info,
                                  const PackedWeightJob& job) -> bool {
      if (!job.has_fc || !WantsWeightsSumI(info)) return false;
      if (!job.is_int2) return true;
      const bool is_supported_texture =
          job.desc.layout ==
          ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
      return job.desc.IsLinearLayout() || !job.is_embedding ||
             is_supported_texture;
    };
    for (const graph::Operation* op : execution_plan) {
      const bool fc = op->GetName() == "FullyConnected";
      if (!fc && op->GetName() != "EmbeddingLookup") continue;
      graph::Tensor weights_tensor = op->inputs[1];
      if (auto it = job_index.find(weights_tensor); it != job_index.end()) {
        if (it->second >= 0) {
          PackedWeightJob& existing = jobs[it->second];
          if (!fc) existing.is_embedding = true;
          if (fc) existing.has_fc = true;
          const auto info = graph::GetInfo(weights_tensor);
          existing.fc = info.ok() && compute_want_sum_i(*info, existing);
        }
        continue;
      }
      job_index[weights_tensor] = -1;
      if (conversion_ctx_.graph_tensor_to_handle.contains(weights_tensor)) {
        continue;
      }
      const auto info = graph::GetInfo(weights_tensor);
      if (!info.ok()) continue;
      if ((info->type != Type::kI4 && info->type != Type::kI2) ||
          info->buffer == nullptr ||
          conversion_ctx_.external_weight_names.contains(info->name)) {
        continue;
      }
      if (!info->name.empty() &&
          conversion_ctx_.shared_weight_descriptors != nullptr &&
          conversion_ctx_.shared_weight_descriptors->contains(info->name)) {
        continue;
      }
      auto bhwc = ShapeToBHWC(info->shape);
      if (!bhwc.ok()) continue;
      PackedWeightJob job;
      job.tensor = weights_tensor;
      job.is_int2 = (info->type == Type::kI2);
      job.has_fc = fc;
      job.is_embedding = !fc || info->name == "model.embed_tokens.weight" ||
                         info->name == "token_embd.weight";
      if (!job.is_embedding) {
        for (const std::weak_ptr<graph::Operation>& weak_op : info->consumers) {
          if (std::shared_ptr<graph::Operation> consumer = weak_op.lock();
              consumer != nullptr && consumer->GetName() == "EmbeddingLookup") {
            job.is_embedding = true;
            break;
          }
        }
      }
      job.ohwi = ml_drift::OHWI(bhwc->b, bhwc->h, bhwc->w, bhwc->c);
      job.desc =
          job.is_int2
              ? model_builder_->GetFullyConnectedInt2WeightsDesc(job.ohwi)
              : model_builder_->GetFullyConnectedInt4WeightsDesc(job.ohwi);
      job.elements_count =
          ml_drift::GetTotalElementsCountForLayout(job.desc, job.ohwi);
      job.fc = compute_want_sum_i(*info, job);
      job_index[weights_tensor] = static_cast<int>(jobs.size());
      jobs.push_back(std::move(job));
    }
    // Largest first, so the big tensors (e.g. the tied head) start early.
    absl::c_sort(jobs, [](const PackedWeightJob& a, const PackedWeightJob& b) {
      return a.ohwi.DimensionsProduct() > b.ohwi.DimensionsProduct();
    });
    return jobs;
  }

  absl::node_hash_map<graph::Tensor, PreparedInt4Weights> PrepareInt4Weights(
      absl::Span<const graph::Operation* const> execution_plan) {
    absl::node_hash_map<graph::Tensor, PreparedInt4Weights> prepared;
    std::vector<PackedWeightJob> jobs = CollectPackedWeightJobs(execution_plan);
    if (jobs.empty()) return prepared;
    const size_t max_threads = GetWeightPrepThreads();
    std::vector<const PackedWeightJob*> small_jobs;
    for (const PackedWeightJob& job : jobs) {
      PreparedInt4Weights& p = prepared[job.tensor];
      p.is_int2 = job.is_int2;
      p.desc = job.desc;
      p.ohwi = job.ohwi;
      p.elements_count = job.elements_count;
      if (ShouldPackSoloParallel(job) && max_threads > 1) {
        const auto info = graph::GetInfo(job.tensor);
        if (!info.ok() || (*info).buffer == nullptr) {
          p.status = absl::InternalError("Missing buffer during weight prep.");
          continue;
        }
        const graph::TensorInformation& info_ref = *info;
        const LockedBufferSpan<const int8_t> raw =
            info_ref.buffer->Lock().As<const int8_t>();
        RearrangePackedWeightJob(
            job, absl::MakeConstSpan(raw.data(), raw.size()), max_threads, &p);
      } else {
        small_jobs.push_back(&job);
      }
    }
    if (!small_jobs.empty()) {
      std::vector<const PackedWeightJob*> valid_small_jobs;
      std::vector<LockedBufferSpan<const int8_t>> raw_spans;
      std::vector<PreparedInt4Weights*> out_ptrs;
      valid_small_jobs.reserve(small_jobs.size());
      raw_spans.reserve(small_jobs.size());
      out_ptrs.reserve(small_jobs.size());
      for (const PackedWeightJob* job : small_jobs) {
        const auto info = graph::GetInfo(job->tensor);
        PreparedInt4Weights& out = prepared[job->tensor];
        if (!info.ok() || (*info).buffer == nullptr) {
          out.status =
              absl::InternalError("Missing buffer during weight prep.");
          continue;
        }
        const graph::TensorInformation& info_ref = *info;
        valid_small_jobs.push_back(job);
        raw_spans.push_back(info_ref.buffer->Lock().As<const int8_t>());
        out_ptrs.push_back(&out);
      }
      const size_t num_workers = std::min(max_threads, out_ptrs.size());
      ParallelFor(
          num_workers, /*min_grain=*/1,
          [&](size_t w_begin, size_t w_end) {
            internal::ScopedOuterParallelPrepackWorker outer_guard;
            for (size_t w = w_begin; w < w_end; ++w) {
              for (size_t j = w; j < out_ptrs.size(); j += num_workers) {
                RearrangePackedWeightJob(
                    *valid_small_jobs[j],
                    absl::MakeConstSpan(raw_spans[j].data(),
                                        raw_spans[j].size()),
                    /*max_threads=*/1, out_ptrs[j]);
              }
            }
          },
          num_workers);
    }
    return prepared;
  }

  // Streams INT4 and INT2 weight preparation so `shared_weights_->on_pack`
  // persists and releases each packed descriptor immediately, keeping transient
  // heap bounded on Apple / iOS devices.
  absl::Status StreamPrepareInt4Weights(
      absl::Span<const graph::Operation* const> execution_plan) {
    std::vector<PackedWeightJob> jobs = CollectPackedWeightJobs(execution_plan);
    if (jobs.empty()) return absl::OkStatus();
    const size_t max_threads = GetWeightPrepThreads();
    const ml_drift::GpuInfo& gpu_info = model_builder_->gpu_info();
    const ml_drift::DataType float_type = conversion_ctx_.default_float_type;
    const ml_drift::TensorStorageType default_2d_storage =
        DefaultBlockwiseScaleStorageType(gpu_info, float_type);

    absl::Duration int4_time = absl::ZeroDuration();
    absl::Duration int2_time = absl::ZeroDuration();
    size_t int4_bytes = 0;
    size_t int2_bytes = 0;
    int int4_count = 0;
    int int2_count = 0;

    auto register_prepared = [&](PackedWeightJob& job,
                                 PreparedInt4Weights& p) -> absl::Status {
      LRT_TENSOR_RETURN_IF_ERROR(p.status);
      LRT_TENSOR_ASSIGN_OR_RETURN(graph::TensorInformation& info,
                                  graph::GetInfo(job.tensor));
      size_t packed_bytes = p.rearranged.size();
      if (p.sum_i.has_value()) {
        packed_bytes += p.sum_i->size() * sizeof(int32_t);
        LRT_TENSOR_RETURN_IF_ERROR(
            AddWeightsSumI(job.tensor, info, std::move(*p.sum_i)));
      }
      if (job.is_int2) {
        int2_bytes += packed_bytes;
        ++int2_count;
      } else {
        int4_bytes += packed_bytes;
        ++int4_count;
      }
      ml_drift::TensorDescriptor weights_td =
          CreatePackedWeightsTensorDescriptor(
              job.desc, job.ohwi, static_cast<size_t>(job.elements_count),
              job.is_embedding);
      if (p.rearranged.size() == weights_td.GetMemorySizeInBytes()) {
        weights_td.SetData(std::move(p.rearranged));
      } else {
        weights_td.UploadDataRaw(absl::MakeConstSpan(p.rearranged));
        std::vector<uint8_t>().swap(p.rearranged);
      }
      ml_drift::GpuModelBuilder::TensorHandle gpu_handle;
      if (!info.name.empty() &&
          conversion_ctx_.shared_weight_descriptors != nullptr) {
        gpu_handle = model_builder_->AddConstantTensor(
            StoreSharedWeightDescriptor(info.name, std::move(weights_td)));
      } else {
        gpu_handle = model_builder_->AddConstantTensor(std::move(weights_td));
      }
      conversion_ctx_.graph_tensor_to_handle[job.tensor] = gpu_handle;
      name_to_tensor_map_[info.name] = job.tensor;
      if (!info.name.empty()) {
        conversion_ctx_.weight_name_to_handle.insert({info.name, gpu_handle});
      }
      // Stream blockwise scales and zero-points in the same unpack pass as the
      // weights so lazy weights only unpack once and release before the next
      // batch.
      if (conversion_ctx_.shared_weight_descriptors != nullptr &&
          !info.name.empty() && info.quantization != nullptr) {
        auto bwq = info.quantization->As<const BlockwiseQuantization>();
        if (bwq.ok() && bwq->quantized_dimension == 0 && bwq->block_size > 0 &&
            job.ohwi.o > 1 && job.ohwi.i % bwq->block_size == 0) {
          const int num_blocks = job.ohwi.i / bwq->block_size;
          if (num_blocks > 1 && bwq->block_size % 4 == 0) {
            const ml_drift::OHWI scale_shape(job.ohwi.o, 1, 1, num_blocks);
            if (bwq->NumBlocks() ==
                static_cast<size_t>(scale_shape.DimensionsProduct())) {
              const std::string scale_key = absl::StrCat(
                  info.name, ":scale:", static_cast<int>(float_type));
              const std::string zp_key = absl::StrCat(
                  info.name, ":zp:", static_cast<int>(float_type));
              bwq->EnsureMaterialized();
              const bool need_scale =
                  !conversion_ctx_.shared_weight_descriptors->contains(
                      scale_key);
              const bool need_zp =
                  !bwq->zero_points.IsSymmetric() &&
                  !conversion_ctx_.shared_weight_descriptors->contains(zp_key);
              if (need_scale || need_zp) {
                PreparedBlockwiseScale prepared_scale =
                    BuildPreparedBlockwiseScale(gpu_info, *bwq, scale_shape,
                                                float_type, default_2d_storage);
                LRT_TENSOR_RETURN_IF_ERROR(StorePreparedBlockwiseScale(
                    scale_key, zp_key, prepared_scale));
              }
            }
          }
        }
      }
      bool only_fc_consumers = !job.is_embedding;
      if (only_fc_consumers) {
        for (const std::weak_ptr<graph::Operation>& weak_op : info.consumers) {
          if (std::shared_ptr<graph::Operation> consumer = weak_op.lock();
              consumer != nullptr && consumer->GetName() != "FullyConnected") {
            only_fc_consumers = false;
            break;
          }
        }
      }
      // Evict lazy payloads (and recursively any fused part buffers) before
      // `MaybeReleaseFusedHostBuffer` clears `info.buffer`.
      if (only_fc_consumers && info.buffer != nullptr && !info.name.empty() &&
          conversion_ctx_.shared_weight_descriptors != nullptr &&
          conversion_ctx_.shared_weight_descriptors->contains(info.name)) {
        info.buffer->ReleaseMaterializedData();
      }
      MaybeReleaseFusedHostBuffer(job.tensor, info);
      return absl::OkStatus();
    };

    std::vector<PackedWeightJob*> small_int4_jobs;
    std::vector<PackedWeightJob*> small_int2_jobs;
    for (PackedWeightJob& job : jobs) {
      if (ShouldPackSoloParallel(job)) {
        const absl::Time t0 = absl::Now();
        PreparedInt4Weights p;
        {
          LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& info,
                                      graph::GetInfo(job.tensor));
          const LockedBufferSpan<const int8_t> raw =
              info.buffer->Lock().As<const int8_t>();
          RearrangePackedWeightJob(
              job, absl::MakeConstSpan(raw.data(), raw.size()), max_threads,
              &p);
        }
        const absl::Duration elapsed = absl::Now() - t0;
        if (job.is_int2) {
          int2_time += elapsed;
        } else {
          int4_time += elapsed;
        }
        LRT_TENSOR_RETURN_IF_ERROR(register_prepared(job, p));
      } else if (job.is_int2) {
        small_int2_jobs.push_back(&job);
      } else {
        small_int4_jobs.push_back(&job);
      }
    }

    auto run_small_batches =
        [&](absl::Span<PackedWeightJob* const> batch_jobs,
            absl::Duration& type_time) -> absl::Status {
      constexpr size_t kMaxBatchJobs = 16;
      constexpr size_t kMaxBatchBytes = 64ULL << 20;
      size_t start = 0;
      while (start < batch_jobs.size()) {
        size_t end = start;
        size_t batch_bytes = 0;
        while (end < batch_jobs.size() && (end - start) < kMaxBatchJobs) {
          const PackedWeightJob* job = batch_jobs[end];
          const size_t est =
              job->is_int2 ? (static_cast<size_t>(job->elements_count) + 3) / 4
                           : (static_cast<size_t>(job->elements_count) + 1) / 2;
          if (end > start && batch_bytes + est > kMaxBatchBytes) break;
          batch_bytes += est;
          ++end;
        }
        const size_t batch_len = end - start;
        const absl::Time t0 = absl::Now();
        std::vector<PreparedInt4Weights> batch_out(batch_len);
        {
          std::vector<LockedBufferSpan<const int8_t>> raw_spans;
          raw_spans.reserve(batch_len);
          for (size_t i = 0; i < batch_len; ++i) {
            LRT_TENSOR_ASSIGN_OR_RETURN(
                const graph::TensorInformation& info,
                graph::GetInfo(batch_jobs[start + i]->tensor));
            raw_spans.push_back(info.buffer->Lock().As<const int8_t>());
          }
          const size_t num_workers = std::min(max_threads, batch_len);
          ParallelFor(
              num_workers, /*min_grain=*/1,
              [&](size_t w_begin, size_t w_end) {
                internal::ScopedOuterParallelPrepackWorker outer_guard;
                for (size_t w = w_begin; w < w_end; ++w) {
                  for (size_t i = w; i < batch_len; i += num_workers) {
                    RearrangePackedWeightJob(
                        *batch_jobs[start + i],
                        absl::MakeConstSpan(raw_spans[i].data(),
                                            raw_spans[i].size()),
                        /*max_threads=*/1, &batch_out[i]);
                  }
                }
              },
              num_workers);
        }
        type_time += absl::Now() - t0;
        for (size_t i = 0; i < batch_len; ++i) {
          LRT_TENSOR_RETURN_IF_ERROR(
              register_prepared(*batch_jobs[start + i], batch_out[i]));
        }
        start = end;
      }
      return absl::OkStatus();
    };

    LRT_TENSOR_RETURN_IF_ERROR(run_small_batches(small_int4_jobs, int4_time));
    LRT_TENSOR_RETURN_IF_ERROR(run_small_batches(small_int2_jobs, int2_time));
    if (int4_count > 0) {
      conversion_ctx_.AddWeightPackStat("int4", int4_time, int4_bytes,
                                        int4_count);
    }
    if (int2_count > 0) {
      conversion_ctx_.AddWeightPackStat("int2", int2_time, int2_bytes,
                                        int2_count);
    }
    return absl::OkStatus();
  }

  // Registers the `sum_i` of FullyConnected weights that are already mapped to
  // a GPU tensor (by an EmbeddingLookup with the same weights): the prepared
  // `sum_i`, or the one of the shared weights cache.
  absl::Status AddSumIOfMappedWeights(
      const graph::Tensor& weights_tensor,
      absl::node_hash_map<graph::Tensor, PreparedInt4Weights>& prepared_int4) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info,
                                graph::GetInfo(weights_tensor));
    if (auto it = prepared_int4.find(weights_tensor);
        it != prepared_int4.end() && it->second.sum_i.has_value()) {
      std::vector<int32_t> sum_i = std::move(*it->second.sum_i);
      it->second.sum_i.reset();
      return AddWeightsSumI(weights_tensor, info, std::move(sum_i));
    }
    if (!info.name.empty() &&
        conversion_ctx_.shared_weight_descriptors != nullptr) {
      ReuseSharedWeightsSumI(weights_tensor, info.name);
    }
    return absl::OkStatus();
  }

  struct PreparedBlockwiseScale {
    ml_drift::TensorDescriptor scale_td;
    std::optional<ml_drift::TensorDescriptor> zp_td;
    absl::Status status;
  };

  static ml_drift::TensorStorageType DefaultBlockwiseScaleStorageType(
      const ml_drift::GpuInfo& gpu_info, ml_drift::DataType float_type) {
    float probe_vals[8] = {};
    ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kFloat32> probe;
    probe.shape = ml_drift::OHWI(4, 1, 1, 2);
    probe.spanned_data = absl::MakeSpan(probe_vals, 8);
    return ml_drift::ScaleOrZeroPointToTensorDesc(gpu_info, probe, float_type)
        .GetStorageType();
  }

  static PreparedBlockwiseScale BuildPreparedBlockwiseScale(
      const ml_drift::GpuInfo& gpu_info, const BlockwiseQuantization& bwq,
      const ml_drift::OHWI& scale_shape, ml_drift::DataType float_type,
      ml_drift::TensorStorageType default_2d_storage) {
    PreparedBlockwiseScale result;
    const size_t expected_blocks =
        static_cast<size_t>(scale_shape.DimensionsProduct());
    if (bwq.scales.size() != expected_blocks) {
      result.status = absl::InternalError(absl::StrCat(
          "Expected ", expected_blocks, " blockwise scales, got ",
          bwq.scales.size()));
      return result;
    }
    result.scale_td =
        ml_drift_runner_internal::BuildBlockwiseScaleOrZpDescriptor(
            gpu_info, bwq.scales, scale_shape, float_type, default_2d_storage);
    if (!bwq.zero_points.IsSymmetric()) {
      absl::StatusOr<std::vector<float>> zp_floats =
          bwq.zero_points.template CopyAs<float>(bwq.scales.size());
      if (!zp_floats.ok()) {
        result.status = zp_floats.status();
        return result;
      }
      result.zp_td =
          ml_drift_runner_internal::BuildBlockwiseScaleOrZpDescriptor(
              gpu_info, *zp_floats, scale_shape, float_type,
              default_2d_storage);
    }
    return result;
  }

  absl::Status StorePreparedBlockwiseScale(const std::string& scale_key,
                                           const std::string& zp_key,
                                           PreparedBlockwiseScale& prepared,
                                           size_t* scale_bytes = nullptr) {
    LRT_TENSOR_RETURN_IF_ERROR(prepared.status);
    if (!conversion_ctx_.shared_weight_descriptors->contains(scale_key)) {
      if (scale_bytes != nullptr) {
        *scale_bytes += prepared.scale_td.GetData().size();
      }
      StoreSharedWeightDescriptor(scale_key, std::move(prepared.scale_td));
    }
    if (prepared.zp_td.has_value() && !zp_key.empty() &&
        !conversion_ctx_.shared_weight_descriptors->contains(zp_key)) {
      if (scale_bytes != nullptr) {
        *scale_bytes += prepared.zp_td->GetData().size();
      }
      StoreSharedWeightDescriptor(zp_key, std::move(*prepared.zp_td));
    }
    return absl::OkStatus();
  }

  // Pre-packs blockwise quantization scale and zero-point descriptors in
  // parallel into `shared_weight_descriptors` (streaming straight to `.mdwc`
  // when `shared_weights_->on_pack` is active) so `AppendGpuModelBuilder`
  // hits the shared cache in O(1) instead of packing hundreds of MB serially.
  absl::Status PrepareBlockwiseScalesAndZps(
      absl::Span<const graph::Operation* const> execution_plan) {
    if (conversion_ctx_.shared_weight_descriptors == nullptr) {
      return absl::OkStatus();
    }
    const ml_drift::DataType float_type = conversion_ctx_.default_float_type;
    const bool streaming =
        shared_weights_ != nullptr && shared_weights_->on_pack;
    // When streaming INT4/INT2 packing without GPU weight conversion, defer
    // those tensors' scales to `StreamPrepareInt4Weights` so lazy GGUF buffers
    // unpack once per batch and release immediately.
    const bool defer_packed_weights =
        streaming && !GpuWeightConversionEnabled();
    struct ScaleJob {
      std::string scale_key;
      std::string zp_key;
      ml_drift::OHWI shape;
      const BlockwiseQuantization* bwq = nullptr;
    };
    std::vector<ScaleJob> jobs;
    absl::flat_hash_set<std::string> seen_keys;
    for (const graph::Operation* op : execution_plan) {
      if (op->GetName() != "FullyConnected" &&
          op->GetName() != "EmbeddingLookup") {
        continue;
      }
      const auto info = graph::GetInfo(op->inputs[1]);
      if (!info.ok() || info->name.empty() || info->quantization == nullptr) {
        continue;
      }
      auto bwq = info->quantization->As<const BlockwiseQuantization>();
      if (!bwq.ok() || bwq->quantized_dimension != 0 || bwq->block_size <= 0) {
        continue;
      }
      auto bhwc = ShapeToBHWC(info->shape);
      if (!bhwc.ok()) continue;
      const ml_drift::OHWI ohwi(bhwc->b, bhwc->h, bhwc->w, bhwc->c);
      if (ohwi.o <= 1 || ohwi.i % bwq->block_size != 0) continue;
      const int num_blocks = ohwi.i / bwq->block_size;
      if (num_blocks <= 1 || bwq->block_size % 4 != 0) continue;
      const ml_drift::OHWI scale_shape(ohwi.o, 1, 1, num_blocks);
      const size_t expected_blocks =
          static_cast<size_t>(scale_shape.DimensionsProduct());
      if (bwq->NumBlocks() != expected_blocks) {
        continue;
      }
      if (bwq->scales.empty() &&
          (bwq->IsMaterialized() || info->buffer == nullptr)) {
        continue;
      }
      if (defer_packed_weights &&
          (info->type == Type::kI4 || info->type == Type::kI2) &&
          info->buffer != nullptr &&
          !conversion_ctx_.external_weight_names.contains(info->name) &&
          !conversion_ctx_.shared_weight_descriptors->contains(info->name)) {
        continue;
      }
      std::string scale_key =
          absl::StrCat(info->name, ":scale:", static_cast<int>(float_type));
      if (!seen_keys.insert(scale_key).second) continue;
      const bool may_have_zp =
          !bwq->zero_points.IsSymmetric() || bwq->MayHaveFloatZeroPoints();
      std::string zp_key =
          absl::StrCat(info->name, ":zp:", static_cast<int>(float_type));
      const bool scale_cached =
          conversion_ctx_.shared_weight_descriptors->contains(scale_key);
      const bool zp_cached =
          !may_have_zp ||
          conversion_ctx_.shared_weight_descriptors->contains(zp_key);
      if (scale_cached && zp_cached) continue;
      jobs.push_back(ScaleJob{.scale_key = std::move(scale_key),
                              .zp_key = std::move(zp_key),
                              .shape = scale_shape,
                              .bwq = &(*bwq)});
    }
    if (jobs.empty()) return absl::OkStatus();

    const ml_drift::GpuInfo& gpu_info = model_builder_->gpu_info();
    const ml_drift::TensorStorageType default_2d_storage =
        DefaultBlockwiseScaleStorageType(gpu_info, float_type);

    size_t max_threads = 16;
    if (const char* e = std::getenv("LRT_WEIGHT_PREP_THREADS"); e != nullptr) {
      int parsed = 0;
      if (absl::SimpleAtoi(e, &parsed)) {
        max_threads = static_cast<size_t>(std::max(1, parsed));
      }
    }
    const size_t batch_cap = streaming ? 32 : jobs.size();
    const absl::Time scale_start = absl::Now();
    size_t scale_bytes = 0;
    for (size_t begin = 0; begin < jobs.size();) {
      const size_t end = std::min(begin + batch_cap, jobs.size());
      const size_t batch_len = end - begin;
      std::vector<PreparedBlockwiseScale> out(batch_len);
      const size_t num_workers = std::min(max_threads, batch_len);
      ParallelFor(
          num_workers, /*min_grain=*/1,
          [&](size_t w_begin, size_t w_end) {
            for (size_t w = w_begin; w < w_end; ++w) {
              for (size_t i = w; i < batch_len; i += num_workers) {
                const ScaleJob& job = jobs[begin + i];
                job.bwq->EnsureMaterialized();
                out[i] = BuildPreparedBlockwiseScale(
                    gpu_info, *job.bwq, job.shape, float_type,
                    default_2d_storage);
              }
            }
          },
          num_workers);
      for (size_t i = 0; i < batch_len; ++i) {
        const ScaleJob& job = jobs[begin + i];
        LRT_TENSOR_RETURN_IF_ERROR(StorePreparedBlockwiseScale(
            job.scale_key, job.zp_key, out[i], &scale_bytes));
      }
      begin = end;
    }
    conversion_ctx_.AddWeightPackStat("scales", absl::Now() - scale_start,
                                      scale_bytes,
                                      static_cast<int>(jobs.size()));
    return absl::OkStatus();
  }

  absl::Status RearrangeAndMapConstantWeights() {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        std::vector<const graph::Operation*> execution_plan,
        GetExecutionPlan(flat_outputs_));
    LRT_TENSOR_RETURN_IF_ERROR(PrepareBlockwiseScalesAndZps(execution_plan));
    if (GpuWeightConversionEnabled()) {
      const absl::Time register_start = absl::Now();
      LRT_TENSOR_RETURN_IF_ERROR(RegisterGpuWeightConversions(execution_plan));
      const absl::Duration register_time = absl::Now() - register_start;
      if (!gpu_weight_conversions_.empty()) {
        // Bytes: the raw weights that `RunGpuWeightConversions` uploads.
        size_t raw_bytes = 0;
        for (const GpuWeightConversion& c : gpu_weight_conversions_) {
          raw_bytes += c.raw_bytes;
        }
        conversion_ctx_.AddWeightPackStat(
            "gpu_register", register_time, raw_bytes,
            static_cast<int>(gpu_weight_conversions_.size()));
      }
    }
    absl::node_hash_map<graph::Tensor, PreparedInt4Weights> prepared_int4;
    if (shared_weights_ != nullptr && shared_weights_->on_pack) {
      LRT_TENSOR_RETURN_IF_ERROR(StreamPrepareInt4Weights(execution_plan));
    } else {
      const absl::Time pack_start = absl::Now();
      prepared_int4 = PrepareInt4Weights(execution_plan);
      const absl::Duration pack_time = absl::Now() - pack_start;
      size_t int4_bytes = 0;
      size_t int2_bytes = 0;
      int int4_count = 0;
      int int2_count = 0;
      for (const auto& [tensor, p] : prepared_int4) {
        size_t bytes = p.rearranged.size();
        if (p.sum_i.has_value()) {
          bytes += p.sum_i->size() * sizeof(int32_t);
        }
        if (p.is_int2) {
          int2_bytes += bytes;
          ++int2_count;
        } else {
          int4_bytes += bytes;
          ++int4_count;
        }
      }
      if (int4_count > 0) {
        conversion_ctx_.AddWeightPackStat("int4", pack_time, int4_bytes,
                                          int4_count);
      }
      if (int2_count > 0) {
        conversion_ctx_.AddWeightPackStat("int2", pack_time, int2_bytes,
                                          int2_count);
      }
    }

    for (const graph::Operation* op : execution_plan) {
      if (op->GetName() == "FullyConnected" ||
          op->GetName() == "EmbeddingLookup") {
        auto weights_tensor = op->inputs[1];
        if (conversion_ctx_.graph_tensor_to_handle.contains(weights_tensor)) {
          // Weights mapped by an earlier EmbeddingLookup can still need the
          // `sum_i` of a tied FullyConnected.
          if (op->GetName() == "FullyConnected" &&
              !conversion_ctx_.weight_sum_i.contains(weights_tensor)) {
            LRT_TENSOR_RETURN_IF_ERROR(
                AddSumIOfMappedWeights(weights_tensor, prepared_int4));
          }
          continue;
        }
        LRT_TENSOR_ASSIGN_OR_RETURN(auto& info,
                                    graph::GetInfo(weights_tensor));
        if (conversion_ctx_.external_weight_names.contains(info.name)) {
          continue;
        }
        // Constant weights (including fused weights or bufferless weight stubs
        // for passes reusing `SharedWeightsCache`) may have no host buffer when
        // an earlier compilation already uploaded their packed GPU copy.
        const bool is_fused_weight = absl::StrContains(info.name, '+');
        const bool is_leaf_weight =
            weights_tensor.group != nullptr &&
            weights_tensor.group->producer == nullptr;
        // Reuse a packed descriptor of the shared cache before looking at the
        // host buffer, so that recompiling a graph whose host weights were
        // released or replaced by bufferless stubs still finds them.
        if (is_leaf_weight && !info.name.empty() &&
            conversion_ctx_.shared_weight_descriptors != nullptr) {
          auto shared_it =
              conversion_ctx_.shared_weight_descriptors->find(info.name);
          if (shared_it != conversion_ctx_.shared_weight_descriptors->end()) {
            auto gpu_handle = model_builder_->AddConstantTensor(
                CopyDescriptorWithoutData(shared_it->second));
            conversion_ctx_.graph_tensor_to_handle[weights_tensor] = gpu_handle;
            name_to_tensor_map_[info.name] = weights_tensor;
            conversion_ctx_.weight_name_to_handle.insert(
                {info.name, gpu_handle});
            if (op->GetName() == "FullyConnected") {
              ReuseSharedWeightsSumI(weights_tensor, info.name);
            }
            MaybeReleaseFusedHostBuffer(weights_tensor, info);
            continue;
          }
        }
        if (info.buffer == nullptr) {
          if (is_fused_weight) {
            return absl::FailedPreconditionError(absl::StrCat(
                "The host data of fused weight '", info.name,
                "' was released after an earlier upload; compile this graph "
                "with the SharedWeightsCache that holds its packed weights."));
          }
          continue;
        }

        auto buffer = info.buffer;
        auto lock = buffer->Lock();
        auto raw_span = std::move(lock).As<const int8_t>();

        if (info.type != Type::kI4 && info.type != Type::kI2) {
          // Constant weights consumed by FullyConnected / EmbeddingLookup
          // kernels must be stored in the kernel's packed weights layout.
          // The generic constant path in `GetTensorHandle` creates a plain
          // linear tensor instead, which silently produces wrong results, so
          // pack them here the same way the ML Drift numerical test bridge
          // does. INT4 and INT2 keep their dedicated fast path below.
          MlDriftDummyWeightCreator weight_creator(model_builder_.get());
          Tensor<MlDriftMixinTag> weights_wrapper(weights_tensor);
          ml_drift::GpuModelBuilder::TensorHandle gpu_handle;
          ml_drift::TensorDescriptor* shared_desc_out = nullptr;
          ml_drift::TensorDescriptor local_shared_desc;
          if (!info.name.empty() &&
              conversion_ctx_.shared_weight_descriptors != nullptr) {
            shared_desc_out = &local_shared_desc;
          }
          std::optional<std::vector<int32_t>> sum_i;
          const absl::Time pack_start = absl::Now();
          if (info.type == Type::kFP32) {
            absl::Span<const float> data(
                reinterpret_cast<const float*>(raw_span.data()),
                raw_span.size() / sizeof(float));
            LRT_TENSOR_ASSIGN_OR_RETURN(
                gpu_handle,
                weight_creator.CreateFcWeights<float>(
                    weights_wrapper, data, conversion_ctx_.default_float_type,
                    shared_desc_out));
          } else if (info.type == Type::kI8) {
            if (op->GetName() == "FullyConnected" && WantsWeightsSumI(info)) {
              LRT_TENSOR_ASSIGN_OR_RETURN(auto bhwc_s, ShapeToBHWC(info.shape));
              const ml_drift::OHWI ohwi(bhwc_s.b, bhwc_s.h, bhwc_s.w, bhwc_s.c);
              LRT_TENSOR_ASSIGN_OR_RETURN(
                  sum_i, ComputeInt8WeightsSumI(raw_span, ohwi));
            }
            LRT_TENSOR_ASSIGN_OR_RETURN(
                gpu_handle,
                weight_creator.CreateFcWeights<int8_t>(
                    weights_wrapper, raw_span,
                    conversion_ctx_.default_float_type, shared_desc_out));
          } else {
            return absl::UnimplementedError(absl::StrCat(
                "Unsupported constant weights type for ", op->GetName(), ": ",
                info.type, " (tensor '", info.name, "')"));
          }
          conversion_ctx_.AddWeightPackStat(
              info.type == Type::kFP32 ? "fp" : "int8",
              absl::Now() - pack_start,
              gpu_handle.tensor_desc.GetMemorySizeInBytes());
          if (shared_desc_out != nullptr) {
            StoreSharedWeightDescriptor(info.name,
                                        std::move(local_shared_desc));
          }
          if (sum_i.has_value()) {
            LRT_TENSOR_RETURN_IF_ERROR(
                AddWeightsSumI(weights_tensor, info, std::move(*sum_i)));
          }
          conversion_ctx_.graph_tensor_to_handle[weights_tensor] = gpu_handle;
          name_to_tensor_map_[info.name] = weights_tensor;
          if (!info.name.empty()) {
            conversion_ctx_.weight_name_to_handle.insert(
                {info.name, gpu_handle});
          }
          MaybeReleaseFusedHostBuffer(weights_tensor, info);
          continue;
        }

        const bool is_int2 = (info.type == Type::kI2);
        ml_drift::OHWI weights_ohwi;
        ml_drift::WeightsDescription desc;
        size_t elements_count = 0;
        std::vector<uint8_t> rearranged_data;
        bool is_embedding = (op->GetName() == "EmbeddingLookup" ||
                             info.name == "model.embed_tokens.weight" ||
                             info.name == "token_embd.weight");
        if (!is_embedding) {
          for (const std::weak_ptr<graph::Operation>& weak_op :
               info.consumers) {
            if (std::shared_ptr<graph::Operation> consumer = weak_op.lock();
                consumer != nullptr &&
                consumer->GetName() == "EmbeddingLookup") {
              is_embedding = true;
              break;
            }
          }
        }
        bool is_supported_texture = false;
        std::optional<std::vector<int32_t>> sum_i;

        if (auto prepared_it = prepared_int4.find(weights_tensor);
            prepared_it != prepared_int4.end()) {
          PreparedInt4Weights& prepared = prepared_it->second;
          LRT_TENSOR_RETURN_IF_ERROR(prepared.status);
          weights_ohwi = prepared.ohwi;
          desc = prepared.desc;
          if (is_int2 && desc.type != ml_drift::DataType::kUint2 &&
              desc.type != ml_drift::DataType::kInt2) {
            return absl::UnimplementedError(
                "Only INT4 and INT2 weight rearrangement is supported in "
                "pre-rearrange.");
          }
          elements_count = static_cast<size_t>(prepared.elements_count);
          rearranged_data = std::move(prepared.rearranged);
          is_supported_texture =
              desc.layout ==
              ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
          if (op->GetName() == "FullyConnected" && prepared.sum_i.has_value()) {
            sum_i = std::move(prepared.sum_i);
          }
        } else if (!is_int2) {
          return absl::InternalError(absl::StrCat("INT4 weights '", info.name,
                                                  "' were not prepared."));
        } else {
          const absl::Time int2_start = absl::Now();
          LRT_TENSOR_ASSIGN_OR_RETURN(auto bhwc_shape, ShapeToBHWC(info.shape));
          weights_ohwi = ml_drift::OHWI(bhwc_shape.b, bhwc_shape.h,
                                        bhwc_shape.w, bhwc_shape.c);
          desc = model_builder_->GetFullyConnectedInt2WeightsDesc(weights_ohwi);
          is_supported_texture =
              desc.layout ==
              ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
          const bool want_sum_i =
              op->GetName() == "FullyConnected" && WantsWeightsSumI(info) &&
              (desc.IsLinearLayout() || !is_embedding || is_supported_texture);
          elements_count = static_cast<size_t>(
              ml_drift::GetTotalElementsCountForLayout(desc, weights_ohwi));
          LRT_TENSOR_ASSIGN_OR_RETURN(
              ml_drift_runner_internal::RearrangedInt4Weights rearranged,
              ml_drift_runner_internal::RearrangeInt2Weights(
                  absl::MakeConstSpan(
                      reinterpret_cast<const uint8_t*>(raw_span.data()),
                      raw_span.size()),
                  weights_ohwi, desc, static_cast<int>(elements_count),
                  want_sum_i));
          rearranged_data = std::move(rearranged.data);
          sum_i = std::move(rearranged.sum_i);
          conversion_ctx_.AddWeightPackStat(
              "int2", absl::Now() - int2_start, rearranged_data.size());
        }
        if (sum_i.has_value()) {
          LRT_TENSOR_RETURN_IF_ERROR(
              AddWeightsSumI(weights_tensor, info, std::move(*sum_i)));
        }

        ml_drift::TensorDescriptor weights_td =
            CreatePackedWeightsTensorDescriptor(
                desc, weights_ohwi, static_cast<size_t>(elements_count),
                is_embedding);

        if (rearranged_data.size() == weights_td.GetMemorySizeInBytes()) {
          weights_td.SetData(std::move(rearranged_data));
        } else {
          weights_td.UploadDataRaw(absl::MakeConstSpan(rearranged_data));
        }

        ml_drift::GpuModelBuilder::TensorHandle gpu_handle;
        if (!info.name.empty() &&
            conversion_ctx_.shared_weight_descriptors != nullptr) {
          gpu_handle = model_builder_->AddConstantTensor(
              StoreSharedWeightDescriptor(info.name, std::move(weights_td)));
        } else {
          gpu_handle = model_builder_->AddConstantTensor(std::move(weights_td));
        }
        conversion_ctx_.graph_tensor_to_handle[weights_tensor] = gpu_handle;
        name_to_tensor_map_[info.name] = weights_tensor;
        if (!info.name.empty()) {
          conversion_ctx_.weight_name_to_handle.insert({info.name, gpu_handle});
        }
        MaybeReleaseFusedHostBuffer(weights_tensor, info);
      }
    }
    return absl::OkStatus();
  }

  bool GpuWeightConversionEnabled() const {
    if (!convert_weights_on_gpu_) return false;
    if (force_gpu_weight_conversion_) return true;
    const ml_drift::GpuInfo& gpu_info = model_builder_->gpu_info();
    // Only exercised on OpenCL so far.
    return gpu_info.IsApiOpenCl() &&
           ml_drift::WeightsManager::IsGpuWeightsPreparationSupported(gpu_info);
  }

  // Registers the constant INT4 / INT8 / INT2 FullyConnected / EmbeddingLookup
  // weights that the GPU can rearrange into exactly the tensors of the CPU
  // paths below (see `GetGpuWeightConversionSpec`; the others keep the CPU
  // paths). Each one gets a dataless constant tensor in the model, plus a
  // `sum_i` tensor when a FullyConnected consumer needs it, and
  // `RunGpuWeightConversions` fills them in `BuildModel()`.
  absl::Status RegisterGpuWeightConversions(
      absl::Span<const graph::Operation* const> execution_plan) {
    const ml_drift_runner_internal::FcAndEmbeddingWeights weights =
        ml_drift_runner_internal::CollectFcAndEmbeddingWeights(execution_plan);
    for (const graph::Tensor& weights_tensor : weights.ordered) {
      if (conversion_ctx_.graph_tensor_to_handle.contains(weights_tensor)) {
        continue;
      }
      absl::StatusOr<const graph::TensorInformation&> tensor_info =
          graph::GetInfo(weights_tensor);
      if (!tensor_info.ok()) continue;
      const graph::TensorInformation& info = *tensor_info;
      // External weights and weights of the shared cache aren't packed here.
      if (conversion_ctx_.external_weight_names.contains(info.name) ||
          (conversion_ctx_.shared_weight_descriptors != nullptr &&
           conversion_ctx_.shared_weight_descriptors->contains(info.name))) {
        continue;
      }
      // Like the CPU path: `sum_i` for weights with a FullyConnected consumer.
      const bool wants_sum_i =
          weights.used_by_fully_connected.contains(weights_tensor) &&
          WantsWeightsSumI(info);
      std::optional<ml_drift_runner_internal::GpuWeightConversionSpec> spec =
          ml_drift_runner_internal::GetGpuWeightConversionSpec(
              *model_builder_, info, wants_sum_i);
      if (!spec.has_value()) continue;

      // Keep the raw values alive until `RunGpuWeightConversions`.
      GpuWeightConversion c;
      c.raw = info.buffer->Lock().As<const int8_t>();
      if (c.raw.size() < spec->raw_bytes) {
        return absl::InvalidArgumentError(
            absl::StrCat("Weights '", info.name, "' have ", c.raw.size(),
                         " bytes, expected ", spec->raw_bytes));
      }
      c.weights_tensor = weights_tensor;
      c.buffer = info.buffer;
      c.name = info.name;
      c.ohwi = spec->ohwi;
      c.desc = spec->desc;
      c.src_type = spec->src_type;
      c.raw_bytes = spec->raw_bytes;

      // Map the weights to a dataless constant tensor of the model.
      c.handle = model_builder_->AddConstantTensor(std::move(spec->dst_desc));
      conversion_ctx_.graph_tensor_to_handle[weights_tensor] = c.handle;
      name_to_tensor_map_[info.name] = weights_tensor;
      conversion_ctx_.weight_name_to_handle.insert({info.name, c.handle});
      if (spec->needs_sum_i) {
        c.sum_i = model_builder_->AddConstantTensor(
            ml_drift_runner_internal::GpuWeightsSumIDescriptor(c.ohwi.o));
        conversion_ctx_.weight_sum_i[weights_tensor] = *c.sum_i;
        conversion_ctx_.weight_name_to_handle.insert(
            {WeightsSumIName(info.name), *c.sum_i});
      }
      gpu_weight_conversions_.push_back(std::move(c));
    }
    return absl::OkStatus();
  }

  // Rearranges the weights of `RegisterGpuWeightConversions` on the GPU into
  // GPU tensors that become immutable external tensors of the model (through
  // the `SharedWeightsCache` if there is one).
  //
  // Weights that another runner sharing the `SharedWeightsCache` converted in
  // the meantime (e.g. both runners ran `Prepare()` before either ran
  // `BuildModel()`) are taken from the cache instead of converted again.
  //
  // The conversions run in batches, one batch after another: a batch uploads
  // the raw weights of at most `max_gpu_weight_conversion_batch_bytes_` (or
  // of one larger tensor), converts them all together with one ML Drift
  // conversion model, and waits for it before freeing the raw GPU copies. That
  // bounds the GPU memory of the raw weights by the batch size instead of the
  // size of all the weights.
  absl::Status RunGpuWeightConversions(
      ml_drift::ExternalTensorsInfo* external_tensors) {
    if (gpu_weight_conversions_.empty()) return absl::OkStatus();
    const absl::Time start = absl::Now();
    // Split off the weights that the cache already has: they are only bound.
    std::vector<GpuWeightConversion> to_convert;
    std::vector<GpuWeightConversion> in_cache;
    for (GpuWeightConversion& c : gpu_weight_conversions_) {
      const bool cached =
          shared_weights_ != nullptr &&
          shared_weights_->tensors.contains(c.name) &&
          (!c.sum_i.has_value() ||
           shared_weights_->tensors.contains(WeightsSumIName(c.name)));
      (cached ? in_cache : to_convert).push_back(std::move(c));
    }
    gpu_weight_conversions_.clear();

    const ml_drift::GpuInfo& gpu_info = model_builder_->gpu_info();
    // The converted tensors of the current batch, by main model value id.
    absl::flat_hash_map<ml_drift::ValueId, std::unique_ptr<BackendTensor>>
        dst_tensors;
    // Binds the converted (or cached) tensors to the model and drops the host
    // weights immediately so fused host buffers are freed batch-by-batch as
    // GPU tensors are allocated instead of overlapping across the entire model.
    const auto bind = [&](GpuWeightConversion& c,
                          bool converted) -> absl::Status {
      const auto take = [&](const ml_drift::GpuModelBuilder::TensorHandle& h)
          -> std::unique_ptr<BackendTensor> {
        if (!converted) return nullptr;
        return std::move(dst_tensors.at(h.id));
      };
      LRT_TENSOR_RETURN_IF_ERROR(BindConvertedWeights(
          c.name, c.handle, take(c.handle), external_tensors));
      if (c.sum_i.has_value()) {
        LRT_TENSOR_RETURN_IF_ERROR(
            BindConvertedWeights(WeightsSumIName(c.name), *c.sum_i,
                                 take(*c.sum_i), external_tensors));
      }
      c.raw = LockedBufferSpan<const int8_t>::Empty();
      c.buffer.reset();
      if (absl::StatusOr<graph::TensorInformation&> info =
              graph::GetInfo(c.weights_tensor);
          info.ok()) {
        MaybeReleaseFusedHostBuffer(c.weights_tensor, *info);
      }
      return absl::OkStatus();
    };
    size_t converted_bytes = 0;
    // Every batch is non-empty: `GpuWeightConversionBatchEnd` > `begin`.
    for (size_t begin = 0; begin < to_convert.size();) {
      const size_t end = ml_drift_runner_internal::GpuWeightConversionBatchEnd(
          to_convert, begin, max_gpu_weight_conversion_batch_bytes_);
      const absl::Span<const GpuWeightConversion> batch =
          absl::MakeConstSpan(to_convert).subspan(begin, end - begin);
      // Allocate the converted tensors of the batch.
      for (const GpuWeightConversion& c : batch) {
        LRT_TENSOR_RETURN_IF_ERROR(
            AllocateConvertedTensor(c.handle, &dst_tensors, &converted_bytes));
        if (c.sum_i.has_value()) {
          LRT_TENSOR_RETURN_IF_ERROR(AllocateConvertedTensor(
              *c.sum_i, &dst_tensors, &converted_bytes));
        }
      }
      // Bind the raw weights (uploaded now) and the converted tensors to the
      // conversion model of the batch.
      LRT_TENSOR_ASSIGN_OR_RETURN(
          ml_drift_runner_internal::GpuWeightConversionModel conversion,
          ml_drift_runner_internal::CreateGpuWeightConversionModel(gpu_info,
                                                                   batch));
      ml_drift::ExternalTensorsInfo conversion_tensors;
      std::vector<std::unique_ptr<BackendTensor>> raw_tensors;
      for (const auto& upload : conversion.uploads) {
        LRT_TENSOR_ASSIGN_OR_RETURN(
            const ml_drift::TensorDescriptor raw_desc,
            ml_drift_runner_internal::RawWeightsDescriptor(conversion.model,
                                                           upload));
        auto tensor = std::make_unique<BackendTensor>();
        LRT_TENSOR_RETURN_IF_ERROR(
            Backend::CreateConstTensor(env_ptr_, raw_desc, tensor.get()));
        conversion_tensors.immutable_tensors[upload.input_id] = tensor.get();
        raw_tensors.push_back(std::move(tensor));
      }
      for (const auto& [conversion_id, main_id] : conversion.io_mapping) {
        auto it = dst_tensors.find(main_id);
        if (it == dst_tensors.end()) {
          return absl::InternalError("Missing converted weights tensor");
        }
        conversion_tensors.immutable_tensors[conversion_id] = it->second.get();
      }
      // Convert, and wait before `raw_tensors` frees the raw weights.
      InferenceContext conversion_context;
      LRT_TENSOR_RETURN_IF_ERROR(Backend::InitInferenceContext(
          &conversion_context, gpu_model_info_, &conversion.model, env_ptr_,
          conversion_tensors));
      LRT_TENSOR_RETURN_IF_ERROR(
          Backend::AddToQueue(&conversion_context, env_ptr_));
      LRT_TENSOR_RETURN_IF_ERROR(Backend::WaitForCompletion(env_ptr_));
      for (size_t i = begin; i < end; ++i) {
        LRT_TENSOR_RETURN_IF_ERROR(bind(to_convert[i], /*converted=*/true));
      }
      dst_tensors.clear();
      begin = end;
    }
    for (GpuWeightConversion& c : in_cache) {
      LRT_TENSOR_RETURN_IF_ERROR(bind(c, /*converted=*/false));
    }
    if (!to_convert.empty()) {
      conversion_ctx_.AddWeightPackStat("gpu_convert", absl::Now() - start,
                                        converted_bytes,
                                        static_cast<int>(to_convert.size()));
    }
    return absl::OkStatus();
  }

  // Allocates the GPU tensor (without data) that a conversion model writes the
  // converted weights (or `sum_i`) of the model tensor `handle` into, in
  // `dst_tensors` by main model value id, and adds its size to
  // `converted_bytes`.
  absl::Status AllocateConvertedTensor(
      const ml_drift::GpuModelBuilder::TensorHandle& handle,
      absl::flat_hash_map<ml_drift::ValueId, std::unique_ptr<BackendTensor>>*
          dst_tensors,
      size_t* converted_bytes) {
    auto tensor = std::make_unique<BackendTensor>();
    LRT_TENSOR_RETURN_IF_ERROR(Backend::CreateConstTensor(
        env_ptr_, CopyDescriptorWithoutData(handle.tensor_desc), tensor.get()));
    *converted_bytes += handle.tensor_desc.GetMemorySizeInBytes();
    (*dst_tensors)[handle.id] = std::move(tensor);
    return absl::OkStatus();
  }

  // Makes the GPU-converted weights (or `sum_i`) `name` of the model tensor
  // `handle` an immutable external tensor of the model. With a
  // `SharedWeightsCache`, the cache owns it: the tensor the cache already has
  // under `name` if any (`tensor`, if not null, is then dropped), else
  // `tensor`. Without a cache, this runner owns `tensor`.
  absl::Status BindConvertedWeights(
      const std::string& name,
      const ml_drift::GpuModelBuilder::TensorHandle& handle,
      std::unique_ptr<BackendTensor> tensor,
      ml_drift::ExternalTensorsInfo* external_tensors) {
    // The model must not upload (dataless) constant data for it.
    auto const_it = gpu_model_.const_tensors.find(handle.id);
    if (const_it != gpu_model_.const_tensors.end()) {
      gpu_model_.tensors[handle.id] =
          CopyDescriptorWithoutData(const_it->second);
      gpu_model_.const_tensors.erase(const_it);
    }
    if (shared_weights_ != nullptr) {
      auto it = shared_weights_->tensors.find(name);
      if (it != shared_weights_->tensors.end()) {
        ++shared_weights_->cache_hits;
      } else {
        if (tensor == nullptr) {
          return absl::InternalError(
              absl::StrCat("Weights '", name, "' were not converted"));
        }
        // Read the converted weights back before caching them, so that a
        // failure leaves the cache unchanged.
        if (shared_weights_->on_upload) {
          ml_drift::TensorDescriptor converted;
          LRT_TENSOR_RETURN_IF_ERROR(
              Backend::TensorToDescriptor(tensor.get(), env_ptr_, &converted));
          shared_weights_->on_upload(name, converted);
        }
        shared_weights_->descriptors.insert_or_assign(
            name, CopyDescriptorWithoutData(handle.tensor_desc));
        it = shared_weights_->tensors.emplace(name, std::move(tensor)).first;
        ++shared_weights_->cache_misses;
      }
      external_tensors->immutable_tensors[handle.id] = it->second.get();
      return absl::OkStatus();
    }
    if (tensor == nullptr) {
      return absl::InternalError(
          absl::StrCat("Weights '", name, "' were not converted"));
    }
    external_tensors->immutable_tensors[handle.id] = tensor.get();
    converted_weight_tensors_.push_back(std::move(tensor));
    return absl::OkStatus();
  }

  // Drops the host copy of a fused constant weight once the shared cache holds
  // its packed descriptor, if the cache allows it
  // (`SharedWeightsCache::release_fused_host_buffers`). Released tensors stay
  // visible to `GetWeightTensors()`.
  void MaybeReleaseFusedHostBuffer(const graph::Tensor& weights_tensor,
                                   graph::TensorInformation& info) {
    if (shared_weights_ == nullptr ||
        !shared_weights_->release_fused_host_buffers ||
        !absl::StrContains(info.name, '+') ||
        !shared_weights_->descriptors.contains(info.name)) {
      return;
    }
    info.buffer = nullptr;
    released_weight_tensors_.insert(weights_tensor);
  }

  absl::StatusOr<ml_drift::GpuModelBuilder::TensorHandle> FindTensorHandle(
      const std::string& name, absl::string_view tensor_role = "Tensor") const {
    if (build_state_released_) {
      auto it = compiled_tensor_handles_.find(name);
      if (it != compiled_tensor_handles_.end()) {
        return it->second;
      }
      if (known_unmapped_tensors_.contains(name)) {
        return absl::NotFoundError(
            absl::StrCat(tensor_role, " handle not found: ", name));
      }
      return absl::NotFoundError(
          absl::StrCat(tensor_role, " not found: ", name));
    }
    auto raw_tensor_ptr = FindTensorByName(name);
    if (!raw_tensor_ptr) {
      return absl::NotFoundError(
          absl::StrCat(tensor_role, " not found: ", name));
    }
    auto it = conversion_ctx_.graph_tensor_to_handle.find(*raw_tensor_ptr);
    if (it == conversion_ctx_.graph_tensor_to_handle.end()) {
      return absl::NotFoundError(
          absl::StrCat(tensor_role, " handle not found: ", name));
    }
    return it->second;
  }

  // Drops the host copy of the packed data of the `SharedWeightsCache`
  // descriptors whose GPU tensors exist, keeping the dataless descriptors
  // (which later runners still use to map the weights). Only when the cache
  // allows releasing host buffers (`release_fused_host_buffers`), and not for
  // weights still uploaded from `host_data`.
  void ReleaseUploadedSharedDescriptorData() {
    if (shared_weights_ == nullptr ||
        !shared_weights_->release_fused_host_buffers) {
      return;
    }
    for (auto& [weight_name, desc] : shared_weights_->descriptors) {
      if (!desc.GetData().empty() &&
          shared_weights_->tensors.contains(weight_name) &&
          !shared_weights_->host_data.contains(weight_name)) {
        desc = CopyDescriptorWithoutData(desc);
      }
    }
  }

  // After `BuildModel()`: keeps what `Run()` and the tensor accessors still
  // need (the dataless handles of the named tensors, the names without a
  // handle, the weight name -> value id map `weight_name_to_value_id_` and the
  // value ids of the weight tensors `weight_value_ids_`), then
  // frees the model builder, the `GpuModel` and the graph bookkeeping of the
  // compilation (`build_state_released_` makes the accessors use what was
  // kept).
  void CaptureCompiledHandlesAndReleaseBuilder() {
    compiled_tensor_handles_.reserve(name_to_tensor_map_.size());
    for (const auto& [name, graph_tensor] : name_to_tensor_map_) {
      auto it = conversion_ctx_.graph_tensor_to_handle.find(graph_tensor);
      if (it != conversion_ctx_.graph_tensor_to_handle.end()) {
        ml_drift::GpuModelBuilder::TensorHandle h = it->second;
        h.tensor_desc = CopyDescriptorWithoutData(h.tensor_desc);
        compiled_tensor_handles_.emplace(name, std::move(h));
      } else {
        known_unmapped_tensors_.insert(name);
      }
    }
    weight_name_to_value_id_.reserve(
        conversion_ctx_.weight_name_to_handle.size());
    for (const auto& [name, handle] : conversion_ctx_.weight_name_to_handle) {
      weight_name_to_value_id_.emplace(name, handle.id);
    }
    for (const auto& [graph_tensor, handle] :
         conversion_ctx_.graph_tensor_to_handle) {
      auto info = graph::GetInfo(graph_tensor);
      if ((info.ok() && info->buffer != nullptr) ||
          released_weight_tensors_.contains(graph_tensor)) {
        weight_value_ids_.push_back(handle.id);
      }
    }
    build_state_released_ = true;

    model_builder_.reset();
    gpu_model_ = ml_drift::GpuModel{};
    flat_inputs_.clear();
    flat_inputs_.shrink_to_fit();
    flat_outputs_.clear();
    flat_outputs_.shrink_to_fit();
    name_to_tensor_map_.clear();
    released_weight_tensors_.clear();
    graph_input_ids_.clear();
    graph_input_ids_.shrink_to_fit();
    graph_output_ids_.clear();
    graph_output_ids_.shrink_to_fit();
    conversion_ctx_.gpu_model_builder = nullptr;
    conversion_ctx_.shared_weight_descriptors = nullptr;
    conversion_ctx_.graph_tensor_to_handle.clear();
    conversion_ctx_.constant_handles_in_tensor_layout.clear();
    conversion_ctx_.weight_name_to_handle.clear();
    conversion_ctx_.external_weight_names.clear();
    conversion_ctx_.weight_sum_i.clear();
  }

 protected:
  std::vector<TensorHandle> flat_inputs_;
  std::vector<TensorHandle> flat_outputs_;

  absl::flat_hash_map<std::string, graph::Tensor> name_to_tensor_map_;
  absl::flat_hash_map<std::string, ml_drift::GpuModelBuilder::TensorHandle>
      compiled_tensor_handles_;
  absl::flat_hash_set<std::string> known_unmapped_tensors_;
  absl::flat_hash_map<std::string, ml_drift::ValueId> weight_name_to_value_id_;
  std::vector<ml_drift::ValueId> weight_value_ids_;
  bool build_state_released_ = false;
  // Fused weights whose host buffers `MaybeReleaseFusedHostBuffer` dropped.
  absl::flat_hash_set<graph::Tensor> released_weight_tensors_;

  Environment env_;
  Environment* env_ptr_ = &env_;
  SharedWeightsCache* shared_weights_ = nullptr;
  std::unique_ptr<ml_drift::GpuModelBuilder> model_builder_;
  ml_drift::GpuModel gpu_model_;
  InferenceContext inference_context_;
  MlDriftBuildContext conversion_ctx_;
  // ML Drift may use INT8-activation convolutions (constructor argument or
  // `LRT_ALLOW_8BIT_CONVS=1`).
  bool allow_8bit_convs_ = false;
  ml_drift::CreateGpuModelInfo gpu_model_info_;

  std::vector<ml_drift::ValueId> graph_input_ids_;
  std::vector<ml_drift::ValueId> graph_output_ids_;

  using GpuWeightConversion = ml_drift_runner_internal::GpuWeightConversion;
  std::vector<GpuWeightConversion> gpu_weight_conversions_;
  // GPU-converted weights of runners without a `SharedWeightsCache`.
  std::vector<std::unique_ptr<BackendTensor>> converted_weight_tensors_;
  size_t max_gpu_weight_conversion_batch_bytes_ = size_t{256} << 20;
  bool convert_weights_on_gpu_ = false;
  bool force_gpu_weight_conversion_ = false;

  absl::Duration compilation_time_;
  CompileTiming compile_timing_;
  bool env_initialized_ = false;
  bool prepared_ = false;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_COMPILED_ML_DRIFT_MODEL_RUNNER_H_

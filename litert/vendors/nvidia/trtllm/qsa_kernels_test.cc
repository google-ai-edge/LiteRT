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

// The QSA kernels (qsa_indexer.h, qsa_sparse_attention.h) against the host
// references of qsa_reference.h: the GPU delegate's own test cases, edge
// cases (start positions on and off block boundaries, the ring cleared by a
// completed block, right-padded chunks, fewer blocks than block_topk, ties,
// repeated and out-of-range selections, rows that see nothing), random
// shapes and types, the 27B's shapes up to a 128k context, and the row model
// of the batched and tree steps.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/trtllm/qsa_indexer.h"
#include "litert/vendors/nvidia/trtllm/qsa_reference.h"
#include "litert/vendors/nvidia/trtllm/qsa_sparse_attention.h"

namespace {

namespace ref = litert::nvidia::qsa_reference;

struct CudaDeleter {
  void operator()(void* pointer) const { cudaFree(pointer); }
};
using Device = std::unique_ptr<void, CudaDeleter>;

Device Allocate(size_t bytes) {
  void* pointer = nullptr;
  EXPECT_EQ(cudaMalloc(&pointer, std::max<size_t>(bytes, 4)), cudaSuccess);
  return Device(pointer);
}

Device UploadBytes(const void* data, size_t bytes) {
  Device device = Allocate(bytes);
  EXPECT_EQ(cudaMemcpy(device.get(), data, bytes, cudaMemcpyHostToDevice),
            cudaSuccess);
  return device;
}

template <typename T>
Device Upload(const std::vector<T>& values) {
  return UploadBytes(values.data(), values.size() * sizeof(T));
}

template <typename T>
std::vector<T> Download(const Device& device, size_t count) {
  std::vector<T> values(count);
  EXPECT_EQ(cudaMemcpy(values.data(), device.get(), count * sizeof(T),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  return values;
}

// Floats in a kernel's element type: the bytes to upload and the values the
// kernel then sees.
struct Typed {
  std::vector<uint8_t> bytes;
  std::vector<float> held;
};

Typed AsType(const std::vector<float>& values, int32_t type) {
  Typed typed;
  typed.held.resize(values.size());
  const size_t width = type == kLiteRtNvidiaQsaFloat32 ? 4 : 2;
  typed.bytes.resize(values.size() * width);
  for (size_t i = 0; i < values.size(); ++i) {
    if (type == kLiteRtNvidiaQsaFloat32) {
      std::memcpy(typed.bytes.data() + 4 * i, &values[i], 4);
      typed.held[i] = values[i];
    } else if (type == kLiteRtNvidiaQsaFloat16) {
      const uint16_t bits = ref::FloatToHalfBits(values[i]);
      std::memcpy(typed.bytes.data() + 2 * i, &bits, 2);
      typed.held[i] = ref::HalfBitsToFloat(bits);
    } else {
      const uint16_t bits = ref::FloatToBf16Bits(values[i]);
      std::memcpy(typed.bytes.data() + 2 * i, &bits, 2);
      typed.held[i] = ref::Bf16BitsToFloat(bits);
    }
  }
  return typed;
}

std::vector<float> FromType(const std::vector<uint8_t>& bytes, int32_t type) {
  const size_t width = type == kLiteRtNvidiaQsaFloat32 ? 4 : 2;
  std::vector<float> values(bytes.size() / width);
  for (size_t i = 0; i < values.size(); ++i) {
    if (type == kLiteRtNvidiaQsaFloat32) {
      std::memcpy(&values[i], bytes.data() + 4 * i, 4);
    } else {
      uint16_t bits;
      std::memcpy(&bits, bytes.data() + 2 * i, 2);
      values[i] = type == kLiteRtNvidiaQsaFloat16 ? ref::HalfBitsToFloat(bits)
                                                  : ref::Bf16BitsToFloat(bits);
    }
  }
  return values;
}

class Random {
 public:
  explicit Random(uint32_t seed) : state_(seed * 2654435761u + 12345u) {}
  // Uniform in [lo, hi).
  float Uniform(float lo, float hi) {
    state_ = state_ * 1664525u + 1013904223u;
    return lo + (hi - lo) * static_cast<float>(state_ >> 8) / 16777216.0f;
  }
  int Int(int lo, int hi) {  // [lo, hi]
    state_ = state_ * 1664525u + 1013904223u;
    return lo +
           static_cast<int>((state_ >> 8) % static_cast<uint32_t>(hi - lo + 1));
  }
  std::vector<float> Values(size_t count, float lo, float hi) {
    std::vector<float> values(count);
    for (auto& value : values) value = Uniform(lo, hi);
    return values;
  }

 private:
  uint32_t state_;
};

// The GPU delegate's test pattern.
std::vector<float> Pattern(int count, int seed) {
  std::vector<float> values(count);
  for (int i = 0; i < count; ++i) {
    values[i] = 0.25f * std::sin(0.37f * static_cast<float>(i + seed));
  }
  return values;
}

// A mask [rows, len] in bool bytes and as floats (0 visible, `fill` not).
struct HostMask {
  int rows = 0;
  int64_t len = 0;
  std::vector<uint8_t> bools;
  std::vector<float> floats;
  float fill = -10000.0f;

  HostMask(int rows_in, int64_t len_in, float fill_in = -10000.0f)
      : rows(rows_in),
        len(len_in),
        bools(rows_in * len_in, 0),
        floats(rows_in * len_in, fill_in),
        fill(fill_in) {}
  void Set(int row, int64_t column, bool visible) {
    bools[row * len + column] = visible ? 1 : 0;
    floats[row * len + column] = visible ? 0.0f : fill;
  }
  // Row t at position pos0 + t sees [lo, pos0 + t], for the first `valid`
  // rows; the others see nothing.
  void Causal(int pos0, int valid, int window = 0) {
    for (int t = 0; t < rows; ++t) {
      for (int64_t c = 0; c < len; ++c) {
        const int64_t position = pos0 + t;
        const bool in_window = window <= 0 || c > position - window;
        Set(t, c, t < valid && c <= position && in_window);
      }
    }
  }
  ref::Mask View(bool use_bool) const {
    ref::Mask mask;
    mask.len = len;
    if (use_bool) {
      mask.bools = bools.data();
    } else {
      mask.floats = floats.data();
    }
    return mask;
  }
  Device Upload(bool use_bool) const {
    return use_bool ? UploadBytes(bools.data(), bools.size())
                    : UploadBytes(floats.data(), floats.size() * 4);
  }
};

// ---------------------------------------------------------------------------
// odml.qsa_indexer_update
// ---------------------------------------------------------------------------

struct IndexerCase {
  ref::IndexerShape shape;
  int start_pos = 0;
  int32_t q_type = kLiteRtNvidiaQsaFloat32;
  bool bool_mask = true;
  std::vector<float> q, k_new, ring, pool, norm, cos, sin;
  HostMask mask{1, 4};
};

struct IndexerRun {
  std::vector<float> ring, pool;
  std::vector<int32_t> topk;
};

IndexerRun RunIndexer(const IndexerCase& c) {
  const auto& s = c.shape;
  const int32_t state_type =
      s.fp16_state ? kLiteRtNvidiaQsaFloat16 : kLiteRtNvidiaQsaFloat32;
  LiteRtNvidiaQsaIndexerParams params = {};
  params.rows = s.rows;
  params.index_heads = s.index_heads;
  params.index_dim = s.index_dim;
  params.rotary_dim = s.rotary_dim;
  params.max_blocks = s.max_blocks;
  params.block_topk = s.block_topk;
  params.mask_len = static_cast<int32_t>(s.mask_len);
  params.q_type = c.q_type;
  params.state_type = state_type;
  params.mask_type =
      c.bool_mask ? kLiteRtNvidiaQsaMaskBool : kLiteRtNvidiaQsaMaskFloat;
  params.rms_norm_eps = s.eps;
  params.scale = s.scale;
  EXPECT_TRUE(LiteRtNvidiaQsaIndexerSupports(&params));
  const Typed q = AsType(c.q, c.q_type);
  const Typed k_new = AsType(c.k_new, c.q_type);
  const Typed ring = AsType(c.ring, state_type);
  const Typed pool = AsType(c.pool, state_type);
  Device q_d = Upload(q.bytes), k_d = Upload(k_new.bytes);
  Device ring_d = Upload(ring.bytes), pool_d = Upload(pool.bytes);
  Device norm_d = Upload(c.norm), cos_d = Upload(c.cos), sin_d = Upload(c.sin);
  Device start_d = Upload(std::vector<int32_t>{c.start_pos});
  Device mask_d = c.mask.Upload(c.bool_mask);
  Device topk_d = Allocate(static_cast<size_t>(s.rows) * s.block_topk * 4);
  Device work_d = Allocate(LiteRtNvidiaQsaIndexerWorkspaceBytes(&params));
  EXPECT_EQ(LiteRtNvidiaLaunchQsaIndexerUpdate(
                &params, q_d.get(), k_d.get(), ring_d.get(), pool_d.get(),
                static_cast<const float*>(norm_d.get()),
                static_cast<const float*>(cos_d.get()),
                static_cast<const float*>(sin_d.get()),
                static_cast<const int32_t*>(start_d.get()), mask_d.get(),
                static_cast<int32_t*>(topk_d.get()), work_d.get(), nullptr),
            cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  IndexerRun run;
  run.ring = FromType(Download<uint8_t>(ring_d, ring.bytes.size()), state_type);
  run.pool = FromType(Download<uint8_t>(pool_d, pool.bytes.size()), state_type);
  run.topk =
      Download<int32_t>(topk_d, static_cast<size_t>(s.rows) * s.block_topk);
  return run;
}

// Rounds the inputs the kernel reads in a narrower type, so the reference
// sees what the kernel sees.
void HoldInTypes(IndexerCase& c) {
  c.q = AsType(c.q, c.q_type).held;
  c.k_new = AsType(c.k_new, c.q_type).held;
  if (c.shape.fp16_state) {
    c.ring = AsType(c.ring, kLiteRtNvidiaQsaFloat16).held;
    c.pool = AsType(c.pool, kLiteRtNvidiaQsaFloat16).held;
  }
}

void ExpectClose(const std::vector<float>& got, const std::vector<float>& want,
                 float relative, const std::string& what) {
  ASSERT_EQ(got.size(), want.size()) << what;
  for (size_t i = 0; i < got.size(); ++i) {
    ASSERT_NEAR(got[i], want[i], relative * (1.0f + std::fabs(want[i])))
        << what << " at " << i;
  }
}

// Checks the GPU's ring and pool against the reference, and its selection
// against the scores of the GPU's own pool (so a last-bit difference in a
// pool value cannot move the comparison).
void CheckIndexer(IndexerCase c, const std::string& what) {
  HoldInTypes(c);
  const IndexerRun run = RunIndexer(c);
  std::vector<float> ring = c.ring;
  std::vector<float> pool = c.pool;
  const auto mask = c.mask.View(c.bool_mask);
  ref::IndexerUpdate(c.shape, c.q, c.k_new, ring, pool, c.norm, c.cos, c.sin,
                     c.start_pos, mask);
  const float state_tolerance = c.shape.fp16_state ? 2e-3f : 2e-5f;
  ExpectClose(run.ring, ring, state_tolerance, what + " ring");
  ExpectClose(run.pool, pool, state_tolerance, what + " pool");
  const ref::IndexerResult scored =
      ref::IndexerScores(c.shape, c.q, run.pool, c.start_pos, mask);
  for (int t = 0; t < c.shape.rows; ++t) {
    std::string why;
    EXPECT_TRUE(ref::IsTopk(
        scored.scores.data() + static_cast<size_t>(t) * scored.bound,
        scored.bound,
        run.topk.data() + static_cast<size_t>(t) * c.shape.block_topk,
        c.shape.block_topk, 2e-5, &why))
        << what << " row " << t << ": " << why;
  }
}

IndexerCase MakeIndexerCase(const ref::IndexerShape& shape, int start_pos,
                            uint32_t seed) {
  IndexerCase c;
  c.shape = shape;
  c.start_pos = start_pos;
  Random random(seed);
  const int dim = shape.index_dim;
  c.q = random.Values(static_cast<size_t>(shape.rows) * shape.index_heads * dim,
                      -1.0f, 1.0f);
  c.k_new = random.Values(static_cast<size_t>(shape.rows) * dim, -1.0f, 1.0f);
  c.ring = random.Values(4 * dim, -1.0f, 1.0f);
  c.pool =
      random.Values(static_cast<size_t>(shape.max_blocks) * dim, -1.0f, 1.0f);
  c.norm = random.Values(dim, 0.5f, 1.5f);
  c.cos.resize(static_cast<size_t>(shape.max_blocks) * shape.rotary_dim);
  c.sin.resize(c.cos.size());
  for (int b = 0; b < shape.max_blocks; ++b) {
    for (int i = 0; i < shape.rotary_dim; ++i) {
      const int pair = i % (shape.rotary_dim / 2);
      const double angle =
          b * 4.0 * std::pow(10000.0, -2.0 * pair / shape.rotary_dim);
      c.cos[static_cast<size_t>(b) * shape.rotary_dim + i] =
          static_cast<float>(std::cos(angle));
      c.sin[static_cast<size_t>(b) * shape.rotary_dim + i] =
          static_cast<float>(std::sin(angle));
    }
  }
  c.mask = HostMask(shape.rows, shape.mask_len);
  c.mask.Causal(std::max(start_pos, 0), shape.rows);
  return c;
}

// The GPU delegate's EndToEndIndexer1DNormWeightRotaryAndInPlaceStateAliasing:
// a decode token completes block 4 (tokens 16..19), five blocks compete for
// two places, and blocks 1 and 3 are identical (a tie).
TEST(QsaIndexerTest, DecodeCompletesBlockAndBreaksTiesToTheLowerBlock) {
  IndexerCase c;
  c.shape.rows = 1;
  c.shape.index_heads = 5;
  c.shape.index_dim = 16;
  c.shape.rotary_dim = 8;
  c.shape.max_blocks = 8;
  c.shape.block_topk = 2;
  c.shape.mask_len = 32;
  c.shape.scale = 1.0f / std::sqrt(16.0f);
  c.start_pos = 19;
  c.q = Pattern(5 * 16, 2);
  c.k_new = Pattern(16, 5);
  c.ring = Pattern(4 * 16, 9);
  c.pool = Pattern(8 * 16, 11);
  for (int d = 0; d < 16; ++d) c.pool[3 * 16 + d] = c.pool[1 * 16 + d];
  c.norm = Pattern(16, 31);
  for (float& w : c.norm) w += 0.8f;
  c.cos.resize(8 * 8);
  c.sin.resize(8 * 8);
  for (int i = 0; i < 64; ++i) {
    c.cos[i] = std::cos(0.13f * (i + 1));
    c.sin[i] = std::sin(0.13f * (i + 1));
  }
  c.mask = HostMask(1, 32);
  for (int t = 0; t <= 19; ++t) c.mask.Set(0, t, true);
  for (bool use_bool : {true, false}) {
    c.bool_mask = use_bool;
    const IndexerRun run = RunIndexer(c);
    std::vector<float> ring = c.ring, pool = c.pool;
    const auto result =
        ref::IndexerUpdate(c.shape, c.q, c.k_new, ring, pool, c.norm, c.cos,
                           c.sin, c.start_pos, c.mask.View(use_bool));
    ExpectClose(run.ring, std::vector<float>(64, 0.0f), 0.0f, "cleared ring");
    ExpectClose(run.pool, pool, 2e-5f, "pool");
    EXPECT_EQ(run.topk, result.topk);
  }
}

// The GPU delegate's PaddedPrefillDoesNotCorruptRingOrBlockPool: eight rows
// at position 0 of which five are real; block 0 completes, the ring keeps
// token 4, block 1 stays as it was.
TEST(QsaIndexerTest, PaddedPrefillLeavesPaddingOutOfRingAndPool) {
  IndexerCase c;
  c.shape.rows = 8;
  c.shape.index_heads = 2;
  c.shape.index_dim = 8;
  c.shape.rotary_dim = 8;
  c.shape.max_blocks = 4;
  c.shape.block_topk = 2;
  c.shape.mask_len = 16;
  c.shape.scale = 0.25f;
  c.q = Pattern(8 * 2 * 8, 1);
  c.k_new = Pattern(8 * 8, 2);
  for (int t = 5; t < 8; ++t) {
    for (int d = 0; d < 8; ++d) c.k_new[t * 8 + d] = 999.0f;
  }
  c.ring.assign(4 * 8, -7.0f);
  c.pool = Pattern(4 * 8, 9);
  c.norm.assign(8, 1.0f);
  c.cos.assign(4 * 8, 1.0f);
  c.sin.assign(4 * 8, 0.0f);
  c.mask = HostMask(8, 16);
  c.mask.Causal(0, 5);
  const IndexerRun run = RunIndexer(c);
  std::vector<float> want_ring(32, 0.0f);
  for (int d = 0; d < 8; ++d) want_ring[d] = c.k_new[4 * 8 + d];
  ExpectClose(run.ring, want_ring, 0.0f, "ring");
  std::vector<float> want_pool = c.pool;
  double square = 0.0;
  std::vector<double> pooled(8);
  for (int d = 0; d < 8; ++d) {
    for (int r = 0; r < 4; ++r) pooled[d] += c.k_new[r * 8 + d] / 4.0;
    square += pooled[d] * pooled[d];
  }
  for (int d = 0; d < 8; ++d) {
    want_pool[d] = static_cast<float>(pooled[d] / std::sqrt(square / 8 + 1e-6));
  }
  ExpectClose(run.pool, want_pool, 2e-5f, "pool");
  CheckIndexer(c, "padded prefill");
}

// The GPU delegate's IndexerShortContextFastPathAndLargePoolTopkMatchesCpu:
// four complete blocks for eight places (every one taken, no scores), then
// 35 for eight.
TEST(QsaIndexerTest, ShortContextTakesEveryBlockAndLongContextSelects) {
  for (int start_pos : {18, 142}) {
    IndexerCase c;
    c.shape.rows = 1;
    c.shape.index_heads = 8;
    c.shape.index_dim = 32;
    c.shape.rotary_dim = 16;
    c.shape.max_blocks = 64;
    c.shape.block_topk = 8;
    c.shape.mask_len = 256;
    c.shape.scale = 1.0f / std::sqrt(32.0f);
    c.start_pos = start_pos;
    c.q = Pattern(8 * 32, 2);
    c.k_new = Pattern(32, 5);
    c.ring = Pattern(4 * 32, 9);
    c.pool = Pattern(64 * 32, 11);
    c.norm.assign(32, 1.0f);
    c.cos.assign(64 * 16, 1.0f);
    c.sin.assign(64 * 16, 0.0f);
    c.mask = HostMask(1, 256);
    for (int t = 0; t <= start_pos; ++t) c.mask.Set(0, t, true);
    const IndexerRun run = RunIndexer(c);
    std::vector<float> ring = c.ring, pool = c.pool;
    const auto result =
        ref::IndexerUpdate(c.shape, c.q, c.k_new, ring, pool, c.norm, c.cos,
                           c.sin, c.start_pos, c.mask.View(true));
    EXPECT_EQ(run.topk, result.topk) << "start_pos " << start_pos;
    if (start_pos == 18) {
      EXPECT_EQ(run.topk, (std::vector<int32_t>{0, 1, 2, 3, -1, -1, -1, -1}));
    }
  }
}

// Every score ties (zero queries: relu(0) everywhere): the lowest blocks win.
TEST(QsaIndexerTest, AllTiedScoresTakeTheLowestBlocks) {
  ref::IndexerShape shape;
  shape.rows = 2;
  shape.index_heads = 4;
  shape.index_dim = 64;
  shape.rotary_dim = 32;
  shape.max_blocks = 200;
  shape.block_topk = 16;
  shape.mask_len = 800;
  IndexerCase c = MakeIndexerCase(shape, 401, 7);
  std::fill(c.q.begin(), c.q.end(), 0.0f);
  const IndexerRun run = RunIndexer(c);
  for (int t = 0; t < 2; ++t) {
    for (int i = 0; i < 16; ++i) EXPECT_EQ(run.topk[t * 16 + i], i);
  }
}

// Start positions on and off block boundaries, short and long steps,
// completed and cleared rings, fewer blocks than places, both mask kinds,
// every query and state type.
TEST(QsaIndexerTest, RandomStepsMatchTheReference) {
  int seed = 0;
  for (int rows : {1, 2, 3, 4, 5, 8, 13, 64}) {
    for (int start_pos : {0, 1, 3, 4, 5, 7, 37, 250}) {
      ++seed;
      Random random(seed);
      ref::IndexerShape shape;
      shape.rows = rows;
      shape.index_heads = std::vector<int>{1, 4, 8}[seed % 3];
      shape.index_dim = std::vector<int>{16, 64, 128}[(seed / 3) % 3];
      shape.rotary_dim = std::min(shape.index_dim, 8 << (seed % 3));
      shape.max_blocks = 96;
      shape.block_topk = std::vector<int>{4, 16, 64, 512}[seed % 4];
      shape.mask_len = 4 * shape.max_blocks + (seed % 2) * 3;
      shape.scale = 1.0f / std::sqrt(static_cast<float>(shape.index_dim));
      shape.fp16_state = seed % 5 == 0;
      IndexerCase c = MakeIndexerCase(shape, start_pos, seed);
      c.q_type = (seed % 3 == 0)   ? kLiteRtNvidiaQsaFloat32
                 : (seed % 3 == 1) ? kLiteRtNvidiaQsaFloat16
                                   : kLiteRtNvidiaQsaBFloat16;
      c.bool_mask = seed % 2 == 0;
      // Right padding on some chunks.
      const int valid = rows > 2 && seed % 4 == 1 ? rows - 2 : rows;
      c.mask =
          HostMask(rows, shape.mask_len, seed % 3 == 0 ? -INFINITY : -10000.0f);
      c.mask.Causal(start_pos, valid);
      CheckIndexer(c, "rows " + std::to_string(rows) + " start " +
                          std::to_string(start_pos));
    }
  }
}

// A negative start position counts as 0.
TEST(QsaIndexerTest, NegativeStartPositionIsZero) {
  ref::IndexerShape shape;
  shape.rows = 6;
  shape.index_heads = 2;
  shape.index_dim = 16;
  shape.rotary_dim = 16;
  shape.max_blocks = 8;
  shape.block_topk = 2;
  shape.mask_len = 32;
  IndexerCase c = MakeIndexerCase(shape, -5, 3);
  c.mask.Causal(0, 6);
  CheckIndexer(c, "negative start");
}

// The 27B's indexer: 8 heads of 128 dims, 64 rotary dims, 512 blocks of a
// budget of 2,048 tokens; decode, a verify of 4 rows and a 64-row chunk at
// 32k, decode at the 128k context, and beyond it (160k: the selection that
// does not hold a row's scores in registers).
TEST(QsaIndexerTest, ModelShapesMatchTheReference) {
  struct Case {
    int max_blocks;
    int rows;
    int start_pos;
  };
  for (const Case& shape_case :
       {Case{8192, 1, 30001}, Case{8192, 4, 29998}, Case{8192, 64, 4096},
        Case{32768, 1, 131000}, Case{40000, 2, 159990}}) {
    ref::IndexerShape shape;
    shape.rows = shape_case.rows;
    shape.index_heads = 8;
    shape.index_dim = 128;
    shape.rotary_dim = 64;
    shape.max_blocks = shape_case.max_blocks;
    shape.block_topk = 512;
    shape.mask_len = 4 * shape_case.max_blocks;
    shape.scale = 1.0f / std::sqrt(128.0f);
    IndexerCase c = MakeIndexerCase(shape, shape_case.start_pos, 11);
    c.q_type = kLiteRtNvidiaQsaBFloat16;
    CheckIndexer(c, "model shape " + std::to_string(shape_case.max_blocks) +
                        " x " + std::to_string(shape_case.rows));
  }
}

// ---------------------------------------------------------------------------
// The indexer's row model (LiteRtNvidiaLaunchQsaIndexerRows)
// ---------------------------------------------------------------------------

// One slot of a row-model indexer case: its state before the step and its
// committed tokens.
struct SlotState {
  int context = 0;
  std::vector<float> ring, pool;
};

// A decode step of row `row` against a slot's state (updated), as the
// single-sequence reference computes it: its scores and selection.
ref::IndexerResult DecodeRow(const ref::IndexerShape& rows_shape,
                             const IndexerCase& c, int row, int position,
                             std::vector<float>& ring,
                             std::vector<float>& pool) {
  ref::IndexerShape shape = rows_shape;
  shape.rows = 1;
  const int dim = shape.index_dim;
  const std::vector<float> q(
      c.q.begin() + static_cast<size_t>(row) * shape.index_heads * dim,
      c.q.begin() + static_cast<size_t>(row + 1) * shape.index_heads * dim);
  const std::vector<float> k(
      c.k_new.begin() + static_cast<size_t>(row) * dim,
      c.k_new.begin() + static_cast<size_t>(row + 1) * dim);
  HostMask mask(1, shape.mask_len);
  mask.Causal(position, 1);
  return ref::IndexerUpdate(shape, q, k, ring, pool, c.norm, c.cos, c.sin,
                            position, mask.View(true));
}

struct RowsRun {
  std::vector<float> ring, pool;
  std::vector<int32_t> topk;
};

RowsRun RunIndexerRows(const IndexerCase& c,
                       const std::vector<SlotState>& slots,
                       const std::vector<int32_t>& slot_of,
                       const std::vector<int32_t>& context_len,
                       const std::vector<uint64_t>& ancestors, bool score) {
  const auto& s = c.shape;
  const int32_t state_type =
      s.fp16_state ? kLiteRtNvidiaQsaFloat16 : kLiteRtNvidiaQsaFloat32;
  LiteRtNvidiaQsaIndexerParams params = {};
  params.rows = s.rows;
  params.index_heads = s.index_heads;
  params.index_dim = s.index_dim;
  params.rotary_dim = s.rotary_dim;
  params.max_blocks = s.max_blocks;
  params.block_topk = s.block_topk;
  params.q_type = c.q_type;
  params.state_type = state_type;
  params.rms_norm_eps = s.eps;
  params.scale = s.scale;
  std::vector<float> ring, pool;
  for (const auto& slot : slots) {
    ring.insert(ring.end(), slot.ring.begin(), slot.ring.end());
    pool.insert(pool.end(), slot.pool.begin(), slot.pool.end());
  }
  const Typed q = AsType(c.q, c.q_type);
  const Typed k_new = AsType(c.k_new, c.q_type);
  const Typed ring_t = AsType(ring, state_type);
  const Typed pool_t = AsType(pool, state_type);
  Device q_d = Upload(q.bytes), k_d = Upload(k_new.bytes);
  Device ring_d = Upload(ring_t.bytes), pool_d = Upload(pool_t.bytes);
  Device norm_d = Upload(c.norm), cos_d = Upload(c.cos), sin_d = Upload(c.sin);
  Device slot_d = Upload(slot_of), context_d = Upload(context_len);
  Device ancestors_d = Allocate(8);
  LiteRtNvidiaQsaRows rows = {};
  rows.context_len = static_cast<const int32_t*>(context_d.get());
  rows.slot = static_cast<const int32_t*>(slot_d.get());
  if (!ancestors.empty()) {
    ancestors_d = Upload(ancestors);
    rows.ancestors = static_cast<const uint64_t*>(ancestors_d.get());
  }
  EXPECT_TRUE(LiteRtNvidiaQsaIndexerRowsSupports(&params, &rows));
  Device topk_d = Allocate(static_cast<size_t>(s.rows) * s.block_topk * 4);
  Device work_d = Allocate(LiteRtNvidiaQsaIndexerRowsWorkspaceBytes(&params));
  EXPECT_EQ(LiteRtNvidiaLaunchQsaIndexerRows(
                &params, &rows, 4 * s.index_dim,
                static_cast<int64_t>(s.max_blocks) * s.index_dim,
                score ? q_d.get() : nullptr, k_d.get(), ring_d.get(),
                pool_d.get(), static_cast<const float*>(norm_d.get()),
                static_cast<const float*>(cos_d.get()),
                static_cast<const float*>(sin_d.get()),
                score ? static_cast<int32_t*>(topk_d.get()) : nullptr,
                work_d.get(), nullptr),
            cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  RowsRun run;
  run.ring =
      FromType(Download<uint8_t>(ring_d, ring_t.bytes.size()), state_type);
  run.pool =
      FromType(Download<uint8_t>(pool_d, pool_t.bytes.size()), state_type);
  if (score) {
    run.topk =
        Download<int32_t>(topk_d, static_cast<size_t>(s.rows) * s.block_topk);
  }
  return run;
}

// A case of `rows` rows with random queries and keys, and `slots` random
// states with the given contexts (the ring holding the incomplete block's
// keys, the pool the complete blocks).
IndexerCase MakeRowsCase(const ref::IndexerShape& shape,
                         std::vector<SlotState>* slots,
                         const std::vector<int>& contexts, uint32_t seed) {
  IndexerCase c = MakeIndexerCase(shape, 0, seed);
  Random random(seed + 1);
  const int dim = shape.index_dim;
  slots->clear();
  for (int context : contexts) {
    SlotState slot;
    slot.context = context;
    slot.ring.assign(4 * dim, 0.0f);
    for (int t = 0; t < context % 4; ++t) {
      for (int d = 0; d < dim; ++d)
        slot.ring[t * dim + d] = random.Uniform(-1, 1);
    }
    slot.pool.assign(static_cast<size_t>(shape.max_blocks) * dim, 0.0f);
    for (int b = 0; b < context / 4; ++b) {
      for (int d = 0; d < dim; ++d) {
        slot.pool[static_cast<size_t>(b) * dim + d] = random.Uniform(-1, 1);
      }
    }
    if (shape.fp16_state) {
      slot.ring = AsType(slot.ring, kLiteRtNvidiaQsaFloat16).held;
      slot.pool = AsType(slot.pool, kLiteRtNvidiaQsaFloat16).held;
    }
    slots->push_back(std::move(slot));
  }
  HoldInTypes(c);
  return c;
}

void ExpectTopk(const ref::IndexerResult& want, const int32_t* got, int k,
                const std::string& what) {
  std::string why;
  EXPECT_TRUE(ref::IsTopk(want.scores.data(), want.bound, got, k, 2e-5, &why))
      << what << ": " << why;
}

void CheckLinearRows(bool fp16_state, int32_t q_type) {
  ref::IndexerShape shape;
  shape.index_heads = 4;
  shape.index_dim = 32;
  shape.rotary_dim = 16;
  shape.max_blocks = 64;
  shape.block_topk = 8;
  shape.mask_len = 256;
  shape.scale = 0.2f;
  shape.fp16_state = fp16_state;
  // Three slots: 3 rows from 41 (completing a block), one row from 99
  // (completing one), 4 rows from 62; and a slot of one row in its dense
  // start.
  const std::vector<int> contexts = {41, 99, 62, 9};
  const std::vector<int32_t> slot_of = {0, 0, 0, 1, 2, 2, 2, 2, 3};
  shape.rows = static_cast<int>(slot_of.size());
  std::vector<SlotState> slots;
  IndexerCase c = MakeRowsCase(shape, &slots, contexts, 77);
  c.q_type = q_type;
  HoldInTypes(c);
  std::vector<int32_t> context_len;
  for (int32_t s : slot_of) context_len.push_back(contexts[s]);
  const RowsRun run = RunIndexerRows(c, slots, slot_of, context_len, {}, true);
  const int dim = shape.index_dim;
  for (int s = 0; s < static_cast<int>(slots.size()); ++s) {
    std::vector<float> ring = slots[s].ring, pool = slots[s].pool;
    int rank = 0;
    for (int r = 0; r < shape.rows; ++r) {
      if (slot_of[r] != s) continue;
      const auto want =
          DecodeRow(shape, c, r, contexts[s] + rank++, ring, pool);
      ExpectTopk(want,
                 run.topk.data() + static_cast<size_t>(r) * shape.block_topk,
                 shape.block_topk, "row " + std::to_string(r));
    }
    const float tolerance = fp16_state ? 2e-3f : 2e-5f;
    ExpectClose(std::vector<float>(run.ring.begin() + s * 4 * dim,
                                   run.ring.begin() + (s + 1) * 4 * dim),
                ring, tolerance, "ring of slot " + std::to_string(s));
    ExpectClose(
        std::vector<float>(
            run.pool.begin() + static_cast<size_t>(s) * shape.max_blocks * dim,
            run.pool.begin() +
                static_cast<size_t>(s + 1) * shape.max_blocks * dim),
        pool, tolerance, "pool of slot " + std::to_string(s));
  }
}

TEST(QsaIndexerTest, RowModelSlotsMatchDecodeStepByStep) {
  CheckLinearRows(/*fp16_state=*/false, kLiteRtNvidiaQsaFloat32);
  CheckLinearRows(/*fp16_state=*/true, kLiteRtNvidiaQsaBFloat16);
}

TEST(QsaIndexerTest, RowModelTreeScoresEachPathThenCommitsOne) {
  ref::IndexerShape shape;
  shape.index_heads = 4;
  shape.index_dim = 32;
  shape.rotary_dim = 16;
  shape.max_blocks = 64;
  shape.block_topk = 8;
  shape.mask_len = 256;
  shape.scale = 0.2f;
  shape.rows = 7;
  // From context 58: 0 is the root (58), 1 and 2 its children (59), 3 under
  // 1 and 4 under 2 (60, completing block 14 along either path), 5 under 4
  // (61), 6 under 3 (61).
  const int parent[7] = {-1, 0, 0, 1, 2, 4, 3};
  const int context = 58;
  std::vector<SlotState> slots;
  IndexerCase c = MakeRowsCase(shape, &slots, {context}, 91);
  std::vector<uint64_t> ancestors(7, 0);
  for (int j = 0; j < 7; ++j) {
    for (int a = j; a >= 0; a = parent[a]) ancestors[j] |= uint64_t{1} << a;
  }
  const std::vector<int32_t> slot_of(7, 0), context_len(7, context);
  const RowsRun run =
      RunIndexerRows(c, slots, slot_of, context_len, ancestors, true);
  // The state stays.
  EXPECT_EQ(run.ring, slots[0].ring);
  EXPECT_EQ(run.pool, slots[0].pool);
  for (int r = 0; r < 7; ++r) {
    std::vector<float> ring = slots[0].ring, pool = slots[0].pool;
    std::vector<int> path;
    for (int a = r; a >= 0; a = parent[a]) path.insert(path.begin(), a);
    ref::IndexerResult want;
    for (size_t i = 0; i < path.size(); ++i) {
      want = DecodeRow(shape, c, path[i], context + static_cast<int>(i), ring,
                       pool);
    }
    ExpectTopk(want,
               run.topk.data() + static_cast<size_t>(r) * shape.block_topk,
               shape.block_topk, "tree row " + std::to_string(r));
  }
  // The path 0, 2, 4, 5 taken: a linear launch over its rows writes them.
  const std::vector<int> taken = {0, 2, 4, 5};
  IndexerCase commit = c;
  commit.shape.rows = static_cast<int>(taken.size());
  commit.k_new.clear();
  const int dim = shape.index_dim;
  for (int r : taken) {
    commit.k_new.insert(commit.k_new.end(), c.k_new.begin() + r * dim,
                        c.k_new.begin() + (r + 1) * dim);
  }
  const RowsRun committed =
      RunIndexerRows(commit, slots, std::vector<int32_t>(taken.size(), 0),
                     std::vector<int32_t>(taken.size(), context), {}, false);
  std::vector<float> ring = slots[0].ring, pool = slots[0].pool;
  for (size_t i = 0; i < taken.size(); ++i) {
    DecodeRow(shape, c, taken[i], context + static_cast<int>(i), ring, pool);
  }
  ExpectClose(committed.ring, ring, 2e-5f, "committed ring");
  ExpectClose(committed.pool, pool, 2e-5f, "committed pool");
}

// ---------------------------------------------------------------------------
// odml.sparse_sdpa_transposed
// ---------------------------------------------------------------------------

struct AttentionCase {
  ref::AttentionShape shape;
  int32_t q_type = kLiteRtNvidiaQsaFloat32;
  int32_t cache_type = kLiteRtNvidiaQsaFloat32;
  bool bool_mask = true;
  std::vector<float> q, k, v;
  std::vector<int32_t> topk;
  HostMask mask{1, 4};
};

std::vector<float> RunAttention(const AttentionCase& c,
                                const LiteRtNvidiaQsaRows* rows = nullptr,
                                int64_t cache_elements = 0) {
  const auto& s = c.shape;
  LiteRtNvidiaQsaAttentionParams params = {};
  params.rows = s.rows;
  params.q_heads = s.q_heads;
  params.kv_heads = s.kv_heads;
  params.head_dim = s.head_dim;
  params.cache_len = s.cache_len;
  params.block_topk = s.block_topk;
  params.q_type = c.q_type;
  params.cache_type = c.cache_type;
  params.value_columns = s.value_columns ? 1 : 0;
  params.mask_type = rows != nullptr ? kLiteRtNvidiaQsaMaskRows
                     : c.bool_mask   ? kLiteRtNvidiaQsaMaskBool
                                     : kLiteRtNvidiaQsaMaskFloat;
  params.scale = s.scale;
  EXPECT_TRUE(LiteRtNvidiaQsaAttentionSupports(&params));
  const Typed q = AsType(c.q, c.q_type);
  const Typed k = AsType(c.k, c.cache_type);
  const Typed v = AsType(c.v, c.cache_type);
  Device q_d = Upload(q.bytes), k_d = Upload(k.bytes), v_d = Upload(v.bytes);
  Device topk_d = Upload(c.topk);
  Device mask_d = rows != nullptr ? Allocate(4) : c.mask.Upload(c.bool_mask);
  const size_t out_count = static_cast<size_t>(s.rows) * s.q_heads * s.head_dim;
  const size_t width = c.q_type == kLiteRtNvidiaQsaFloat32 ? 4 : 2;
  Device out_d = Allocate(out_count * width);
  Device work_d = Allocate(LiteRtNvidiaQsaAttentionWorkspaceBytes(&params));
  EXPECT_EQ(LiteRtNvidiaLaunchQsaAttention(
                &params, q_d.get(), k_d.get(), v_d.get(),
                static_cast<const int32_t*>(topk_d.get()), mask_d.get(), rows,
                out_d.get(), work_d.get(), nullptr),
            cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  return FromType(Download<uint8_t>(out_d, out_count * width), c.q_type);
}

void CheckAttention(AttentionCase c, const std::string& what) {
  c.q = AsType(c.q, c.q_type).held;
  c.k = AsType(c.k, c.cache_type).held;
  c.v = AsType(c.v, c.cache_type).held;
  const std::vector<float> got = RunAttention(c);
  const auto mask = c.mask.View(c.bool_mask);
  const std::vector<float> want = ref::SparseAttention(
      c.shape, c.q, c.k, c.v, c.topk,
      [&](int row, int token) { return mask.Visible(row, token); });
  // Several rows over FP16 caches run on tensor cores with FP16 queries and
  // probabilities.
  const bool tensor_cores = c.shape.rows > 1 &&
                            c.cache_type == kLiteRtNvidiaQsaFloat16 &&
                            !c.shape.value_columns;
  const float tolerance = c.q_type == kLiteRtNvidiaQsaBFloat16 ? 1e-2f
                          : c.q_type == kLiteRtNvidiaQsaFloat16 || tensor_cores
                              ? 2e-3f
                              : 2e-5f;
  ExpectClose(got, want, tolerance, what);
}

AttentionCase MakeAttentionCase(const ref::AttentionShape& shape,
                                uint32_t seed) {
  AttentionCase c;
  c.shape = shape;
  Random random(seed);
  c.q = random.Values(
      static_cast<size_t>(shape.q_heads) * shape.rows * shape.head_dim, -1.0f,
      1.0f);
  const size_t cache =
      static_cast<size_t>(shape.kv_heads) * shape.cache_len * shape.head_dim;
  c.k = random.Values(cache, -1.0f, 1.0f);
  c.v = random.Values(cache, -1.0f, 1.0f);
  c.mask = HostMask(shape.rows, shape.cache_len);
  return c;
}

// Each row's selection: distinct complete blocks below its position, in
// random order, then -1 (or fewer when not enough exist).
void SelectRandomBlocks(AttentionCase& c, int pos0, uint32_t seed) {
  Random random(seed);
  const auto& s = c.shape;
  c.topk.assign(static_cast<size_t>(s.rows) * s.block_topk, -1);
  for (int r = 0; r < s.rows; ++r) {
    const int complete = (pos0 + r + 1) / 4;
    std::vector<int> blocks(complete);
    for (int b = 0; b < complete; ++b) blocks[b] = b;
    for (int i = complete - 1; i > 0; --i) {
      std::swap(blocks[i], blocks[random.Int(0, i)]);
    }
    for (int i = 0; i < std::min(complete, s.block_topk); ++i) {
      c.topk[static_cast<size_t>(r) * s.block_topk + i] = blocks[i];
    }
  }
}

// The GPU delegate's SparseSdpaSplitKvGqa4VerifyAndPrefillMatchesCpu: depth
// 256, 4 query heads per KV head, NaN in masked values, one selection past
// the visible keys.
TEST(QsaAttentionTest, GpuDelegateGqaCaseWithNanInHiddenValues) {
  for (int rows : {1, 5, 16}) {
    ref::AttentionShape shape;
    shape.rows = rows;
    shape.q_heads = 8;
    shape.kv_heads = 2;
    shape.head_dim = 256;
    shape.cache_len = 128;
    shape.block_topk = 16;
    shape.scale = 1.0f / 16.0f;
    AttentionCase c;
    c.shape = shape;
    c.q = Pattern(8 * rows * 256, 1 + rows);
    c.k = Pattern(2 * 128 * 256, 7);
    c.v = Pattern(2 * 128 * 256, 13);
    for (int kv = 0; kv < 2; ++kv) {
      for (int t = 115; t < 128; ++t) {
        for (int d = 0; d < 256; ++d) c.v[(kv * 128 + t) * 256 + d] = NAN;
      }
    }
    c.topk.assign(rows * 16, -1);
    for (int r = 0; r < rows; ++r) {
      for (int i = 0; i < 16; ++i) {
        c.topk[r * 16 + i] = (i == 15 && r == 0) ? 24 : i + (r % 6);
      }
    }
    c.mask = HostMask(rows, 128);
    for (int r = 0; r < rows; ++r) {
      for (int t = 0; t <= 98 + r; ++t) c.mask.Set(r, t, true);
    }
    for (bool use_bool : {true, false}) {
      c.bool_mask = use_bool;
      CheckAttention(c, "rows " + std::to_string(rows));
    }
  }
}

// Repeated entries count once, entries outside the cache are ignored, a row
// that sees no key outputs zero, and the trailing incomplete block is always
// attended.
TEST(QsaAttentionTest, RepeatedOutOfRangeAndEmptyRows) {
  ref::AttentionShape shape;
  shape.rows = 3;
  shape.q_heads = 4;
  shape.kv_heads = 2;
  shape.head_dim = 64;
  shape.cache_len = 64;
  shape.block_topk = 6;
  shape.scale = 0.125f;
  AttentionCase c = MakeAttentionCase(shape, 5);
  c.topk = {2, 2, 2, 9, -1, 400,  // row 0: block 2 three times
            0, 1, 3, 3, 5,  99,   // row 1
            0, 1, 2, 3, 4,  5};   // row 2 sees nothing
  c.mask.Causal(30, 2);
  CheckAttention(c, "edge selections");
  const auto got = RunAttention(c);
  for (int i = 2 * 4 * 64; i < 3 * 4 * 64; ++i) EXPECT_EQ(got[i], 0.0f);
}

// Random shapes, groups, selections, masks (causal, sliding window, none),
// value layouts and types.
TEST(QsaAttentionTest, RandomCasesMatchTheReference) {
  int seed = 100;
  for (int dim : {64, 128, 256}) {
    for (int group : {1, 2, 3, 4, 6, 8, 12}) {
      ++seed;
      Random random(seed);
      ref::AttentionShape shape;
      shape.rows = std::vector<int>{1, 2, 3, 4, 5, 8, 17}[seed % 7];
      shape.kv_heads = std::vector<int>{1, 2, 4}[seed % 3];
      shape.q_heads = shape.kv_heads * group;
      shape.head_dim = dim;
      shape.cache_len = std::vector<int>{64, 1027, 4096}[(seed / 3) % 3];
      shape.block_topk = std::vector<int>{1, 8, 64, 512}[seed % 4];
      shape.scale = 1.0f / std::sqrt(static_cast<float>(dim));
      shape.value_columns = seed % 2 == 1;
      AttentionCase c = MakeAttentionCase(shape, seed);
      c.q_type = (seed % 3 == 0)   ? kLiteRtNvidiaQsaFloat32
                 : (seed % 3 == 1) ? kLiteRtNvidiaQsaFloat16
                                   : kLiteRtNvidiaQsaBFloat16;
      c.cache_type =
          seed % 2 == 0 ? kLiteRtNvidiaQsaFloat16 : kLiteRtNvidiaQsaFloat32;
      c.bool_mask = seed % 4 < 2;
      const int pos0 = random.Int(0, shape.cache_len - shape.rows);
      const int window = seed % 5 == 0 ? 37 : 0;
      c.mask = HostMask(shape.rows, shape.cache_len,
                        seed % 3 == 0 ? -INFINITY : -10000.0f);
      c.mask.Causal(pos0, shape.rows, window);
      SelectRandomBlocks(c, pos0, seed);
      CheckAttention(
          c, "dim " + std::to_string(dim) + " group " + std::to_string(group));
    }
  }
}

// The 27B's attention: 24 query heads on 4 KV heads of 256, FP16 caches,
// 512 selected blocks; decode, verify of 4 and 8 rows at 32k, decode at
// 128k.
TEST(QsaAttentionTest, ModelShapesMatchTheReference) {
  struct Case {
    int cache_len;
    int rows;
  };
  // 32,771 keys: the Qwen 3.8 export's cache, past a multiple of 4.
  for (const Case& shape_case :
       {Case{32768, 1}, Case{32768, 4}, Case{32768, 8}, Case{131072, 1},
        Case{32771, 1}, Case{32771, 8}}) {
    ref::AttentionShape shape;
    shape.rows = shape_case.rows;
    shape.q_heads = 24;
    shape.kv_heads = 4;
    shape.head_dim = 256;
    shape.cache_len = shape_case.cache_len;
    shape.block_topk = 512;
    shape.scale = 1.0f / 16.0f;
    AttentionCase c = MakeAttentionCase(shape, 77);
    c.q_type = kLiteRtNvidiaQsaBFloat16;
    c.cache_type = kLiteRtNvidiaQsaFloat16;
    const int pos0 = shape.cache_len - 40;
    c.mask.Causal(pos0, shape.rows);
    SelectRandomBlocks(c, pos0, 78);
    CheckAttention(c, "model " + std::to_string(shape_case.cache_len) + " x " +
                          std::to_string(shape_case.rows));
  }
}

// The row model: a linear chain gives what the mask gives; a tree's rows see
// their ancestors' keys at their scratch indices; two slots in one buffer.
// The row model on the CUDA-core path (FP32 caches) and on the tensor-core
// path (FP16 caches, values as rows).
void CheckRowModel(int32_t cache_type) {
  const bool tensor_cores = cache_type == kLiteRtNvidiaQsaFloat16;
  const float tolerance = tensor_cores ? 2e-3f : 2e-5f;
  ref::AttentionShape shape;
  shape.rows = 6;
  shape.q_heads = 8;
  shape.kv_heads = 2;
  shape.head_dim = 128;
  shape.cache_len = 256;
  shape.block_topk = 8;
  shape.scale = 0.1f;
  const int context = 101;  // committed keys [0, 101)

  // Linear chain: rows at 101..106 written at their positions.
  AttentionCase c = MakeAttentionCase(shape, 9);
  c.cache_type = cache_type;
  c.k = AsType(c.k, cache_type).held;
  c.v = AsType(c.v, cache_type).held;
  c.mask.Causal(context, 6);
  SelectRandomBlocks(c, context, 10);
  std::vector<int32_t> context_len(6, context);
  std::vector<int32_t> write_index(6);
  for (int j = 0; j < 6; ++j) write_index[j] = context + j;
  Device context_d = Upload(context_len), write_d = Upload(write_index);
  LiteRtNvidiaQsaRows rows = {};
  rows.context_len = static_cast<const int32_t*>(context_d.get());
  rows.write_index = static_cast<const int32_t*>(write_d.get());
  ExpectClose(RunAttention(c, &rows), RunAttention(c),
              tensor_cores ? 1e-3f : 1e-6f, "linear chain");

  // A tree: 0 is the root, 1 and 2 its children, 3 under 1, 4 and 5 under
  // 2; rows write to scratch indices 200 + j.
  const int parent[6] = {-1, 0, 0, 1, 2, 2};
  std::vector<uint64_t> ancestors(6, 0);
  for (int j = 0; j < 6; ++j) {
    for (int a = j; a >= 0; a = parent[a]) ancestors[j] |= uint64_t{1} << a;
    write_index[j] = 200 + j;
  }
  Device ancestors_d = Upload(ancestors);
  Device tree_write_d = Upload(write_index);
  rows.ancestors = static_cast<const uint64_t*>(ancestors_d.get());
  rows.write_index = static_cast<const int32_t*>(tree_write_d.get());
  const auto visible = [&](int row, int token) {
    if (token < context) return true;
    for (int j = 0; j < 6; ++j) {
      if (((ancestors[row] >> j) & 1u) != 0 && write_index[j] == token) {
        return true;
      }
    }
    return false;
  };
  const std::vector<float> tree_want =
      ref::SparseAttention(c.shape, c.q, c.k, c.v, c.topk, visible);
  ExpectClose(RunAttention(c, &rows), tree_want, tolerance, "tree");

  // Two slots: rows 0..2 in slot 0, 3..5 in slot 1, each a linear chain of
  // its own; slot 1's caches follow slot 0's in the same buffers.
  AttentionCase slots = c;
  const size_t per_slot =
      static_cast<size_t>(shape.kv_heads) * shape.cache_len * shape.head_dim;
  Random random(12);
  const std::vector<float> k1 =
      AsType(random.Values(per_slot, -1.0f, 1.0f), cache_type).held;
  const std::vector<float> v1 =
      AsType(random.Values(per_slot, -1.0f, 1.0f), cache_type).held;
  slots.k.insert(slots.k.end(), k1.begin(), k1.end());
  slots.v.insert(slots.v.end(), v1.begin(), v1.end());
  std::vector<int32_t> slot_of = {0, 0, 0, 1, 1, 1};
  std::vector<int32_t> slot_context = {context, context, context, 60, 60, 60};
  std::vector<int32_t> slot_write = {context, context + 1, context + 2,
                                     60,      61,          62};
  Device slot_d = Upload(slot_of), slot_context_d = Upload(slot_context);
  Device slot_write_d = Upload(slot_write);
  LiteRtNvidiaQsaRows slot_rows = {};
  slot_rows.context_len = static_cast<const int32_t*>(slot_context_d.get());
  slot_rows.write_index = static_cast<const int32_t*>(slot_write_d.get());
  slot_rows.slot = static_cast<const int32_t*>(slot_d.get());
  slot_rows.slot_stride = static_cast<int64_t>(per_slot);
  const std::vector<float> got = RunAttention(slots, &slot_rows);
  for (int slot = 0; slot < 2; ++slot) {
    AttentionCase one = c;
    one.shape.rows = 3;
    one.q.clear();
    for (int h = 0; h < shape.q_heads; ++h) {
      for (int r = 0; r < 3; ++r) {
        const float* row =
            c.q.data() + (static_cast<size_t>(h) * 6 + slot * 3 + r) * 128;
        one.q.insert(one.q.end(), row, row + 128);
      }
    }
    one.topk.assign(c.topk.begin() + slot * 3 * 8,
                    c.topk.begin() + (slot * 3 + 3) * 8);
    if (slot == 1) {
      one.k = k1;
      one.v = v1;
    }
    const int base = slot_context[slot * 3];
    const auto want = ref::SparseAttention(
        one.shape, one.q, one.k, one.v, one.topk,
        [&](int row, int token) { return token <= base + row; });
    const std::vector<float> slice(
        got.begin() + static_cast<size_t>(slot) * 3 * 8 * 128,
        got.begin() + static_cast<size_t>(slot + 1) * 3 * 8 * 128);
    ExpectClose(slice, want, tolerance, "slot " + std::to_string(slot));
  }
}

TEST(QsaAttentionTest, RowModelLinearChainTreeAndSlots) {
  CheckRowModel(kLiteRtNvidiaQsaFloat32);
  CheckRowModel(kLiteRtNvidiaQsaFloat16);
}

// Batched decode: a row per slot, each slot's caches its own.
TEST(QsaAttentionTest, RowModelBatchedDecode) {
  ref::AttentionShape shape;
  shape.rows = 5;
  shape.q_heads = 8;
  shape.kv_heads = 2;
  shape.head_dim = 128;
  shape.cache_len = 512;
  shape.block_topk = 16;
  shape.scale = 0.1f;
  for (int32_t cache_type :
       {kLiteRtNvidiaQsaFloat32, kLiteRtNvidiaQsaFloat16}) {
    const size_t per_slot =
        static_cast<size_t>(shape.kv_heads) * shape.cache_len * shape.head_dim;
    AttentionCase c = MakeAttentionCase(shape, 31);
    c.cache_type = cache_type;
    Random random(32);
    c.k = AsType(random.Values(per_slot * 5, -1.0f, 1.0f), cache_type).held;
    c.v = AsType(random.Values(per_slot * 5, -1.0f, 1.0f), cache_type).held;
    const std::vector<int32_t> slot_of = {3, 0, 4, 1, 2};
    const std::vector<int32_t> context_len = {300, 17, 511, 64, 129};
    std::vector<int32_t> write_index(5);
    for (int r = 0; r < 5; ++r) write_index[r] = context_len[r];
    c.mask.Causal(0, 0);
    SelectRandomBlocks(c, 0, 33);
    for (int r = 0; r < 5; ++r) {
      // Complete blocks below each row's own position.
      Random pick(40 + r);
      const int complete = (context_len[r] + 1) / 4;
      for (int i = 0; i < shape.block_topk; ++i) {
        c.topk[static_cast<size_t>(r) * shape.block_topk + i] =
            i < complete ? pick.Int(0, complete - 1) : -1;
      }
    }
    Device slot_d = Upload(slot_of), context_d = Upload(context_len);
    Device write_d = Upload(write_index);
    LiteRtNvidiaQsaRows rows = {};
    rows.context_len = static_cast<const int32_t*>(context_d.get());
    rows.write_index = static_cast<const int32_t*>(write_d.get());
    rows.slot = static_cast<const int32_t*>(slot_d.get());
    rows.slot_stride = static_cast<int64_t>(per_slot);
    const std::vector<float> got = RunAttention(c, &rows);
    for (int r = 0; r < 5; ++r) {
      AttentionCase one = c;
      one.shape.rows = 1;
      one.q.clear();
      for (int h = 0; h < shape.q_heads; ++h) {
        const float* row =
            c.q.data() + (static_cast<size_t>(h) * 5 + r) * shape.head_dim;
        one.q.insert(one.q.end(), row, row + shape.head_dim);
      }
      one.topk.assign(c.topk.begin() + r * shape.block_topk,
                      c.topk.begin() + (r + 1) * shape.block_topk);
      one.k.assign(c.k.begin() + slot_of[r] * per_slot,
                   c.k.begin() + (slot_of[r] + 1) * per_slot);
      one.v.assign(c.v.begin() + slot_of[r] * per_slot,
                   c.v.begin() + (slot_of[r] + 1) * per_slot);
      const int position = context_len[r];
      const auto want = ref::SparseAttention(
          one.shape, one.q, one.k, one.v, one.topk,
          [&](int, int token) { return token <= position; });
      const size_t width = static_cast<size_t>(shape.q_heads) * shape.head_dim;
      const std::vector<float> slice(got.begin() + r * width,
                                     got.begin() + (r + 1) * width);
      ExpectClose(slice, want,
                  cache_type == kLiteRtNvidiaQsaFloat16 ? 2e-3f : 2e-5f,
                  "batched row " + std::to_string(r));
    }
  }
}

}  // namespace

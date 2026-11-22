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

#include "litert/vendors/nvidia/experimental/megakernel/megakernel.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <type_traits>

#include "litert/vendors/nvidia/experimental/megakernel/block_sync.cuh"
#include "litert/vendors/nvidia/experimental/megakernel/decode_attention.cuh"
#include "litert/vendors/nvidia/experimental/megakernel/int4_block_gemv.cuh"
#include "litert/vendors/nvidia/experimental/megakernel/program.h"
#include "litert/vendors/nvidia/experimental/megakernel/qkv_norm_rope.cuh"
#include "litert/vendors/nvidia/trtllm/decode_attention.h"
#include "litert/vendors/nvidia/trtllm/int4_block_gemv.h"

namespace litert::nvidia::megakernel {
namespace {

namespace attention = litert::nvidia::megakernel::decode_attention;
namespace gemv = litert::nvidia::megakernel::int4_block;
namespace rope = litert::nvidia::megakernel::qkv_norm_rope;

// A CTA: eight consumer warps, which run the instructions' items, and a
// producer warp, which streams the projections' weights into a ring in
// shared memory ahead of them, across instruction and layer boundaries.
constexpr int kConsumerThreads = 256;
constexpr int kConsumerWarps = kConsumerThreads / 32;
constexpr int kThreads = kLiteRtNvidiaMegakernelThreads;
constexpr int kProducerWarp = kConsumerWarps;
static_assert(kThreads == kConsumerThreads + 32, "a producer warp");
// The consumers' barrier: the producer never waits at it.
using ConsumerSync = NamedSync<1, kConsumerThreads>;

constexpr int kBlock = gemv::kBlock;  // input dims per scale
constexpr unsigned kFull = 0xffffffffu;
constexpr int kMaxModelDim = 3072;
constexpr int kMaxTopK = 16;
constexpr int kMaxExperts = 128;  // one warp routes them, four a lane
// decode_attention.cu: a block per chunk of keys up to kMaxSplits per head.
constexpr int kMaxSplits = 168;
// The mask fill of odml.sdpa_transposed's decomposition.
constexpr float kAttentionFill = -10000.0f;
// A consumer thread's quads of a model-dim vector (the prologues).
constexpr int kQuads = kMaxModelDim / 4 / kConsumerThreads;
// The attention items the kernel is built for: (depth, row cap, chunk).
constexpr int kLocalDepth = 256;
constexpr int kLocalRows = 4;
constexpr int kGlobalDepth = 512;
constexpr int kGlobalRows = 8;
constexpr int kChunk = 64;
// A wait longer than this is a deadlock: the launch traps.
constexpr uint64_t kWaitLimitNs = 2000000000ull;
// The weight ring: kStages stages of kStageBytes, each a unit of up to
// kMaxUnitRows rows of one projection (weights, then scales).
constexpr int kStages = 10;
constexpr int kStageBytes = 6400;
// Units in flight from DRAM per CTA. Two or three already reach the DRAM's
// rate; every unit more only queues in front of the step's other loads (the
// norms' inputs, the attention's keys, the counters), whose L2 hits then
// take microseconds instead of a fraction of one (measured with 2 to 10
// stages in flight: 0.36 us at 2, 2.6 us at 10).
// The other stages hold units that have arrived, ready for the consumers.
constexpr int kInFlight = 3;
constexpr int kMaxUnitRows = 8;  // a dense projection's
// The experts' units: their gate's or up's rows, and their down
// projection's rows of one slot (narrow: an expert's hidden size of at most
// kMaxDownBlocks blocks).
constexpr int kExpertRows = 4;
constexpr int kDownRows = 16;
constexpr int kMaxDownBlocks = 22;
// A CTA's items of a projection whose items take several units (gate and
// up, or the slots), and their parts' row sums, in shared memory.
constexpr int kMaxCtaItems = 32;
constexpr int kPartialFloats = 512;

static_assert(kConsumerThreads == attention::ThreadsFor<kLocalDepth>() &&
                  kConsumerThreads == attention::ThreadsFor<kGlobalDepth>() &&
                  kConsumerThreads == attention::kCombineThreads &&
                  kConsumerThreads == rope::kThreads,
              "the consumers run the per-op kernels' blocks as they are");
static_assert(kQuads * 4 * kConsumerThreads == kMaxModelDim, "quads");

__device__ __forceinline__ void Sync() { ConsumerSync::Sync(); }

// The launches' persistent state (LiteRtNvidiaMegakernelStateBytes).
struct State {
  uint32_t epoch;     // steps completed
  uint32_t finished;  // CTAs done with the step, over all steps
  uint32_t reserved[30];
  uint32_t done[kLiteRtNvidiaMegakernelMaxInstructions];
};

// --- Shapes shared by the builder and the kernel -----------------------------

__host__ __device__ constexpr int RoundUp16(int bytes) {
  return (bytes + 15) / 16 * 16;
}

__host__ __device__ constexpr int Gcd(int a, int b) {
  return b == 0 ? a : Gcd(b, a % b);
}

// The compact scales' row maxima a unit of `unit_rows` rows copies: the 16
// bytes from its first row & ~7, or (16 rows) its own.
__host__ __device__ constexpr int MaxWindowBytes(int unit_rows) {
  return unit_rows <= 8 ? 16 : RoundUp16(unit_rows * 2);
}

// A unit's bytes in its stage: the weights, the scales from rows *
// row_bytes, the maxima's window after them.
__host__ __device__ constexpr int UnitBytes(int unit_rows, int cols,
                                           bool compact) {
  return compact ? unit_rows * cols / 2 + RoundUp16(unit_rows * cols / kBlock) +
                       MaxWindowBytes(unit_rows)
                 : unit_rows * (cols / 2 + cols / kBlock * 2);
}

// A dense projection's unit (its rows, of `cols` inputs, with FP16
// scales).
struct UnitShape {
  int rows;
};

// The rows of a unit of a dense projection of `rows` rows and `cols`
// inputs: as many as a stage holds, at most kMaxUnitRows, a power of two
// whose scales end on 16 bytes (bulk copies move multiples of 16 bytes
// between 16-byte addresses), dividing `rows`; 0 if there is none.
__host__ __device__ inline UnitShape UnitShapeFor(int rows, int cols) {
  UnitShape s{0};
  const int scale_row = cols / kBlock * 2;
  const int align = 16 / Gcd(scale_row, 16);
  int unit = kStageBytes / UnitBytes(1, cols, false);
  unit = unit < kMaxUnitRows ? unit : kMaxUnitRows;
  int pow2 = 1;
  while (pow2 * 2 <= unit) pow2 *= 2;
  unit = pow2;
  if (unit % align != 0 || rows % unit != 0 || (cols / 2) % 16 != 0) return s;
  s.rows = unit;
  return s;
}

// Dynamic shared memory of a CTA: the layer input x and h = x +
// rms_norm(o) (BF16), the routing of two layers, the items' partial sums, a
// reduction scratch, the control words, the ring's barriers, `work` (a
// projection's input, the attention's blocks, the prologues' staging) and
// the ring.
struct SharedLayout {
  int x;
  int h;
  int route;
  int partial;
  int reduce;
  int control;
  int barriers;
  int work;
  int ring;
  int total;
};

constexpr int kReduceFloats = 64;
constexpr int kControlWords = 16;
// Experts, weights, weights times the experts' scales; rsqrt(mean(h^2) +
// eps).
constexpr int kRouteBytes = 2 * kMaxTopK * 12 + 16;

__host__ __device__ inline SharedLayout LayoutShared(int model_dim,
                                                     int work_bytes) {
  SharedLayout s{};
  int at = 0;
  s.x = at;
  at += RoundUp16(model_dim * 2);
  s.h = at;
  at += RoundUp16(model_dim * 2);
  s.route = at;
  at += kRouteBytes;
  s.partial = at;
  at += kPartialFloats * 4 + kMaxCtaItems * 4;
  s.reduce = at;
  at += kReduceFloats * 4;
  s.control = at;
  at += kControlWords * 4;
  s.barriers = at;
  at += 2 * kStages * 8;
  at = (at + 127) / 128 * 128;
  s.work = at;
  at += (work_bytes + 127) / 128 * 128;
  s.ring = at;
  at += kStages * kStageBytes;
  s.total = at;
  return s;
}

bool IsProjection(int opcode) {
  return opcode == kLiteRtNvidiaMegakernelOpQkv ||
         opcode == kLiteRtNvidiaMegakernelOpOutput ||
         opcode == kLiteRtNvidiaMegakernelOpRouter ||
         opcode == kLiteRtNvidiaMegakernelOpDenseGateUp ||
         opcode == kLiteRtNvidiaMegakernelOpExpertHidden ||
         opcode == kLiteRtNvidiaMegakernelOpDenseDown ||
         opcode == kLiteRtNvidiaMegakernelOpExpertDown ||
         opcode == kLiteRtNvidiaMegakernelOpHead;
}

__device__ __forceinline__ bool IsProjectionOp(int opcode) {
  return opcode == kLiteRtNvidiaMegakernelOpQkv ||
         opcode == kLiteRtNvidiaMegakernelOpOutput ||
         opcode == kLiteRtNvidiaMegakernelOpRouter ||
         opcode == kLiteRtNvidiaMegakernelOpDenseGateUp ||
         opcode == kLiteRtNvidiaMegakernelOpExpertHidden ||
         opcode == kLiteRtNvidiaMegakernelOpDenseDown ||
         opcode == kLiteRtNvidiaMegakernelOpExpertDown ||
         opcode == kLiteRtNvidiaMegakernelOpHead;
}

// --- Small device helpers -----------------------------------------------------

__device__ __forceinline__ uint64_t GlobalTimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ uint32_t SmId() {
  uint32_t id;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}

__device__ __forceinline__ uint32_t LoadAcquire(const uint32_t* address) {
  uint32_t value;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(address)
               : "memory");
  return value;
}

// Waits until every CTA has finished instruction `index` in this step.
__device__ __forceinline__ void WaitDone(const State* state, int index,
                                         uint32_t target) {
  const uint32_t* done = &state->done[index];
  const uint64_t since = GlobalTimer();
  while (static_cast<int32_t>(LoadAcquire(done) - target) < 0) {
    // A program whose waits never end (a CTA not resident, a broken state)
    // fails the launch instead of holding the GPU.
    if (GlobalTimer() - since > kWaitLimitNs) __trap();
  }
}

__device__ __forceinline__ uint32_t SharedAddress(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

// A 16-byte copy from global (L2: .cg, never a stale L1 line) to shared
// memory, in flight until CopyAsyncWait.
__device__ __forceinline__ void CopyAsync16(void* shared, const void* global) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :
               : "r"(SharedAddress(shared)), "l"(global)
               : "memory");
}

// This thread's copies are in shared memory (the caller syncs).
__device__ __forceinline__ void CopyAsyncWait() {
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void InitBarrier(uint64_t* barrier) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(
                   SharedAddress(barrier))
               : "memory");
}

// The stage's copies of `bytes` in all complete the barrier's phase.
__device__ __forceinline__ void ExpectBytes(uint64_t* barrier, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
          SharedAddress(barrier)),
      "r"(bytes)
      : "memory");
}

// An L2 policy for data read once per step (the weights): evicted first, so
// that streaming them does not push out of L2 what the step reads again
// soon (its code, the norms' scales, the activations, the counters).
__device__ __forceinline__ uint64_t EvictFirstPolicy() {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n"
               : "=l"(policy));
  return policy;
}

// The copy's target is named as the CTA's shared memory, not the cluster's:
// on sm_120 the cluster form is a call into the driver's runtime
// (__cuda_syscall_cp_async_bulk_unicast), whose stack the driver reserves for
// every thread the device can hold (14.4 KB each, 1.65 GiB on an RTX 5080,
// at the kernel's first launch), while the CTA form is one instruction.
__device__ __forceinline__ void BulkCopy(void* target, const void* source,
                                         uint32_t bytes, uint64_t* barrier,
                                         uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
      ".L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(SharedAddress(target)),
      "l"(source), "r"(bytes), "r"(SharedAddress(barrier)), "l"(policy)
      : "memory");
}

__device__ __forceinline__ void Arrive(uint64_t* barrier) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(
                   SharedAddress(barrier))
               : "memory");
}

// Waits until the phase of `barrier` with parity `phase` has completed.
__device__ __forceinline__ void WaitParity(uint64_t* barrier, uint32_t phase) {
  asm volatile(
      "{\n"
      ".reg .pred done;\n"
      "WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
      "@!done bra WAIT;\n"
      "}\n" ::"r"(SharedAddress(barrier)),
      "r"(phase)
      : "memory");
}

__device__ __forceinline__ int WarpMax(int value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value = max(value, __shfl_xor_sync(kFull, value, offset));
  }
  return value;
}

__device__ __forceinline__ float Bf16(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

// The XOR butterfly sum of a warp (every lane ends with the same bits).
__device__ __forceinline__ float ButterflySum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(kFull, value, offset);
  }
  return value;
}

// Element k of an FP32 projection input of `blocks` blocks of 32 in shared
// memory: float4 q = (k % 32) / 4 of block b = k / 32 is at q * blocks + b,
// so that the lanes of a warp, a block each, read consecutive float4s.
__device__ __forceinline__ int Swizzled(int k, int blocks) {
  return (((k & 31) >> 2) * blocks + (k >> 5)) * 4 + (k & 3);
}

// A cache's row for the token: the ring's slot or the position clamped to
// the cache (megakernel_steps.cu's CacheRow).
__device__ __forceinline__ int32_t CacheRow(int32_t position, int32_t len,
                                            int32_t ring) {
  if (ring != 0) return ((position % len) + len) % len;
  return position < 0 ? 0 : (position >= len ? len - 1 : position);
}

// --- The work of a CTA ---------------------------------------------------------

// Everything an instruction's code needs, in registers or shared memory.
struct Context {
  const LiteRtNvidiaMegakernelProgram* program;
  const LiteRtNvidiaMegakernelStep* step;
  char* workspace;
  State* state;
  uint32_t target;      // every CTA's count of an instruction this step
  __nv_bfloat16* x_s;
  __nv_bfloat16* h_s;
  int32_t* route_s;     // [2][kMaxTopK] experts of a layer, by its parity
  float* weight_s;      // [2][kMaxTopK] its routing weights
  float* scaled_s;      // [2][kMaxTopK] the weights times the experts' scales
  float* partial_s;     // [kPartialFloats]: items' parts' row sums
  volatile int* counter_s;  // [kMaxCtaItems]: items' parts delivered
  float* reduce_s;      // [kReduceFloats]
  volatile int* control;  // [kControlWords]: see kControl*
  uint64_t* full;       // [kStages]: a stage's copies are in
  uint64_t* empty;      // [kStages]: a stage's unit is consumed
  float* work;          // the instruction's shared memory
  char* ring;           // [kStages][kStageBytes]
  int cta;
  int ctas;
  // The CTA's place in the round robin of the current projection's items.
  // Each projection's items continue the round robin where the previous
  // one's left off, so that the CTAs that take one item more than the
  // others differ from one projection to the next.
  int vcta;
  uint64_t* wait_stamp;      // the trace's stamps of the waits' end (2)
  uint64_t* prologue_stamp;  // the trace's stamp of the prologue's end
  float* hsq_s;              // the layer's rsqrt(mean(h^2) + eps)
};

// The control words.
constexpr int kControlNextItem = 0;  // the consumers' next unit
constexpr int kControlRouted = 1;    // [2]: layer + 1 routed, by parity
constexpr int kControlRouteClaim = 3;  // the last layer + 1 a warp routes
// [kStages]: the unit each stage was last armed for. A consumer may take
// its unit's stage two phases ahead of it (its warp took a later item
// early); a parity wait cannot tell those phases apart, so it first waits
// until the producer has armed the stage for its unit.
constexpr int kControlArmed = 4;
static_assert(kControlArmed + kStages <= kControlWords, "control words");

__device__ __forceinline__ void PrologueDone(const Context& c) {
  if (c.prologue_stamp != nullptr && threadIdx.x == 0) {
    *c.prologue_stamp = GlobalTimer();
  }
}

template <typename T>
__device__ __forceinline__ T* At(const Context& c, uint64_t offset) {
  return reinterpret_cast<T*>(c.workspace + offset);
}

// The other items go to whole consumer groups, item i to CTA i % ctas.
template <typename F>
__device__ __forceinline__ void ForItems(const Context& c, int items, F&& f) {
  for (int i = c.cta; i < items; i += c.ctas) {
    f(i);
    Sync();  // the item's shared memory is the next one's
  }
}

// The items of an instruction of `items` that go to round-robin place
// `vcta` of `ctas`: vcta, vcta + ctas, ...
__device__ __forceinline__ int ItemsFor(int vcta, int ctas, int items) {
  return vcta < items ? (items - vcta + ctas - 1) / ctas : 0;
}

// A projection's items for this CTA: c.vcta, c.vcta + ctas, ...
__device__ __forceinline__ int ItemsOf(const Context& c, int items) {
  return ItemsFor(c.vcta, c.ctas, items);
}

// --- Staging -------------------------------------------------------------------

// A BF16 input in global memory written in this step by other CTAs into
// `act` (16-byte word q of block b at q * blocks + b), every copy in flight
// at once. The caller syncs.
__device__ __forceinline__ void StageBf16(const __nv_bfloat16* src, int count,
                                          __nv_bfloat16* act) {
  const int blocks = count / kBlock;
  for (int i = threadIdx.x; i < count / 8; i += kConsumerThreads) {
    CopyAsync16(act + ((i & 3) * blocks + (i >> 2)) * 8, src + 8 * i);
  }
  CopyAsyncWait();
}

// The experts' hidden rows (FP32 [slots, hidden], this step's) into `act`,
// each slot's row Swizzled at act + slot * hidden. The caller syncs.
__device__ __forceinline__ void StageExpertHidden(const float* src, int slots,
                                                  int hidden, float* act) {
  const int row_quads = hidden / 4;
  const int blocks = hidden / kBlock;
  for (int i = threadIdx.x; i < slots * row_quads; i += kConsumerThreads) {
    const int slot = i / row_quads;
    const int k = (i - slot * row_quads) * 4;
    CopyAsync16(act + slot * hidden + Swizzled(k, blocks), src + 4 * i);
  }
  CopyAsyncWait();
}

// --- The prologues: the small math of a model-dim vector -----------------------
//
// A consumer thread owns quads (4 consecutive elements) threadIdx.x + j *
// kConsumerThreads, j < kQuads, of every model-dim vector. Its loads of a
// prologue are issued together: the weights' vectors (norm scales) before
// the instruction's wait, the step's vectors after it, so that a prologue
// costs one round trip; its sums are block reductions in shared memory
// (FP32, in a fixed order). The rounding points are the graph's: BF16 where
// it holds BF16 (x, h, the attention residual's norm), FP32 inside the
// fused kernels (the block output, the fused norms).

__device__ __forceinline__ int QuadOf(int j) {
  return threadIdx.x + j * kConsumerThreads;
}

// Quad q of an FP32 vector the step does not write (L1 may keep it), of one
// it writes (L2), and of BF16 vectors in global or shared memory.
__device__ __forceinline__ float4 WeightQuad(const float* v, int q) {
  return __ldg(reinterpret_cast<const float4*>(v) + q);
}
__device__ __forceinline__ float4 StepQuad(const float* v, int q) {
  return __ldcg(reinterpret_cast<const float4*>(v) + q);
}
__device__ __forceinline__ float4 Bf16Quad(uint2 bits) {
  const float2 lo =
      __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits.x));
  const float2 hi =
      __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits.y));
  return make_float4(lo.x, lo.y, hi.x, hi.y);
}
__device__ __forceinline__ float4 StepBf16Quad(const __nv_bfloat16* v, int q) {
  return Bf16Quad(__ldcg(reinterpret_cast<const uint2*>(v) + q));
}
__device__ __forceinline__ float4 SharedBf16Quad(const __nv_bfloat16* v,
                                                 int q) {
  return Bf16Quad(reinterpret_cast<const uint2*>(v)[q]);
}
__device__ __forceinline__ void StoreBf16Quad(__nv_bfloat16* v, int q,
                                              float4 x) {
  const __nv_bfloat162 lo = __floats2bfloat162_rn(x.x, x.y);
  const __nv_bfloat162 hi = __floats2bfloat162_rn(x.z, x.w);
  uint2 bits;
  bits.x = *reinterpret_cast<const uint32_t*>(&lo);
  bits.y = *reinterpret_cast<const uint32_t*>(&hi);
  reinterpret_cast<uint2*>(v)[q] = bits;
}

__device__ __forceinline__ float SumSquares(float4 v) {
  return fmaf(v.w, v.w, fmaf(v.z, v.z, fmaf(v.y, v.y, v.x * v.x)));
}

// The block sums of kN values per thread: each warp's butterfly, then every
// thread adds the warps' sums in warp order. Every consumer thread returns
// the same sums.
template <int kN>
__device__ __forceinline__ void BlockSums(const Context& c, float (&v)[kN]) {
  static_assert(kN * kConsumerWarps <= kReduceFloats, "reduce_s");
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x / 32;
#pragma unroll
  for (int n = 0; n < kN; ++n) v[n] = ButterflySum(v[n]);
  Sync();  // reduce_s is free
  if (lane == 0) {
#pragma unroll
    for (int n = 0; n < kN; ++n) c.reduce_s[n * kConsumerWarps + warp] = v[n];
  }
  Sync();
#pragma unroll
  for (int n = 0; n < kN; ++n) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < kConsumerWarps; ++w) {
      total += c.reduce_s[n * kConsumerWarps + w];
    }
    v[n] = total;
  }
}

// x (a quad per j, FP32 of BF16 values) times the norm's scale into the
// projection input `act` (FP32, Swizzled); returns the factor
// rsqrt(mean(x^2) + eps) * multiplier the projection's sums take.
__device__ __forceinline__ float NormedInput(const Context& c,
                                             const float4 (&x)[kQuads],
                                             const float4 (&scale)[kQuads],
                                             float multiplier, float* act) {
  const int dim = c.program->model_dim;
  const int blocks = dim / kBlock;
  float squares[1] = {0.0f};
#pragma unroll
  for (int j = 0; j < kQuads; ++j) {
    const int q = QuadOf(j);
    if (4 * q >= dim) continue;
    squares[0] += SumSquares(x[j]);
    *reinterpret_cast<float4*>(act + Swizzled(4 * q, blocks)) =
        make_float4(__fmul_rn(x[j].x, scale[j].x), __fmul_rn(x[j].y, scale[j].y),
                    __fmul_rn(x[j].z, scale[j].z), __fmul_rn(x[j].w, scale[j].w));
  }
  BlockSums(c, squares);
  return rsqrtf(squares[0] / static_cast<float>(dim) + c.program->norm_eps) *
         multiplier;
}

// A norm's scale, this thread's quads (zero past the vector).
__device__ __forceinline__ void LoadScale(const Context& c, const float* scale,
                                          float4 (&out)[kQuads]) {
  const int dim = c.program->model_dim;
#pragma unroll
  for (int j = 0; j < kQuads; ++j) {
    const int q = QuadOf(j);
    out[j] = 4 * q < dim ? WeightQuad(scale, q) : make_float4(0, 0, 0, 0);
  }
}

// The instruction's inputs are in: thread 0 waits for the instructions it
// depends on; the consumers' unit counter and the items' counters restart.
__device__ __forceinline__ void Begin(const Context& c,
                                      const LiteRtNvidiaMegakernelInstruction& ins) {
  if (threadIdx.x == 0) {
    for (int w = 0; w < kLiteRtNvidiaMegakernelMaxWaits; ++w) {
      if (ins.waits[w] >= 0) WaitDone(c.state, ins.waits[w], c.target);
    }
    c.control[kControlNextItem] = 0;
  }
  if (threadIdx.x < kMaxCtaItems) c.counter_s[threadIdx.x] = 0;
  Sync();
  if (c.wait_stamp != nullptr && threadIdx.x == 0) {
    c.wait_stamp[0] = GlobalTimer();
    c.wait_stamp[1] = c.wait_stamp[0];  // no prologue
  }
}

// The layer input x of layer l (BF16, into x_s; CTA 0 also writes it to
// the scratch, which the lab compares) and the input of the projection
// that takes it, x times `scale` into `act`; returns that projection's
// factor. Layer 0's x is the step's embeddings; another layer's is the
// previous layer block's output (MoeBlockCombineKernel<false>, decode):
// x = (h + rms(rms(dense) * dense_post + rms(moe) * moe_post) * sum_post) *
// multiplier, from h (shared memory), the dense feed-forward's output and
// the experts' weighted sum (this step's). Begins the instruction once the
// weights' vectors are in flight.
__device__ float LayerInputNormed(const Context& c,
                                  const LiteRtNvidiaMegakernelInstruction& ins,
                                  int l, const float* scale, float* act) {
  const auto& program = *c.program;
  const int dim = program.model_dim;
  const float eps = program.norm_eps;
  float4 norm[kQuads];
  LoadScale(c, scale, norm);
  float4 x[kQuads];
  if (l == 0) {
    Begin(c, ins);
#pragma unroll
    for (int j = 0; j < kQuads; ++j) {
      const int q = QuadOf(j);
      x[j] = make_float4(0, 0, 0, 0);
      if (4 * q >= dim) continue;
      const float* e = c.step->embeddings + 4 * q;  // any alignment
      x[j] = make_float4(Bf16(e[0]), Bf16(e[1]), Bf16(e[2]), Bf16(e[3]));
      StoreBf16Quad(c.x_s, q, x[j]);
    }
  } else {
    const auto& prev = program.layer[l - 1];
    const auto& scratch = program.scratch[l - 1];
    float4 dense_post[kQuads];
    float4 moe_post[kQuads];
    float4 sum_post[kQuads];
    LoadScale(c, prev.dense_post, dense_post);
    LoadScale(c, prev.moe_post, moe_post);
    LoadScale(c, prev.sum_post, sum_post);
    Begin(c, ins);
    const __nv_bfloat16* dense_out = At<__nv_bfloat16>(c, scratch.dense_out);
    const float* experts_out = At<float>(c, scratch.experts_out);
    float4 dense[kQuads];
    float4 moe[kQuads];
    float sums[2] = {0.0f, 0.0f};
#pragma unroll
    for (int j = 0; j < kQuads; ++j) {
      const int q = QuadOf(j);
      dense[j] = moe[j] = make_float4(0, 0, 0, 0);
      if (4 * q >= dim) continue;
      dense[j] = StepBf16Quad(dense_out, q);
      moe[j] = StepQuad(experts_out, q);
    }
#pragma unroll
    for (int j = 0; j < kQuads; ++j) {
      sums[0] += SumSquares(dense[j]);
      sums[1] += SumSquares(moe[j]);
    }
    BlockSums(c, sums);
    const float inv_dim = 1.0f / static_cast<float>(dim);
    const float dense_factor = rsqrtf(__fmaf_rn(sums[0], inv_dim, eps));
    const float moe_factor = rsqrtf(__fmaf_rn(sums[1], inv_dim, eps));
    // dv * dense_factor * dense_post + mv * moe_factor * moe_post.
    auto combine = [&](float d, float m, float dp, float mp) {
      return __fmaf_rn(__fmul_rn(d, dense_factor), dp,
                       __fmul_rn(__fmul_rn(m, moe_factor), mp));
    };
    float squares[1] = {0.0f};
#pragma unroll
    for (int j = 0; j < kQuads; ++j) {
      dense[j] = make_float4(
          combine(dense[j].x, moe[j].x, dense_post[j].x, moe_post[j].x),
          combine(dense[j].y, moe[j].y, dense_post[j].y, moe_post[j].y),
          combine(dense[j].z, moe[j].z, dense_post[j].z, moe_post[j].z),
          combine(dense[j].w, moe[j].w, dense_post[j].w, moe_post[j].w));
      squares[0] += SumSquares(dense[j]);
    }
    BlockSums(c, squares);
    const float sum_factor = rsqrtf(__fmaf_rn(squares[0], inv_dim, eps));
    const float multiplier = prev.residual_multiplier;
    auto out = [&](float cv, float sp, float h) {
      return Bf16(__fmul_rn(__fmaf_rn(__fmul_rn(cv, sum_factor), sp, h),
                            multiplier));
    };
#pragma unroll
    for (int j = 0; j < kQuads; ++j) {
      const int q = QuadOf(j);
      x[j] = make_float4(0, 0, 0, 0);
      if (4 * q >= dim) continue;
      const float4 h = SharedBf16Quad(c.h_s, q);
      x[j] = make_float4(out(dense[j].x, sum_post[j].x, h.x),
                         out(dense[j].y, sum_post[j].y, h.y),
                         out(dense[j].z, sum_post[j].z, h.z),
                         out(dense[j].w, sum_post[j].w, h.w));
    }
#pragma unroll
    for (int j = 0; j < kQuads; ++j) {
      const int q = QuadOf(j);
      if (4 * q < dim) StoreBf16Quad(c.x_s, q, x[j]);
    }
  }
  if (c.cta == 0) {
    __nv_bfloat16* xs = l < program.layers
                            ? At<__nv_bfloat16>(c, program.scratch[l].x)
                            : At<__nv_bfloat16>(c, program.final_x);
#pragma unroll
    for (int j = 0; j < kQuads; ++j) {
      const int q = QuadOf(j);
      if (4 * q < dim) StoreBf16Quad(xs, q, x[j]);
    }
  }
  return NormedInput(c, x, norm, 1.0f, act);
}

// The graph's rms_norm reciprocal of a BF16 vector whose squares (each
// BF16(v * v)) this thread summed into `squares`: r = BF16(1 /
// BF16(sqrt(BF16(BF16(sum / dim) + BF16(eps))))).
__device__ __forceinline__ float GraphNormReciprocal(const Context& c,
                                                     float squares) {
  float sums[1] = {squares};
  BlockSums(c, sums);
  const float mean = Bf16(sums[0] / static_cast<float>(c.program->model_dim));
  return Bf16(1.0f / Bf16(sqrtf(Bf16(mean + Bf16(c.program->norm_eps)))));
}

__device__ __forceinline__ float Bf16Squares(float4 v) {
  return Bf16(v.x * v.x) + Bf16(v.y * v.y) + Bf16(v.z * v.z) + Bf16(v.w * v.w);
}

// h = x + rms_norm(o) * post_attention_norm as the graph computes it
// (megakernel_steps.cu's NormResidualKernel) into h_s (CTA 0 writes it to
// the scratch), and the router's input h times its norm's scale into
// `act`; returns the router's factor. The mean of h^2 goes to hsq_s for
// the dense and the experts' norms.
__device__ float AttentionResidualNormed(
    const Context& c, const LiteRtNvidiaMegakernelInstruction& ins,
    float* act) {
  const auto& program = *c.program;
  const int l = ins.layer;
  const auto& layer = program.layer[l];
  const int dim = program.model_dim;
  float4 post[kQuads];
  float4 norm[kQuads];
  LoadScale(c, layer.post_attention_norm, post);
  LoadScale(c, layer.router_norm, norm);
  Begin(c, ins);
  const __nv_bfloat16* o = At<__nv_bfloat16>(c, program.scratch[l].o);
  float4 v[kQuads];
  float squares = 0.0f;
#pragma unroll
  for (int j = 0; j < kQuads; ++j) {
    const int q = QuadOf(j);
    v[j] = 4 * q < dim ? StepBf16Quad(o, q) : make_float4(0, 0, 0, 0);
  }
#pragma unroll
  for (int j = 0; j < kQuads; ++j) squares += Bf16Squares(v[j]);
  const float r = GraphNormReciprocal(c, squares);
  auto residual = [&](float x, float ov, float p) {
    return Bf16(x + Bf16(Bf16(ov * r) * Bf16(p)));
  };
  float4 h[kQuads];
#pragma unroll
  for (int j = 0; j < kQuads; ++j) {
    const int q = QuadOf(j);
    h[j] = make_float4(0, 0, 0, 0);
    if (4 * q >= dim) continue;
    const float4 x = SharedBf16Quad(c.x_s, q);
    h[j] = make_float4(residual(x.x, v[j].x, post[j].x),
                       residual(x.y, v[j].y, post[j].y),
                       residual(x.z, v[j].z, post[j].z),
                       residual(x.w, v[j].w, post[j].w));
    StoreBf16Quad(c.h_s, q, h[j]);
  }
  if (c.cta == 0) {
    __nv_bfloat16* hs = At<__nv_bfloat16>(c, program.scratch[l].h);
#pragma unroll
    for (int j = 0; j < kQuads; ++j) {
      const int q = QuadOf(j);
      if (4 * q < dim) StoreBf16Quad(hs, q, h[j]);
    }
  }
  // rsqrt(mean(h^2) + eps), which the dense and the experts' norms reuse.
  const float factor = NormedInput(c, h, norm, 1.0f, act);
  if (threadIdx.x == 0) *c.hsq_s = factor;
  return factor * layer.router_multiplier;
}

// h (shared memory) times a norm's scale into `act`, for the dense and the
// experts' projections of the layer whose router computed rsqrt(mean(h^2)
// + eps) (hsq_s); returns that factor.
__device__ float HiddenNormed(const Context& c,
                              const LiteRtNvidiaMegakernelInstruction& ins,
                              const float* scale, float* act) {
  const int dim = c.program->model_dim;
  const int blocks = dim / kBlock;
  float4 norm[kQuads];
  LoadScale(c, scale, norm);
  Begin(c, ins);
  const float factor = *c.hsq_s;
#pragma unroll
  for (int j = 0; j < kQuads; ++j) {
    const int q = QuadOf(j);
    if (4 * q >= dim) continue;
    const float4 h = SharedBf16Quad(c.h_s, q);
    *reinterpret_cast<float4*>(act + Swizzled(4 * q, blocks)) =
        make_float4(__fmul_rn(h.x, norm[j].x), __fmul_rn(h.y, norm[j].y),
                    __fmul_rn(h.z, norm[j].z), __fmul_rn(h.w, norm[j].w));
  }
  Sync();  // the products are in
  return factor;
}

// --- The weight stream ------------------------------------------------------------

// RouteRow's routing (int4_block_gemv.cuh) of one row of router logits by a
// warp: the softmax over the experts, the top_k by probability (ties to the
// lower index) and the chosen probabilities renormalized to sum 1, into
// `indices` and `weights` (shared memory). RouteRow's arrays are indexed by
// round and expert, which puts them in local memory; here the indices stay
// static.
__device__ void RouteTopK(const __nv_bfloat16* src, int experts, int top_k,
                          int lane, int32_t* indices, float* weights) {
  constexpr int kPerLane = kMaxExperts / 32;
  float values[kPerLane];
  float largest = -INFINITY;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    const int e = lane + i * 32;
    values[i] = e < experts ? __bfloat162float(__ldcg(src + e)) : -INFINITY;
    largest = fmaxf(largest, values[i]);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    largest = fmaxf(largest, __shfl_xor_sync(kFull, largest, offset));
  }
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    values[i] = lane + i * 32 < experts ? expf(values[i] - largest) : 0.0f;
    sum += values[i];
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_xor_sync(kFull, sum, offset);
  }
  const float inv_sum = 1.0f / sum;
  float chosen_sum = 0.0f;
  float mine = 0.0f;  // lane j's: round j's probability
  for (int j = 0; j < top_k; ++j) {
    float best = -1.0f;
    int best_e = experts;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int e = lane + i * 32;
      if (e < experts && (values[i] > best || (values[i] == best && e < best_e))) {
        best = values[i];
        best_e = e;
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      const float other = __shfl_xor_sync(kFull, best, offset);
      const int other_e = __shfl_xor_sync(kFull, best_e, offset);
      if (other > best || (other == best && other_e < best_e)) {
        best = other;
        best_e = other_e;
      }
    }
    const float probability = best * inv_sum;
    chosen_sum += probability;
    if (lane == j) mine = probability;
    if (lane == 0) indices[j] = best_e;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      if (lane + i * 32 == best_e) values[i] = -1.0f;
    }
  }
  const float inv = chosen_sum > 0.0f ? 1.0f / chosen_sum : 0.0f;
  if (lane < top_k) weights[lane] = mine * inv;
}

// The router's top_k of layer l (RouteTopK) into the routing slot of its
// parity, by a consumer warp once every CTA has finished the router
// (instruction `router`): the first warp of the CTA out of work in the
// layer's dense gate and up projections claims it, so that the producer can
// stream the routed experts' weights while the others finish. The weights
// times the experts' scales go to `scaled`; CTA 0 writes the routing to the
// scratch.
__device__ void RouteLayer(const Context& c, int l, int router) {
  const int slot = l & 1;
  const int lane = threadIdx.x & 31;
  int claimed = 0;
  if (lane == 0) {
    claimed = atomicMax(const_cast<int*>(&c.control[kControlRouteClaim]),
                        l + 1) < l + 1;
  }
  if (!__shfl_sync(kFull, claimed, 0)) return;
  if (lane == 0) WaitDone(c.state, router, c.target);
  __syncwarp();
  const auto& layer = c.program->layer[l];
  const auto& scratch = c.program->scratch[l];
  int32_t* route = c.route_s + slot * kMaxTopK;
  float* weights = c.weight_s + slot * kMaxTopK;
  RouteTopK(At<__nv_bfloat16>(c, scratch.router), layer.experts, layer.top_k,
            lane, route, weights);
  __syncwarp();
  if (lane < layer.top_k) {
    c.scaled_s[slot * kMaxTopK + lane] =
        weights[lane] * __ldg(layer.per_expert_scale + route[lane]);
    if (c.cta == 0) {
      At<int32_t>(c, scratch.routing)[lane] = route[lane];
      At<float>(c, scratch.routing + layer.top_k * 4)[lane] = weights[lane];
    }
  }
  __threadfence_block();
  __syncwarp();
  if (lane == 0) c.control[kControlRouted + slot] = l + 1;
}

// Waits (one lane) until layer l's routing is in shared memory.
__device__ __forceinline__ void WaitRouted(const Context& c, int l) {
  const uint64_t since = GlobalTimer();
  while (c.control[kControlRouted + (l & 1)] != l + 1) {
    __nanosleep(128);
    if (GlobalTimer() - since > kWaitLimitNs) __trap();
  }
}

// The kinds of projection instructions, by how their units map to rows.
enum UnitKind : int {
  kDenseUnits = 0,   // item i: rows i * R of one array
  kGatedUnits = 1,   // item i, part p: rows p * part_rows + i * R
  kExpertUnits = 2,  // item i: slot i % top_k, rows (i / top_k) * R of the
                     // slot's expert; part p: the gate's or the up's arrays
  kDownUnits = 3,    // item i, part p: slot p, rows i * R of its expert
};

__device__ __forceinline__ int KindOf(int opcode) {
  return opcode == kLiteRtNvidiaMegakernelOpDenseGateUp ? kGatedUnits
         : opcode == kLiteRtNvidiaMegakernelOpExpertHidden ? kExpertUnits
         : opcode == kLiteRtNvidiaMegakernelOpExpertDown   ? kDownUnits
                                                           : kDenseUnits;
}

// What the producer needs to place a projection instruction's units,
// computed once per instruction, in registers (no arrays: an indexed array
// would live in local memory, a round trip per unit).
struct StreamPlan {
  const uint8_t* weights0;  // part 0's (or the only) arrays at row 0
  const uint8_t* weights1;  // part 1's (the experts' up projections)
  const char* scales0;
  const char* scales1;
  const char* maxima0;  // compact scales' row maxima (FP16)
  const char* maxima1;
  int kind;
  int unit_rows;
  int rows;        // a part's (an expert's) rows
  int part_rows;   // kGatedUnits: the up rows' offset
  int row_bytes;
  int scale_row;   // a row's scale bytes
  int max_bytes;   // the maxima window: 0 (FP16 scales), 16 or rows * 2
};

__device__ StreamPlan PlanOf(const Context& c,
                             const LiteRtNvidiaMegakernelInstruction& ins) {
  const auto& program = *c.program;
  const int dim = program.model_dim;
  StreamPlan p{};
  p.kind = KindOf(ins.opcode);
  p.unit_rows = ins.tile_rows;
  auto dense = [&](const uint8_t* w, const void* s, int rows, int cols) {
    p.weights0 = p.weights1 = w;
    p.scales0 = p.scales1 = static_cast<const char*>(s);
    p.rows = rows;
    p.row_bytes = cols / 2;
    p.scale_row = cols / kBlock * 2;
  };
  if (ins.layer >= program.layers) {
    dense(program.head.weights, program.head.scales, program.head.rows, dim);
    return p;
  }
  const auto& layer = program.layer[ins.layer];
  const LiteRtNvidiaMoeArena& a = layer.arena;
  switch (ins.opcode) {
    case kLiteRtNvidiaMegakernelOpQkv:
      dense(layer.qkv.weights, layer.qkv.scales, layer.qkv.rows, dim);
      break;
    case kLiteRtNvidiaMegakernelOpOutput:
      dense(layer.o.weights, layer.o.scales, dim, layer.o.cols);
      break;
    case kLiteRtNvidiaMegakernelOpRouter:
      dense(a.router, a.router_scales, layer.experts, dim);
      break;
    case kLiteRtNvidiaMegakernelOpDenseGateUp:
      dense(a.dense_gate_up, a.dense_gate_up_scales, layer.dense_hidden, dim);
      p.part_rows = layer.dense_hidden;
      break;
    case kLiteRtNvidiaMegakernelOpDenseDown:
      dense(a.dense_down, a.dense_down_scales, dim, layer.dense_hidden);
      break;
    case kLiteRtNvidiaMegakernelOpExpertHidden:
      p.weights0 = a.gate;
      p.weights1 = a.up;
      p.scales0 = static_cast<const char*>(a.gate_scales);
      p.scales1 = static_cast<const char*>(a.up_scales);
      p.maxima0 = static_cast<const char*>(a.gate_scale_max);
      p.maxima1 = static_cast<const char*>(a.up_scale_max);
      p.rows = layer.expert_hidden;
      p.row_bytes = dim / 2;
      p.scale_row = dim / kBlock;
      p.max_bytes = MaxWindowBytes(p.unit_rows);
      break;
    default:  // the experts' down projections
      p.weights0 = p.weights1 = a.down;
      p.scales0 = p.scales1 = static_cast<const char*>(a.down_scales);
      p.maxima0 = p.maxima1 = static_cast<const char*>(a.down_scale_max);
      p.rows = dim;
      p.row_bytes = layer.expert_hidden / 2;
      p.scale_row = layer.expert_hidden / kBlock;
      p.max_bytes = MaxWindowBytes(p.unit_rows);
      break;
  }
  return p;
}

// The producer warp: every unit of every projection instruction of this
// CTA, in the order the consumers take them, into the ring (unit k into
// stage k % kStages once unit k - kStages is consumed, and once unit k -
// kInFlight has arrived). Lane 0 issues; the routed experts' units wait for
// their layer's routing.
__device__ void Produce(const Context& c) {
  const auto& program = *c.program;
  if ((threadIdx.x & 31) != 0) return;
  const uint64_t policy = EvictFirstPolicy();
  uint32_t k = 0;
  int stage = 0;
  uint32_t phase = 0;  // (k / kStages) & 1
  int rotation = 0;    // the projections' items so far, mod ctas
  for (int i = 0; i < program.instruction_count; ++i) {
    const LiteRtNvidiaMegakernelInstruction ins = program.instruction[i];
    if (!IsProjectionOp(ins.opcode)) continue;
    const int vcta = (c.cta + c.ctas - rotation) % c.ctas;
    rotation = (rotation + ins.items) % c.ctas;
    const StreamPlan p = PlanOf(c, ins);
    const int top_k =
        ins.layer < program.layers ? program.layer[ins.layer].top_k : 1;
    const int32_t* route = c.route_s + (ins.layer & 1) * kMaxTopK;
    if (p.kind == kExpertUnits || p.kind == kDownUnits) WaitRouted(c, ins.layer);
    const int items = ItemsFor(vcta, c.ctas, ins.items);
    const int unit = p.unit_rows;
    // kExpertUnits: item vcta + j * ctas, its slot and first row, kept
    // up to date as j grows.
    int slot = vcta % top_k;
    int o0 = vcta / top_k * unit;
    const int slot_step = c.ctas % top_k;
    const int o0_step = c.ctas / top_k * unit;
    for (int j = 0; j < items; ++j) {
      const int item = vcta + j * c.ctas;
      for (int part = 0; part < ins.parts; ++part, ++k) {
        int first;
        int rows;
        if (p.kind == kDenseUnits || p.kind == kGatedUnits) {
          first = part * p.part_rows + item * unit;
          rows = min(unit, p.rows - item * unit);
        } else if (p.kind == kExpertUnits) {
          first = route[slot] * p.rows + o0;
          rows = min(unit, p.rows - o0);
        } else {
          first = route[part] * p.rows + item * unit;
          rows = min(unit, p.rows - item * unit);
        }
        const bool up = p.kind == kExpertUnits && part == 1;
        const uint8_t* weights = up ? p.weights1 : p.weights0;
        const char* scales = up ? p.scales1 : p.scales0;
        const char* maxima = up ? p.maxima1 : p.maxima0;
        char* target = c.ring + stage * kStageBytes;
        WaitParity(&c.empty[stage], phase ^ 1);
        if (k >= kInFlight) {
          // Unit k - kInFlight has arrived (its stage is still its own: the
          // producer refills it only for unit k - kInFlight + kStages).
          const int back = stage >= kInFlight ? stage - kInFlight
                                              : stage - kInFlight + kStages;
          WaitParity(&c.full[back], stage >= kInFlight ? phase : phase ^ 1);
        }
        const uint32_t weight_bytes = rows * p.row_bytes;
        const uint32_t scale_bytes = rows * p.scale_row;
        ExpectBytes(&c.full[stage], weight_bytes + scale_bytes + p.max_bytes);
        BulkCopy(target, weights + static_cast<size_t>(first) * p.row_bytes,
                 weight_bytes, &c.full[stage], policy);
        BulkCopy(target + unit * p.row_bytes,
                 scales + static_cast<size_t>(first) * p.scale_row, scale_bytes,
                 &c.full[stage], policy);
        if (p.max_bytes > 0) {
          BulkCopy(target + unit * p.row_bytes + RoundUp16(unit * p.scale_row),
                   maxima + static_cast<size_t>(first & ~7) * 2, p.max_bytes,
                   &c.full[stage], policy);
        }
        // Published after the copies are armed: a consumer that sees it
        // waits on this unit's phase.
        c.control[kControlArmed + stage] = static_cast<int>(k);
        if (++stage == kStages) {
          stage = 0;
          phase ^= 1;
        }
      }
      slot += slot_step;
      o0 += o0_step;
      if (slot >= top_k) {
        slot -= top_k;
        o0 += unit;
      }
    }
  }
}

// --- Projections from the ring --------------------------------------------------------

// Word q (8 values) of block b of a projection input in shared memory: FP32
// in Swizzled's layout, or BF16 (16-byte word q of block b at q * blocks +
// b).
__device__ __forceinline__ void InputWord(const float* act, int blocks, int b,
                                          int q, float v[8]) {
  const float4 lo =
      *reinterpret_cast<const float4*>(act + ((2 * q) * blocks + b) * 4);
  const float4 hi =
      *reinterpret_cast<const float4*>(act + ((2 * q + 1) * blocks + b) * 4);
  v[0] = lo.x;
  v[1] = lo.y;
  v[2] = lo.z;
  v[3] = lo.w;
  v[4] = hi.x;
  v[5] = hi.y;
  v[6] = hi.z;
  v[7] = hi.w;
}

__device__ __forceinline__ void InputWord(const __nv_bfloat16* act, int blocks,
                                          int b, int q, float v[8]) {
  const uint4 bits = *reinterpret_cast<const uint4*>(act + (q * blocks + b) * 8);
  const uint32_t parts[4] = {bits.x, bits.y, bits.z, bits.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 pair = __bfloat1622float2(
        *reinterpret_cast<const __nv_bfloat162*>(&parts[j]));
    v[2 * j] = pair.x;
    v[2 * j + 1] = pair.y;
  }
}

// A lane's share of a unit: kWords 16-byte words (32 weights each) of its
// rows and blocks, with their scales, in registers, so that the stage is
// free again before the math.
template <int kWords>
struct LaneShare {
  uint4 words[kWords];
  float scales[kWords];
};

// The scale of block b of row r of a unit at `stage`: FP16, or a compact
// fraction of the row's maximum.
template <bool kCompact>
__device__ __forceinline__ float ScaleOf(const char* stage, int unit_rows,
                                         int row_bytes, int blocks, int r,
                                         int b, float row_max) {
  const char* scales = stage + unit_rows * row_bytes;
  if constexpr (kCompact) {
    return row_max *
           static_cast<float>(reinterpret_cast<const uint8_t*>(scales)[r * blocks + b]);
  } else {
    return __half2float(reinterpret_cast<const __half*>(scales)[r * blocks + b]);
  }
}

// Row r's compact maximum / 255 (`shift` = the unit's first row & 7).
__device__ __forceinline__ float RowMax(const char* stage, int unit_rows,
                                        int row_bytes, int blocks, int shift,
                                        int r) {
  const __half* maxima = reinterpret_cast<const __half*>(
      stage + unit_rows * row_bytes + RoundUp16(unit_rows * blocks));
  return __half2float(maxima[shift + r]) * (1.0f / 255.0f);
}

// The dot of a 32-weight word with 32 inputs (four 8-value words of `act`'s
// block b): two chains (even and odd weights) in order, as the per-op
// kernels' inner loop; FP32.
template <typename Act>
__device__ __forceinline__ float WordDot(uint4 word, const Act* act, int blocks,
                                         int b) {
  const uint32_t parts[4] = {word.x, word.y, word.z, word.w};
  float even = 0.0f;
  float odd = 0.0f;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    float v[8];
    InputWord(act, blocks, b, q, v);
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      even = fmaf(v[2 * n], gemv::Nibble(parts[q], 2 * n), even);
      odd = fmaf(v[2 * n + 1], gemv::Nibble(parts[q], 2 * n + 1), odd);
    }
  }
  return even + odd;
}

// Waits until unit k of the stream is in its stage; returns the stage. The
// consumer first waits for the producer to arm the stage for unit k: a
// consumer may get to a stage two phases ahead of its unit (its warp took a
// later unit early), which a parity wait cannot tell apart.
__device__ __forceinline__ int WaitUnit(const Context& c, uint32_t k) {
  const int stage = k % kStages;
  if (c.control[kControlArmed + stage] != static_cast<int>(k)) {
    int sleep = 64;
    while (c.control[kControlArmed + stage] != static_cast<int>(k)) {
      __nanosleep(sleep);
      sleep = sleep < 1024 ? sleep * 2 : sleep;
    }
  }
  WaitParity(&c.full[stage], (k / kStages) & 1);
  return stage;
}

// The stage is free for the producer (the warp has its share in registers).
__device__ __forceinline__ void FreeStage(const Context& c, int stage) {
  __syncwarp();
  if ((threadIdx.x & 31) == 0) Arrive(&c.empty[stage]);
}

// This warp's next unit of the CTA's `units` (in order), or -1.
__device__ __forceinline__ int NextUnit(const Context& c, int units) {
  int u = 0;
  if ((threadIdx.x & 31) == 0) {
    u = atomicAdd(const_cast<int*>(&c.control[kControlNextItem]), 1);
  }
  u = __shfl_sync(kFull, u, 0);
  return u < units ? u : -1;
}

// Lane r's row r of `v` (the indices stay static: v stays in registers).
template <int kRows>
__device__ __forceinline__ float RowOf(const float v[kRows]) {
  const int lane = threadIdx.x & 31;
  float mine = 0.0f;
#pragma unroll
  for (int r = 0; r < kRows; ++r) mine = lane == r ? v[r] : mine;
  return mine;
}

// Delivers part `part` of item j (the CTA's j-th) of an instruction of
// `parts` units per item: lanes r < kRows hold row r's sum, which goes to
// the item's partials in shared memory. Returns true in the warp that
// delivers the item's last part: it then reads every part's (in order).
template <int kRows>
__device__ __forceinline__ bool DeliverPart(const Context& c, int j, int part,
                                            int parts, float sum) {
  const int lane = threadIdx.x & 31;
  if (lane < kRows) c.partial_s[(j * parts + part) * kRows + lane] = sum;
  __threadfence_block();
  __syncwarp();
  int before = 0;
  if (lane == 0) before = atomicAdd(const_cast<int*>(&c.counter_s[j]), 1);
  before = __shfl_sync(kFull, before, 0);
  __threadfence_block();
  return before == parts - 1;
}

// A projection whose units are kRows rows by the input's blocks, lane l
// taking blocks l, l + 32, ... of every row (each block's inputs once for
// all rows). `deliver(j, part, item, rows, sums)` gets each unit's row sums
// (every lane holds all of them).
template <int kRows, int kLaneBlocks, bool kCompact, typename Act,
          typename Deliver>
__device__ __forceinline__ void ProjectUnits(
    const Context& c, const LiteRtNvidiaMegakernelInstruction& ins,
    uint32_t base, int out_rows, int cols, const Act* act, Deliver&& deliver) {
  const int blocks = cols / kBlock;
  const int row_bytes = cols / 2;
  const int parts = ins.parts;
  const int units = ItemsOf(c, ins.items) * parts;
  const int lane = threadIdx.x & 31;
  const int top_k = ins.layer < c.program->layers
                        ? c.program->layer[ins.layer].top_k
                        : 1;
  for (int u; (u = NextUnit(c, units)) >= 0;) {
    const int j = u / parts;
    const int part = u - j * parts;
    const int item = c.vcta + j * c.ctas;
    // The unit's first row in its part (or slot's expert): the compact
    // maxima's window starts at that row & ~7.
    const int row0 = KindOf(ins.opcode) == kExpertUnits
                         ? item / top_k * kRows
                         : item * kRows;
    const int rows = min(kRows, out_rows - row0);
    const int stage = WaitUnit(c, base + u);
    const char* at = c.ring + stage * kStageBytes;
    LaneShare<kRows * kLaneBlocks> share;
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const float row_max =
          kCompact && r < rows
              ? RowMax(at, kRows, row_bytes, blocks, row0 & 7, r)
              : 0.0f;
#pragma unroll
      for (int i = 0; i < kLaneBlocks; ++i) {
        const int b = lane + 32 * i;
        const int w = r * kLaneBlocks + i;
        if (r < rows && b < blocks) {
          share.words[w] =
              reinterpret_cast<const uint4*>(at)[r * blocks + b];
          share.scales[w] =
              ScaleOf<kCompact>(at, kRows, row_bytes, blocks, r, b, row_max);
        } else {
          share.words[w] = make_uint4(0, 0, 0, 0);
          share.scales[w] = 0.0f;
        }
      }
    }
    FreeStage(c, stage);
    // Per block: each 8-input word of the block once, for every row's two
    // chains (even and odd weights, in order, as WordDot); then the block's
    // dot times its scale into the row's sum.
    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.0f;
#pragma unroll
    for (int i = 0; i < kLaneBlocks; ++i) {
      const int b = lane + 32 * i;
      if (b >= blocks) break;
      float even[kRows];
      float odd[kRows];
#pragma unroll
      for (int r = 0; r < kRows; ++r) even[r] = odd[r] = 0.0f;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float v[8];
        InputWord(act, blocks, b, q, v);
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
          const uint4 word = share.words[r * kLaneBlocks + i];
          const uint32_t bits =
              q == 0 ? word.x : q == 1 ? word.y : q == 2 ? word.z : word.w;
#pragma unroll
          for (int n = 0; n < 4; ++n) {
            even[r] = fmaf(v[2 * n], gemv::Nibble(bits, 2 * n), even[r]);
            odd[r] = fmaf(v[2 * n + 1], gemv::Nibble(bits, 2 * n + 1), odd[r]);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        acc[r] = fmaf(even[r] + odd[r], share.scales[r * kLaneBlocks + i],
                      acc[r]);
      }
    }
    float sums[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) sums[r] = ButterflySum(acc[r]);
    deliver(j, part, item, rows, sums);
  }
}

// A dense projection's rows: BF16(sum * factor) into `out`; with a gate
// (two parts per item, the gate rows' unit and the up rows'),
// BF16(Gelu(gate * factor) * up * factor).
template <int kRows, int kLaneBlocks, typename Act>
__device__ void DenseItemsOf(const Context& c,
                             const LiteRtNvidiaMegakernelInstruction& ins,
                             uint32_t base, int out_rows, int cols,
                             const Act* act, float factor, int gate,
                             __nv_bfloat16* out) {
  const int lane = threadIdx.x & 31;
  ProjectUnits<kRows, kLaneBlocks, false>(
      c, ins, base, out_rows, cols, act,
      [&](int j, int part, int item, int rows, const float* sums) {
        const int n0 = item * kRows;
        if (gate == 0) {
          const float value = RowOf<kRows>(sums) * factor;
          if (lane < rows) out[n0 + lane] = __float2bfloat16_rn(value);
          return;
        }
        if (!DeliverPart<kRows>(c, j, part, 2, RowOf<kRows>(sums))) return;
        if (lane < rows) {
          const float g = c.partial_s[(j * 2) * kRows + lane] * factor;
          const float u = c.partial_s[(j * 2 + 1) * kRows + lane];
          out[n0 + lane] = __float2bfloat16_rn(gemv::Gelu(g, gate) * u * factor);
        }
      });
}

// The instantiation for the projection's unit rows and input blocks (the
// builder admits these).
template <typename Act>
__device__ void DenseItems(const Context& c,
                           const LiteRtNvidiaMegakernelInstruction& ins,
                           uint32_t base, int out_rows, int cols,
                           const Act* act, float factor, int gate,
                           __nv_bfloat16* out) {
  if (ins.tile_rows == 4) {
    DenseItemsOf<4, 3>(c, ins, base, out_rows, cols, act, factor, gate, out);
  } else if (ins.tile_rows == 2) {
    DenseItemsOf<2, 4>(c, ins, base, out_rows, cols, act, factor, gate, out);
  } else {
    DenseItemsOf<1, 8>(c, ins, base, out_rows, cols, act, factor, gate, out);
  }
}

// The experts' gated projections (MoeHiddenKernel): hidden[slot][o] =
// Gelu(gate * f) * up * f, FP32. Item i is slot i % top_k's rows from (i /
// top_k) * 4; its units are the gate rows' and the up rows'.
__device__ void ExpertHiddenItems(const Context& c,
                                  const LiteRtNvidiaMegakernelInstruction& ins,
                                  uint32_t base,
                                  const LiteRtNvidiaMegakernelLayer& layer,
                                  const float* act, float factor,
                                  float* hidden) {
  constexpr int kRows = kExpertRows;
  const int lane = threadIdx.x & 31;
  ProjectUnits<kRows, 3, true>(
      c, ins, base, layer.expert_hidden, c.program->model_dim, act,
      [&](int j, int part, int item, int rows, const float* sums) {
        if (!DeliverPart<kRows>(c, j, part, 2, RowOf<kRows>(sums))) return;
        const int slot = item % layer.top_k;
        const int o0 = item / layer.top_k * kRows;
        if (lane < rows) {
          const float g = c.partial_s[(j * 2) * kRows + lane] * factor;
          const float u = c.partial_s[(j * 2 + 1) * kRows + lane];
          hidden[static_cast<size_t>(slot) * layer.expert_hidden + o0 + lane] =
              gemv::Gelu(g, layer.expert_gate) * u * factor;
        }
      });
}

// The experts' down projections (MoeDownKernel) of a tile of kDownRows
// output rows: a unit per slot (its expert's rows), lane 2r + h taking row
// r's blocks h, h + 2, ...; the slots' sums times their routing weights and
// experts' scales are added in slot order by the warp that delivers the
// tile's last slot.
__device__ void ExpertDownItems(const Context& c,
                                const LiteRtNvidiaMegakernelInstruction& ins,
                                uint32_t base,
                                const LiteRtNvidiaMegakernelLayer& layer,
                                const float* scaled, const float* act,
                                float* out) {
  constexpr int kRows = kDownRows;
  constexpr int kWords = kMaxDownBlocks / 2;
  const int dim = c.program->model_dim;
  const int hidden = layer.expert_hidden;
  const int blocks = hidden / kBlock;
  const int row_bytes = hidden / 2;
  const int top_k = layer.top_k;
  const int units = ItemsOf(c, ins.items) * top_k;
  const int lane = threadIdx.x & 31;
  const int r = lane >> 1;
  const int h = lane & 1;
  for (int u; (u = NextUnit(c, units)) >= 0;) {
    const int j = u / top_k;
    const int slot = u - j * top_k;
    const int o0 = (c.vcta + j * c.ctas) * kRows;
    const int rows = min(kRows, dim - o0);
    const int stage = WaitUnit(c, base + u);
    const char* at = c.ring + stage * kStageBytes;
    LaneShare<kWords> share;
    const float row_max =
        r < rows ? RowMax(at, kRows, row_bytes, blocks, 0, r) : 0.0f;
#pragma unroll
    for (int i = 0; i < kWords; ++i) {
      const int b = h + 2 * i;
      if (r < rows && b < blocks) {
        share.words[i] = reinterpret_cast<const uint4*>(at)[r * blocks + b];
        share.scales[i] =
            ScaleOf<true>(at, kRows, row_bytes, blocks, r, b, row_max);
      } else {
        share.words[i] = make_uint4(0, 0, 0, 0);
        share.scales[i] = 0.0f;
      }
    }
    FreeStage(c, stage);
    const float* slot_act = act + static_cast<size_t>(slot) * hidden;
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < kWords; ++i) {
      const int b = h + 2 * i;
      if (b >= blocks) break;
      acc = fmaf(WordDot(share.words[i], slot_act, blocks, b), share.scales[i],
                 acc);
    }
    const float sum = acc + __shfl_xor_sync(kFull, acc, 1);
    // Row r's sum goes to partial lane r (DeliverPart's lanes < kRows).
    const float row_sum = __shfl_sync(kFull, sum, (lane & (kRows - 1)) * 2);
    if (!DeliverPart<kRows>(c, j, slot, top_k, row_sum)) continue;
    if (lane < rows) {
      float total = 0.0f;
      for (int s = 0; s < top_k; ++s) {
        total = __fadd_rn(
            total, __fmul_rn(scaled[s], c.partial_s[(j * top_k + s) * kRows + lane]));
      }
      out[o0 + lane] = total;
    }
  }
}

__device__ void ScanMasks(const Context& c);

__device__ void RunQkv(const Context& c,
                       const LiteRtNvidiaMegakernelInstruction& ins,
                       uint32_t base) {
  const auto& program = *c.program;
  const int l = ins.layer;
  const auto& layer = program.layer[l];
  const float factor = LayerInputNormed(c, ins, l, layer.input_norm, c.work);
  if (l == 0) ScanMasks(c);
  PrologueDone(c);
  DenseItems(c, ins, base, layer.qkv.rows, layer.qkv.cols, c.work, factor, 0,
             At<__nv_bfloat16>(c, program.scratch[l].qkv));
}

__device__ void RunNormRope(const Context& c,
                            const LiteRtNvidiaMegakernelInstruction& ins) {
  Begin(c, ins);
  const auto& program = *c.program;
  const auto& layer = program.layer[ins.layer];
  const auto& scratch = program.scratch[ins.layer];
  const int depth = layer.head_dim;
  const int32_t position = *c.step->position;
  const int32_t row = CacheRow(position, layer.cache_len, layer.ring);
  const __nv_bfloat16* qkv = At<__nv_bfloat16>(c, scratch.qkv);
  float* values = c.work;
  float* partial = c.work + rope::kMaxDepth;
  ForItems(c, ins.items, [&](int head) {
    const bool is_q = head < layer.q_heads;
    const bool is_k = !is_q && head < layer.q_heads + layer.kv_heads;
    const int kv = head - layer.q_heads - (is_k ? 0 : layer.kv_heads);
    // A global layer's values are its keys normalized without a scale.
    const int source = is_q || is_k || layer.shared_kv == 0
                           ? head
                           : head - layer.kv_heads;
    __nv_bfloat16* out =
        is_q ? At<__nv_bfloat16>(c, scratch.q) + head * depth
             : At<__nv_bfloat16>(c, is_k ? scratch.k : scratch.v) + kv * depth;
    rope::NormRopeHead<ConsumerSync>(
        qkv + static_cast<size_t>(source) * depth,
        is_q ? layer.q_norm : (is_k ? layer.k_norm : nullptr),
        /*normalize=*/true, /*rotate=*/is_q || is_k, depth, layer.rotary_dim,
        program.norm_eps, layer.rope_min_timescale, layer.rope_max_timescale,
        c.step->position, out, values, partial);
    if (!is_q) {
      // The token's row into the cache, as FP16 of the BF16 values; each
      // thread reads back what it wrote.
      __half* cache = static_cast<__half*>(is_k ? layer.k_cache : layer.v_cache) +
                      (static_cast<size_t>(kv) * layer.cache_len + row) * depth;
      for (int d = threadIdx.x; d < depth; d += kConsumerThreads) {
        cache[d] = __float2half_rn(__bfloat162float(out[d]));
      }
    }
  });
}

// Each CTA's slice of each mask: its last visible key (-1: none) into the
// workspace, which the attention instructions combine. Every CTA does it in
// the step's first instruction, which every attention waits for.
__device__ void ScanMasks(const Context& c) {
  const auto& program = *c.program;
  int* lasts = reinterpret_cast<int*>(c.reduce_s);
  for (int m = 0; m < 2; ++m) {
    const int len = program.mask_len[m];
    const bool* mask = c.step->mask[m];
    int last = -1;
    if (len > 0 && mask != nullptr) {
      const int slice = (len + c.ctas - 1) / c.ctas;
      const int begin = c.cta * slice;
      const int end = min(len, begin + slice);
      for (int j = begin + threadIdx.x; j < end; j += kConsumerThreads) {
        if (mask[j]) last = max(last, j);
      }
    }
    last = WarpMax(last);
    Sync();  // lasts (reduce_s) is free
    if ((threadIdx.x & 31) == 0) lasts[threadIdx.x / 32] = last;
    Sync();
    if (threadIdx.x == 0) {
      for (int w = 0; w < kConsumerWarps; ++w) last = max(last, lasts[w]);
      At<int32_t>(c, program.mask_last)[m * kLiteRtNvidiaMegakernelMaxCtas +
                                        c.cta] = last;
    }
  }
  Sync();
}

// The keys an attention over `mask` reads: up to the end of the chunk that
// holds its last visible key. The per-op launch reads the mask of every
// chunk of the cache and skips those it hides; the chunks past the last
// visible key are such chunks, so bounding the loop there skips the same
// chunks without reading their masks.
__device__ int VisibleKeys(const Context& c, int mask, int cache_len) {
  const int32_t* lasts = At<int32_t>(c, c.program->mask_last) +
                         mask * kLiteRtNvidiaMegakernelMaxCtas;
  int last = -1;
  for (int t = threadIdx.x; t < c.ctas; t += kConsumerThreads) {
    last = max(last, __ldcg(lasts + t));
  }
  last = WarpMax(last);
  int* warps = reinterpret_cast<int*>(c.reduce_s);
  Sync();  // reduce_s is free
  if ((threadIdx.x & 31) == 0) warps[threadIdx.x / 32] = last;
  Sync();
  for (int w = 0; w < kConsumerWarps; ++w) last = max(last, warps[w]);
  return last < 0 ? 0 : min(cache_len, (last / kChunk + 1) * kChunk);
}

// Clang may retain loops despite the unroll hints of the block inlined
// here, and reports them at this function's declaration.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
template <int kDepth, int kRowsCap>
__device__ void AttentionItems(const Context& c,
                               const LiteRtNvidiaMegakernelInstruction& ins) {
  const auto& program = *c.program;
  const auto& layer = program.layer[ins.layer];
  const auto& scratch = program.scratch[ins.layer];
  const int group = layer.q_heads / layer.kv_heads;
  const int splits = scratch.splits;
  const bool* mask = c.step->mask[layer.mask];
  Begin(c, ins);
  if (c.cta >= ins.items) return;
  const int seq = VisibleKeys(c, layer.mask, layer.cache_len);
  PrologueDone(c);
  // Items KV head fastest: the few splits a short context lets see keys go
  // to different CTAs.
  ForItems(c, ins.items, [&](int item) {
    attention::DecodeAttentionPartialBlock<__nv_bfloat16, kDepth, kRowsCap,
                                           attention::Fp16Cache, kChunk,
                                           ConsumerSync>(
        At<__nv_bfloat16>(c, scratch.q),
        static_cast<const __half*>(layer.k_cache),
        static_cast<const __half*>(layer.v_cache), attention::Fp16Cache{},
        mask, /*mask_rows=*/1, group, seq, layer.cache_len,
        /*window_start=*/nullptr, kAttentionFill, splits,
        At<float>(c, scratch.attention_acc), At<float>(c, scratch.attention_max),
        At<float>(c, scratch.attention_sum), item / layer.kv_heads,
        item % layer.kv_heads, splits, c.work);
  });
}
#pragma clang diagnostic pop

template <int kDepth>
__device__ void CombineItems(const Context& c,
                             const LiteRtNvidiaMegakernelInstruction& ins) {
  const auto& program = *c.program;
  const auto& layer = program.layer[ins.layer];
  const auto& scratch = program.scratch[ins.layer];
  const int group = layer.q_heads / layer.kv_heads;
  Begin(c, ins);
  ForItems(c, ins.items, [&](int item) {
    attention::DecodeAttentionCombineBlock<__nv_bfloat16, kDepth, ConsumerSync>(
        At<float>(c, scratch.attention_acc), At<float>(c, scratch.attention_max),
        At<float>(c, scratch.attention_sum), group, scratch.splits,
        At<__nv_bfloat16>(c, scratch.attention_out), item % group,
        item / group, c.work);
  });
}

__device__ void RunOutput(const Context& c,
                          const LiteRtNvidiaMegakernelInstruction& ins,
                          uint32_t base) {
  const auto& program = *c.program;
  const auto& layer = program.layer[ins.layer];
  const auto& scratch = program.scratch[ins.layer];
  auto* act = reinterpret_cast<__nv_bfloat16*>(c.work);
  Begin(c, ins);
  StageBf16(At<__nv_bfloat16>(c, scratch.attention_out), layer.o.cols, act);
  Sync();
  PrologueDone(c);
  DenseItems(c, ins, base, program.model_dim, layer.o.cols, act, 1.0f, 0,
             At<__nv_bfloat16>(c, scratch.o));
}

__device__ void RunRouter(const Context& c,
                          const LiteRtNvidiaMegakernelInstruction& ins,
                          uint32_t base) {
  const auto& program = *c.program;
  const int l = ins.layer;
  const auto& layer = program.layer[l];
  const float factor = AttentionResidualNormed(c, ins, c.work);
  PrologueDone(c);
  DenseItems(c, ins, base, layer.experts, program.model_dim, c.work, factor, 0,
             At<__nv_bfloat16>(c, program.scratch[l].router));
}

__device__ void RunDenseGateUp(const Context& c,
                               const LiteRtNvidiaMegakernelInstruction& ins,
                               uint32_t base, int index) {
  const auto& program = *c.program;
  const int l = ins.layer;
  const auto& layer = program.layer[l];
  const float factor = HiddenNormed(c, ins, layer.dense_norm, c.work);
  PrologueDone(c);
  DenseItems(c, ins, base, layer.dense_hidden, program.model_dim, c.work,
             factor, layer.dense_gate,
             At<__nv_bfloat16>(c, program.scratch[l].dense_hidden));
  // The first warp out of work routes the layer's experts (the next
  // instruction's; its waits[0] is the router).
  if (index + 1 < program.instruction_count) {
    const auto& next = program.instruction[index + 1];
    if (next.opcode == kLiteRtNvidiaMegakernelOpExpertHidden) {
      RouteLayer(c, l, next.waits[0]);
    }
  }
}

__device__ void RunExpertHidden(const Context& c,
                                const LiteRtNvidiaMegakernelInstruction& ins,
                                uint32_t base) {
  const auto& program = *c.program;
  const int l = ins.layer;
  const auto& layer = program.layer[l];
  // Routed in the dense gate and up projections, or here.
  RouteLayer(c, l, ins.waits[0]);
  const float factor = HiddenNormed(c, ins, layer.expert_norm, c.work);
  PrologueDone(c);
  ExpertHiddenItems(c, ins, base, layer, c.work, factor,
                    At<float>(c, program.scratch[l].expert_hidden));
}

__device__ void RunDenseDown(const Context& c,
                             const LiteRtNvidiaMegakernelInstruction& ins,
                             uint32_t base) {
  const auto& program = *c.program;
  const int l = ins.layer;
  const auto& layer = program.layer[l];
  const auto& scratch = program.scratch[l];
  auto* act = reinterpret_cast<__nv_bfloat16*>(c.work);
  Begin(c, ins);
  StageBf16(At<__nv_bfloat16>(c, scratch.dense_hidden), layer.dense_hidden,
            act);
  Sync();
  PrologueDone(c);
  DenseItems(c, ins, base, program.model_dim, layer.dense_hidden, act, 1.0f, 0,
             At<__nv_bfloat16>(c, scratch.dense_out));
}

__device__ void RunExpertDown(const Context& c,
                              const LiteRtNvidiaMegakernelInstruction& ins,
                              uint32_t base) {
  const auto& program = *c.program;
  const int l = ins.layer;
  const auto& layer = program.layer[l];
  const auto& scratch = program.scratch[l];
  // The layer's routing (a warp found it before the experts' hidden
  // projections).
  const int slot = l & 1;
  Begin(c, ins);
  if (threadIdx.x == 0) WaitRouted(c, l);
  StageExpertHidden(At<float>(c, scratch.expert_hidden), layer.top_k,
                    layer.expert_hidden, c.work);
  Sync();
  PrologueDone(c);
  ExpertDownItems(c, ins, base, layer, c.scaled_s + slot * kMaxTopK, c.work,
                  At<float>(c, scratch.experts_out));
}

__device__ void RunHead(const Context& c,
                        const LiteRtNvidiaMegakernelInstruction& ins,
                        uint32_t base) {
  const auto& program = *c.program;
  const int dim = program.model_dim;
  const float factor = LayerInputNormed(c, ins, program.layers,
                                        program.final_norm, c.work);
  if (c.cta == 0) {
    // The activations: the graph's final norm, as FP32 of BF16
    // (megakernel_steps.cu's NormToFloatKernel).
    float squares = 0.0f;
#pragma unroll
    for (int j = 0; j < kQuads; ++j) {
      const int q = QuadOf(j);
      if (4 * q < dim) squares += Bf16Squares(SharedBf16Quad(c.x_s, q));
    }
    const float r = GraphNormReciprocal(c, squares);
    for (int i = threadIdx.x; i < dim; i += kConsumerThreads) {
      c.step->activations[i] =
          Bf16(Bf16(__bfloat162float(c.x_s[i]) * r) *
               Bf16(__ldg(program.final_norm + i)));
    }
  }
  PrologueDone(c);
  const float soft_in = Bf16(program.soft_cap_in);
  const float soft_out = Bf16(program.soft_cap);
  const bool capped = program.soft_cap > 0.0f;
  float* logits = c.step->logits;
  const int lane = threadIdx.x & 31;
  ProjectUnits<4, 3, false>(
      c, ins, base, program.head.rows, dim, c.work,
      [&](int, int, int item, int rows, const float* sums) {
        // The head's BF16 output, then the soft cap as the graph's BF16
        // tensors take it (megakernel_steps.cu's SoftCapKernel).
        const float z = Bf16(RowOf<4>(sums) * factor);
        if (lane < rows) {
          logits[item * 4 + lane] =
              capped ? Bf16(Bf16(tanhf(Bf16(z * soft_in))) * soft_out) : z;
        }
      });
}

// --- The kernel -----------------------------------------------------------------

// As with AttentionItems, for the hints of the code inlined into the kernel.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
__global__ void __launch_bounds__(kThreads, 1)
    MegakernelKernel(const LiteRtNvidiaMegakernelProgram* __restrict__ program,
                     const __grid_constant__ LiteRtNvidiaMegakernelStep step,
                     char* workspace, State* state, uint64_t* trace) {
  // The weight stream takes bulk copies and mbarrier transactions (sm_90 on);
  // the builder refuses older devices, for which the kernel is empty.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  extern __shared__ __align__(128) char shared[];
  __shared__ uint32_t epoch_s;
  const SharedLayout layout =
      LayoutShared(program->model_dim, program->work_bytes);
  Context c;
  c.program = program;
  c.step = &step;
  c.workspace = workspace;
  c.state = state;
  c.x_s = reinterpret_cast<__nv_bfloat16*>(shared + layout.x);
  c.h_s = reinterpret_cast<__nv_bfloat16*>(shared + layout.h);
  c.route_s = reinterpret_cast<int32_t*>(shared + layout.route);
  c.weight_s =
      reinterpret_cast<float*>(shared + layout.route + 2 * kMaxTopK * 4);
  c.scaled_s =
      reinterpret_cast<float*>(shared + layout.route + 2 * kMaxTopK * 8);
  c.hsq_s =
      reinterpret_cast<float*>(shared + layout.route + 2 * kMaxTopK * 12);
  c.partial_s = reinterpret_cast<float*>(shared + layout.partial);
  c.counter_s = reinterpret_cast<volatile int*>(shared + layout.partial +
                                                kPartialFloats * 4);
  c.reduce_s = reinterpret_cast<float*>(shared + layout.reduce);
  c.control = reinterpret_cast<volatile int*>(shared + layout.control);
  c.full = reinterpret_cast<uint64_t*>(shared + layout.barriers);
  c.empty = c.full + kStages;
  c.work = reinterpret_cast<float*>(shared + layout.work);
  c.ring = shared + layout.ring;
  c.cta = blockIdx.x;
  c.ctas = gridDim.x;
  c.vcta = c.cta;
  c.prologue_stamp = nullptr;
  c.wait_stamp = nullptr;
  if (threadIdx.x == 0) {
    epoch_s = LoadAcquire(&state->epoch);
    for (int s = 0; s < kStages; ++s) {
      InitBarrier(&c.full[s]);
      InitBarrier(&c.empty[s]);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    for (int w = 0; w < kControlWords; ++w) c.control[w] = 0;
    for (int s = 0; s < kStages; ++s) c.control[kControlArmed + s] = -1;
  }
  __syncthreads();
  // Every CTA finishes every instruction once per step.
  c.target = (epoch_s + 1) * static_cast<uint32_t>(c.ctas);
  if (threadIdx.x / 32 == kProducerWarp) {
    Produce(c);
    return;
  }
  const int count = program->instruction_count;
  uint64_t* stamps =
      trace != nullptr ? trace + static_cast<size_t>(c.cta) * count * 4 : nullptr;
  if (trace != nullptr && threadIdx.x == 0) {
    trace[static_cast<size_t>(c.ctas) * count * 4 + c.cta] = SmId();
  }
  uint32_t base = 0;  // the stream index of the instruction's first unit
  int rotation = 0;   // the projections' items so far, mod ctas
  for (int i = 0; i < count; ++i) {
    const LiteRtNvidiaMegakernelInstruction ins = program->instruction[i];
    if (IsProjectionOp(ins.opcode)) {
      c.vcta = (c.cta + c.ctas - rotation) % c.ctas;
    }
    if (stamps != nullptr && threadIdx.x == 0) stamps[i * 4] = GlobalTimer();
    // Each instruction's code begins it (Begin) once the loads that do not
    // wait for its inputs are in flight.
    c.wait_stamp = stamps != nullptr ? stamps + i * 4 + 1 : nullptr;
    c.prologue_stamp = stamps != nullptr ? stamps + i * 4 + 2 : nullptr;
    switch (ins.opcode) {
      case kLiteRtNvidiaMegakernelOpQkv:
        RunQkv(c, ins, base);
        break;
      case kLiteRtNvidiaMegakernelOpNormRope:
        RunNormRope(c, ins);
        break;
      case kLiteRtNvidiaMegakernelOpAttention:
        if (program->layer[ins.layer].head_dim == kGlobalDepth) {
          AttentionItems<kGlobalDepth, kGlobalRows>(c, ins);
        } else {
          AttentionItems<kLocalDepth, kLocalRows>(c, ins);
        }
        break;
      case kLiteRtNvidiaMegakernelOpAttentionCombine:
        if (program->layer[ins.layer].head_dim == kGlobalDepth) {
          CombineItems<kGlobalDepth>(c, ins);
        } else {
          CombineItems<kLocalDepth>(c, ins);
        }
        break;
      case kLiteRtNvidiaMegakernelOpOutput:
        RunOutput(c, ins, base);
        break;
      case kLiteRtNvidiaMegakernelOpRouter:
        RunRouter(c, ins, base);
        break;
      case kLiteRtNvidiaMegakernelOpDenseGateUp:
        RunDenseGateUp(c, ins, base, i);
        break;
      case kLiteRtNvidiaMegakernelOpExpertHidden:
        RunExpertHidden(c, ins, base);
        break;
      case kLiteRtNvidiaMegakernelOpDenseDown:
        RunDenseDown(c, ins, base);
        break;
      case kLiteRtNvidiaMegakernelOpExpertDown:
        RunExpertDown(c, ins, base);
        break;
      case kLiteRtNvidiaMegakernelOpHead:
        RunHead(c, ins, base);
        break;
      default:
        break;
    }
    if (IsProjectionOp(ins.opcode)) {
      base += ItemsOf(c, ins.items) * ins.parts;
      rotation = (rotation + ins.items) % c.ctas;
    }
    Sync();
    if (threadIdx.x == 0) {
      if (stamps != nullptr) stamps[i * 4 + 3] = GlobalTimer();
      __threadfence();
      atomicAdd(&state->done[i], 1u);
    }
  }
  if (threadIdx.x == 0) {
    __threadfence();
    const uint32_t before = atomicAdd(&state->finished, 1u);
    if (before + 1 == c.target) {
      // The last CTA of the step: the next launch's epoch.
      atomicExch(&state->epoch, epoch_s + 1);
    }
  }
#endif  // __CUDA_ARCH__ >= 900
}
#pragma clang diagnostic pop

// --- The builder ----------------------------------------------------------------

uint64_t Take(uint64_t* at, uint64_t bytes) {
  const uint64_t offset = *at;
  *at += (bytes + 255) / 256 * 256;
  return offset;
}

bool LayerSupported(const LiteRtNvidiaMegakernelModel& m,
                    const LiteRtNvidiaMegakernelLayer& l) {
  if (l.q_heads <= 0 || l.kv_heads <= 0 || l.q_heads % l.kv_heads != 0 ||
      l.cache_len <= 0 || l.mask < 0 || l.mask > 1) {
    return false;
  }
  const int group = l.q_heads / l.kv_heads;
  const int kv_rows = l.shared_kv != 0 ? 1 : 2;
  // The attention items the kernel is built for, chunks of 64 keys at every
  // cache length: decode_attention.cu takes chunks of 32 below 16 *
  // kMaxSplits keys for many rows (the 26B's global layers at a 2,048-token
  // context), which only reorders the softmax's sums.
  const bool local = l.head_dim == kLocalDepth && group <= kLocalRows;
  const bool global = l.head_dim == kGlobalDepth && group > kLocalRows &&
                      group <= kGlobalRows;
  if (!local && !global) return false;
  const int dim = m.model_dim;
  // Every projection's rows go through the ring in units the kernel's
  // instantiations take (DenseItems; the experts' fixed units).
  auto dense = [&](int rows, int cols) {
    const int unit = UnitShapeFor(rows, cols).rows;
    const int lane_blocks = (cols / kBlock + 31) / 32;
    return (unit == 4 && lane_blocks <= 3) || (unit == 2 && lane_blocks <= 4) ||
           (unit == 1 && lane_blocks <= 8);
  };
  const bool units =
      dense(l.qkv.rows, dim) && dense(dim, l.o.cols) && dense(l.experts, dim) &&
      dense(l.dense_hidden, dim) && dense(dim, l.dense_hidden) &&
      (dim / kBlock + 31) / 32 <= 3 &&
      l.expert_hidden % kExpertRows == 0 &&
      UnitBytes(kExpertRows, dim, true) <= kStageBytes &&
      (kExpertRows * (dim / kBlock)) % 16 == 0 && dim % kDownRows == 0 &&
      l.expert_hidden / kBlock <= kMaxDownBlocks &&
      UnitBytes(kDownRows, l.expert_hidden, true) <= kStageBytes &&
      (kDownRows * (l.expert_hidden / kBlock)) % 16 == 0;
  return units && l.rotary_dim >= 0 && l.rotary_dim <= l.head_dim &&
         l.rotary_dim % 2 == 0 && l.head_dim <= rope::kMaxDepth &&
         l.qkv.rows == (l.q_heads + kv_rows * l.kv_heads) * l.head_dim &&
         l.qkv.cols == dim && l.o.rows == dim &&
         l.o.cols == l.q_heads * l.head_dim && l.o.cols % kBlock == 0 &&
         l.o.cols <= 8192 && l.experts > 0 && l.experts <= kMaxExperts &&
         l.top_k > 0 && l.top_k <= kMaxTopK && l.top_k <= l.experts &&
         l.expert_hidden % kBlock == 0 && l.expert_hidden > 0 &&
         l.expert_hidden <= 32 * kBlock && l.dense_hidden % kBlock == 0 &&
         l.dense_hidden > 0 && (l.dense_gate == 1 || l.dense_gate == 2) &&
         (l.expert_gate == 1 || l.expert_gate == 2) &&
         l.arena.compact_scales != 0 && l.arena.expert_major != 0;
}

// The pointers the bulk copies read from start on 16 bytes.
bool Aligned(const void* pointer) {
  return reinterpret_cast<uintptr_t>(pointer) % 16 == 0;
}

int TilesOf(int rows, int tile_rows) { return (rows + tile_rows - 1) / tile_rows; }

// The kernel streams with bulk copies and mbarrier transactions: sm_90 on.
cudaError_t CheckDevice() {
  int device = 0;
  int major = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status == cudaSuccess) {
    status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                    device);
  }
  if (status != cudaSuccess) return status;
  return major >= 9 ? cudaSuccess : cudaErrorNotSupported;
}

}  // namespace
}  // namespace litert::nvidia::megakernel

using namespace litert::nvidia::megakernel;

extern "C" size_t LiteRtNvidiaMegakernelStateBytes(void) {
  return sizeof(State);
}

extern "C" size_t LiteRtNvidiaMegakernelTraceBytes(
    const LiteRtNvidiaMegakernelProgram* program) {
  if (program == nullptr) return 0;
  return (static_cast<size_t>(program->ctas) * program->instruction_count * 4 +
          program->ctas) *
         sizeof(uint64_t);
}

extern "C" cudaError_t LiteRtNvidiaMegakernelBuildProgram(
    const LiteRtNvidiaMegakernelModel* model, int32_t ctas,
    LiteRtNvidiaMegakernelProgram* program) {
  if (model == nullptr || program == nullptr || model->layers <= 0 ||
      model->layers > kLiteRtNvidiaMegakernelMaxLayers ||
      model->layer == nullptr || model->model_dim <= 0 ||
      model->model_dim % kBlock != 0 || model->model_dim > kMaxModelDim ||
      model->vocab <= 0 || model->head.rows != model->vocab ||
      model->head.cols != model->model_dim || ctas < 0 ||
      ctas > kLiteRtNvidiaMegakernelMaxCtas) {
    return cudaErrorInvalidValue;
  }
  for (int l = 0; l < model->layers; ++l) {
    if (!LayerSupported(*model, model->layer[l])) return cudaErrorNotSupported;
  }
  if (UnitShapeFor(model->vocab, model->model_dim).rows <= 0) {
    return cudaErrorNotSupported;
  }
  // With pointers, every projection's weights and scales start on 16 bytes
  // (bulk copies); a description without pointers builds the shapes alone.
  for (int l = 0; l < model->layers; ++l) {
    const auto& layer = model->layer[l];
    const auto& a = layer.arena;
    for (const void* p :
         {static_cast<const void*>(layer.qkv.weights), layer.qkv.scales,
          static_cast<const void*>(layer.o.weights), layer.o.scales,
          static_cast<const void*>(a.router), a.router_scales,
          static_cast<const void*>(a.dense_gate_up), a.dense_gate_up_scales,
          static_cast<const void*>(a.dense_down), a.dense_down_scales,
          static_cast<const void*>(a.gate), a.gate_scales, a.gate_scale_max,
          static_cast<const void*>(a.up), a.up_scales, a.up_scale_max,
          static_cast<const void*>(a.down), a.down_scales, a.down_scale_max}) {
      if (p != nullptr && !Aligned(p)) return cudaErrorNotSupported;
    }
  }
  if ((model->head.weights != nullptr && !Aligned(model->head.weights)) ||
      (model->head.scales != nullptr && !Aligned(model->head.scales)) ||
      (model->final_norm != nullptr && !Aligned(model->final_norm))) {
    return cudaErrorNotSupported;
  }
  // The prologues read the norms' vectors in 16-byte quads.
  for (int l = 0; l < model->layers; ++l) {
    const auto& layer = model->layer[l];
    for (const float* v : {layer.input_norm, layer.post_attention_norm,
                           layer.router_norm, layer.dense_norm, layer.expert_norm,
                           layer.dense_post, layer.moe_post, layer.sum_post}) {
      if (v != nullptr && !Aligned(v)) return cudaErrorNotSupported;
    }
  }
  std::memset(program, 0, sizeof(*program));
  auto& p = *program;
  p.version = kLiteRtNvidiaMegakernelVersion;
  p.threads = kThreads;
  p.layers = model->layers;
  p.model_dim = model->model_dim;
  p.vocab = model->vocab;
  p.norm_eps = model->norm_eps;
  p.soft_cap = model->soft_cap;
  p.soft_cap_in = model->soft_cap_in;
  p.final_norm = model->final_norm;
  p.head = model->head;
  const int dim = model->model_dim;
  // The workspace: each layer's scratch, then the tail's.
  uint64_t at = 0;
  int max_activation = dim;
  int work_bytes = 0;
  for (int l = 0; l < model->layers; ++l) {
    const LiteRtNvidiaMegakernelLayer& layer = model->layer[l];
    p.layer[l] = layer;
    auto& s = p.scratch[l];
    const int depth = layer.head_dim;
    const int group = layer.q_heads / layer.kv_heads;
    s.chunk = kChunk;
    s.splits = std::min((layer.cache_len + kChunk - 1) / kChunk, kMaxSplits);
    const uint64_t qkv_values =
        static_cast<uint64_t>(layer.q_heads + 2 * layer.kv_heads) * depth;
    s.x = Take(&at, dim * 2);
    s.qkv = Take(&at, static_cast<uint64_t>(layer.qkv.rows) * 2);
    s.qkv_values = Take(&at, qkv_values * 2);
    s.q = Take(&at, static_cast<uint64_t>(layer.q_heads) * depth * 2);
    s.k = Take(&at, static_cast<uint64_t>(layer.kv_heads) * depth * 2);
    s.v = Take(&at, static_cast<uint64_t>(layer.kv_heads) * depth * 2);
    // The partials as the per-op launch lays out its workspace: acc, then
    // the maxima, then the sums, sized as it asks.
    const uint64_t partials =
        static_cast<uint64_t>(layer.kv_heads) * s.splits * group;
    const uint64_t attention_bytes = std::max<uint64_t>(
        LiteRtNvidiaDecodeAttentionWorkspaceBytes(layer.kv_heads, group,
                                                  layer.cache_len, depth),
        partials * (depth + 2) * 4);
    s.attention_acc = Take(&at, attention_bytes);
    s.attention_max = s.attention_acc + partials * depth * 4;
    s.attention_sum = s.attention_max + partials * 4;
    s.attention_out = Take(&at, static_cast<uint64_t>(layer.q_heads) * depth * 2);
    s.o = Take(&at, dim * 2);
    s.h = Take(&at, dim * 2);
    s.router = Take(&at, static_cast<uint64_t>(layer.experts) * 2);
    s.routing = Take(&at, static_cast<uint64_t>(layer.top_k) * 8);
    s.expert_hidden =
        Take(&at, static_cast<uint64_t>(layer.top_k) * layer.expert_hidden * 4);
    s.experts_out = Take(&at, static_cast<uint64_t>(dim) * 4);
    s.dense_hidden = Take(&at, static_cast<uint64_t>(layer.dense_hidden) * 2);
    s.dense_out = Take(&at, dim * 2);
    const LiteRtNvidiaMoeShape shape = {1,   layer.top_k,         layer.experts,
                                        dim, layer.expert_hidden, layer.expert_gate};
    s.moe = Take(&at, LiteRtNvidiaMoeBlockWorkspaceBytes(
                          &shape, layer.dense_hidden, layer.dense_gate));
    max_activation = std::max({max_activation, layer.o.cols, layer.dense_hidden});
    // The shared memory the layer's items take in `work`.
    const int attention_shared =
        depth == kGlobalDepth
            ? static_cast<int>(attention::SharedBytesFor<kGlobalDepth, kGlobalRows,
                                                         false, kChunk>())
            : static_cast<int>(attention::SharedBytesFor<kLocalDepth, kLocalRows,
                                                         false, kChunk>());
    const int combine_shared =
        (depth == kGlobalDepth ? attention::CombineSharedFloats<kGlobalDepth>()
                               : attention::CombineSharedFloats<kLocalDepth>()) *
        4;
    work_bytes = std::max({work_bytes, attention_shared, combine_shared,
                           (rope::kMaxDepth + rope::kThreads / 32) * 4,
                           layer.top_k * layer.expert_hidden * 4,
                           2 * dim * 4});
  }
  for (int l = 0; l < model->layers; ++l) {
    const auto& layer = model->layer[l];
    p.mask_len[layer.mask] = std::max(p.mask_len[layer.mask], layer.cache_len);
  }
  p.mask_last = Take(&at, 2 * kLiteRtNvidiaMegakernelMaxCtas * 4);
  p.final_x = Take(&at, dim * 2);
  p.head_raw = Take(&at, static_cast<uint64_t>(model->vocab) * 2);
  p.workspace_bytes = at;
  p.max_activation = max_activation;
  // BF16 projection inputs (the o and dense down projections).
  work_bytes = std::max(work_bytes, max_activation * 2);
  p.work_bytes = work_bytes;
  p.shared_bytes = LayoutShared(dim, work_bytes).total;

  if (ctas == 0) {
    cudaError_t status = CheckDevice();
    if (status != cudaSuccess) return status;
    status = cudaFuncSetAttribute(
        MegakernelKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        p.shared_bytes);
    if (status != cudaSuccess) return status;
    int device = 0;
    int sms = 0;
    int per_sm = 0;
    if ((status = cudaGetDevice(&device)) != cudaSuccess ||
        (status = cudaDeviceGetAttribute(
             &sms, cudaDevAttrMultiProcessorCount, device)) != cudaSuccess ||
        (status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
             &per_sm, MegakernelKernel, kThreads, p.shared_bytes)) !=
            cudaSuccess) {
      return status;
    }
    if (per_sm <= 0) return cudaErrorNotSupported;
    ctas = std::min(sms * per_sm, kLiteRtNvidiaMegakernelMaxCtas);
  }
  p.ctas = ctas;

  // The instructions, layer by layer. A projection's items are tiles of its
  // unit's rows; `parts` units per item.
  int n = 0;
  auto add = [&](int opcode, int layer, int items, int tile_rows, int parts,
                 std::initializer_list<int> waits) {
    auto& ins = p.instruction[n];
    ins.opcode = opcode;
    ins.layer = layer;
    ins.items = items;
    ins.tile_rows = tile_rows;
    ins.parts = parts;
    int w = 0;
    for (const int wait : waits) ins.waits[w++] = wait;
    for (; w < kLiteRtNvidiaMegakernelMaxWaits; ++w) ins.waits[w] = -1;
    return n++;
  };
  int dense_down = -1;
  int expert_down = -1;
  for (int l = 0; l < model->layers; ++l) {
    const auto& layer = model->layer[l];
    if (n + 11 > kLiteRtNvidiaMegakernelMaxInstructions) {
      return cudaErrorNotSupported;
    }
    const int group = layer.q_heads / layer.kv_heads;
    const int qkv_rows = UnitShapeFor(layer.qkv.rows, dim).rows;
    const int qkv = add(kLiteRtNvidiaMegakernelOpQkv, l,
                        TilesOf(layer.qkv.rows, qkv_rows), qkv_rows, 1,
                        {dense_down, expert_down});
    const int norm_rope = add(kLiteRtNvidiaMegakernelOpNormRope, l,
                              layer.q_heads + 2 * layer.kv_heads, 0, 0, {qkv});
    const int attention_items = p.scratch[l].splits * layer.kv_heads;
    const int partial = add(kLiteRtNvidiaMegakernelOpAttention, l,
                            attention_items, 0, 0, {norm_rope});
    const int combine = add(kLiteRtNvidiaMegakernelOpAttentionCombine, l,
                            group * layer.kv_heads, 0, 0, {partial});
    const int o_rows = UnitShapeFor(dim, layer.o.cols).rows;
    const int output = add(kLiteRtNvidiaMegakernelOpOutput, l,
                           TilesOf(dim, o_rows), o_rows, 1, {combine});
    const int router_rows = UnitShapeFor(layer.experts, dim).rows;
    const int router = add(kLiteRtNvidiaMegakernelOpRouter, l,
                           TilesOf(layer.experts, router_rows), router_rows, 1,
                           {output});
    const int dense_rows = UnitShapeFor(layer.dense_hidden, dim).rows;
    const int gate_up = add(kLiteRtNvidiaMegakernelOpDenseGateUp, l,
                            TilesOf(layer.dense_hidden, dense_rows), dense_rows,
                            2, {output});
    // waits[0] is the router: the producer routes the layer once it is done.
    const int hidden = add(kLiteRtNvidiaMegakernelOpExpertHidden, l,
                           layer.top_k * TilesOf(layer.expert_hidden, kExpertRows),
                           kExpertRows, 2, {router});
    const int down_rows = UnitShapeFor(dim, layer.dense_hidden).rows;
    dense_down = add(kLiteRtNvidiaMegakernelOpDenseDown, l,
                     TilesOf(dim, down_rows), down_rows, 1, {gate_up});
    expert_down = add(kLiteRtNvidiaMegakernelOpExpertDown, l,
                      TilesOf(dim, kDownRows), kDownRows, layer.top_k, {hidden});
  }
  const int head_rows = UnitShapeFor(model->vocab, dim).rows;
  if (head_rows != 4 || (dim / kBlock + 31) / 32 > 3) return cudaErrorNotSupported;
  add(kLiteRtNvidiaMegakernelOpHead, model->layers,
      TilesOf(model->vocab, head_rows), head_rows, 1, {dense_down, expert_down});
  p.instruction_count = n;
  for (int i = 0; i < n; ++i) {
    const auto& ins = p.instruction[i];
    if (!IsProjection(ins.opcode)) continue;
    if (ins.tile_rows <= 0) return cudaErrorNotSupported;
    // An item of several parts keeps its parts' sums in shared memory.
    const int items = (ins.items + ctas - 1) / ctas;
    if (ins.parts > 1 && (items > kMaxCtaItems ||
                          items * ins.parts * ins.tile_rows > kPartialFloats)) {
      return cudaErrorNotSupported;
    }
  }
  return cudaSuccess;
}

extern "C" cudaError_t LiteRtNvidiaMegakernelLaunch(
    const LiteRtNvidiaMegakernelProgram* program,
    const LiteRtNvidiaMegakernelProgram* device_program,
    const LiteRtNvidiaMegakernelStep* step, void* workspace, void* state,
    uint64_t* trace, cudaStream_t stream) {
  if (program == nullptr || device_program == nullptr || step == nullptr ||
      workspace == nullptr || state == nullptr ||
      program->version != kLiteRtNvidiaMegakernelVersion || program->ctas <= 0) {
    return cudaErrorInvalidValue;
  }
  static bool attribute_set = false;
  static int attribute_bytes = 0;
  if (!attribute_set || attribute_bytes < program->shared_bytes) {
    cudaError_t status = CheckDevice();
    if (status != cudaSuccess) return status;
    status = cudaFuncSetAttribute(
        MegakernelKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        program->shared_bytes);
    if (status != cudaSuccess) return status;
    attribute_set = true;
    attribute_bytes = program->shared_bytes;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(program->ctas);
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = program->shared_bytes;
  config.stream = stream;
  // The CTAs wait for each other: all of them must be resident at once.
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeCooperative;
  attribute.val.cooperative = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(&config, MegakernelKernel, device_program, *step,
                            static_cast<char*>(workspace),
                            static_cast<State*>(state), trace);
}

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


#include "litert/vendors/nvidia/trtllm/hyper_connection.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace {

// CTAs of sixteen warps: of every stream, as many CTAs read the activations
// and the norm weights as its pieces have tasks for their warps, so larger
// CTAs read less of them twice.
constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;
constexpr unsigned kFullWarp = 0xffffffffu;
constexpr int kChunk = 8;  // BF16 values per 16-byte load
// A thread loads at most this many chunks of a stream (its norm): streams of
// up to kThreads * kMaxStreamChunks * kChunk values.
constexpr int kMaxStreamChunks = 2;
constexpr int kMaxRows = 8;
constexpr int kMaxStreams = 8;
// A row of the down projection (low_rank rows of streams * hidden values) is
// summed in pieces, a warp each: every stream in pieces of about
// kPieceValues values; at most kMaxSplits pieces a row.
constexpr int kMaxSplits = 16;
constexpr int kPieceValues = 2560;
// The steps of its first task of each projection whose weights a thread
// copies into shared memory when the kernel starts, without waiting: the
// weights do not depend on anything the kernel computes, so the wait for
// them overlaps the norm (and, for the up projection, the down projection).
constexpr int kHeld = 3;

// Weights in BF16 rows, or INT4 in blocks of 32 (two's complement nibbles,
// low first, a half scale a block: the CUDA subbyte GEMV's layout).
struct Weights {
  const void* values;
  const __half* scales;  // INT4 only, [rows][columns / 32]
};

// One launch's shapes, its plan (a function of the shapes alone: the order
// of every sum is the same on every device and at every grid size) and its
// pointers.
struct Launch {
  int32_t rows;
  int32_t streams;
  int32_t hidden;
  int32_t low_rank;
  int32_t width;  // streams * hidden
  float down_scale;
  float eps;
  int32_t stream_chunks;  // hidden / kChunk
  int32_t stream_steps;   // hidden / step
  int32_t stream_pieces;  // the pieces of a stream of a down row
  int32_t piece;          // the steps of a piece
  int32_t splits;         // streams * stream_pieces
  int32_t stream_visits;  // low_rank * stream_pieces / kWarps
  int32_t up_steps;       // low_rank / step
  int32_t lanes;          // the lanes of a warp that share an up row
  int32_t lanes_shift;    // lanes == 1 << lanes_shift
  int32_t xn_floats;      // a row's normed stream in shared memory
  int32_t t_floats;       // a row's low-rank gates in shared memory
  int32_t group_ctas;     // the CTAs of a group of rows
  // 0: the whole mix, the projections meeting at a grid barrier. 1: the norm
  // and the down projection; 2: the rest, from `normed` and `partial`.
  int32_t phase;
  const __nv_bfloat16* x;
  const float* norm_weight;
  Weights down;
  Weights up;
  __nv_bfloat16* mixed;
  __nv_bfloat16* normed;
  float* partial;   // [rows][low_rank][kMaxSplits]
  unsigned* state;  // [0] arrivals at the grid barrier, [1] CTAs done
};

__device__ __forceinline__ void Unpack(const uint4& raw, float (&out)[kChunk]) {
  const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
  for (int i = 0; i < kChunk / 2; ++i) {
    out[2 * i] = __uint_as_float(words[i] << 16);
    out[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000u);
  }
}

__device__ __forceinline__ float WarpSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(kFullWarp, value, offset);
  }
  return value;
}

__device__ __forceinline__ float Nibble(uint32_t word, int index) {
  const uint32_t bits = (word >> (4 * index)) & 0xfu;
  return static_cast<float>(static_cast<int>((bits ^ 8u) - 8u));
}

// A step of a weight row: 16 bytes, 32 INT4 values and their scale or 8 BF16
// values.
struct Step {
  uint4 raw;
  float scale;
};

template <bool kInt4>
__device__ __forceinline__ const void* StepWeights(const Weights& w,
                                                   int columns, int row,
                                                   int step) {
  if constexpr (kInt4) {
    return static_cast<const uint8_t*>(w.values) +
           static_cast<size_t>(row) * (columns / 2) +
           static_cast<size_t>(step) * 16;
  } else {
    return static_cast<const __nv_bfloat16*>(w.values) +
           static_cast<size_t>(row) * columns +
           static_cast<size_t>(step) * kChunk;
  }
}

__device__ __forceinline__ const __half* StepScale(const Weights& w,
                                                   int columns, int row,
                                                   int step) {
  return w.scales + static_cast<size_t>(row) * (columns / 32) + step;
}

template <bool kInt4>
__device__ __forceinline__ Step LoadStep(const Weights& w, int columns, int row,
                                         int step) {
  Step loaded;
  loaded.raw = __ldg(static_cast<const uint4*>(
      StepWeights<kInt4>(w, columns, row, step)));
  loaded.scale = 1.0f;
  if constexpr (kInt4) {
    loaded.scale = __half2float(__ldg(StepScale(w, columns, row, step)));
  }
  return loaded;
}

// Copies from global to shared memory that the thread does not wait for
// (compute capability 8.0): the copies of a group are there after the wait
// for it.
__device__ __forceinline__ void CopyAsync16(void* shared_pointer,
                                            const void* global_pointer) {
  const uint32_t address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_pointer));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(address), "l"(global_pointer)
               : "memory");
}

__device__ __forceinline__ void CopyAsync4(void* shared_pointer,
                                           const void* global_pointer) {
  const uint32_t address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_pointer));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
               :
               : "r"(address), "l"(global_pointer)
               : "memory");
}

__device__ __forceinline__ void CommitAsync() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// Waits for all groups of copies, or for all but the most recent one.
__device__ __forceinline__ void WaitAsync() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ void WaitAsyncButOne() {
  asm volatile("cp.async.wait_group 1;\n" ::: "memory");
}

// Where a task's values are in shared memory. Lanes read the float4 runs of
// their steps at the same time, so the runs that a warp reads together are
// neighbours (lane after lane), not the runs of one step: 32 lanes reading
// 32 values apart would all fall on the same banks. A step's run q is at
// values + q * run_pitch, the task's next step at + step_pitch, a CTA's next
// row at + row_pitch (floats).
struct Values {
  const float* values;
  int run_pitch;
  int step_pitch;
  int row_pitch;
};

// acc[r] += the step's weights times the step's values of row r, for every
// row a CTA holds.
template <bool kInt4, int kRows>
__device__ __forceinline__ void Accumulate(const Step& step, const float* v,
                                           const Values& at,
                                           float (&acc)[kRows]) {
  if constexpr (kInt4) {
    const uint32_t words[4] = {step.raw.x, step.raw.y, step.raw.z, step.raw.w};
    float block[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) block[r] = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      float4 lo[kRows], hi[kRows];
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const float* run = v + r * at.row_pitch + 2 * w * at.run_pitch;
        lo[r] = *reinterpret_cast<const float4*>(run);
        hi[r] = *reinterpret_cast<const float4*>(run + at.run_pitch);
      }
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        block[r] = fmaf(Nibble(words[w], 0), lo[r].x, block[r]);
        block[r] = fmaf(Nibble(words[w], 1), lo[r].y, block[r]);
        block[r] = fmaf(Nibble(words[w], 2), lo[r].z, block[r]);
        block[r] = fmaf(Nibble(words[w], 3), lo[r].w, block[r]);
        block[r] = fmaf(Nibble(words[w], 4), hi[r].x, block[r]);
        block[r] = fmaf(Nibble(words[w], 5), hi[r].y, block[r]);
        block[r] = fmaf(Nibble(words[w], 6), hi[r].z, block[r]);
        block[r] = fmaf(Nibble(words[w], 7), hi[r].w, block[r]);
      }
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = fmaf(block[r], step.scale, acc[r]);
  } else {
    float weights[kChunk];
    Unpack(step.raw, weights);
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const float* run = v + r * at.row_pitch;
      const float4 lo = *reinterpret_cast<const float4*>(run);
      const float4 hi = *reinterpret_cast<const float4*>(run + at.run_pitch);
      acc[r] = fmaf(weights[0], lo.x, acc[r]);
      acc[r] = fmaf(weights[1], lo.y, acc[r]);
      acc[r] = fmaf(weights[2], lo.z, acc[r]);
      acc[r] = fmaf(weights[3], lo.w, acc[r]);
      acc[r] = fmaf(weights[4], hi.x, acc[r]);
      acc[r] = fmaf(weights[5], hi.y, acc[r]);
      acc[r] = fmaf(weights[6], hi.z, acc[r]);
      acc[r] = fmaf(weights[7], hi.w, acc[r]);
    }
  }
}

// The steps first, first + stride, ... below `end` of weight row `row`.
struct Task {
  int row;
  int first;
  int end;
  int stride;
};

// A thread's first kHeld steps of a task in shared memory: step i's weights
// at weights[i * kThreads] and, for INT4, the four bytes that hold its scale
// at scales[i * kThreads] (the scales are two bytes apart).
struct Staged {
  uint4* weights;
  uint32_t* scales;
};

template <bool kInt4>
__device__ __forceinline__ void Stage(const Weights& w, int columns,
                                      const Task& task, const Staged& staged) {
#pragma unroll
  for (int i = 0; i < kHeld; ++i) {
    const int at = task.first + i * task.stride;
    if (at < task.end) {
      CopyAsync16(staged.weights + i * kThreads,
                  StepWeights<kInt4>(w, columns, task.row, at));
      if constexpr (kInt4) {
        CopyAsync4(staged.scales + i * kThreads,
                   reinterpret_cast<const void*>(
                       reinterpret_cast<uintptr_t>(
                           StepScale(w, columns, task.row, at)) &
                       ~static_cast<uintptr_t>(3)));
      }
    }
  }
}

// acc[r] += the task's dot product with the rows' values: its first kHeld
// steps from `staged` when given, the rest loaded here, the next step's
// weights asked for before the current one's products.
template <bool kInt4, int kRows>
__device__ __forceinline__ void Dot(const Weights& w, int columns,
                                    const Task& task, const Staged* staged,
                                    const Values& values, float (&acc)[kRows]) {
  int at = task.first;
  const float* v = values.values;
  if (staged != nullptr) {
#pragma unroll
    for (int i = 0; i < kHeld; ++i) {
      if (at < task.end) {
        Step step;
        step.raw = staged->weights[i * kThreads];
        step.scale = 1.0f;
        if constexpr (kInt4) {
          const uint32_t pair = staged->scales[i * kThreads];
          const bool high = (reinterpret_cast<uintptr_t>(
                                 StepScale(w, columns, task.row, at)) &
                             2) != 0;
          step.scale = __half2float(__ushort_as_half(
              static_cast<unsigned short>(high ? pair >> 16 : pair & 0xffffu)));
        }
        Accumulate<kInt4, kRows>(step, v, values, acc);
      }
      at += task.stride;
      v += values.step_pitch;
    }
  }
  if (at >= task.end) return;
  Step current = LoadStep<kInt4>(w, columns, task.row, at);
  while (true) {
    const int next = at + task.stride;
    const bool more = next < task.end;
    Step ahead = current;
    if (more) ahead = LoadStep<kInt4>(w, columns, task.row, next);
    Accumulate<kInt4, kRows>(current, v, values, acc);
    if (!more) break;
    current = ahead;
    at = next;
    v += values.step_pitch;
  }
}

// The place (in floats) of run `run` of step `step` among steps that lanes
// read `group` at a time (a power of two, 1 << shift), `runs` runs a step.
__device__ __forceinline__ int RunAt(int step, int run, int runs, int shift) {
  return ((((step >> shift) * runs + run) << shift) +
          (step & ((1 << shift) - 1))) *
         4;
}

// The place in a row of `xn` of half `half` of chunk `chunk` of a stream (the
// down projection's lanes read 32 steps at a time).
template <bool kInt4>
__device__ __forceinline__ int ChunkAt(int chunk, int half) {
  return kInt4 ? RunAt(chunk / 4, (chunk % 4) * 2 + half, 8, 5)
               : RunAt(chunk, half, 2, 5);
}

// Stream `stream` of the CTA's rows, normed into shared memory `xn`
// ([kRows][xn_floats], FP32 holding the BF16 values of `normed`, at
// ChunkAt): the squares in a fixed order (a thread's chunks, the warp, the
// warps).
template <bool kInt4, int kRows>
__device__ __forceinline__ void NormStream(const Launch& a, int row0, int rows,
                                           int stream, float* xn,
                                           float* scratch) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const float* norm_weight =
      a.norm_weight + static_cast<size_t>(stream) * a.hidden;
  float4 weights[kMaxStreamChunks][2];
#pragma unroll
  for (int j = 0; j < kMaxStreamChunks; ++j) {
    const int c = threadIdx.x + j * kThreads;
    if (c < a.stream_chunks) {
      const float4* at = reinterpret_cast<const float4*>(norm_weight) + 2 * c;
      weights[j][0] = __ldg(at);
      weights[j][1] = __ldg(at + 1);
    }
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    if (r >= rows) break;
    const __nv_bfloat16* x = a.x + static_cast<size_t>(row0 + r) * a.width +
                             static_cast<size_t>(stream) * a.hidden;
    float v[kMaxStreamChunks][kChunk];
    float squares = 0.0f;
#pragma unroll
    for (int j = 0; j < kMaxStreamChunks; ++j) {
      const int c = threadIdx.x + j * kThreads;
      if (c < a.stream_chunks) {
        Unpack(__ldg(reinterpret_cast<const uint4*>(x + c * kChunk)), v[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < kMaxStreamChunks; ++j) {
      const int c = threadIdx.x + j * kThreads;
      if (c < a.stream_chunks) {
#pragma unroll
        for (int i = 0; i < kChunk; ++i) {
          squares = fmaf(v[j][i], v[j][i], squares);
        }
      }
    }
    squares = WarpSum(squares);
    if (lane == 0) scratch[warp] = squares;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) total += scratch[w];
    const float factor = rsqrtf(total / static_cast<float>(a.hidden) + a.eps);
#pragma unroll
    for (int j = 0; j < kMaxStreamChunks; ++j) {
      const int c = threadIdx.x + j * kThreads;
      if (c < a.stream_chunks) {
        const float4 lo = weights[j][0];
        const float4 hi = weights[j][1];
        const __nv_bfloat162 pairs[4] = {
            __floats2bfloat162_rn(v[j][0] * factor * lo.x,
                                  v[j][1] * factor * lo.y),
            __floats2bfloat162_rn(v[j][2] * factor * lo.z,
                                  v[j][3] * factor * lo.w),
            __floats2bfloat162_rn(v[j][4] * factor * hi.x,
                                  v[j][5] * factor * hi.y),
            __floats2bfloat162_rn(v[j][6] * factor * hi.z,
                                  v[j][7] * factor * hi.w)};
        float rounded[kChunk];
        Unpack(*reinterpret_cast<const uint4*>(pairs), rounded);
        float* out = xn + r * a.xn_floats;
        *reinterpret_cast<float4*>(out + ChunkAt<kInt4>(c, 0)) =
            make_float4(rounded[0], rounded[1], rounded[2], rounded[3]);
        *reinterpret_cast<float4*>(out + ChunkAt<kInt4>(c, 1)) =
            make_float4(rounded[4], rounded[5], rounded[6], rounded[7]);
      }
    }
    // The scratch is free for the next row; the normed values are in place.
    __syncthreads();
  }
}

// Every CTA's results so far are written before any CTA goes on: the
// arrivals reach `target` (the launch is cooperative: the whole grid is
// resident).
__device__ __forceinline__ void GridBarrier(unsigned* arrivals,
                                            unsigned target) {
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(arrivals, 1u);
    // Volatile reads: a loop without side effects may be assumed to end.
    const volatile unsigned* seen = arrivals;
    while (*seen < target) {
    }
  }
  __syncthreads();
}

// grid: the CTAs of each group of kRows rows. The warps of a group take the
// tasks of the two projections in turn: of the down projection a piece of
// one stream of a low-rank output's row (the eight warps of a CTA the same
// piece of eight outputs, so a CTA norms and holds one stream at a time,
// and the CTAs of a stream write its `normed` between them); of the up
// projection the rows of a few hidden channels.
template <bool kInt4, int kRows>
__global__ void __launch_bounds__(kThreads) HcMixKernel(Launch a) {
  constexpr int kValues = kInt4 ? 32 : kChunk;
  extern __shared__ __align__(16) unsigned char shared[];
  __shared__ float scratch[kWarps];
  constexpr int kRuns = kValues / 4;
  float* xn = reinterpret_cast<float*>(shared);  // [kRows][xn_floats]
  float* t = xn + kRows * a.xn_floats;           // [kRows][t_floats]
  // [down, up][kHeld][kThreads] weights, then as many scales.
  uint4* staged_weights = reinterpret_cast<uint4*>(t + kRows * a.t_floats);
  uint32_t* staged_scales =
      reinterpret_cast<uint32_t*>(staged_weights + 2 * kHeld * kThreads);
  const Staged staged_down = {staged_weights + threadIdx.x,
                              staged_scales + threadIdx.x};
  const Staged staged_up = {staged_weights + kHeld * kThreads + threadIdx.x,
                            staged_scales + kHeld * kThreads + threadIdx.x};
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int group = blockIdx.x / a.group_ctas;
  const int cta = blockIdx.x - group * a.group_ctas;
  const int row0 = group * kRows;
  const int rows = a.rows - row0 < kRows ? a.rows - row0 : kRows;
  // The up projection's rows of this lane: `lanes` lanes share a row
  // (stream s of hidden channel h), the rows of a channel are neighbours.
  const int sub = lane / a.lanes;
  const int position = lane - sub * a.lanes;
  const int channel = sub / a.streams;
  const int up_stream = sub - channel * a.streams;
  const int channels = 32 / (a.lanes * a.streams);
  const int up_tasks = a.hidden / channels;
  const int first_task = cta * kWarps + warp;
  const auto up_task = [&](int task) {
    return Task{up_stream * a.hidden + task * channels + channel, position,
                a.up_steps, a.lanes};
  };
  const Values up_values = {t + position * 4, a.lanes * 4,
                            kRuns * a.lanes * 4, a.t_floats};
  // The down projection's task `task` (of low_rank * splits): output
  // task % low_rank, piece task / low_rank.
  const int visits = a.low_rank * a.splits / kWarps;
  const auto down_task = [&](int task, Values* values) {
    const int split = task / a.low_rank;
    const int stream = split / a.stream_pieces;
    const int begin = (split - stream * a.stream_pieces) * a.piece;
    const int end =
        begin + a.piece < a.stream_steps ? begin + a.piece : a.stream_steps;
    const int origin = stream * a.stream_steps;
    *values = {xn + RunAt(begin + lane, 0, kRuns, 5), 32 * 4, kRuns * 32 * 4,
               a.xn_floats};
    return Task{task - split * a.low_rank, origin + begin + lane, origin + end,
                32};
  };
  const bool hold_down =
      a.phase != 2 && first_task < a.low_rank * a.splits;
  const bool hold_up = a.phase != 1 && first_task < up_tasks;
  if (hold_down) {
    Values unused;
    Stage<kInt4>(a.down, a.width, down_task(first_task, &unused), staged_down);
  }
  CommitAsync();
  if (hold_up) Stage<kInt4>(a.up, a.low_rank, up_task(first_task), staged_up);
  CommitAsync();
  if (a.phase != 2) {
    int normed_stream = -1;
    bool wrote = lane == 0;
    for (int visit = cta; visit < visits; visit += a.group_ctas) {
      const int stream = visit / a.stream_visits;
      if (stream != normed_stream) {
        NormStream<kInt4, kRows>(a, row0, rows, stream, xn, scratch);
        normed_stream = stream;
      }
      // This visit's share of the stream's `normed`.
      const int share = visit - stream * a.stream_visits;
      for (int c = share + threadIdx.x * a.stream_visits; c < a.stream_chunks;
           c += kThreads * a.stream_visits) {
        wrote = true;
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
          if (r >= rows) break;
          const float* in = xn + r * a.xn_floats;
          const float4 lo =
              *reinterpret_cast<const float4*>(in + ChunkAt<kInt4>(c, 0));
          const float4 hi =
              *reinterpret_cast<const float4*>(in + ChunkAt<kInt4>(c, 1));
          const __nv_bfloat162 pairs[4] = {__floats2bfloat162_rn(lo.x, lo.y),
                                           __floats2bfloat162_rn(lo.z, lo.w),
                                           __floats2bfloat162_rn(hi.x, hi.y),
                                           __floats2bfloat162_rn(hi.z, hi.w)};
          *reinterpret_cast<uint4*>(
              a.normed + static_cast<size_t>(row0 + r) * a.width +
              static_cast<size_t>(stream) * a.hidden + c * kChunk) =
              *reinterpret_cast<const uint4*>(pairs);
        }
      }
      const int task_index = visit * kWarps + warp;
      Values values;
      const Task task = down_task(task_index, &values);
      float acc[kRows];
#pragma unroll
      for (int r = 0; r < kRows; ++r) acc[r] = 0.0f;
      // The staged weights of the down projection are there.
      if (visit == cta) WaitAsyncButOne();
      Dot<kInt4, kRows>(a.down, a.width, task,
                        visit == cta && hold_down ? &staged_down : nullptr,
                        values, acc);
      const int split = task_index / a.low_rank;
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const float sum = WarpSum(acc[r]);
        if (lane == 0 && r < rows) {
          a.partial[(static_cast<size_t>(row0 + r) * a.low_rank + task.row) *
                        kMaxSplits +
                    split] = sum;
        }
      }
    }
    if (a.phase == 1) return;
    // What this thread wrote is there for the others past the barrier.
    if (wrote) __threadfence();
    GridBarrier(a.state, gridDim.x);
  }
  // Other CTAs wrote what follows: it is read from the L2, not from what
  // this multiprocessor may hold of an earlier launch. First the normed
  // values of the first up task's rows, then the pieces.
  const float mean = 1.0f / static_cast<float>(a.streams);
  const auto normed_value = [&](const Task& task, int r) {
    const unsigned short bits = __ldcg(reinterpret_cast<const unsigned short*>(
        a.normed + static_cast<size_t>(row0 + r) * a.width + task.row));
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
  };
  float first_values[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    first_values[r] = 0.0f;
    if (r < rows && first_task < up_tasks) {
      first_values[r] = normed_value(up_task(first_task), r);
    }
  }
  // Every CTA: t = silu(scale * (the pieces' sum)) of its rows.
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    if (r >= rows) break;
    for (int j = threadIdx.x; j < a.low_rank; j += kThreads) {
      const float4* pieces = reinterpret_cast<const float4*>(
          a.partial +
          (static_cast<size_t>(row0 + r) * a.low_rank + j) * kMaxSplits);
      float4 parts[kMaxSplits / 4];
#pragma unroll
      for (int run = 0; run < kMaxSplits / 4; ++run) {
        if (run * 4 < a.splits) parts[run] = __ldcg(pieces + run);
      }
      float sum = 0.0f;
#pragma unroll
      for (int run = 0; run < kMaxSplits / 4; ++run) {
        if (run * 4 < a.splits) sum += parts[run].x;
        if (run * 4 + 1 < a.splits) sum += parts[run].y;
        if (run * 4 + 2 < a.splits) sum += parts[run].z;
        if (run * 4 + 3 < a.splits) sum += parts[run].w;
      }
      const float d = sum * a.down_scale;
      // Where the up projection's lanes read value j.
      t[r * a.t_floats +
        RunAt(j / kValues, (j % kValues) / 4, kRuns, a.lanes_shift) + j % 4] =
          d / (1.0f + __expf(-d));
    }
  }
  __syncthreads();
  WaitAsync();
  // The up projection, the gates and the mean.
  for (int index = first_task; index < up_tasks; index += kWarps * a.group_ctas) {
    const Task task = up_task(index);
    float values[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      values[r] = first_values[r];
      if (r < rows && index != first_task) values[r] = normed_value(task, r);
    }
    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.0f;
    Dot<kInt4, kRows>(a.up, a.low_rank, task,
                      index == first_task && hold_up ? &staged_up : nullptr,
                      up_values, acc);
    const int h = index * channels + channel;
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      float dot = acc[r];
      for (int offset = 1; offset < a.lanes; offset *= 2) {
        dot += __shfl_xor_sync(kFullWarp, dot, offset);
      }
      float gated = values[r] / (1.0f + __expf(-dot));
      for (int offset = a.lanes; offset < a.lanes * a.streams; offset *= 2) {
        gated += __shfl_xor_sync(kFullWarp, gated, offset);
      }
      if (position == 0 && up_stream == 0 && r < rows) {
        a.mixed[static_cast<size_t>(row0 + r) * a.hidden + h] =
            __float2bfloat16_rn(gated * mean);
      }
    }
  }
  // The last CTA leaves the state zero for the next launch: every CTA is
  // past the barrier.
  if (a.phase == 0) {
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(a.state + 1, 1u) == gridDim.x - 1) {
      a.state[0] = 0;
      a.state[1] = 0;
    }
  }
}

int FloorPowerOfTwo(int value) {
  int power = 1;
  while (power * 2 <= value) power *= 2;
  return power;
}

// The dynamic shared memory of a CTA at most (every device of compute
// capability 8.0 and later offers it).
constexpr size_t kMaxSharedBytes = 96 * 1024;

// Fills the launch's plan from its shapes (steps of 32 INT4 values or 8
// BF16 values) and returns the dynamic shared memory of a CTA of `rows`
// rows.
size_t Plan(Launch& a, bool int4, int rows) {
  const int values = int4 ? 32 : kChunk;
  a.stream_chunks = a.hidden / kChunk;
  a.stream_steps = a.hidden / values;
  a.up_steps = a.low_rank / values;
  const int lanes_most = 32 / a.streams < a.up_steps / 2
                             ? 32 / a.streams
                             : (a.up_steps / 2 < 1 ? 1 : a.up_steps / 2);
  a.lanes = FloorPowerOfTwo(lanes_most);
  a.lanes_shift = 0;
  while ((1 << a.lanes_shift) < a.lanes) ++a.lanes_shift;
  // Whole groups of steps (32 for the down projection, `lanes` for the up).
  a.xn_floats = (a.stream_steps + 31) / 32 * 32 * values;
  a.t_floats = (a.up_steps + a.lanes - 1) / a.lanes * a.lanes * values;
  int pieces = (a.hidden + kPieceValues / 2) / kPieceValues;
  if (pieces > kMaxSplits / a.streams) pieces = kMaxSplits / a.streams;
  if (pieces < 1) pieces = 1;
  a.piece = (a.stream_steps + pieces - 1) / pieces;
  a.stream_pieces = (a.stream_steps + a.piece - 1) / a.piece;
  a.splits = a.streams * a.stream_pieces;
  a.stream_visits = a.low_rank * a.stream_pieces / kWarps;
  return static_cast<size_t>(rows) * (a.xn_floats + a.t_floats) *
             sizeof(float) +
         2 * kHeld * kThreads * (sizeof(uint4) + sizeof(uint32_t));
}

Launch Shapes(const LiteRtNvidiaHcMixParams& p) {
  Launch a = {};
  a.rows = p.rows;
  a.streams = p.streams;
  a.hidden = p.hidden;
  a.low_rank = p.low_rank;
  a.width = p.streams * p.hidden;
  a.down_scale = p.down_scale;
  a.eps = p.eps;
  return a;
}

// The rows a CTA holds: two of a launch of several, when they fit.
int GroupRows(const LiteRtNvidiaHcMixParams& p) {
  Launch a = Shapes(p);
  return p.rows > 1 && Plan(a, p.int4 != 0, 2) <= kMaxSharedBytes ? 2 : 1;
}

// The device's multiprocessors and whether it launches cooperative grids,
// asked once.
struct Device {
  int multiprocessors = 0;
  bool cooperative = false;
};

const Device& CurrentDevice() {
  static const Device device = [] {
    Device d;
    int id = 0, cooperative = 0;
    if (cudaGetDevice(&id) != cudaSuccess ||
        cudaDeviceGetAttribute(&d.multiprocessors,
                               cudaDevAttrMultiProcessorCount,
                               id) != cudaSuccess ||
        cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch,
                               id) != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      return Device{};
    }
    d.cooperative = cooperative != 0;
    return d;
  }();
  return device;
}

// The CTAs of the kernel that a multiprocessor holds at once with
// `shared_bytes` of dynamic shared memory (0: it cannot run).
template <bool kInt4, int kRows>
int ResidentCtas(size_t shared_bytes) {
  // The kernel's limit is raised to the most any launch asked for.
  static std::atomic<size_t> allowed{48 * 1024};
  if (shared_bytes > allowed.load(std::memory_order_relaxed)) {
    if (cudaFuncSetAttribute(HcMixKernel<kInt4, kRows>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shared_bytes)) != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      return 0;
    }
    allowed.store(shared_bytes, std::memory_order_relaxed);
  }
  int resident = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &resident, HcMixKernel<kInt4, kRows>, kThreads, shared_bytes) !=
      cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    return 0;
  }
  return resident;
}

template <bool kInt4, int kRows>
cudaError_t LaunchMix(Launch a, bool one_launch, cudaStream_t stream) {
  const Device& device = CurrentDevice();
  const size_t shared_bytes = Plan(a, kInt4, kRows);
  const int resident = ResidentCtas<kInt4, kRows>(shared_bytes);
  if (device.multiprocessors < 1 || resident < 1) {
    return cudaErrorInvalidConfiguration;
  }
  // As many CTAs a group as its larger projection has tasks for its warps,
  // at most two a multiprocessor, the whole grid resident.
  const int groups = (a.rows + kRows - 1) / kRows;
  const int down_tasks = a.low_rank * a.splits;
  const int up_tasks = a.hidden / (32 / (a.lanes * a.streams));
  const int tasks = down_tasks > up_tasks ? down_tasks : up_tasks;
  const int most =
      device.multiprocessors * (resident < 2 ? resident : 2) / groups;
  int group_ctas = (tasks + kWarps - 1) / kWarps;
  if (group_ctas > most) group_ctas = most;
  if (group_ctas < 1) {
    // More groups than the device holds CTAs: no barrier.
    group_ctas = 1;
    one_launch = false;
  }
  a.group_ctas = group_ctas;
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(groups * group_ctas);
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = shared_bytes;
  config.stream = stream;
  if (one_launch && device.cooperative && a.state != nullptr) {
    cudaLaunchAttribute cooperative = {};
    cooperative.id = cudaLaunchAttributeCooperative;
    cooperative.val.cooperative = 1;
    config.attrs = &cooperative;
    config.numAttrs = 1;
    a.phase = 0;
    return cudaLaunchKernelEx(&config, HcMixKernel<kInt4, kRows>, a);
  }
  a.phase = 1;
  const cudaError_t status =
      cudaLaunchKernelEx(&config, HcMixKernel<kInt4, kRows>, a);
  if (status != cudaSuccess) return status;
  a.phase = 2;
  return cudaLaunchKernelEx(&config, HcMixKernel<kInt4, kRows>, a);
}

bool Aligned16(const void* pointer) {
  return reinterpret_cast<uintptr_t>(pointer) % 16 == 0;
}

}  // namespace

extern "C" bool LiteRtNvidiaHcMixSupports(const LiteRtNvidiaHcMixParams* p) {
  static_assert(kMaxStreams == 8, "the stream counts below");
  if (p == nullptr || p->rows < 1 || p->rows > kMaxRows || p->hidden < kChunk ||
      p->hidden % kChunk != 0 ||
      p->hidden > kThreads * kMaxStreamChunks * kChunk ||
      // The warps of a CTA take the same piece of as many outputs.
      p->low_rank < kWarps || p->low_rank % kWarps != 0 ||
      p->low_rank > p->hidden || !(p->eps >= 0.0f) ||
      (p->streams != 1 && p->streams != 2 && p->streams != 4 &&
       p->streams != 8) ||
      // INT4 blocks of 32 lie within a stream and within the low rank.
      (p->int4 != 0 && (p->low_rank % 32 != 0 || p->hidden % 32 != 0))) {
    return false;
  }
  Launch a = Shapes(*p);
  return Plan(a, p->int4 != 0, 1) <= kMaxSharedBytes;
}

extern "C" size_t LiteRtNvidiaHcMixWorkspaceBytes(
    const LiteRtNvidiaHcMixParams* p) {
  if (!LiteRtNvidiaHcMixSupports(p)) return 0;
  return static_cast<size_t>(p->rows) * p->low_rank * kMaxSplits *
         sizeof(float);
}

extern "C" size_t LiteRtNvidiaHcMixStateBytes() { return 2 * sizeof(unsigned); }

extern "C" cudaError_t LiteRtNvidiaLaunchHcMix(
    const LiteRtNvidiaHcMixParams* p, const __nv_bfloat16* x,
    const float* norm_weight, const void* w_down, const __half* down_scales,
    const void* w_up, const __half* up_scales, __nv_bfloat16* mixed,
    __nv_bfloat16* normed, void* state, void* workspace, cudaStream_t stream) {
  const bool int4 = p != nullptr && p->int4 != 0;
  if (!LiteRtNvidiaHcMixSupports(p) || x == nullptr ||
      norm_weight == nullptr || w_down == nullptr || w_up == nullptr ||
      (int4 && (down_scales == nullptr || up_scales == nullptr)) ||
      mixed == nullptr || normed == nullptr || workspace == nullptr ||
      // The kernel reads and writes 16 bytes at a time.
      !Aligned16(x) || !Aligned16(norm_weight) || !Aligned16(w_down) ||
      !Aligned16(w_up) || !Aligned16(normed) || !Aligned16(workspace) ||
      (int4 && (!Aligned16(down_scales) || !Aligned16(up_scales)))) {
    return cudaErrorInvalidValue;
  }
  Launch a = Shapes(*p);
  a.x = x;
  a.norm_weight = norm_weight;
  a.down = Weights{w_down, down_scales};
  a.up = Weights{w_up, up_scales};
  a.mixed = mixed;
  a.normed = normed;
  a.partial = static_cast<float*>(workspace);
  a.state = static_cast<unsigned*>(state);
  const bool one_launch = state != nullptr;
  if (GroupRows(*p) == 2) {
    return int4 ? LaunchMix<true, 2>(a, one_launch, stream)
                : LaunchMix<false, 2>(a, one_launch, stream);
  }
  return int4 ? LaunchMix<true, 1>(a, one_launch, stream)
              : LaunchMix<false, 1>(a, one_launch, stream);
}

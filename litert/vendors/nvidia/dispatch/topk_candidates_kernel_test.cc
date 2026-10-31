// Copyright 2026 The ODML Authors. Licensed under the Apache License,
// Version 2.0.

#include "litert/vendors/nvidia/dispatch/topk_candidates_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "cuda_fp16.h"
#include <gtest/gtest.h>
#include "cuda_runtime_api.h"

namespace litert::nvidia {
namespace {

struct Candidate {
  int32_t id;
  float value;
  bool operator==(const Candidate& other) const {
    return id == other.id && value == other.value;
  }
};

// The k largest by logit then index: what the CPU sampler's selection
// gives whenever its k-th value is unique, and what the kernel gives always.
std::vector<Candidate> OrderedTopK(const std::vector<float>& values, int k) {
  std::vector<int32_t> order(values.size());
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + k, order.end(),
                    [&](int32_t a, int32_t b) {
                      if (values[a] != values[b]) return values[a] > values[b];
                      return a < b;
                    });
  std::vector<Candidate> out;
  for (int i = 0; i < k; ++i) out.push_back({order[i], values[order[i]]});
  return out;
}

// LiteRT-LM's TopKTokenIds for k <= 1024 (runtime/components/
// sampling_cpu_util.cc): a min-heap of the first k, a strictly greater value
// evicts the minimum, the survivors sorted by logit then index.
std::vector<Candidate> HeapTopK(const std::vector<float>& values, int k) {
  struct LogitIndex {
    float logit;
    int index;
  };
  auto min_heap_comp = [](const LogitIndex& l, const LogitIndex& r) {
    return l.logit > r.logit;
  };
  std::vector<LogitIndex> heap;
  for (int i = 0; i < k; ++i) heap.push_back({values[i], i});
  std::make_heap(heap.begin(), heap.end(), min_heap_comp);
  for (int i = k; i < static_cast<int>(values.size()); ++i) {
    if (values[i] > heap.front().logit) {
      std::pop_heap(heap.begin(), heap.end(), min_heap_comp);
      heap.back() = {values[i], i};
      std::push_heap(heap.begin(), heap.end(), min_heap_comp);
    }
  }
  std::sort(heap.begin(), heap.end(),
            [](const LogitIndex& a, const LogitIndex& b) {
              if (a.logit != b.logit) return a.logit > b.logit;
              return a.index < b.index;
            });
  std::vector<Candidate> out;
  for (const LogitIndex& entry : heap)
    out.push_back({entry.index, entry.logit});
  return out;
}

// Whether the k-th largest value is unique, so that any correct selection
// gives the same set.
bool CutoffIsUnique(const std::vector<float>& values, int k) {
  if (k >= static_cast<int>(values.size())) return true;
  std::vector<float> sorted(values);
  std::partial_sort(sorted.begin(), sorted.begin() + k + 1, sorted.end(),
                    std::greater<float>());
  return sorted[k - 1] != sorted[k];
}

struct GpuResult {
  std::vector<std::vector<Candidate>> rows;
  std::vector<int32_t> flags;
  cudaError_t status = cudaSuccess;
};

GpuResult RunGpu(const std::vector<float>& values, size_t rows, int k,
                 bool fp16 = false) {
  GpuResult result;
  const size_t count = values.size() / rows;
  std::vector<__half> half_values;
  if (fp16) {
    for (float value : values) half_values.push_back(__float2half_rn(value));
  }
  const size_t input_bytes =
      values.size() * (fp16 ? sizeof(__half) : sizeof(float));
  const void* input = fp16 ? static_cast<const void*>(half_values.data())
                           : static_cast<const void*>(values.data());
  void* device_values = nullptr;
  int32_t* device_ids = nullptr;
  float* device_out = nullptr;
  int32_t* device_flags = nullptr;
  const size_t slots = rows * static_cast<size_t>(std::max(k, 1));
  EXPECT_EQ(cudaMalloc(&device_values, input_bytes), cudaSuccess);
  EXPECT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_ids),
                       slots * sizeof(int32_t)),
            cudaSuccess);
  EXPECT_EQ(
      cudaMalloc(reinterpret_cast<void**>(&device_out), slots * sizeof(float)),
      cudaSuccess);
  EXPECT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_flags),
                       rows * sizeof(int32_t)),
            cudaSuccess);
  EXPECT_EQ(
      cudaMemcpy(device_values, input, input_bytes, cudaMemcpyHostToDevice),
      cudaSuccess);
  result.status =
      fp16 ? LiteRtNvidiaLaunchF16TopKCandidateRows(
                 static_cast<const uint16_t*>(device_values), rows, count, k,
                 device_ids, device_out, device_flags, nullptr)
           : LiteRtNvidiaLaunchF32TopKCandidateRows(
                 static_cast<const float*>(device_values), rows, count, k,
                 device_ids, device_out, device_flags, nullptr);
  if (result.status == cudaSuccess) {
    result.status = cudaDeviceSynchronize();
  }
  if (result.status == cudaSuccess) {
    std::vector<int32_t> ids(slots);
    std::vector<float> out(slots);
    result.flags.resize(rows);
    EXPECT_EQ(cudaMemcpy(ids.data(), device_ids, slots * sizeof(int32_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(cudaMemcpy(out.data(), device_out, slots * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(cudaMemcpy(result.flags.data(), device_flags,
                         rows * sizeof(int32_t), cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (size_t row = 0; row < rows; ++row) {
      std::vector<Candidate> candidates;
      for (int i = 0; i < k; ++i) {
        candidates.push_back({ids[row * k + i], out[row * k + i]});
      }
      result.rows.push_back(std::move(candidates));
    }
  } else {
    cudaGetLastError();
  }
  cudaFree(device_values);
  cudaFree(device_ids);
  cudaFree(device_out);
  cudaFree(device_flags);
  return result;
}

std::vector<float> RandomLogits(size_t n, std::mt19937& gen) {
  std::normal_distribution<float> dist(0.0f, 4.0f);
  std::vector<float> values(n);
  for (float& value : values) value = dist(gen);
  return values;
}

std::vector<float> RowOf(const std::vector<float>& values, size_t rows,
                         size_t row) {
  const size_t count = values.size() / rows;
  return std::vector<float>(values.begin() + row * count,
                            values.begin() + (row + 1) * count);
}

TEST(TopKCandidatesKernelTest, MatchesTheCpuSelectionOnRandomLogits) {
  constexpr size_t kVocab = 262144;
  constexpr size_t kRows = 3;
  std::mt19937 gen(20261027);
  int unique_cutoffs = 0;
  for (int k : {1, 7, 64, 1024}) {
    const std::vector<float> values = RandomLogits(kRows * kVocab, gen);
    const GpuResult gpu = RunGpu(values, kRows, k);
    ASSERT_EQ(gpu.status, cudaSuccess) << "k=" << k;
    for (size_t row = 0; row < kRows; ++row) {
      const std::vector<float> row_values = RowOf(values, kRows, row);
      EXPECT_EQ(gpu.flags[row], 0);
      EXPECT_EQ(gpu.rows[row], OrderedTopK(row_values, k)) << "k=" << k;
      if (CutoffIsUnique(row_values, k)) {
        ++unique_cutoffs;
        EXPECT_EQ(gpu.rows[row], HeapTopK(row_values, k)) << "k=" << k;
      }
    }
  }
  // Float32 draws from a normal distribution tie almost never.
  EXPECT_GE(unique_cutoffs, 10);
}

TEST(TopKCandidatesKernelTest, Fp16LogitsAreWidenedExactlyAndTieByIndex) {
  constexpr size_t kVocab = 262144;
  std::mt19937 gen(27);
  std::vector<float> values = RandomLogits(kVocab, gen);
  // What the kernel reads: the float16 values, widened.
  for (float& value : values) value = __half2float(__float2half_rn(value));
  for (int k : {1, 64, 1024}) {
    const GpuResult gpu = RunGpu(values, 1, k, /*fp16=*/true);
    ASSERT_EQ(gpu.status, cudaSuccess);
    EXPECT_EQ(gpu.flags[0], 0);
    EXPECT_EQ(gpu.rows[0], OrderedTopK(values, k)) << "k=" << k;
    if (CutoffIsUnique(values, k)) {
      EXPECT_EQ(gpu.rows[0], HeapTopK(values, k)) << "k=" << k;
    }
  }
}

TEST(TopKCandidatesKernelTest, TiesAtTheCutoffTakeTheLowestIndices) {
  constexpr size_t kVocab = 70000;
  std::vector<float> values(kVocab, 1.0f);
  // Five values above the cutoff, then a cutoff value shared by many.
  const std::vector<int32_t> maxima = {65000, 30, 4000, 12345, 69999};
  for (size_t i = 0; i < maxima.size(); ++i) {
    values[maxima[i]] = 10.0f - static_cast<float>(i);
  }
  const GpuResult gpu = RunGpu(values, 1, 8);
  ASSERT_EQ(gpu.status, cudaSuccess);
  const std::vector<Candidate> expected = {
      {65000, 10.0f}, {30, 9.0f}, {4000, 8.0f}, {12345, 7.0f},
      {69999, 6.0f},  {0, 1.0f},  {1, 1.0f},    {2, 1.0f}};
  EXPECT_EQ(gpu.rows[0], expected);
  EXPECT_EQ(gpu.rows[0], OrderedTopK(values, 8));
}

TEST(TopKCandidatesKernelTest, AllEqualLogitsGiveTheFirstK) {
  std::vector<float> values(5000, 0.25f);
  const GpuResult gpu = RunGpu(values, 2, 4);
  ASSERT_EQ(gpu.status, cudaSuccess);
  for (size_t row = 0; row < 2; ++row) {
    const std::vector<Candidate> expected = {
        {0, 0.25f}, {1, 0.25f}, {2, 0.25f}, {3, 0.25f}};
    EXPECT_EQ(gpu.rows[row], expected);
  }
}

TEST(TopKCandidatesKernelTest, MaskedRowsKeepNegativeInfinityByIndex) {
  constexpr size_t kVocab = 9000;
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<float> values(kVocab, -inf);
  values[8000] = 2.0f;
  values[100] = 3.0f;
  values[5] = 1.0f;
  const GpuResult gpu = RunGpu(values, 1, 6);
  ASSERT_EQ(gpu.status, cudaSuccess);
  EXPECT_EQ(gpu.flags[0], 0);
  const std::vector<Candidate> expected = {
      {100, 3.0f}, {8000, 2.0f}, {5, 1.0f}, {0, -inf}, {1, -inf}, {2, -inf}};
  EXPECT_EQ(gpu.rows[0], expected);
  EXPECT_EQ(gpu.rows[0], OrderedTopK(values, 6));
}

TEST(TopKCandidatesKernelTest, SignedZerosAreOneValue) {
  std::vector<float> values(3000, -1.0f);
  values[7] = -0.0f;
  values[3] = 0.0f;
  values[11] = -0.0f;
  values[2000] = 0.0f;
  const GpuResult gpu = RunGpu(values, 1, 4);
  ASSERT_EQ(gpu.status, cudaSuccess);
  const std::vector<int32_t> ids = {gpu.rows[0][0].id, gpu.rows[0][1].id,
                                    gpu.rows[0][2].id, gpu.rows[0][3].id};
  EXPECT_EQ(ids, (std::vector<int32_t>{3, 7, 11, 2000}));
  for (const Candidate& candidate : gpu.rows[0]) {
    EXPECT_EQ(candidate.value, 0.0f);
  }
}

TEST(TopKCandidatesKernelTest, NaNFlagsTheRow) {
  std::mt19937 gen(3);
  std::vector<float> values = RandomLogits(2 * 20000, gen);
  values[20000 + 777] = std::numeric_limits<float>::quiet_NaN();
  const GpuResult gpu = RunGpu(values, 2, 16);
  ASSERT_EQ(gpu.status, cudaSuccess);
  EXPECT_EQ(gpu.flags[0], 0);
  EXPECT_EQ(gpu.flags[1], kLiteRtNvidiaTopKCandidatesNaN);
  EXPECT_EQ(gpu.rows[0], OrderedTopK(RowOf(values, 2, 0), 16));
}

TEST(TopKCandidatesKernelTest, KEqualsCountAndKOne) {
  std::mt19937 gen(5);
  const std::vector<float> values = RandomLogits(8, gen);
  const GpuResult all = RunGpu(values, 1, 8);
  ASSERT_EQ(all.status, cudaSuccess);
  EXPECT_EQ(all.rows[0], OrderedTopK(values, 8));
  const GpuResult one = RunGpu(values, 1, 1);
  ASSERT_EQ(one.status, cudaSuccess);
  EXPECT_EQ(one.rows[0], OrderedTopK(values, 1));
  EXPECT_EQ(one.rows[0], HeapTopK(values, 1));
}

TEST(TopKCandidatesKernelTest, RejectsBadArguments) {
  const std::vector<float> values(64, 1.0f);
  EXPECT_EQ(RunGpu(values, 1, 0).status, cudaErrorInvalidValue);
  EXPECT_EQ(RunGpu(values, 1, LiteRtNvidiaTopKCandidatesMaxK() + 1).status,
            cudaErrorInvalidValue);
  EXPECT_EQ(RunGpu(values, 1, 65).status, cudaErrorInvalidValue);
  EXPECT_EQ(RunGpu(values, 1, 64).status, cudaSuccess);
  EXPECT_EQ(LiteRtNvidiaTopKCandidatesMaxK(), 1024);
}

}  // namespace
}  // namespace litert::nvidia

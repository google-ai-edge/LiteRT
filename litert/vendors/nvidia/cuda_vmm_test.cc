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

#include "litert/vendors/nvidia/cuda_vmm.h"

#include <cstdint>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"

namespace litert::nvidia {
namespace {

class CudaVmmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
    auto granule = CudaVmmGranule();
    ASSERT_TRUE(granule.HasValue()) << granule.Error().Message();
    granule_ = *granule;
  }
  void TearDown() override { cudaStreamDestroy(stream_); }

  cudaStream_t stream_ = nullptr;
  uint64_t granule_ = 0;
};

TEST_F(CudaVmmTest, GranuleIsAPowerOfTwo) {
  EXPECT_GT(granule_, 0u);
  EXPECT_EQ(granule_ & (granule_ - 1), 0u);
}

TEST_F(CudaVmmTest, BlocksHoldWhatAStreamCopiesIntoThem) {
  // Blocks are created, filled, read and released in turn: every mapping
  // follows the unmapping of the previous block.
  std::vector<uint8_t> written(granule_);
  std::vector<uint8_t> read(granule_);
  for (int round = 0; round < 64; ++round) {
    auto block = CudaVmmBlock::Create(granule_ * (1 + round % 3));
    ASSERT_TRUE(block.HasValue()) << block.Error().Message();
    EXPECT_NE((*block)->handle(), 0u);
    EXPECT_EQ((*block)->size(), granule_ * (1 + round % 3));
    EXPECT_EQ(reinterpret_cast<uintptr_t>((*block)->address()) % granule_, 0u);
    for (size_t i = 0; i < written.size(); ++i) {
      written[i] = static_cast<uint8_t>(i * 31 + round);
    }
    auto* last = static_cast<uint8_t*>((*block)->address()) + (*block)->size() -
                 granule_;
    ASSERT_EQ(cudaMemcpyAsync(last, written.data(), granule_,
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(read.data(), last, granule_,
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess) << "round " << round;
    ASSERT_EQ(read, written) << "round " << round;
  }
}

TEST_F(CudaVmmTest, AThreadWithoutACudaContextReleasesABlock) {
  size_t free_before = 0;
  size_t total = 0;
  ASSERT_EQ(cudaMemGetInfo(&free_before, &total), cudaSuccess);
  auto block = CudaVmmBlock::Create(64 * granule_);
  ASSERT_TRUE(block.HasValue()) << block.Error().Message();
  size_t free_with_block = 0;
  ASSERT_EQ(cudaMemGetInfo(&free_with_block, &total), cudaSuccess);
  EXPECT_LE(free_with_block + 64 * granule_, free_before);
  std::thread([owned = std::move(*block)]() mutable { owned.reset(); }).join();
  size_t free_after = 0;
  ASSERT_EQ(cudaMemGetInfo(&free_after, &total), cudaSuccess);
  EXPECT_GE(free_after, free_with_block + 64 * granule_);
}

TEST_F(CudaVmmTest, RejectsSizesThatAreNotGranules) {
  EXPECT_FALSE(CudaVmmBlock::Create(0).HasValue());
  EXPECT_FALSE(CudaVmmBlock::Create(granule_ + 4096).HasValue());
}

}  // namespace
}  // namespace litert::nvidia

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

#include "litert/vendors/nvidia/dispatch/runtime_cache.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/c/litert_common.h"
#include "NvInfer.h"

namespace litert::nvidia {
namespace {

constexpr int kWidth = 32;
constexpr int kMaxRows = 4;
constexpr size_t kBytes = kWidth * kMaxRows * sizeof(float);

class TestLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    // Rejected candidates intentionally exercise the SDK's error path.
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "TensorRT-RTX: %s\n", message);
    }
  }
};

class RuntimeCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    char directory_template[] = "/tmp/litert_nvidia_runtime_cache_XXXXXX";
    char* directory = mkdtemp(directory_template);
    ASSERT_NE(directory, nullptr);
    directory_ = directory;
    path_ = directory_ + "/cache.bin";

    std::unique_ptr<nvinfer1::IBuilder> builder(
        nvinfer1::createInferBuilder(logger_));
    ASSERT_NE(builder, nullptr);
    const auto flags =
        1U << static_cast<unsigned>(
            nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(
        builder->createNetworkV2(flags));
    std::unique_ptr<nvinfer1::IBuilderConfig> build_config(
        builder->createBuilderConfig());
    ASSERT_NE(network, nullptr);
    ASSERT_NE(build_config, nullptr);
    auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT,
                                    nvinfer1::Dims2{-1, kWidth});
    ASSERT_NE(input, nullptr);
    auto* sum = network->addElementWise(*input, *input,
                                        nvinfer1::ElementWiseOperation::kSUM);
    ASSERT_NE(sum, nullptr);
    sum->getOutput(0)->setName("output");
    network->markOutput(*sum->getOutput(0));
    // The builder owns optimization profiles; do not delete this pointer.
    auto* profile = builder->createOptimizationProfile();
    ASSERT_NE(profile, nullptr);
    ASSERT_TRUE(profile->setDimensions("input",
                                       nvinfer1::OptProfileSelector::kMIN,
                                       nvinfer1::Dims2{1, kWidth}));
    ASSERT_TRUE(profile->setDimensions("input",
                                       nvinfer1::OptProfileSelector::kOPT,
                                       nvinfer1::Dims2{2, kWidth}));
    ASSERT_TRUE(profile->setDimensions("input",
                                       nvinfer1::OptProfileSelector::kMAX,
                                       nvinfer1::Dims2{kMaxRows, kWidth}));
    ASSERT_GE(build_config->addOptimizationProfile(profile), 0);
    ASSERT_TRUE(build_config->setNbComputeCapabilities(1));
    ASSERT_TRUE(build_config->setComputeCapability(
        nvinfer1::ComputeCapability::kCURRENT, 0));
    std::unique_ptr<nvinfer1::IHostMemory> plan(
        builder->buildSerializedNetwork(*network, *build_config));
    ASSERT_NE(plan, nullptr);
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    ASSERT_NE(runtime_, nullptr);
    engine_.reset(runtime_->deserializeCudaEngine(plan->data(), plan->size()));
    ASSERT_NE(engine_, nullptr);
    config_.reset(engine_->createRuntimeConfig());
    ASSERT_NE(config_, nullptr);
    cache_.reset(config_->createRuntimeCache());
    ASSERT_NE(cache_, nullptr);
    ASSERT_TRUE(config_->setRuntimeCache(*cache_));
    // Test-only: finish specialization before serializing either input shape.
    config_->setDynamicShapesKernelSpecializationStrategy(
        nvinfer1::DynamicShapesKernelSpecializationStrategy::kEAGER);
    context_.reset(engine_->createExecutionContext(config_.get()));
    ASSERT_NE(context_, nullptr);
    ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&input_, kBytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&output_, kBytes), cudaSuccess);
    ASSERT_TRUE(context_->setTensorAddress("input", input_));
    ASSERT_TRUE(context_->setTensorAddress("output", output_));
    RunShape(1);
    ASSERT_FALSE(HasFatalFailure());
    candidate_ = Serialize();
    ASSERT_FALSE(candidate_.empty());
    ASSERT_TRUE(Accepts(candidate_));
  }

  void TearDown() override {
    if (stream_ != nullptr) {
      EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    }
    context_.reset();
    config_.reset();
    cache_.reset();
    engine_.reset();
    runtime_.reset();
    if (input_ != nullptr) EXPECT_EQ(cudaFree(input_), cudaSuccess);
    if (output_ != nullptr) EXPECT_EQ(cudaFree(output_), cudaSuccess);
    if (stream_ != nullptr) EXPECT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
    if (!directory_.empty()) {
      // This is only the private directory returned by mkdtemp in SetUp.
      std::error_code error;
      std::filesystem::remove_all(directory_, error);
      EXPECT_FALSE(error) << error.message();
    }
  }

  void RunShape(int rows) {
    ASSERT_TRUE(
        context_->setInputShape("input", nvinfer1::Dims2{rows, kWidth}));
    std::array<float, kWidth * kMaxRows> input;
    for (size_t i = 0; i < input.size(); ++i) input[i] = i * 0.125f;
    ASSERT_EQ(cudaMemcpyAsync(input_, input.data(), kBytes,
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_TRUE(context_->enqueueV3(stream_));
    std::array<float, kWidth * kMaxRows> output{};
    ASSERT_EQ(
        cudaMemcpyAsync(output.data(), output_, rows * kWidth * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    for (int i = 0; i < rows * kWidth; ++i) {
      EXPECT_FLOAT_EQ(output[i], input[i] * 2.0f);
    }
  }

  std::vector<uint8_t> Serialize() {
    std::unique_ptr<nvinfer1::IHostMemory> bytes(cache_->serialize());
    if (!bytes || bytes->data() == nullptr || bytes->size() == 0) return {};
    const auto* begin = static_cast<const uint8_t*>(bytes->data());
    return {begin, begin + bytes->size()};
  }

  bool Accepts(const std::vector<uint8_t>& bytes) {
    std::unique_ptr<nvinfer1::IRuntimeCache> probe(
        config_->createRuntimeCache());
    return probe && probe->deserialize(bytes.data(), bytes.size());
  }

  std::vector<uint8_t> Read(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
  }

  void Persist(const std::vector<uint8_t>& bytes) {
    auto saved = PersistValidatedRuntimeCache(*config_, bytes.data(),
                                              bytes.size(), path_);
    ASSERT_TRUE(saved) << saved.Error().Message();
  }

  void ExpectNoTemporaryFiles() {
    for (const auto& entry : std::filesystem::directory_iterator(directory_)) {
      EXPECT_EQ(entry.path().filename().string().find(".tmp."),
                std::string::npos)
          << entry.path();
    }
  }

  TestLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IRuntimeCache> cache_;
  std::unique_ptr<nvinfer1::IRuntimeConfig> config_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_ = nullptr;
  void* input_ = nullptr;
  void* output_ = nullptr;
  std::string directory_;
  std::string path_;
  std::vector<uint8_t> candidate_;
};

TEST_F(RuntimeCacheTest, PersistsOriginalValidCandidate) {
  Persist(candidate_);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(Read(path_), candidate_);
  EXPECT_TRUE(Accepts(Read(path_)));
  ExpectNoTemporaryFiles();
  // A validation probe must not replace or disturb the active execution cache.
  RunShape(1);
}

TEST_F(RuntimeCacheTest, RejectedCandidatesPreservePreviousBytesAndInode) {
  Persist(candidate_);
  ASSERT_FALSE(HasFatalFailure());
  struct stat before{};
  ASSERT_EQ(stat(path_.c_str(), &before), 0);
  auto corrupt = candidate_;
  std::fill(corrupt.begin(), corrupt.end(), 0);
  auto truncated = candidate_;
  truncated.resize(1);
  for (const auto& invalid : {corrupt, truncated}) {
    auto saved = PersistValidatedRuntimeCache(*config_, invalid.data(),
                                              invalid.size(), path_);
    ASSERT_FALSE(saved);
    EXPECT_EQ(static_cast<LiteRtStatus>(saved.Error().StatusValue()),
              kLiteRtStatusErrorInvalidArgument);
    EXPECT_EQ(Read(path_), candidate_);
    struct stat after{};
    ASSERT_EQ(stat(path_.c_str(), &after), 0);
    EXPECT_EQ(after.st_ino, before.st_ino);
    ExpectNoTemporaryFiles();
  }
  EXPECT_TRUE(Accepts(Read(path_)));
  RunShape(1);
}

TEST_F(RuntimeCacheTest, RejectsEmptyAndInvalidCandidatesWithoutCreatingFile) {
  const std::array<uint8_t, 4> invalid{};
  EXPECT_FALSE(PersistValidatedRuntimeCache(*config_, nullptr, 0, path_));
  EXPECT_FALSE(PersistValidatedRuntimeCache(*config_, invalid.data(),
                                            invalid.size(), path_));
  EXPECT_FALSE(std::filesystem::exists(path_));
  ExpectNoTemporaryFiles();
}

TEST_F(RuntimeCacheTest, PersistsValidReplacementAfterAnotherInputShape) {
  Persist(candidate_);
  ASSERT_FALSE(HasFatalFailure());
  struct stat before{};
  ASSERT_EQ(stat(path_.c_str(), &before), 0);
  RunShape(kMaxRows);
  ASSERT_FALSE(HasFatalFailure());
  const auto next = Serialize();
  ASSERT_FALSE(next.empty());
  ASSERT_TRUE(Accepts(next));
  Persist(next);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(Read(path_), next);
  EXPECT_TRUE(Accepts(Read(path_)));
  struct stat after{};
  ASSERT_EQ(stat(path_.c_str(), &after), 0);
  EXPECT_NE(after.st_ino, before.st_ino);
  // SDK serialization may canonicalize or reuse generic kernels. Neither byte
  // inequality nor growth is required for a valid replacement to be written.
  ExpectNoTemporaryFiles();
  RunShape(1);
}

TEST_F(RuntimeCacheTest, TemporaryFileCreationFailureLeavesExistingFileAlone) {
  Persist(candidate_);
  ASSERT_FALSE(HasFatalFailure());
  // ENOTDIR works even when tests run as root, unlike permission-bit tests.
  auto saved = PersistValidatedRuntimeCache(
      *config_, candidate_.data(), candidate_.size(), path_ + "/child");
  ASSERT_FALSE(saved);
  EXPECT_EQ(static_cast<LiteRtStatus>(saved.Error().StatusValue()),
            kLiteRtStatusErrorFileIO);
  EXPECT_EQ(Read(path_), candidate_);
  ExpectNoTemporaryFiles();
}

TEST_F(RuntimeCacheTest, RenameFailurePreservesDestinationAndRemovesTemporary) {
  const std::string destination = directory_ + "/existing_directory";
  ASSERT_TRUE(std::filesystem::create_directory(destination));
  const std::string previous_path = destination + "/cache.bin";
  auto initial = PersistValidatedRuntimeCache(*config_, candidate_.data(),
                                              candidate_.size(), previous_path);
  ASSERT_TRUE(initial) << initial.Error().Message();
  // The temporary file can be fully written, but a file cannot replace this
  // directory. This exercises post-write failure and cleanup without chmod.
  auto saved = PersistValidatedRuntimeCache(*config_, candidate_.data(),
                                            candidate_.size(), destination);
  ASSERT_FALSE(saved);
  EXPECT_EQ(static_cast<LiteRtStatus>(saved.Error().StatusValue()),
            kLiteRtStatusErrorFileIO);
  EXPECT_EQ(Read(previous_path), candidate_);
  ExpectNoTemporaryFiles();
  Persist(candidate_);
}

}  // namespace
}  // namespace litert::nvidia

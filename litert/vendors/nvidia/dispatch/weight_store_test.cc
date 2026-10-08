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

#include "litert/vendors/nvidia/dispatch/weight_store.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/nvidia/bytecode.h"
#include "litert/vendors/nvidia/compiler/weight_holder_calibration.h"
#include "litert/vendors/nvidia/cuda_vmm.h"
#include "tflite/schema/schema_generated.h"

namespace litert::nvidia {
namespace {

class ScopedEnvironment {
 public:
  ScopedEnvironment(const char* name, const char* value) : name_(name) {
    if (const char* previous = std::getenv(name)) {
      previous_ = previous;
    }
    setenv(name, value, 1);
  }
  ~ScopedEnvironment() {
    if (previous_) {
      setenv(name_.c_str(), previous_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::optional<std::string> previous_;
};

TensorRtAotFileIdentity Identity(const std::string& path) {
  struct stat status{};
  EXPECT_EQ(stat(path.c_str(), &status), 0);
  return {static_cast<uint64_t>(status.st_dev),
          static_cast<uint64_t>(status.st_ino),
          status.st_mtim.tv_sec,
          status.st_mtim.tv_nsec,
          status.st_ctim.tv_sec,
          status.st_ctim.tv_nsec};
}

// Moves the modification time of the file, as rewriting it would.
void Touch(const std::string& path) {
  const struct timespec times[2] = {{0, UTIME_OMIT}, {1000000000, 0}};
  ASSERT_EQ(utimensat(AT_FDCWD, path.c_str(), times, 0), 0);
}

// The device memory of weight stores that describe pieces of a file, without
// an engine.
class EngineWeightMemoryTest : public ::testing::Test {
 protected:
  static constexpr uint64_t kFileBytes = 80 << 20;
  // The segment the stores of the tests have in common, large enough for its
  // device memory to stand out from what else happens on the device.
  static constexpr uint64_t kCommonBytes = 64 << 20;

  void SetUp() override {
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
    auto granule = CudaVmmGranule();
    ASSERT_TRUE(granule.HasValue()) << granule.Error().Message();
    granule_ = *granule;
    ASSERT_EQ(kCommonBytes % granule_, 0);
    file_.resize(kFileBytes);
    uint32_t state = 12345;
    for (auto& byte : file_) {
      state = state * 1664525u + 1013904223u;
      byte = static_cast<uint8_t>(state >> 24);
    }
    path_ = ::testing::TempDir() + "/weight_store_source.XXXXXX";
    const int fd = mkstemp(path_.data());
    ASSERT_GE(fd, 0);
    ASSERT_EQ(write(fd, file_.data(), file_.size()),
              static_cast<ssize_t>(file_.size()));
    close(fd);
    runs_ = {std::vector<uint8_t>(8192, 0x11), std::vector<uint8_t>(4096, 0x22),
             std::vector<uint8_t>(4096, 0x33)};
  }

  void TearDown() override {
    cudaStreamDestroy(stream_);
    unlink(path_.c_str());
  }

  // Weight data of the common segment and a segment of two granules between
  // three private granules:
  //   [0, G) private, [G, G + common) common, one private granule,
  //   two granules of the second segment, one private granule.
  TensorRtWeightStore Store() const {
    TensorRtWeightStore store;
    store.source_path = path_;
    store.source_size = file_.size();
    store.source_identity = Identity(path_);
    store.granule = granule_;
    store.weight_data_size = 5 * granule_ + kCommonBytes;
    TensorRtWeightSegment common;
    common.holder_name = "common";
    common.size = kCommonBytes;
    common.payload_offset = granule_;
    common.key = {0x1111, 0x2222};
    common.pieces = {{5000, kCommonBytes - 4096, 0},
                     {70000001, 2345, kCommonBytes - 4096 + 128}};
    TensorRtWeightSegment second;
    second.holder_name = "second";
    second.size = 2 * granule_;
    second.payload_offset = 2 * granule_ + kCommonBytes;
    second.key = {0x3333, 0x4444};
    second.pieces = {{72 << 20, 3 << 20, 128}};
    store.segments = {common, second};
    store.private_runs = {
        {4096, runs_[0].data(), runs_[0].size()},
        {granule_ + kCommonBytes + 8192, runs_[1].data(), runs_[1].size()},
        {4 * granule_ + kCommonBytes, runs_[2].data(), runs_[2].size()}};
    return store;
  }

  std::vector<uint8_t> Read(const EngineWeightMemory::Range& range) {
    std::vector<uint8_t> bytes(range.size);
    EXPECT_EQ(cudaMemcpyAsync(bytes.data(), range.block->address(), range.size,
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    return bytes;
  }

  bool HoldsFile(const std::vector<uint8_t>& segment,
                 const TensorRtWeightPiece& piece) const {
    return std::equal(segment.begin() + piece.segment_offset,
                      segment.begin() + piece.segment_offset + piece.size,
                      file_.begin() + piece.source_offset);
  }

  size_t FreeDeviceBytes() {
    size_t free_bytes = 0;
    size_t total = 0;
    EXPECT_EQ(cudaMemGetInfo(&free_bytes, &total), cudaSuccess);
    return free_bytes;
  }

  cudaStream_t stream_ = nullptr;
  uint64_t granule_ = 0;
  std::vector<uint8_t> file_;
  std::string path_;
  std::vector<std::vector<uint8_t>> runs_;
};

TEST_F(EngineWeightMemoryTest, FillsSegmentsFromTheFileAndTheRestFromRuns) {
  const TensorRtWeightStore store = Store();
  auto memory = EngineWeightMemory::Create(store, stream_);
  ASSERT_TRUE(memory.HasValue()) << memory.Error().Message();
  EXPECT_EQ((*memory)->weight_data_size(), store.weight_data_size);
  EXPECT_EQ((*memory)->private_bytes(), 3 * granule_);
  EXPECT_EQ((*memory)->segment_bytes(), kCommonBytes + 2 * granule_);
  EXPECT_EQ((*memory)->shared_segment_bytes(), 0);
  const auto& ranges = (*memory)->ranges();
  ASSERT_EQ(ranges.size(), 5);
  uint64_t position = 0;
  for (const auto& range : ranges) {
    EXPECT_EQ(range.offset, position);
    EXPECT_EQ(range.block->size(), range.size);
    position += range.size;
  }
  EXPECT_EQ(position, store.weight_data_size);

  std::vector<uint8_t> expected(granule_, 0);
  std::fill_n(expected.begin() + 4096, 8192, 0x11);
  EXPECT_EQ(Read(ranges[0]), expected);
  const std::vector<uint8_t> common = Read(ranges[1]);
  EXPECT_TRUE(HoldsFile(common, store.segments[0].pieces[0]));
  EXPECT_TRUE(HoldsFile(common, store.segments[0].pieces[1]));
  expected.assign(granule_, 0);
  std::fill_n(expected.begin() + 8192, 4096, 0x22);
  EXPECT_EQ(Read(ranges[2]), expected);
  EXPECT_TRUE(HoldsFile(Read(ranges[3]), store.segments[1].pieces[0]));
  expected.assign(granule_, 0);
  std::fill_n(expected.begin(), 4096, 0x33);
  EXPECT_EQ(Read(ranges[4]), expected);
}

TEST_F(EngineWeightMemoryTest, EnginesShareTheSegmentsWithEqualKeys) {
  const TensorRtWeightStore store = Store();
  auto first = EngineWeightMemory::Create(store, stream_);
  ASSERT_TRUE(first.HasValue()) << first.Error().Message();
  // Another engine has the common segment elsewhere in its weight data and
  // nothing else.
  TensorRtWeightStore other = Store();
  other.weight_data_size = 2 * granule_ + kCommonBytes;
  other.segments.pop_back();
  other.segments[0].payload_offset = 2 * granule_;
  other.private_runs.resize(1);
  const size_t free_before = FreeDeviceBytes();
  auto second = EngineWeightMemory::Create(other, stream_);
  ASSERT_TRUE(second.HasValue()) << second.Error().Message();
  const size_t free_after = FreeDeviceBytes();
  EXPECT_EQ((*second)->shared_segment_bytes(), kCommonBytes);
  EXPECT_EQ((*second)->private_bytes(), 2 * granule_);
  ASSERT_EQ((*second)->ranges().size(), 2);
  EXPECT_EQ((*second)->ranges()[1].block, (*first)->ranges()[1].block);
  // The second engine costs the device its private granules only.
  EXPECT_LT(
      static_cast<int64_t>(free_before) - static_cast<int64_t>(free_after),
      static_cast<int64_t>(kCommonBytes / 2));

  // The segment lives as long as an engine holds it.
  first->reset();
  const std::vector<uint8_t> common = Read((*second)->ranges()[1]);
  EXPECT_TRUE(HoldsFile(common, other.segments[0].pieces[0]));
  EXPECT_TRUE(HoldsFile(common, other.segments[0].pieces[1]));
  second->reset();
  auto third = EngineWeightMemory::Create(other, stream_);
  ASSERT_TRUE(third.HasValue()) << third.Error().Message();
  EXPECT_EQ((*third)->shared_segment_bytes(), 0);
  EXPECT_TRUE(
      HoldsFile(Read((*third)->ranges()[1]), other.segments[0].pieces[0]));
}

TEST_F(EngineWeightMemoryTest, EveryEngineGetsItsOwnCopyWhenSharingIsOff) {
  ScopedEnvironment off("LITERT_NVIDIA_DISPATCH_SHARED_WEIGHT_SEGMENTS", "0");
  const TensorRtWeightStore store = Store();
  auto first = EngineWeightMemory::Create(store, stream_);
  ASSERT_TRUE(first.HasValue()) << first.Error().Message();
  const size_t free_before = FreeDeviceBytes();
  auto second = EngineWeightMemory::Create(store, stream_);
  ASSERT_TRUE(second.HasValue()) << second.Error().Message();
  const size_t free_after = FreeDeviceBytes();
  EXPECT_EQ((*second)->shared_segment_bytes(), 0);
  EXPECT_NE((*second)->ranges()[1].block, (*first)->ranges()[1].block);
  EXPECT_GT(
      static_cast<int64_t>(free_before) - static_cast<int64_t>(free_after),
      static_cast<int64_t>(kCommonBytes / 2));
  EXPECT_TRUE(
      HoldsFile(Read((*second)->ranges()[1]), store.segments[0].pieces[0]));
}

TEST_F(EngineWeightMemoryTest, RejectsAModelFileThatChanged) {
  const TensorRtWeightStore store = Store();
  Touch(path_);
  auto memory = EngineWeightMemory::Create(store, stream_);
  ASSERT_FALSE(memory.HasValue());
  EXPECT_EQ(memory.Error().Status(), kLiteRtStatusErrorInvalidArgument);
  // The store of plans built from the file as it is now loads.
  memory = EngineWeightMemory::Create(Store(), stream_);
  EXPECT_TRUE(memory.HasValue());
}

TEST_F(EngineWeightMemoryTest, RejectsStoresThatDoNotAddUp) {
  TensorRtWeightStore store = Store();
  store.granule = granule_ / 2;
  EXPECT_FALSE(EngineWeightMemory::Create(store, stream_).HasValue());
  store = Store();
  store.segments[1].payload_offset = granule_ + kCommonBytes - granule_;
  EXPECT_FALSE(EngineWeightMemory::Create(store, stream_).HasValue());
  store = Store();
  store.weight_data_size -= 2 * granule_;
  EXPECT_FALSE(EngineWeightMemory::Create(store, stream_).HasValue());
  store = Store();
  store.private_runs[0].offset = granule_ - 4096;
  EXPECT_FALSE(EngineWeightMemory::Create(store, stream_).HasValue());
  store = Store();
  store.segments[1].pieces[0].size = 2 * granule_;
  EXPECT_FALSE(EngineWeightMemory::Create(store, stream_).HasValue());
  store = Store();
  store.segments[1].pieces[0].source_offset = kFileBytes - 4096;
  EXPECT_FALSE(EngineWeightMemory::Create(store, stream_).HasValue());
  // Nothing of a failed store stays behind for the next one.
  auto memory = EngineWeightMemory::Create(Store(), stream_);
  ASSERT_TRUE(memory.HasValue()) << memory.Error().Message();
  EXPECT_EQ((*memory)->shared_segment_bytes(), 0);
}

// Packs signed INT4 values two per byte, low nibble first (TFLite layout).
std::vector<uint8_t> PackInt4(const std::vector<int8_t>& values) {
  std::vector<uint8_t> packed((values.size() + 1) / 2);
  for (size_t i = 0; i < values.size(); ++i) {
    const uint8_t nibble = static_cast<uint8_t>(values[i]) & 0x0F;
    packed[i / 2] |= i % 2 == 0 ? nibble : nibble << 4;
  }
  return packed;
}

constexpr auto kCudaBufferType = static_cast<LiteRtTensorBufferType>(
    kLiteRtTensorBufferTypeUserCustomBuffer + 1);

// A float tensor on the device. Only its metadata is mocked: TensorRT, CUDA
// and the dispatch library are real.
struct TestBuffer {
  explicit TestBuffer(const std::vector<int32_t>& dims) {
    type.element_type = kLiteRtElementTypeFloat32;
    type.layout.rank = dims.size();
    for (size_t i = 0; i < dims.size(); ++i) {
      type.layout.dimensions[i] = dims[i];
      elements *= dims[i];
    }
    EXPECT_EQ(cudaMalloc(&device_ptr, bytes()), cudaSuccess);
  }
  ~TestBuffer() { cudaFree(device_ptr); }
  size_t bytes() const { return elements * sizeof(float); }
  LiteRtTensorBuffer handle() {
    return reinterpret_cast<LiteRtTensorBuffer>(this);
  }

  size_t elements = 1;
  void* device_ptr = nullptr;
  LiteRtRankedTensorType type{};
  LiteRtTensorBufferHandle registered = 0;
};

LiteRtRuntimeContext RuntimeContext() {
  LiteRtRuntimeContext context{};
  context.create_tensor_buffer_requirements =
      LiteRtCreateTensorBufferRequirements;
  context.get_tensor_buffer_type = [](LiteRtTensorBuffer,
                                      LiteRtTensorBufferType* type) {
    *type = kCudaBufferType;
    return kLiteRtStatusOk;
  };
  context.get_tensor_buffer_tensor_type = [](LiteRtTensorBuffer buffer,
                                             LiteRtRankedTensorType* type) {
    *type = reinterpret_cast<TestBuffer*>(buffer)->type;
    return kLiteRtStatusOk;
  };
  context.get_tensor_buffer_size = [](LiteRtTensorBuffer buffer, size_t* size) {
    *size = reinterpret_cast<TestBuffer*>(buffer)->bytes();
    return kLiteRtStatusOk;
  };
  context.get_tensor_buffer_packed_size = context.get_tensor_buffer_size;
  context.get_tensor_buffer_offset = [](LiteRtTensorBuffer, size_t* offset) {
    *offset = 0;
    return kLiteRtStatusOk;
  };
  context.get_tensor_buffer_custom_tensor_buffer_handle =
      [](LiteRtTensorBuffer buffer, HwMemoryHandle* handle) {
        *handle = reinterpret_cast<TestBuffer*>(buffer)->device_ptr;
        return kLiteRtStatusOk;
      };
  return context;
}

// A model file with INT4 projection weights, compiled by the compiler plugin
// into plans without those weights and run through the dispatch library.
// As in a language model, partition 0 projects many activation rows (the
// tensor-core GEMM of a prefill) and partition 1 one row (the GEMV of a
// decode step) by the same weights; partition 1 has a second projection.
class WeightStoreDispatchTest : public ::testing::Test {
 protected:
  static constexpr int32_t kColumns = 256;
  static constexpr int32_t kPrefillRows = 1024;
  static constexpr int32_t kSecondChannels = 1024;
  static constexpr uint64_t kFirstOffset = 8192;
  static constexpr uint64_t kSecondOffset = 1 << 20;
  static constexpr size_t kFileBytes = 2 << 20;

  void SetUp() override {
    if (!TensorRtWeightHoldersSupported()) {
      GTEST_SKIP() << "Needs the weights manager of TensorRT-RTX 1.7.";
    }
    // Enough output channels for the GEMM to take the prefill projection: it
    // wants three blocks for every two multiprocessors, and has a block for
    // every 128 rows and 128 channels.
    int device = 0;
    int multiprocessors = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    ASSERT_EQ(cudaDeviceGetAttribute(&multiprocessors,
                                     cudaDevAttrMultiProcessorCount, device),
              cudaSuccess);
    first_channels_ = (multiprocessors * 3 + 15) / 16 * 128;
    ASSERT_LE(kFirstOffset + first_channels_ * kColumns / 2, kSecondOffset);

    const char* root = std::getenv("TEST_TMPDIR");
    ASSERT_NE(root, nullptr);
    std::string pattern = std::string(root) + "/nvidia-weight-store-XXXXXX";
    ASSERT_NE(mkdtemp(pattern.data()), nullptr);
    directory_ = pattern;

    first_ = Values(first_channels_, 5);
    second_ = Values(kSecondChannels, 11);
    std::vector<uint8_t> file(kFileBytes, 0x5a);
    const std::vector<uint8_t> first_packed = PackInt4(first_);
    const std::vector<uint8_t> second_packed = PackInt4(second_);
    std::memcpy(file.data() + kFirstOffset, first_packed.data(),
                first_packed.size());
    std::memcpy(file.data() + kSecondOffset, second_packed.data(),
                second_packed.size());
    source_ = (directory_ / "model.bin").string();
    std::ofstream(source_, std::ios::binary)
        .write(reinterpret_cast<const char*>(file.data()), file.size());
    // The weights of a model are views of its memory-mapped file.
    const int fd = open(source_.c_str(), O_RDONLY);
    ASSERT_GE(fd, 0);
    void* mapping = mmap(nullptr, kFileBytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    ASSERT_NE(mapping, MAP_FAILED);
    mapping_ = static_cast<const uint8_t*>(mapping);

    ASSERT_EQ(
        setenv("LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR", directory_.c_str(), 1),
        0);
    ASSERT_EQ(
        setenv("LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH", source_.c_str(), 1), 0);
    ASSERT_EQ(LiteRtDispatchGetApi(&api_), kLiteRtStatusOk);

    prefill_input_ = std::make_unique<TestBuffer>(
        std::vector<int32_t>{kPrefillRows, kColumns});
    prefill_output_ = std::make_unique<TestBuffer>(
        std::vector<int32_t>{kPrefillRows, first_channels_});
    decode_input_ =
        std::make_unique<TestBuffer>(std::vector<int32_t>{1, 1, kColumns});
    decode_first_output_ = std::make_unique<TestBuffer>(
        std::vector<int32_t>{1, 1, first_channels_});
    decode_second_output_ = std::make_unique<TestBuffer>(
        std::vector<int32_t>{1, 1, kSecondChannels});
  }

  void TearDown() override {
    for (auto context : contexts_) {
      EXPECT_EQ(api_.interface->invocation_context_destroy(context),
                kLiteRtStatusOk);
    }
    if (device_ != nullptr) {
      EXPECT_EQ(api_.interface->device_context_destroy(device_),
                kLiteRtStatusOk);
    }
    if (mapping_ != nullptr) {
      munmap(const_cast<uint8_t*>(mapping_), kFileBytes);
    }
    unsetenv("LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR");
    unsetenv("LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH");
    if (!directory_.empty()) {
      std::filesystem::remove_all(directory_);
    }
  }

  static std::vector<int8_t> Values(int32_t channels, int seed) {
    std::vector<int8_t> values(static_cast<size_t>(channels) * kColumns);
    uint32_t state = seed;
    for (auto& value : values) {
      state = state * 1664525u + 1013904223u;
      value = static_cast<int8_t>((state >> 28) & 0xF) - 8;
    }
    return values;
  }

  static float Scale(int partition, int projection, int32_t channel) {
    return 0.001f * (1 + channel % 7) * (1 + partition) * (1 + projection);
  }

  // A projection of `input` by INT4 weights that are a view of the model
  // file at `offset`.
  void AddProjection(LiteRtSubgraphT& graph, LiteRtTensorT& input,
                     std::vector<int32_t> output_dims, uint64_t offset,
                     int partition, int projection) {
    const int32_t channels = output_dims.back();
    auto& weights = graph.EmplaceTensor();
    weights.SetType(
        MakeRankedTensorType(kLiteRtElementTypeInt4, {channels, kColumns}));
    weights.SetName("weights" + std::to_string(projection));
    SetWeightsFromUnownedBuffer(
        weights.Weights(),
        litert::BufferRef<uint8_t>(
            mapping_ + offset, static_cast<size_t>(channels) * kColumns / 2));
    std::vector<float> scales(channels);
    for (int32_t channel = 0; channel < channels; ++channel) {
      scales[channel] = Scale(partition, projection, channel);
    }
    const std::vector<int64_t> zero_points(channels, 0);
    weights.SetQarams(MakePerChannelQuantization(scales, zero_points,
                                                 /*quantized_dim=*/0, weights));
    auto& output = graph.EmplaceTensor();
    output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                        std::move(output_dims)));
    output.SetName("output" + std::to_string(projection));
    graph.Outputs().push_back(&output);
    auto& fc = graph.EmplaceOp();
    fc.SetOpCode(kLiteRtOpCodeTflFullyConnected);
    tflite::FullyConnectedOptionsT fc_options;
    fc_options.keep_num_dims = true;
    tflite::BuiltinOptionsUnion options;
    options.Set(std::move(fc_options));
    litert::internal::SetTflOptions(fc, std::move(options));
    litert::internal::AttachInput(&input, fc);
    litert::internal::AttachInput(&weights, fc);
    litert::internal::AttachOutput(&output, fc);
  }

  void Compile() {
    LiteRtCompilerPlugin plugin = nullptr;
    ASSERT_EQ(LiteRtCreateCompilerPlugin(LrtGetCompilerContext(), &plugin,
                                         nullptr, nullptr),
              kLiteRtStatusOk);
    LiteRtModelT model;
    {
      auto& graph = model.EmplaceSubgraph();
      auto& input = graph.EmplaceTensor();
      input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                         {kPrefillRows, kColumns}));
      input.SetName("input");
      graph.Inputs().push_back(&input);
      AddProjection(graph, input, {kPrefillRows, first_channels_}, kFirstOffset,
                    /*partition=*/0, /*projection=*/0);
    }
    {
      auto& graph = model.EmplaceSubgraph();
      auto& input = graph.EmplaceTensor();
      input.SetType(
          MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 1, kColumns}));
      input.SetName("input");
      graph.Inputs().push_back(&input);
      AddProjection(graph, input, {1, 1, first_channels_}, kFirstOffset,
                    /*partition=*/1, /*projection=*/0);
      AddProjection(graph, input, {1, 1, kSecondChannels}, kSecondOffset,
                    /*partition=*/1, /*projection=*/1);
    }
    LiteRtCompiledResult result = nullptr;
    const LiteRtStatus status =
        LiteRtCompilerPluginCompile(plugin, nullptr, &model, &result);
    locators_.clear();
    if (status == kLiteRtStatusOk) {
      LiteRtParamIndex modules = 0;
      EXPECT_EQ(LiteRtCompiledResultNumByteCodeModules(result, &modules),
                kLiteRtStatusOk);
      for (LiteRtParamIndex i = 0; i < modules; ++i) {
        const void* data = nullptr;
        size_t size = 0;
        EXPECT_EQ(LiteRtGetCompiledResultByteCode(result, i, &data, &size),
                  kLiteRtStatusOk);
        const auto* bytes = static_cast<const uint8_t*>(data);
        locators_.emplace_back(bytes, bytes + size);
      }
      LiteRtDestroyCompiledResult(result);
    }
    LiteRtDestroyCompilerPlugin(plugin);
    ASSERT_EQ(status, kLiteRtStatusOk);
    ASSERT_EQ(locators_.size(), 2);
  }

  static std::string Function(int partition) {
    return "tensorrt_partition_" + std::to_string(partition);
  }

  // The shard of a partition, read from the file its locator names.
  struct Shard {
    std::vector<uint8_t> bytes;
    std::optional<TensorRtBytecode> bytecode;
  };
  void ReadShard(int partition, Shard& shard) {
    auto locator = TryParseTensorRtAotLocator(locators_[partition].data(),
                                              locators_[partition].size());
    ASSERT_TRUE(locator.HasValue());
    ASSERT_TRUE(locator->has_value());
    std::ifstream file((*locator)->path, std::ios::binary);
    shard.bytes.assign(std::istreambuf_iterator<char>(file), {});
    auto parsed = ParseTensorRtBytecode(shard.bytes.data(), shard.bytes.size(),
                                        Function(partition).c_str());
    ASSERT_TRUE(parsed.HasValue()) << parsed.Error().Message();
    shard.bytecode = std::move(*parsed);
  }

  // Whether the shard holds the row of packed weights that the model file
  // has at `offset`.
  bool HoldsModelBytes(const Shard& shard, uint64_t offset) const {
    const uint8_t* row = mapping_ + offset;
    return std::search(shard.bytes.begin(), shard.bytes.end(), row,
                       row + kColumns / 2) != shard.bytes.end();
  }

  void CreateDevice() {
    ASSERT_EQ(
        api_.interface->device_context_create(&runtime_, nullptr, &device_),
        kLiteRtStatusOk);
    for (TestBuffer* buffer :
         {prefill_input_.get(), prefill_output_.get(), decode_input_.get(),
          decode_first_output_.get(), decode_second_output_.get()}) {
      ASSERT_EQ(api_.interface->register_tensor_buffer(
                    device_, buffer->handle(), &buffer->registered),
                kLiteRtStatusOk);
    }
  }

  LiteRtStatus CreateContext(int partition,
                             LiteRtDispatchInvocationContext* context) {
    LiteRtMemBuffer buffer{};
    buffer.fd = -1;
    buffer.base_addr = locators_[partition].data();
    buffer.size = locators_[partition].size();
    const std::vector<TestBuffer*> outputs =
        partition == 0 ? std::vector<TestBuffer*>{prefill_output_.get()}
                       : std::vector<TestBuffer*>{decode_first_output_.get(),
                                                  decode_second_output_.get()};
    const LiteRtStatus status = api_.interface->invocation_context_create(
        &runtime_, device_, kLiteRtDispatchExecutableTypeMlModel, &buffer,
        Function(partition).c_str(), 1, static_cast<int>(outputs.size()),
        context);
    if (status != kLiteRtStatusOk) {
      return status;
    }
    contexts_.push_back(*context);
    EXPECT_EQ(
        api_.interface->attach_input(
            *context, 0,
            (partition == 0 ? prefill_input_ : decode_input_)->registered),
        kLiteRtStatusOk);
    for (size_t i = 0; i < outputs.size(); ++i) {
      EXPECT_EQ(api_.interface->attach_output(*context, static_cast<int>(i),
                                              outputs[i]->registered),
                kLiteRtStatusOk);
    }
    return status;
  }

  void Destroy(LiteRtDispatchInvocationContext context) {
    contexts_.erase(std::find(contexts_.begin(), contexts_.end(), context));
    EXPECT_EQ(api_.interface->invocation_context_destroy(context),
              kLiteRtStatusOk);
  }

  // Checks the projection of the activation rows `rows` that `output` holds.
  void ExpectProjection(const TestBuffer& output,
                        const std::vector<float>& activations,
                        const std::vector<int32_t>& rows,
                        const std::vector<int8_t>& values, int partition,
                        int projection) {
    std::vector<float> actual(output.elements);
    ASSERT_EQ(cudaMemcpy(actual.data(), output.device_ptr, output.bytes(),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    const int32_t channels = static_cast<int32_t>(values.size() / kColumns);
    for (const int32_t row : rows) {
      const float* activation = activations.data() + row * kColumns;
      for (int32_t channel = 0; channel < channels; ++channel) {
        double expected = 0.0;
        double magnitude = 0.0;
        for (int32_t i = 0; i < kColumns; ++i) {
          const double term =
              static_cast<double>(activation[i]) *
              values[static_cast<size_t>(channel) * kColumns + i];
          expected += term;
          magnitude += std::fabs(term);
        }
        const double scale = Scale(partition, projection, channel);
        ASSERT_NEAR(actual[static_cast<size_t>(row) * channels + channel],
                    expected * scale, 0.02 * magnitude * scale + 1e-4)
            << "partition=" << partition << " projection=" << projection
            << " row=" << row << " channel=" << channel;
      }
    }
  }

  // Runs the engine of `partition` on activations that depend on `round` and
  // checks what it computed.
  void Run(LiteRtDispatchInvocationContext context, int partition, int round) {
    TestBuffer& input = partition == 0 ? *prefill_input_ : *decode_input_;
    std::vector<float> activations(input.elements);
    for (size_t i = 0; i < activations.size(); ++i) {
      activations[i] =
          0.125f *
          (static_cast<int>((i * 7 + i / kColumns + round * 3) % 13) - 6);
    }
    ASSERT_EQ(cudaMemcpy(input.device_ptr, activations.data(), input.bytes(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    for (TestBuffer* output :
         {prefill_output_.get(), decode_first_output_.get(),
          decode_second_output_.get()}) {
      ASSERT_EQ(cudaMemset(output->device_ptr, 0, output->bytes()),
                cudaSuccess);
    }
    // cudaMemset returns before the device is done, and the stream of the
    // engine does not wait for it.
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(api_.interface->invoke(context), kLiteRtStatusOk);
    if (partition == 0) {
      ExpectProjection(*prefill_output_, activations,
                       {0, 1, 127, 128, 640, kPrefillRows - 1}, first_, 0, 0);
    } else {
      ExpectProjection(*decode_first_output_, activations, {0}, first_, 1, 0);
      ExpectProjection(*decode_second_output_, activations, {0}, second_, 1, 1);
    }
  }

  // As run_head.sh configures the backend, with one launch per projection.
  ScopedEnvironment gemv_{"LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS",
                          "cuda_gemv"};
  ScopedEnvironment shared_{"LITERT_NVIDIA_TENSORRT_SHARED_WEIGHTS", "1"};
  ScopedEnvironment precision_{"LITERT_NVIDIA_TENSORRT_FP16_ACTIVATIONS",
                               "bf16"};
  ScopedEnvironment groups_{"LITERT_NVIDIA_TENSORRT_FUSE_GEMV_GROUPS", "0"};
  ScopedEnvironment cache_{"LITERT_NVIDIA_DISPATCH_RUNTIME_CACHE_DIR", ""};
  ScopedEnvironment other_cache_{"LITERT_NVIDIA_TENSORRT_RUNTIME_CACHE_DIR",
                                 ""};
  int32_t first_channels_ = 0;
  std::filesystem::path directory_;
  std::string source_;
  const uint8_t* mapping_ = nullptr;
  std::vector<int8_t> first_;
  std::vector<int8_t> second_;
  std::vector<std::vector<uint8_t>> locators_;
  LiteRtRuntimeContext runtime_ = RuntimeContext();
  LiteRtDispatchApi api_{};
  LiteRtDispatchDeviceContext device_ = nullptr;
  std::unique_ptr<TestBuffer> prefill_input_;
  std::unique_ptr<TestBuffer> prefill_output_;
  std::unique_ptr<TestBuffer> decode_input_;
  std::unique_ptr<TestBuffer> decode_first_output_;
  std::unique_ptr<TestBuffer> decode_second_output_;
  std::vector<LiteRtDispatchInvocationContext> contexts_;
};

TEST_F(WeightStoreDispatchTest, PlansLeaveTheirPluginWeightsInTheModelFile) {
  Compile();
  ASSERT_FALSE(HasFatalFailure());
  Shard prefill;
  Shard decode;
  ReadShard(0, prefill);
  ReadShard(1, decode);
  ASSERT_FALSE(HasFatalFailure());
  const size_t first_bytes =
      static_cast<size_t>(first_channels_) * kColumns / 2;
  const size_t second_bytes = kSecondChannels * kColumns / 2;
  // The packed weights are in neither shard.
  EXPECT_FALSE(HoldsModelBytes(prefill, kFirstOffset));
  EXPECT_FALSE(HoldsModelBytes(decode, kFirstOffset));
  EXPECT_FALSE(HoldsModelBytes(decode, kSecondOffset));
  EXPECT_EQ(prefill.bytecode->version, kTensorRtBytecodeVersionWithWeightStore);
  EXPECT_TRUE(prefill.bytecode->refit_weights.empty());
  ASSERT_TRUE(prefill.bytecode->weight_store.has_value());
  ASSERT_TRUE(decode.bytecode->weight_store.has_value());
  const TensorRtWeightStore& one = *prefill.bytecode->weight_store;
  const TensorRtWeightStore& two = *decode.bytecode->weight_store;
  EXPECT_EQ(one.source_path, std::filesystem::canonical(source_).string());
  EXPECT_EQ(one.source_size, kFileBytes);
  EXPECT_TRUE(one.source_identity == Identity(source_));
  ASSERT_EQ(one.segments.size(), 1);
  ASSERT_EQ(one.segments[0].pieces.size(), 1);
  EXPECT_EQ(one.segments[0].pieces[0].source_offset, kFirstOffset);
  EXPECT_EQ(one.segments[0].pieces[0].size, first_bytes);
  EXPECT_EQ(one.segments[0].size, one.granule);
  EXPECT_EQ(one.segments[0].payload_offset % one.granule, 0);
  // The decode engine reads the segment of the prefill engine, whose weights
  // it multiplies one activation row by, and a segment of its own.
  ASSERT_EQ(two.segments.size(), 2);
  const bool first_is_common = two.segments[0].key == one.segments[0].key;
  const TensorRtWeightSegment& common = two.segments[first_is_common ? 0 : 1];
  const TensorRtWeightSegment& own = two.segments[first_is_common ? 1 : 0];
  EXPECT_TRUE(common.key == one.segments[0].key);
  EXPECT_FALSE(own.key == one.segments[0].key);
  ASSERT_EQ(own.pieces.size(), 1);
  EXPECT_EQ(own.pieces[0].source_offset, kSecondOffset);
  EXPECT_EQ(own.pieces[0].size, second_bytes);
}

TEST_F(WeightStoreDispatchTest, EnginesComputeWithMappedWeights) {
  CreateDevice();
  ASSERT_FALSE(HasFatalFailure());
  for (const char* share : {"1", "0"}) {
    SCOPED_TRACE(share);
    ScopedEnvironment sharing("LITERT_NVIDIA_DISPATCH_SHARED_WEIGHT_SEGMENTS",
                              share);
    // The second compilation finds the shards of the first one.
    Compile();
    ASSERT_FALSE(HasFatalFailure());
    LiteRtDispatchInvocationContext prefill = nullptr;
    LiteRtDispatchInvocationContext decode = nullptr;
    ASSERT_EQ(CreateContext(0, &prefill), kLiteRtStatusOk);
    ASSERT_EQ(CreateContext(1, &decode), kLiteRtStatusOk);
    for (int round = 0; round < 3; ++round) {
      Run(prefill, 0, round);
      Run(decode, 1, round);
      ASSERT_FALSE(HasFatalFailure());
    }
    // The engine that is left keeps the segment both engines read.
    Destroy(prefill);
    Run(decode, 1, 3);
    // An engine that is created again finds it there.
    ASSERT_EQ(CreateContext(0, &prefill), kLiteRtStatusOk);
    Run(prefill, 0, 4);
    Run(decode, 1, 5);
    ASSERT_FALSE(HasFatalFailure());
    Destroy(prefill);
    Destroy(decode);
  }
}

TEST_F(WeightStoreDispatchTest, LazyEnginesMapTheirWeightsAtEveryReload) {
  ScopedEnvironment lazy("LITERT_NVIDIA_DISPATCH_LAZY_AOT_ENGINES", "1");
  Compile();
  ASSERT_FALSE(HasFatalFailure());
  CreateDevice();
  ASSERT_FALSE(HasFatalFailure());
  LiteRtDispatchInvocationContext prefill = nullptr;
  LiteRtDispatchInvocationContext decode = nullptr;
  ASSERT_EQ(CreateContext(0, &prefill), kLiteRtStatusOk);
  ASSERT_EQ(CreateContext(1, &decode), kLiteRtStatusOk);
  for (int round = 0; round < 3; ++round) {
    Run(prefill, 0, round);
    Run(decode, 1, round);
    Run(decode, 1, round + 10);
    ASSERT_FALSE(HasFatalFailure());
  }
  Run(prefill, 0, 20);
}

TEST_F(WeightStoreDispatchTest, AChangedModelFileFailsTheLoad) {
  Compile();
  ASSERT_FALSE(HasFatalFailure());
  CreateDevice();
  ASSERT_FALSE(HasFatalFailure());
  Touch(source_);
  LiteRtDispatchInvocationContext context = nullptr;
  EXPECT_EQ(CreateContext(0, &context), kLiteRtStatusErrorInvalidArgument);
}

TEST_F(WeightStoreDispatchTest, PlansKeepTheirWeightsWhenTheStoreIsOff) {
  ScopedEnvironment off("LITERT_NVIDIA_TENSORRT_WEIGHT_STORE", "0");
  Compile();
  ASSERT_FALSE(HasFatalFailure());
  Shard prefill;
  Shard decode;
  ReadShard(0, prefill);
  ReadShard(1, decode);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_FALSE(prefill.bytecode->weight_store.has_value());
  EXPECT_FALSE(decode.bytecode->weight_store.has_value());
  EXPECT_TRUE(HoldsModelBytes(prefill, kFirstOffset));
  EXPECT_TRUE(HoldsModelBytes(decode, kFirstOffset));
  EXPECT_TRUE(HoldsModelBytes(decode, kSecondOffset));
  CreateDevice();
  ASSERT_FALSE(HasFatalFailure());
  // The engines run without the model file.
  Touch(source_);
  LiteRtDispatchInvocationContext prefill_context = nullptr;
  LiteRtDispatchInvocationContext decode_context = nullptr;
  ASSERT_EQ(CreateContext(0, &prefill_context), kLiteRtStatusOk);
  ASSERT_EQ(CreateContext(1, &decode_context), kLiteRtStatusOk);
  Run(prefill_context, 0, 0);
  Run(decode_context, 1, 0);
}

}  // namespace
}  // namespace litert::nvidia

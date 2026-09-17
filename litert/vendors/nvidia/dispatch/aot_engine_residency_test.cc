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

#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/nvidia/bytecode.h"
#include "litert/vendors/nvidia/cache_layout.h"
#include "NvInfer.h"

namespace litert::nvidia {
namespace {

constexpr int kElements = 32;
constexpr size_t kBytes = kElements * sizeof(float);
constexpr auto kCudaBufferType = static_cast<LiteRtTensorBufferType>(
    kLiteRtTensorBufferTypeUserCustomBuffer + 1);

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

LiteRtRankedTensorType TensorType() {
  LiteRtRankedTensorType type{};
  type.element_type = kLiteRtElementTypeFloat32;
  type.layout.rank = 1;
  type.layout.dimensions[0] = kElements;
  return type;
}

// Only buffer metadata is mocked. TensorRT, CUDA allocations, engine lifetime,
// AOT parsing/validation and the public dispatch function table are real.
struct TestBuffer {
  void* device_ptr = nullptr;
  LiteRtRankedTensorType type = TensorType();
  size_t bytes = kBytes;
  LiteRtTensorBuffer handle() {
    return reinterpret_cast<LiteRtTensorBuffer>(this);
  }
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
    *size = reinterpret_cast<TestBuffer*>(buffer)->bytes;
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

class TestLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR) {
      ADD_FAILURE() << "TensorRT-RTX: " << message;
    }
  }
};

class AotEngineResidencyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(LiteRtDispatchGetApi(&api_), kLiteRtStatusOk);
    char directory_template[] = "/tmp/litert_nvidia_residency_test_XXXXXX";
    char* directory = mkdtemp(directory_template);
    ASSERT_NE(directory, nullptr);
    directory_ = directory;
    ASSERT_EQ(cudaMalloc(&input_.device_ptr, kBytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&output_.device_ptr, kBytes), cudaSuccess);
  }

  void TearDown() override {
    for (auto context : contexts_) {
      EXPECT_EQ(api_.interface->invocation_context_destroy(context),
                kLiteRtStatusOk);
    }
    if (device_) {
      EXPECT_EQ(api_.interface->device_context_destroy(device_),
                kLiteRtStatusOk);
    }
    if (output_.device_ptr) {
      cudaFree(output_.device_ptr);
    }
    if (input_.device_ptr) {
      cudaFree(input_.device_ptr);
    }
    if (update_.device_ptr) {
      cudaFree(update_.device_ptr);
    }
    if (position_.device_ptr) {
      cudaFree(position_.device_ptr);
    }
    for (const auto& path : paths_) {
      unlink(path.c_str());
    }
    if (!directory_.empty()) {
      rmdir(directory_.c_str());
    }
  }

  Expected<std::vector<uint8_t>> BuildBytecode(const std::string& name,
                                               float bias, bool stripped) {
    std::unique_ptr<nvinfer1::IBuilder> builder(
        nvinfer1::createInferBuilder(logger_));
    if (!builder) {
      return Error(kLiteRtStatusErrorRuntimeFailure, "No TRT builder");
    }
    std::unique_ptr<nvinfer1::INetworkDefinition> network(
        builder->createNetworkV2(0));
    std::unique_ptr<nvinfer1::IBuilderConfig> config(
        builder->createBuilderConfig());
    if (!network || !config) {
      return Error(kLiteRtStatusErrorRuntimeFailure, "No TRT network/config");
    }
    std::array<float, kElements> biases;
    biases.fill(bias);
    nvinfer1::Weights weights{nvinfer1::DataType::kFLOAT, biases.data(),
                              kElements};
    auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT,
                                    nvinfer1::Dims{1, {kElements}});
    auto* constant =
        network->addConstant(nvinfer1::Dims{1, {kElements}}, weights);
    if (!input || !constant) {
      return Error(kLiteRtStatusErrorRuntimeFailure, "No TRT input/constant");
    }
    auto* sum = network->addElementWise(*input, *constant->getOutput(0),
                                        nvinfer1::ElementWiseOperation::kSUM);
    if (!sum) {
      return Error(kLiteRtStatusErrorRuntimeFailure, "No TRT sum");
    }
    sum->getOutput(0)->setName("output");
    network->markOutput(*sum->getOutput(0));
    if (stripped) {
      if (!network->setWeightsName(weights, "bias") ||
          !network->markWeightsRefittable("bias") ||
          !config->setNbComputeCapabilities(1) ||
          !config->setComputeCapability(nvinfer1::ComputeCapability::kCURRENT,
                                        0)) {
        return Error(kLiteRtStatusErrorRuntimeFailure,
                     "Cannot configure refit");
      }
      config->setFlag(nvinfer1::BuilderFlag::kREFIT_INDIVIDUAL);
      config->setFlag(nvinfer1::BuilderFlag::kSTRIP_PLAN);
    }
    std::unique_ptr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
      return Error(kLiteRtStatusErrorCompilation, "Build failed");
    }
    if (!stripped) {
      return PackTensorRtBytecode(name, {"input"}, {"output"},
                                  serialized->data(), serialized->size());
    }
    TensorRtSharedWeight shared_weight{TensorRtWeightDataType::kFloat,
                                       kElements, std::vector<uint8_t>(kBytes)};
    std::memcpy(shared_weight.data.data(), biases.data(), kBytes);
    TensorRtBundleEntry entry{name,
                              {"input"},
                              {"output"},
                              serialized->data(),
                              serialized->size(),
                              nullptr,
                              {{"bias", 0}}};
    return PackTensorRtSharedWeightShard({shared_weight}, entry);
  }

  bool Write(const std::string& path, const std::vector<uint8_t>& bytes) {
    paths_.push_back(path);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    file.close();
    return !file.fail() && chmod(path.c_str(), S_IRUSR) == 0;
  }

  Expected<std::vector<uint8_t>> Locator(const std::string& path,
                                         const std::vector<uint8_t>& bytes) {
    struct stat status{};
    if (stat(path.c_str(), &status) != 0) {
      return Error(kLiteRtStatusErrorFileIO, "Cannot stat test artifact");
    }
    TensorRtAotLocator locator;
    locator.path = path;
    locator.artifact_size = bytes.size();
    locator.fingerprint =
        FingerprintTensorRtAotArtifact(bytes.data(), bytes.size());
    locator.file_identity =
        TensorRtAotFileIdentity{static_cast<uint64_t>(status.st_dev),
                                static_cast<uint64_t>(status.st_ino),
                                status.st_mtim.tv_sec,
                                status.st_mtim.tv_nsec,
                                status.st_ctim.tv_sec,
                                status.st_ctim.tv_nsec};
    return PackTensorRtAotLocator(locator);
  }

  void CreateDevice() {
    ASSERT_EQ(
        api_.interface->device_context_create(&runtime_, nullptr, &device_),
        kLiteRtStatusOk);
    ASSERT_EQ(api_.interface->register_tensor_buffer(device_, input_.handle(),
                                                     &input_handle_),
              kLiteRtStatusOk);
    ASSERT_EQ(api_.interface->register_tensor_buffer(device_, output_.handle(),
                                                     &output_handle_),
              kLiteRtStatusOk);
  }

  LiteRtStatus CreateContext(const std::vector<uint8_t>& bytes,
                             const char* name,
                             LiteRtDispatchInvocationContext* context,
                             int num_inputs = 1) {
    LiteRtMemBuffer buffer{};
    buffer.fd = -1;
    buffer.base_addr = bytes.data();
    buffer.size = bytes.size();
    const auto status = api_.interface->invocation_context_create(
        &runtime_, device_, kLiteRtDispatchExecutableTypeMlModel, &buffer, name,
        num_inputs, 1, context);
    if (status == kLiteRtStatusOk) {
      contexts_.push_back(*context);
    }
    return status;
  }

  void Bind(LiteRtDispatchInvocationContext context) {
    // These requirements must remain available while the engine is nonresident.
    auto type = TensorType();
    LiteRtTensorBufferRequirements requirements = nullptr;
    ASSERT_EQ(api_.interface->get_input_requirements(context, 0, &type,
                                                     &requirements),
              kLiteRtStatusOk);
    LiteRtTensorBufferType first_type;
    ASSERT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                  requirements, 0, &first_type),
              kLiteRtStatusOk);
    EXPECT_EQ(first_type, kCudaBufferType);
    LiteRtDestroyTensorBufferRequirements(requirements);
    ASSERT_EQ(api_.interface->get_output_requirements(context, 0, &type,
                                                      &requirements),
              kLiteRtStatusOk);
    LiteRtDestroyTensorBufferRequirements(requirements);
    ASSERT_EQ(api_.interface->attach_input(context, 0, input_handle_),
              kLiteRtStatusOk);
    ASSERT_EQ(api_.interface->attach_output(context, 0, output_handle_),
              kLiteRtStatusOk);
  }

  void Run(LiteRtDispatchInvocationContext context, float input_base,
           float bias) {
    std::array<float, kElements> input;
    for (int i = 0; i < kElements; ++i) {
      input[i] = input_base + i;
    }
    ASSERT_EQ(cudaMemcpy(input_.device_ptr, input.data(), kBytes,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(api_.interface->invoke(context), kLiteRtStatusOk);
    std::array<float, kElements> output;
    ASSERT_EQ(cudaMemcpy(output.data(), output_.device_ptr, kBytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int i = 0; i < kElements; ++i) {
      EXPECT_FLOAT_EQ(output[i], input[i] + bias);
    }
  }

  void PreparePair() {
    CreateDevice();
    if (HasFatalFailure()) {
      return;
    }
    auto a = BuildBytecode("a", 1.0f, false);
    ASSERT_TRUE(a) << a.Error().Message();
    a_bytes_ = std::move(*a);
    auto b = BuildBytecode("b", 2.0f, true);
    ASSERT_TRUE(b) << b.Error().Message();
    const auto a_path = directory_ + "/a.bin";
    const auto b_path = directory_ + "/b.bin";
    ASSERT_TRUE(Write(a_path, a_bytes_));
    ASSERT_TRUE(Write(b_path, *b));
    auto a_locator = Locator(a_path, a_bytes_);
    auto b_locator = Locator(b_path, *b);
    ASSERT_TRUE(a_locator) << a_locator.Error().Message();
    ASSERT_TRUE(b_locator) << b_locator.Error().Message();
    ASSERT_EQ(CreateContext(*a_locator, "a", &a_), kLiteRtStatusOk);
    ASSERT_EQ(CreateContext(*b_locator, "b", &b_), kLiteRtStatusOk);
    Bind(a_);
    Bind(b_);
  }

  Expected<std::vector<uint8_t>> BuildValueCacheBytecode() {
    std::unique_ptr<nvinfer1::IBuilder> builder(
        nvinfer1::createInferBuilder(logger_));
    if (!builder) {
      return Error(kLiteRtStatusErrorRuntimeFailure, "No TRT builder");
    }
    std::unique_ptr<nvinfer1::INetworkDefinition> network(
        builder->createNetworkV2(0));
    std::unique_ptr<nvinfer1::IBuilderConfig> config(
        builder->createBuilderConfig());
    if (!network || !config) {
      return Error(kLiteRtStatusErrorRuntimeFailure, "No TRT network/config");
    }
    const std::string input_name =
        std::string("cache") + kTransposedValueCacheSuffix;
    const std::string output_name =
        std::string("updated") + kTransposedValueCacheSuffix;
    // Physical [B,H,S,D]; LiteRT exposes logical [B,H,D,S] with SD strides.
    auto* cache =
        network->addInput(input_name.c_str(), nvinfer1::DataType::kFLOAT,
                          nvinfer1::Dims{4, {1, 2, 4, 4}});
    auto* update = network->addInput("update", nvinfer1::DataType::kFLOAT,
                                     nvinfer1::Dims{4, {1, 2, 1, 4}});
    auto* position = network->addInput("position", nvinfer1::DataType::kINT32,
                                       nvinfer1::Dims{1, {1}});
    if (!cache || !update || !position) {
      return Error(kLiteRtStatusErrorRuntimeFailure, "No TRT cache inputs");
    }
    auto* layer = network->addKVCacheUpdate(*cache, *update, *position,
                                            nvinfer1::KVCacheMode::kLINEAR);
    if (!layer || !layer->getOutput(0)) {
      return Error(kLiteRtStatusErrorRuntimeFailure, "No TRT KV cache update");
    }
    layer->getOutput(0)->setName(output_name.c_str());
    network->markOutput(*layer->getOutput(0));
    std::unique_ptr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
      return Error(kLiteRtStatusErrorCompilation, "Cache build failed");
    }
    return PackTensorRtBytecode("cache_a", {input_name, "update", "position"},
                                {output_name}, serialized->data(),
                                serialized->size());
  }

  void CheckValueCacheContract(bool lazy) {
    ScopedEnvironment mode("LITERT_NVIDIA_DISPATCH_LAZY_AOT_ENGINES",
                           lazy ? "1" : "0");
    input_.type.layout = LiteRtLayout{4, true, {1, 2, 4, 4}, {32, 16, 1, 4}};
    output_.type = input_.type;
    update_.type.layout = LiteRtLayout{4, false, {1, 2, 1, 4}, {}};
    update_.bytes = 8 * sizeof(float);
    position_.type.element_type = kLiteRtElementTypeInt32;
    position_.type.layout = LiteRtLayout{1, false, {1}, {}};
    position_.bytes = sizeof(int32_t);
    ASSERT_EQ(cudaMalloc(&update_.device_ptr, update_.bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&position_.device_ptr, position_.bytes), cudaSuccess);
    CreateDevice();
    ASSERT_FALSE(HasFatalFailure());
    LiteRtTensorBufferHandle update_handle = 0, position_handle = 0;
    ASSERT_EQ(api_.interface->register_tensor_buffer(device_, update_.handle(),
                                                     &update_handle),
              kLiteRtStatusOk);
    ASSERT_EQ(api_.interface->register_tensor_buffer(
                  device_, position_.handle(), &position_handle),
              kLiteRtStatusOk);

    auto cache_bytes = BuildValueCacheBytecode();
    ASSERT_TRUE(cache_bytes) << cache_bytes.Error().Message();
    auto parsed = ParseTensorRtBytecode(cache_bytes->data(),
                                        cache_bytes->size(), "cache_a");
    ASSERT_TRUE(parsed) << parsed.Error().Message();
    auto other_bytes = PackTensorRtBytecode(
        "cache_b", parsed->input_names, parsed->output_names,
        parsed->engine_data, parsed->engine_size);
    ASSERT_TRUE(other_bytes) << other_bytes.Error().Message();
    const std::string a_path = directory_ + "/cache_a.bin";
    const std::string b_path = directory_ + "/cache_b.bin";
    ASSERT_TRUE(Write(a_path, *cache_bytes));
    ASSERT_TRUE(Write(b_path, *other_bytes));
    auto a_locator = Locator(a_path, *cache_bytes);
    auto b_locator = Locator(b_path, *other_bytes);
    ASSERT_TRUE(a_locator) << a_locator.Error().Message();
    ASSERT_TRUE(b_locator) << b_locator.Error().Message();
    ASSERT_EQ(CreateContext(*a_locator, "cache_a", &a_, 3), kLiteRtStatusOk);
    ASSERT_EQ(CreateContext(*b_locator, "cache_b", &b_, 3), kLiteRtStatusOk);

    TestBuffer unannotated = input_;
    unannotated.type.layout.has_strides = false;
    TestBuffer native = input_;
    native.type.layout = LiteRtLayout{4, true, {1, 2, 4, 4}, {32, 16, 4, 1}};
    TestBuffer invalid = input_;
    ++invalid.type.layout.strides[1];
    LiteRtTensorBufferHandle unannotated_handle = 0, native_handle = 0;
    LiteRtTensorBufferHandle invalid_handle = 0;
    ASSERT_EQ(api_.interface->register_tensor_buffer(
                  device_, unannotated.handle(), &unannotated_handle),
              kLiteRtStatusOk);
    ASSERT_EQ(api_.interface->register_tensor_buffer(device_, native.handle(),
                                                     &native_handle),
              kLiteRtStatusOk);
    EXPECT_EQ(api_.interface->register_tensor_buffer(device_, invalid.handle(),
                                                     &invalid_handle),
              kLiteRtStatusErrorUnsupported);
    for (auto context : {a_, b_}) {
      for (bool output : {false, true}) {
        const auto get_requirements =
            output ? api_.interface->get_output_requirements
                   : api_.interface->get_input_requirements;
        LiteRtTensorBufferRequirements requirements = nullptr;
        ASSERT_EQ(
            get_requirements(context, 0, &unannotated.type, &requirements),
            kLiteRtStatusOk);
        int num_strides = 0;
        const uint32_t* strides = nullptr;
        ASSERT_EQ(LiteRtGetTensorBufferRequirementsStrides(
                      requirements, &num_strides, &strides),
                  kLiteRtStatusOk);
        ASSERT_EQ(num_strides, 4);
        for (int i = 0; i < 4; ++i) {
          EXPECT_EQ(strides[i], input_.type.layout.strides[i]);
        }
        size_t required_bytes = 0;
        ASSERT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(requirements,
                                                              &required_bytes),
                  kLiteRtStatusOk);
        EXPECT_EQ(required_bytes, kBytes);
        LiteRtDestroyTensorBufferRequirements(requirements);
        requirements = nullptr;
        EXPECT_EQ(get_requirements(context, 0, &native.type, &requirements),
                  kLiteRtStatusErrorUnsupported);
        const auto attach = output ? api_.interface->attach_output
                                   : api_.interface->attach_input;
        EXPECT_EQ(attach(context, 0, unannotated_handle),
                  kLiteRtStatusErrorInvalidArgument);
        EXPECT_EQ(attach(context, 0, native_handle),
                  kLiteRtStatusErrorInvalidArgument);
        ASSERT_EQ(attach(context, 0, output ? output_handle_ : input_handle_),
                  kLiteRtStatusOk);
      }
      ASSERT_EQ(api_.interface->attach_input(context, 1, update_handle),
                kLiteRtStatusOk);
      ASSERT_EQ(api_.interface->attach_input(context, 2, position_handle),
                kLiteRtStatusOk);
      // Conversely, a valid SD cache must not bind to a dense update port.
      EXPECT_EQ(api_.interface->attach_input(context, 1, input_handle_),
                kLiteRtStatusErrorInvalidArgument);
    }
    if (!lazy) {
      ASSERT_EQ(unlink(a_path.c_str()), 0);
      ASSERT_EQ(unlink(b_path.c_str()), 0);
    }
    for (int round = 0; round < 3; ++round) {
      auto context = round == 1 ? b_ : a_;
      std::array<float, kElements> cache;
      std::array<float, 8> update;
      for (int i = 0; i < kElements; ++i) cache[i] = round * 100.0f + i;
      for (int i = 0; i < 8; ++i) update[i] = -10.0f * (round + 1) - i;
      const int32_t position = round + 1;
      ASSERT_EQ(cudaMemcpy(input_.device_ptr, cache.data(), kBytes,
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(update_.device_ptr, update.data(), update_.bytes,
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(position_.device_ptr, &position, position_.bytes,
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      // First exercise distinct buffers (dispatch seeds and rebinds the
      // aliased input), then exercise the same buffer on a reloaded engine.
      const bool same_buffer = round == 2;
      ASSERT_EQ(api_.interface->attach_output(
                    context, 0, same_buffer ? input_handle_ : output_handle_),
                kLiteRtStatusOk);
      ASSERT_EQ(api_.interface->invoke(context), kLiteRtStatusOk);
      std::array<float, kElements> result;
      ASSERT_EQ(cudaMemcpy(result.data(),
                           same_buffer ? input_.device_ptr : output_.device_ptr,
                           kBytes, cudaMemcpyDeviceToHost),
                cudaSuccess);
      for (int h = 0; h < 2; ++h) {
        for (int s = 0; s < 4; ++s) {
          for (int d = 0; d < 4; ++d) {
            const int offset = h * 16 + s * 4 + d;
            EXPECT_FLOAT_EQ(result[offset],
                            s == position ? update[h * 4 + d] : cache[offset]);
          }
        }
      }
    }
    EXPECT_EQ(api_.interface->unregister_tensor_buffer(device_, native_handle),
              kLiteRtStatusOk);
    EXPECT_EQ(
        api_.interface->unregister_tensor_buffer(device_, unannotated_handle),
        kLiteRtStatusOk);
  }

  ScopedEnvironment lazy_{"LITERT_NVIDIA_DISPATCH_LAZY_AOT_ENGINES", "1"};
  ScopedEnvironment cache_{"LITERT_NVIDIA_DISPATCH_RUNTIME_CACHE_DIR", ""};
  ScopedEnvironment other_cache_{"LITERT_NVIDIA_TENSORRT_RUNTIME_CACHE_DIR",
                                 ""};
  ScopedEnvironment host_io_{"LITERT_NVIDIA_DISPATCH_PREFER_HOST_IO", "0"};
  ScopedEnvironment graphs_{"LITERT_NVIDIA_DISPATCH_CUDA_GRAPH", "1"};
  ScopedEnvironment arena_{"LITERT_NVIDIA_DISPATCH_SHARED_ARENA", "1"};
  ScopedEnvironment layers_{"LITERT_NVIDIA_DISPATCH_LAYER_PROFILE", "0"};
  TestLogger logger_;
  LiteRtRuntimeContext runtime_ = RuntimeContext();
  LiteRtDispatchApi api_{};
  LiteRtDispatchDeviceContext device_ = nullptr;
  TestBuffer input_, output_, update_, position_;
  LiteRtTensorBufferHandle input_handle_ = 0, output_handle_ = 0;
  LiteRtDispatchInvocationContext a_ = nullptr, b_ = nullptr;
  std::vector<LiteRtDispatchInvocationContext> contexts_;
  std::string directory_;
  std::vector<std::string> paths_;
  std::vector<uint8_t> a_bytes_;
};

TEST_F(AotEngineResidencyTest,
       ReloadsBothPlainAndStrippedPlansWithSameBuffers) {
  PreparePair();
  ASSERT_FALSE(HasFatalFailure());
  for (int round = 0; round < 3; ++round) {
    Run(a_, round * 10.0f, 1.0f);
    Run(a_, round * 10.0f + 1, 1.0f);
    Run(b_, round * 20.0f, 2.0f);
    Run(b_, round * 20.0f + 1, 2.0f);
  }
  Run(a_, 100.0f, 1.0f);
}

TEST_F(AotEngineResidencyTest,
       MissingAndChangedArtifactsFailClosedAndCanRetry) {
  PreparePair();
  ASSERT_FALSE(HasFatalFailure());
  Run(a_, 0.0f, 1.0f);
  Run(b_, 0.0f, 2.0f);
  const std::string original = directory_ + "/a.bin";
  const std::string saved = directory_ + "/a.saved";
  paths_.push_back(saved);
  ASSERT_EQ(rename(original.c_str(), saved.c_str()), 0);
  EXPECT_EQ(api_.interface->invoke(a_), kLiteRtStatusErrorFileIO);
  ASSERT_EQ(rename(saved.c_str(), original.c_str()), 0);
  Run(a_, 10.0f, 1.0f);
  Run(b_, 10.0f, 2.0f);
  ASSERT_EQ(rename(original.c_str(), saved.c_str()), 0);
  auto changed = a_bytes_;
  changed.back() ^= 1;
  ASSERT_TRUE(Write(original, changed));
  EXPECT_EQ(api_.interface->invoke(a_), kLiteRtStatusErrorInvalidArgument);
  ASSERT_EQ(unlink(original.c_str()), 0);
  ASSERT_EQ(rename(saved.c_str(), original.c_str()), 0);
  Run(a_, 20.0f, 1.0f);
}

TEST_F(AotEngineResidencyTest, DefaultEagerModeDoesNotReopenArtifacts) {
  ScopedEnvironment eager("LITERT_NVIDIA_DISPATCH_LAZY_AOT_ENGINES", "0");
  PreparePair();
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(unlink((directory_ + "/a.bin").c_str()), 0);
  ASSERT_EQ(unlink((directory_ + "/b.bin").c_str()), 0);
  Run(a_, 0.0f, 1.0f);
  Run(b_, 0.0f, 2.0f);
  Run(a_, 10.0f, 1.0f);
}

TEST_F(AotEngineResidencyTest, RejectsNonAotBacking) {
  CreateDevice();
  ASSERT_FALSE(HasFatalFailure());
  auto bytes = BuildBytecode("a", 1.0f, false);
  ASSERT_TRUE(bytes) << bytes.Error().Message();
  LiteRtDispatchInvocationContext context = nullptr;
  EXPECT_EQ(CreateContext(*bytes, "a", &context),
            kLiteRtStatusErrorUnsupported);
}

TEST_F(AotEngineResidencyTest, UnusedContextsCanOutliveDeviceContext) {
  PreparePair();
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(api_.interface->device_context_destroy(device_), kLiteRtStatusOk);
  device_ = nullptr;
  // TearDown destroys only metadata-probed, nonresident contexts. Their new
  // residency bookkeeping must not access the now-destroyed device context.
}

TEST_F(AotEngineResidencyTest, EnforcesValueCacheStridesAcrossReloads) {
  CheckValueCacheContract(/*lazy=*/true);
}

TEST_F(AotEngineResidencyTest, EnforcesValueCacheStridesInDefaultEagerMode) {
  CheckValueCacheContract(/*lazy=*/false);
}

TEST_F(AotEngineResidencyTest,
       RefitFailureReleasesEngineAndAllowsOtherInvocations) {
  PreparePair();
  ASSERT_FALSE(HasFatalFailure());
  auto bytes = BuildBytecode("bad", 3.0f, true);
  ASSERT_TRUE(bytes) << bytes.Error().Message();
  auto parsed = ParseTensorRtBytecode(bytes->data(), bytes->size(), "bad");
  ASSERT_TRUE(parsed) << parsed.Error().Message();
  // The serialized engine is valid, but the external refit weight has the
  // wrong count. Metadata probing succeeds; execution-time refit must fail.
  TensorRtSharedWeight wrong_weight{
      TensorRtWeightDataType::kFloat, kElements - 1,
      std::vector<uint8_t>((kElements - 1) * sizeof(float))};
  TensorRtBundleEntry entry{"bad",
                            parsed->input_names,
                            parsed->output_names,
                            parsed->engine_data,
                            parsed->engine_size,
                            nullptr,
                            {{"bias", 0}}};
  auto invalid_refit = PackTensorRtSharedWeightShard({wrong_weight}, entry);
  ASSERT_TRUE(invalid_refit) << invalid_refit.Error().Message();
  const std::string bad_path = directory_ + "/bad.bin";
  ASSERT_TRUE(Write(bad_path, *invalid_refit));
  auto locator = Locator(bad_path, *invalid_refit);
  ASSERT_TRUE(locator) << locator.Error().Message();
  LiteRtDispatchInvocationContext bad = nullptr;
  ASSERT_EQ(CreateContext(*locator, "bad", &bad), kLiteRtStatusOk);
  Bind(bad);
  ASSERT_FALSE(HasFatalFailure());
  Run(a_, 0.0f, 1.0f);
  EXPECT_EQ(api_.interface->invoke(bad), kLiteRtStatusErrorInvalidArgument);
  Run(b_, 10.0f, 2.0f);
  EXPECT_EQ(api_.interface->invoke(bad), kLiteRtStatusErrorInvalidArgument);
  Run(a_, 20.0f, 1.0f);
}

}  // namespace
}  // namespace litert::nvidia

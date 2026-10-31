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

#include "litert/vendors/nvidia/dispatch/greedy_sampler_c_api.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "cuda_fp16.h"
#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/c/internal/litert_custom_tensor_buffer_handlers_def.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/nvidia/cache_layout.h"

namespace litert::nvidia {
namespace {

struct TestEvent {
  cudaEvent_t cuda_event = nullptr;
  LiteRtStatus wait_status = kLiteRtStatusOk;
  int waits = 0;
};

// Only LiteRT buffer metadata is mocked. All device memory, CUDA events,
// sampler state, reductions, and result transfers use the real implementation.
struct TestLogits {
  LiteRtTensorBufferType buffer_type = kNvidiaCudaTensorBufferType;
  LiteRtRankedTensorType tensor_type{};
  void* allocation = nullptr;
  size_t allocation_size = 0;
  size_t offset = 0;
  size_t packed_size = 0;
  TestEvent* event = nullptr;

  LiteRtTensorBuffer handle() {
    return reinterpret_cast<LiteRtTensorBuffer>(this);
  }
};

LiteRtRuntimeContext RuntimeContext() {
  LiteRtRuntimeContext context{};
  context.get_environment_options = [](LiteRtEnvironment,
                                       LiteRtEnvironmentOptions*) {
    return kLiteRtStatusErrorNotFound;
  };
  context.get_tensor_buffer_type = [](LiteRtTensorBuffer buffer,
                                      LiteRtTensorBufferType* type) {
    *type = reinterpret_cast<TestLogits*>(buffer)->buffer_type;
    return kLiteRtStatusOk;
  };
  context.get_tensor_buffer_tensor_type = [](LiteRtTensorBuffer buffer,
                                             LiteRtRankedTensorType* type) {
    *type = reinterpret_cast<TestLogits*>(buffer)->tensor_type;
    return kLiteRtStatusOk;
  };
  context.get_tensor_buffer_size = [](LiteRtTensorBuffer buffer, size_t* size) {
    *size = reinterpret_cast<TestLogits*>(buffer)->allocation_size;
    return kLiteRtStatusOk;
  };
  context.get_tensor_buffer_packed_size = [](LiteRtTensorBuffer buffer,
                                             size_t* size) {
    *size = reinterpret_cast<TestLogits*>(buffer)->packed_size;
    return kLiteRtStatusOk;
  };
  context.get_tensor_buffer_offset = [](LiteRtTensorBuffer buffer,
                                        size_t* offset) {
    *offset = reinterpret_cast<TestLogits*>(buffer)->offset;
    return kLiteRtStatusOk;
  };
  context.get_tensor_buffer_custom_tensor_buffer_handle =
      [](LiteRtTensorBuffer buffer, HwMemoryHandle* handle) {
        *handle = reinterpret_cast<TestLogits*>(buffer)->allocation;
        return kLiteRtStatusOk;
      };
  context.has_tensor_buffer_event = [](LiteRtTensorBuffer buffer, bool* has) {
    *has = reinterpret_cast<TestLogits*>(buffer)->event != nullptr;
    return kLiteRtStatusOk;
  };
  context.get_tensor_buffer_event = [](LiteRtTensorBuffer buffer,
                                       LiteRtEvent* event) {
    *event = reinterpret_cast<LiteRtEvent>(
        reinterpret_cast<TestLogits*>(buffer)->event);
    return kLiteRtStatusOk;
  };
  context.wait_event = [](LiteRtEvent event, int64_t timeout) {
    auto* test_event = reinterpret_cast<TestEvent*>(event);
    EXPECT_EQ(timeout, -1);
    ++test_event->waits;
    if (test_event->wait_status != kLiteRtStatusOk) {
      return test_event->wait_status;
    }
    return cudaEventSynchronize(test_event->cuda_event) == cudaSuccess
               ? kLiteRtStatusOk
               : kLiteRtStatusErrorRuntimeFailure;
  };
  return context;
}

class GreedySamplerCApiTest : public testing::Test {
 protected:
  void SetUp() override {
    LiteRtInterface raw_interface = nullptr;
    ASSERT_EQ(LiteRtDispatchQueryInterface(
                  kLiteRtInterfaceBasic,
                  {LITERT_DISPATCH_ABI_VERSION_MAJOR,
                   LITERT_DISPATCH_ABI_VERSION_MINOR,
                   LITERT_DISPATCH_ABI_VERSION_PATCH},
                  &raw_interface),
              kLiteRtStatusOk);
    interface_ = static_cast<const LiteRtDispatchInterface_V1*>(raw_interface);
    ASSERT_NE(interface_, nullptr);
    ASSERT_EQ(interface_->initialize(&runtime_, nullptr, nullptr),
              kLiteRtStatusOk);
    ASSERT_EQ(LiteRtDispatchNvidiaGreedySamplerCreate(&sampler_),
              kLiteRtStatusOk);
  }

  void TearDown() override {
    LiteRtDispatchNvidiaGreedySamplerDestroy(sampler_);
    if (logits_.allocation) cudaFree(logits_.allocation);
    // Do not leave the module pointing to this fixture's runtime function
    // table.
    EXPECT_EQ(interface_->initialize(nullptr, nullptr, nullptr),
              kLiteRtStatusOk);
  }

  void SetLogits(const std::vector<float>& values, int rows, bool fp16,
                 size_t offset = 0) {
    ASSERT_EQ(values.size() % rows, 0);
    if (logits_.allocation) {
      ASSERT_EQ(cudaFree(logits_.allocation), cudaSuccess);
    }
    logits_ = {};
    logits_.tensor_type.element_type =
        fp16 ? kLiteRtElementTypeFloat16 : kLiteRtElementTypeFloat32;
    logits_.tensor_type.layout = LiteRtLayout{
        3, false, {1, rows, static_cast<int32_t>(values.size() / rows)}, {}};
    logits_.packed_size =
        values.size() * (fp16 ? sizeof(__half) : sizeof(float));
    logits_.offset = offset;
    logits_.allocation_size = offset + logits_.packed_size;
    ASSERT_EQ(cudaMalloc(&logits_.allocation, logits_.allocation_size),
              cudaSuccess);
    std::vector<__half> half_values;
    if (fp16) {
      for (float value : values) half_values.push_back(__float2half_rn(value));
    }
    const void* data = fp16 ? static_cast<const void*>(half_values.data())
                            : static_cast<const void*>(values.data());
    ASSERT_EQ(cudaMemcpy(static_cast<uint8_t*>(logits_.allocation) + offset,
                         data, logits_.packed_size, cudaMemcpyHostToDevice),
              cudaSuccess);
    // A copy from pageable memory returns once staged, before the DMA lands;
    // the sampler's non-blocking stream must not read the fresh allocation
    // first (it did, now and then: index 0 of an all-zero row).
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }

  LiteRtStatus Sample(int rows, int vocab, int32_t* ids) {
    return LiteRtDispatchNvidiaGreedySamplerSampleBatched(
        sampler_, logits_.handle(), rows, vocab, ids);
  }

  LiteRtRuntimeContext runtime_ = RuntimeContext();
  const LiteRtDispatchInterface_V1* interface_ = nullptr;
  LiteRtDispatchNvidiaGreedySampler sampler_ = nullptr;
  TestLogits logits_;
};

TEST_F(GreedySamplerCApiTest, SamplesFp16AndFp32RowsAndReusesGrowingState) {
  constexpr int kVocab = 262144;
  for (bool fp16 : {false, true}) {
    for (int rows : {1, 4, 1}) {
      SCOPED_TRACE(testing::Message() << "fp16=" << fp16 << " rows=" << rows);
      std::vector<float> values(rows * kVocab, -4.0f);
      for (int row = 0; row < rows; ++row) {
        // Equal maxima in distant reduction blocks must select the first.
        values[row * kVocab + 17 + row] = 9.0f;
        values[row * kVocab + kVocab - 1] = 9.0f;
      }
      SetLogits(values, rows, fp16, /*offset=*/16);
      ASSERT_FALSE(HasFatalFailure());
      std::array<int32_t, 6> ids;
      ids.fill(-123);
      ASSERT_EQ(Sample(rows, kVocab, ids.data() + 1), kLiteRtStatusOk);
      EXPECT_EQ(ids[0], -123);
      EXPECT_EQ(ids[rows + 1], -123);
      for (int row = 0; row < rows; ++row) EXPECT_EQ(ids[row + 1], 17 + row);
      // Sampling must not modify its input, including FP16 bit patterns.
      if (fp16) {
        std::vector<__half> actual(values.size());
        ASSERT_EQ(cudaMemcpy(actual.data(),
                             static_cast<uint8_t*>(logits_.allocation) + 16,
                             logits_.packed_size, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (size_t i = 0; i < values.size(); ++i) {
          ASSERT_EQ(__half2float(actual[i]), values[i]) << "index=" << i;
        }
      } else {
        std::vector<float> actual(values.size());
        ASSERT_EQ(cudaMemcpy(actual.data(),
                             static_cast<uint8_t*>(logits_.allocation) + 16,
                             logits_.packed_size, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(actual, values);
      }
    }
  }
}

TEST_F(GreedySamplerCApiTest, PreservesNanInfinityAndFp16TieRulesPerRow) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (bool fp16 : {false, true}) {
    SetLogits({nan, inf, 3, 4, -1, nan, inf, inf, -inf, -inf, -inf, -inf, -4, 9,
               2, 9},
              4, fp16);
    ASSERT_FALSE(HasFatalFailure());
    std::array<int32_t, 4> ids;
    ASSERT_EQ(Sample(4, 4, ids.data()), kLiteRtStatusOk);
    EXPECT_EQ(ids, (std::array<int32_t, 4>{0, 2, 0, 1}));
  }
  SetLogits({1.0f, 1.0001f, -3.0f}, 1, true);
  ASSERT_FALSE(HasFatalFailure());
  int32_t id = -1;
  ASSERT_EQ(Sample(1, 3, &id), kLiteRtStatusOk);
  EXPECT_EQ(id,
            0);  // The first two distinct FP32 values round to the same FP16.
}

TEST_F(GreedySamplerCApiTest, RetainsSingleRowFp32Abi) {
  SetLogits({-4, 7, 7, 2}, 1, false);
  ASSERT_FALSE(HasFatalFailure());
  int32_t id = -1;
  ASSERT_EQ(LiteRtDispatchNvidiaGreedySamplerSampleF32(
                sampler_, logits_.handle(), 4, &id),
            kLiteRtStatusOk);
  EXPECT_EQ(id, 1);
  SetLogits({-4, 7, 7, 2}, 1, true);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(LiteRtDispatchNvidiaGreedySamplerSampleF32(
                sampler_, logits_.handle(), 4, &id),
            kLiteRtStatusErrorUnsupported);
  EXPECT_EQ(id, 1);
  SetLogits({-4, 7, 7, 2}, 2, false);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(LiteRtDispatchNvidiaGreedySamplerSampleF32(
                sampler_, logits_.handle(), 2, &id),
            kLiteRtStatusErrorUnsupported);
}

TEST_F(GreedySamplerCApiTest, AcceptsCanonicalDenseStridesButRejectsPadding) {
  for (bool fp16 : {false, true}) {
    for (int rows : {1, 4}) {
      SCOPED_TRACE(testing::Message() << "fp16=" << fp16 << " rows=" << rows);
      std::vector<float> values(rows * 4, -4.0f);
      for (int row = 0; row < rows; ++row) values[row * 4 + row] = 9.0f;
      SetLogits(values, rows, fp16);
      ASSERT_FALSE(HasFatalFailure());
      logits_.tensor_type.layout.has_strides = true;
      logits_.tensor_type.layout.strides[0] = rows * 4;
      logits_.tensor_type.layout.strides[1] = 4;
      logits_.tensor_type.layout.strides[2] = 1;
      std::array<int32_t, 4> ids;
      ids.fill(-123);
      ASSERT_EQ(Sample(rows, 4, ids.data()), kLiteRtStatusOk);
      for (int row = 0; row < rows; ++row) EXPECT_EQ(ids[row], row);
      if (!fp16 && rows == 1) {
        EXPECT_EQ(LiteRtDispatchNvidiaGreedySamplerSampleF32(
                      sampler_, logits_.handle(), 4, ids.data()),
                  kLiteRtStatusErrorUnsupported);
      }
      ++logits_.tensor_type.layout.strides[1];
      ids.fill(-123);
      EXPECT_EQ(Sample(rows, 4, ids.data()), kLiteRtStatusErrorUnsupported);
      EXPECT_EQ(ids, (std::array<int32_t, 4>{-123, -123, -123, -123}));
    }
  }
}

TEST_F(GreedySamplerCApiTest,
       RejectsUnsupportedTypesLayoutsAndNonDeviceMemory) {
  SetLogits({-4, 7, 7, 2}, 1, false);
  ASSERT_FALSE(HasFatalFailure());
  const auto original = logits_;
  int32_t id = -123;
  for (LiteRtElementType type :
       {kLiteRtElementTypeBFloat16, kLiteRtElementTypeInt32}) {
    logits_.tensor_type.element_type = type;
    EXPECT_EQ(Sample(1, 4, &id), kLiteRtStatusErrorUnsupported);
  }
  logits_ = original;
  logits_.tensor_type.layout.has_strides = true;
  EXPECT_EQ(Sample(1, 4, &id), kLiteRtStatusErrorUnsupported);
  logits_ = original;
  logits_.tensor_type.layout.dimensions[0] = 2;
  EXPECT_EQ(Sample(1, 4, &id), kLiteRtStatusErrorUnsupported);
  logits_ = original;
  EXPECT_EQ(Sample(2, 4, &id), kLiteRtStatusErrorUnsupported);
  EXPECT_EQ(Sample(1, 3, &id), kLiteRtStatusErrorUnsupported);
  logits_.buffer_type = kLiteRtTensorBufferTypeHostMemory;
  EXPECT_EQ(Sample(1, 4, &id), kLiteRtStatusErrorUnsupported);
  logits_ = original;
  std::array<float, 4> host_values{};
  logits_.allocation = host_values.data();
  EXPECT_EQ(Sample(1, 4, &id), kLiteRtStatusErrorUnsupported);
  logits_ = original;
  --logits_.packed_size;
  EXPECT_EQ(Sample(1, 4, &id), kLiteRtStatusErrorUnsupported);
  logits_ = original;
  logits_.offset = 1;
  ++logits_.allocation_size;
  EXPECT_EQ(Sample(1, 4, &id), kLiteRtStatusErrorInvalidArgument);
  logits_ = original;
  --logits_.allocation_size;
  EXPECT_EQ(Sample(1, 4, &id), kLiteRtStatusErrorInvalidArgument);
  logits_ = original;
  EXPECT_EQ(id, -123);
  // A rejected request must not poison the sampler's next valid call.
  ASSERT_EQ(Sample(1, 4, &id), kLiteRtStatusOk);
  EXPECT_EQ(id, 1);
}

// A constraint's mask reaches CUDA logits through a host lock, and the
// dispatch uploads such a write only when an invocation next reads the
// buffer: the sampler has to upload it itself before its kernel reads (the
// fused argmax of an invocation is stale after such a write too).
TEST_F(GreedySamplerCApiTest, SeesHostWritesMadeThroughTheBufferHandlers) {
  LiteRtInterface raw_handlers = nullptr;
  ASSERT_EQ(
      LiteRtDispatchQueryInterface(
          kLiteRtInterfaceCustomTensorBufferHandlers,
          {LITERT_DISPATCH_ABI_VERSION_MAJOR, LITERT_DISPATCH_ABI_VERSION_MINOR,
           LITERT_DISPATCH_ABI_VERSION_PATCH},
          &raw_handlers),
      kLiteRtStatusOk);
  const auto* handlers =
      static_cast<const LiteRtCustomTensorBufferHandlersDef_V1*>(raw_handlers);
  ASSERT_NE(handlers, nullptr);
  // The default; set so that an environment uploading at unlock still tests
  // the deferred upload.
  ASSERT_EQ(setenv("LITERT_NVIDIA_DISPATCH_DEFERRED_UPLOAD", "1", 1), 0);
  constexpr int kVocab = 4096;
  const size_t bytes = kVocab * sizeof(float);
  LiteRtRankedTensorType type{};
  type.element_type = kLiteRtElementTypeFloat32;
  type.layout = LiteRtLayout{3, false, {1, 1, kVocab}, {}};
  HwMemoryInfoPtr info = nullptr;
  ASSERT_EQ(
      handlers->create_func(nullptr, nullptr, &type,
                            kNvidiaCudaTensorBufferType, bytes, bytes, &info),
      kLiteRtStatusOk);
  ASSERT_NE(info, nullptr);
  ASSERT_EQ(handlers->clear_func(info), kLiteRtStatusOk);
  logits_ = {};
  logits_.tensor_type = type;
  logits_.allocation = info->memory_handle;
  logits_.allocation_size = bytes;
  logits_.packed_size = bytes;
  const auto write_favorite = [&](int favorite) {
    void* host = nullptr;
    ASSERT_EQ(
        handlers->lock_func(info, kLiteRtTensorBufferLockModeWrite, &host),
        kLiteRtStatusOk);
    std::vector<float> values(kVocab, 0.0f);
    values[favorite] = 5.0f;
    std::memcpy(host, values.data(), bytes);
    ASSERT_EQ(handlers->unlock_func(info), kLiteRtStatusOk);
  };
  int32_t id = -1;
  write_favorite(300);
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(Sample(1, kVocab, &id), kLiteRtStatusOk);
  EXPECT_EQ(id, 300);
  write_favorite(7);
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(Sample(1, kVocab, &id), kLiteRtStatusOk);
  EXPECT_EQ(id, 7);
  // The handlers own the allocation; the fixture must not free it.
  logits_.allocation = nullptr;
  ASSERT_EQ(handlers->destroy_func(info), kLiteRtStatusOk);
}

TEST_F(GreedySamplerCApiTest, RejectsInvalidArgumentsWithoutWritingIds) {
  SetLogits({1, 2}, 1, false);
  ASSERT_FALSE(HasFatalFailure());
  int32_t id = -123;
  EXPECT_EQ(Sample(0, 2, &id), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(Sample(1, 0, &id), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(Sample(1, 2, nullptr), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtDispatchNvidiaGreedySamplerSampleBatched(
                nullptr, logits_.handle(), 1, 2, &id),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtDispatchNvidiaGreedySamplerSampleBatched(sampler_, nullptr, 1,
                                                           2, &id),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtDispatchNvidiaGreedySamplerSampleBatched(
                sampler_, logits_.handle(), 1,
                std::numeric_limits<size_t>::max(), &id),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(id, -123);
}

TEST_F(GreedySamplerCApiTest, RejectsIncompleteRuntimeContext) {
  SetLogits({1, 2}, 1, false);
  ASSERT_FALSE(HasFatalFailure());
  const auto get_size = runtime_.get_tensor_buffer_size;
  runtime_.get_tensor_buffer_size = nullptr;
  int32_t id = -123;
  EXPECT_EQ(Sample(1, 2, &id), kLiteRtStatusErrorUnsupported);
  EXPECT_EQ(id, -123);
  runtime_.get_tensor_buffer_size = get_size;
  ASSERT_EQ(Sample(1, 2, &id), kLiteRtStatusOk);
  EXPECT_EQ(id, 1);
}

TEST_F(GreedySamplerCApiTest,
       RejectsChangingDeviceWithoutChangingCurrentDevice) {
  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  if (device_count < 2) GTEST_SKIP() << "Requires two CUDA devices";
  SetLogits({1, 2}, 1, false);
  ASSERT_FALSE(HasFatalFailure());
  int32_t id = -123;
  ASSERT_EQ(Sample(1, 2, &id), kLiteRtStatusOk);
  int original_device = -1;
  ASSERT_EQ(cudaGetDevice(&original_device), cudaSuccess);
  const int other_device = (original_device + 1) % device_count;
  ASSERT_EQ(cudaSetDevice(other_device), cudaSuccess);
  void* other_allocation = nullptr;
  const cudaError_t allocation_status =
      cudaMalloc(&other_allocation, logits_.allocation_size);
  if (allocation_status != cudaSuccess) {
    cudaSetDevice(original_device);
    FAIL() << cudaGetErrorString(allocation_status);
  }
  void* original_allocation = logits_.allocation;
  logits_.allocation = other_allocation;
  id = -123;
  EXPECT_EQ(Sample(1, 2, &id), kLiteRtStatusErrorUnsupported);
  EXPECT_EQ(id, -123);
  int current_device = -1;
  EXPECT_EQ(cudaGetDevice(&current_device), cudaSuccess);
  EXPECT_EQ(current_device, other_device);
  logits_.allocation = original_allocation;
  EXPECT_EQ(cudaFree(other_allocation), cudaSuccess);
  EXPECT_EQ(cudaSetDevice(original_device), cudaSuccess);
  EXPECT_EQ(Sample(1, 2, &id), kLiteRtStatusOk);
  EXPECT_EQ(id, 1);
}

TEST_F(GreedySamplerCApiTest, WaitsForProducerEventAndPropagatesWaitFailure) {
  SetLogits({0, 0, 0, 0}, 1, false);
  ASSERT_FALSE(HasFatalFailure());
  cudaStream_t producer = nullptr;
  TestEvent event;
  ASSERT_EQ(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking),
            cudaSuccess);
  ASSERT_EQ(cudaEventCreateWithFlags(&event.cuda_event, cudaEventDisableTiming),
            cudaSuccess);
  const std::array<float, 4> updated{0, 0, 9, 0};
  ASSERT_EQ(
      cudaMemcpyAsync(logits_.allocation, updated.data(), logits_.packed_size,
                      cudaMemcpyHostToDevice, producer),
      cudaSuccess);
  ASSERT_EQ(cudaEventRecord(event.cuda_event, producer), cudaSuccess);
  logits_.event = &event;
  int32_t id = -123;
  EXPECT_EQ(Sample(1, 4, &id), kLiteRtStatusOk);
  EXPECT_EQ(id, 2);
  EXPECT_EQ(event.waits, 1);
  event.wait_status = kLiteRtStatusErrorRuntimeFailure;
  id = -123;
  EXPECT_EQ(Sample(1, 4, &id), kLiteRtStatusErrorRuntimeFailure);
  EXPECT_EQ(id, -123);
  EXPECT_EQ(event.waits, 2);
  logits_.event = nullptr;
  EXPECT_EQ(cudaEventDestroy(event.cuda_event), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(producer), cudaSuccess);
  EXPECT_EQ(Sample(1, 4, &id), kLiteRtStatusOk);
  EXPECT_EQ(id, 2);
}

}  // namespace
}  // namespace litert::nvidia

// Copyright 2026 The ODML Authors. Licensed under the Apache License,
// Version 2.0.

#include "litert/vendors/nvidia/dispatch/topk_candidates_c_api.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
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

struct Candidate {
  int32_t id;
  float value;
  bool operator==(const Candidate& other) const {
    return id == other.id && value == other.value;
  }
};

// The k largest by logit then index.
std::vector<Candidate> OrderedTopK(const float* values, int count, int k) {
  std::vector<int32_t> order(count);
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

class TopKCandidatesCApiTest : public testing::Test {
 protected:
  void SetUp() override {
    LiteRtInterface raw_interface = nullptr;
    ASSERT_EQ(LiteRtDispatchQueryInterface(kLiteRtInterfaceBasic,
                                           {LITERT_DISPATCH_ABI_VERSION_MAJOR,
                                            LITERT_DISPATCH_ABI_VERSION_MINOR,
                                            LITERT_DISPATCH_ABI_VERSION_PATCH},
                                           &raw_interface),
              kLiteRtStatusOk);
    interface_ = static_cast<const LiteRtDispatchInterface_V1*>(raw_interface);
    ASSERT_NE(interface_, nullptr);
    ASSERT_EQ(interface_->initialize(&runtime_, nullptr, nullptr),
              kLiteRtStatusOk);
    ASSERT_EQ(LiteRtDispatchNvidiaTopKCandidatesCreate(&selector_),
              kLiteRtStatusOk);
  }

  void TearDown() override {
    LiteRtDispatchNvidiaTopKCandidatesDestroy(selector_);
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
    // the selector's non-blocking stream must not read the allocation first.
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }

  LiteRtStatus Select(int rows, int vocab, int k, std::vector<int32_t>& ids,
                      std::vector<float>& values, std::vector<int32_t>& flags) {
    ids.assign(static_cast<size_t>(rows) * k, -7);
    values.assign(static_cast<size_t>(rows) * k, -7.0f);
    flags.assign(rows, -7);
    return LiteRtDispatchNvidiaTopKCandidatesSelect(
        selector_, logits_.handle(), rows, vocab, k, ids.data(), values.data(),
        flags.data());
  }

  std::vector<Candidate> Row(const std::vector<int32_t>& ids,
                             const std::vector<float>& values, int row, int k) {
    std::vector<Candidate> out;
    for (int i = 0; i < k; ++i)
      out.push_back({ids[row * k + i], values[row * k + i]});
    return out;
  }

  LiteRtRuntimeContext runtime_ = RuntimeContext();
  const LiteRtDispatchInterface_V1* interface_ = nullptr;
  LiteRtDispatchNvidiaTopKCandidates selector_ = nullptr;
  TestLogits logits_;
};

std::vector<float> RandomLogits(size_t n, std::mt19937& gen) {
  std::normal_distribution<float> dist(0.0f, 4.0f);
  std::vector<float> values(n);
  for (float& value : values) value = dist(gen);
  return values;
}

constexpr int kVocab = 262144;

TEST_F(TopKCandidatesCApiTest, SelectsEveryRowOfFp32AndFp16Logits) {
  std::mt19937 gen(20261027);
  for (bool fp16 : {false, true}) {
    for (int rows : {1, 4}) {
      std::vector<float> values =
          RandomLogits(static_cast<size_t>(rows) * kVocab, gen);
      if (fp16) {
        for (float& value : values)
          value = __half2float(__float2half_rn(value));
      }
      SetLogits(values, rows, fp16);
      ASSERT_FALSE(HasFatalFailure());
      for (int k : {1, 64, 1024}) {
        std::vector<int32_t> ids;
        std::vector<float> out;
        std::vector<int32_t> flags;
        ASSERT_EQ(Select(rows, kVocab, k, ids, out, flags), kLiteRtStatusOk)
            << "fp16=" << fp16 << " rows=" << rows << " k=" << k;
        for (int row = 0; row < rows; ++row) {
          EXPECT_EQ(flags[row], 0);
          EXPECT_EQ(
              Row(ids, out, row, k),
              OrderedTopK(values.data() + static_cast<size_t>(row) * kVocab,
                          kVocab, k))
              << "fp16=" << fp16 << " rows=" << rows << " k=" << k
              << " row=" << row;
        }
      }
    }
  }
  EXPECT_EQ(LiteRtDispatchNvidiaTopKCandidatesMaxK(), 1024);
}

TEST_F(TopKCandidatesCApiTest, AcceptsAnOffsetViewAndIsDeterministic) {
  std::mt19937 gen(7);
  const std::vector<float> values = RandomLogits(2 * 50000, gen);
  SetLogits(values, 2, /*fp16=*/false, /*offset=*/256);
  ASSERT_FALSE(HasFatalFailure());
  std::vector<int32_t> ids, flags;
  std::vector<float> out;
  ASSERT_EQ(Select(2, 50000, 32, ids, out, flags), kLiteRtStatusOk);
  std::vector<int32_t> again_ids, again_flags;
  std::vector<float> again_out;
  ASSERT_EQ(Select(2, 50000, 32, again_ids, again_out, again_flags),
            kLiteRtStatusOk);
  EXPECT_EQ(ids, again_ids);
  EXPECT_EQ(out, again_out);
  for (int row = 0; row < 2; ++row) {
    EXPECT_EQ(Row(ids, out, row, 32),
              OrderedTopK(values.data() + row * 50000, 50000, 32));
  }
}

TEST_F(TopKCandidatesCApiTest, FlagsARowWithANaN) {
  std::mt19937 gen(11);
  std::vector<float> values = RandomLogits(2 * 4096, gen);
  values[4096 + 9] = std::numeric_limits<float>::quiet_NaN();
  SetLogits(values, 2, /*fp16=*/false);
  ASSERT_FALSE(HasFatalFailure());
  std::vector<int32_t> ids, flags;
  std::vector<float> out;
  ASSERT_EQ(Select(2, 4096, 8, ids, out, flags), kLiteRtStatusOk);
  EXPECT_EQ(flags[0], 0);
  EXPECT_EQ(flags[1], LITERT_DISPATCH_NVIDIA_TOPK_CANDIDATES_NAN);
  EXPECT_EQ(Row(ids, out, 0, 8), OrderedTopK(values.data(), 4096, 8));
}

// A constraint's mask reaches CUDA logits through a host lock, and the
// dispatch uploads such a write only when an invocation next reads the
// buffer: the selector has to upload it itself before its kernel reads.
TEST_F(TopKCandidatesCApiTest, SeesHostWritesMadeThroughTheBufferHandlers) {
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
  constexpr int kSmallVocab = 4096;
  const size_t bytes = kSmallVocab * sizeof(float);
  LiteRtRankedTensorType type{};
  type.element_type = kLiteRtElementTypeFloat32;
  type.layout = LiteRtLayout{3, false, {1, 1, kSmallVocab}, {}};
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
  const auto write_favorites = [&](int first, int second) {
    void* host = nullptr;
    ASSERT_EQ(
        handlers->lock_func(info, kLiteRtTensorBufferLockModeWrite, &host),
        kLiteRtStatusOk);
    std::vector<float> values(kSmallVocab, 0.0f);
    values[first] = 5.0f;
    values[second] = 4.0f;
    std::memcpy(host, values.data(), bytes);
    ASSERT_EQ(handlers->unlock_func(info), kLiteRtStatusOk);
  };
  std::vector<int32_t> ids, flags;
  std::vector<float> out;
  write_favorites(300, 2000);
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(Select(1, kSmallVocab, 2, ids, out, flags), kLiteRtStatusOk);
  EXPECT_EQ(ids, (std::vector<int32_t>{300, 2000}));
  EXPECT_EQ(out, (std::vector<float>{5.0f, 4.0f}));
  write_favorites(7, 1);
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(Select(1, kSmallVocab, 2, ids, out, flags), kLiteRtStatusOk);
  EXPECT_EQ(ids, (std::vector<int32_t>{7, 1}));
  // The handlers own the allocation; the fixture must not free it.
  logits_.allocation = nullptr;
  ASSERT_EQ(handlers->destroy_func(info), kLiteRtStatusOk);
}

TEST_F(TopKCandidatesCApiTest, WaitsForProducerEventAndPropagatesWaitFailure) {
  std::mt19937 gen(5);
  const std::vector<float> values = RandomLogits(4096, gen);
  SetLogits(values, 1, /*fp16=*/false);
  ASSERT_FALSE(HasFatalFailure());
  TestEvent event;
  ASSERT_EQ(cudaEventCreate(&event.cuda_event), cudaSuccess);
  ASSERT_EQ(cudaEventRecord(event.cuda_event, nullptr), cudaSuccess);
  logits_.event = &event;
  std::vector<int32_t> ids, flags;
  std::vector<float> out;
  ASSERT_EQ(Select(1, 4096, 4, ids, out, flags), kLiteRtStatusOk);
  EXPECT_EQ(event.waits, 1);
  EXPECT_EQ(Row(ids, out, 0, 4), OrderedTopK(values.data(), 4096, 4));
  event.wait_status = kLiteRtStatusErrorTimeoutExpired;
  EXPECT_EQ(Select(1, 4096, 4, ids, out, flags),
            kLiteRtStatusErrorTimeoutExpired);
  EXPECT_EQ(event.waits, 2);
  EXPECT_EQ(ids[0], -7) << "nothing written after a failed wait";
  logits_.event = nullptr;
  cudaEventDestroy(event.cuda_event);
}

TEST_F(TopKCandidatesCApiTest, RejectsBadArgumentsAndHostBuffers) {
  std::mt19937 gen(9);
  const std::vector<float> values = RandomLogits(4096, gen);
  SetLogits(values, 1, /*fp16=*/false);
  ASSERT_FALSE(HasFatalFailure());
  std::vector<int32_t> ids, flags;
  std::vector<float> out;
  EXPECT_EQ(Select(1, 4096, 0, ids, out, flags),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(Select(1, 4096, LiteRtDispatchNvidiaTopKCandidatesMaxK() + 1, ids,
                   out, flags),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(Select(2, 2048, 8, ids, out, flags), kLiteRtStatusErrorUnsupported)
      << "the buffer's shape is [1, 1, 4096]";
  EXPECT_EQ(LiteRtDispatchNvidiaTopKCandidatesSelect(
                selector_, logits_.handle(), 1, 4096, 8, nullptr, out.data(),
                flags.data()),
            kLiteRtStatusErrorInvalidArgument);
  logits_.buffer_type = kLiteRtTensorBufferTypeHostMemory;
  EXPECT_EQ(Select(1, 4096, 8, ids, out, flags), kLiteRtStatusErrorUnsupported);
  logits_.buffer_type = kNvidiaCudaTensorBufferType;
  logits_.tensor_type.element_type = kLiteRtElementTypeInt8;
  EXPECT_EQ(Select(1, 4096, 8, ids, out, flags), kLiteRtStatusErrorUnsupported);
  logits_.tensor_type.element_type = kLiteRtElementTypeFloat32;
  EXPECT_EQ(Select(1, 4096, 8, ids, out, flags), kLiteRtStatusOk);
  EXPECT_EQ(Row(ids, out, 0, 8), OrderedTopK(values.data(), 4096, 8));
  EXPECT_EQ(LiteRtDispatchNvidiaTopKCandidatesCreate(nullptr),
            kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
}  // namespace litert::nvidia

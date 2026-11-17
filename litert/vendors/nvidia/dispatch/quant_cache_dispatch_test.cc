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

// The quantized KV caches through the dispatch: a decode block over INT4
// caches (compiler/quant_attention_test_graph.h), compiled by the graph
// builder and run through the public dispatch function table with tensor
// buffers of the model's types. TensorRT has no INT4 tensors at an engine's
// boundary, so the engine declares such a cache as the INT8 bytes that hold
// it, and the dispatch binds the buffer LiteRT allocates for the INT4 tensor,
// which has exactly those bytes (litert::internal::GetNumPackedBytes: two
// values per byte), to it. LiteRT-LM passes one buffer for a cache's input
// and its output: the update happens in it.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/core/util/tensor_type_util.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/nvidia/bytecode.h"
#include "litert/vendors/nvidia/compiler/quant_attention_test_graph.h"
#include "litert/vendors/nvidia/compiler/tensorrt_graph_builder.h"
#include "litert/vendors/nvidia/trtllm/quant_attention_reference.h"

namespace litert::nvidia {
namespace {

using quant_reference::AttentionInputs;
using quant_reference::Caches;
using quant_reference::CompareRows;
using quant_reference::FloatToHalfBits;
using quant_reference::HalfBitsToFloat;
using quant_reference::PackInt4;
using quant_reference::Quantize;
using quant_reference::RowsMatch;
using quant_reference::UnpackInt4;
using quant_test::kDepth;
using quant_test::kGlobal;
using quant_test::kGroup;
using quant_test::kHeads;
using quant_test::kQueryHeads;

constexpr int kCacheLen = 320;
constexpr float kSdpaFill = -45856.0f;  // the decomposition's
constexpr auto kCudaBufferType = static_cast<LiteRtTensorBufferType>(
    kLiteRtTensorBufferTypeUserCustomBuffer + 1);

// Only buffer metadata is mocked: a device allocation of the size LiteRT
// packs the tensor's type into. TensorRT, the engine and the dispatch
// function table are real.
struct TestBuffer {
  void* device_ptr = nullptr;
  LiteRtRankedTensorType type{};
  size_t bytes = 0;
  LiteRtTensorBufferHandle handle = 0;

  LiteRtTensorBuffer buffer() {
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

class QuantCacheDispatchTest : public ::testing::Test {
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
    ASSERT_EQ(interface_->device_context_create(&runtime_, nullptr, &device_),
              kLiteRtStatusOk);
  }

  void TearDown() override {
    if (context_) {
      EXPECT_EQ(interface_->invocation_context_destroy(context_),
                kLiteRtStatusOk);
    }
    for (auto& buffer : buffers_) {
      if (buffer.handle != 0) {
        EXPECT_EQ(interface_->unregister_tensor_buffer(device_, buffer.handle),
                  kLiteRtStatusOk);
      }
    }
    if (device_) {
      EXPECT_EQ(interface_->device_context_destroy(device_), kLiteRtStatusOk);
    }
    for (auto& buffer : buffers_) {
      cudaFree(buffer.device_ptr);
    }
  }

  // A buffer of the bytes LiteRT packs a tensor of this type into.
  TestBuffer* AddBuffer(LiteRtElementType element_type,
                        std::initializer_list<int32_t> dims) {
    TestBuffer buffer;
    buffer.type.element_type = element_type;
    buffer.type.layout.rank = dims.size();
    std::copy(dims.begin(), dims.end(), buffer.type.layout.dimensions);
    const auto bytes = litert::internal::GetNumPackedBytes(buffer.type);
    EXPECT_TRUE(bytes.HasValue());
    buffer.bytes = bytes ? *bytes : 0;
    EXPECT_EQ(cudaMalloc(&buffer.device_ptr, buffer.bytes), cudaSuccess);
    buffers_.push_back(buffer);
    return &buffers_.back();
  }

  template <typename T>
  void Upload(TestBuffer* buffer, const std::vector<T>& host) {
    ASSERT_EQ(host.size() * sizeof(T), buffer->bytes);
    ASSERT_EQ(cudaMemcpy(buffer->device_ptr, host.data(), buffer->bytes,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
  }

  std::vector<uint8_t> Download(TestBuffer* buffer) {
    std::vector<uint8_t> host(buffer->bytes);
    EXPECT_EQ(cudaMemcpy(host.data(), buffer->device_ptr, buffer->bytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return host;
  }

  LiteRtRuntimeContext runtime_ = RuntimeContext();
  const LiteRtDispatchInterface_V1* interface_ = nullptr;
  LiteRtDispatchDeviceContext device_ = nullptr;
  LiteRtDispatchInvocationContext context_ = nullptr;
  // Stable addresses: the dispatch keeps the buffers' pointers.
  std::deque<TestBuffer> buffers_;
};

std::vector<uint16_t> HalfBits(const std::vector<float>& values) {
  std::vector<uint16_t> bits(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    bits[i] = FloatToHalfBits(values[i]);
  }
  return bits;
}

TEST_F(QuantCacheDispatchTest, Int4CachesAreBoundAsTheirBytesInPlace) {
  const quant_test::Scales& scales = kGlobal;
  quant_test::Graph graph;
  quant_test::AddDecodeBlock(&graph, scales, kCacheLen, /*float32=*/false);
  auto built = BuildTensorRtEngine(
      litert::compiler::Subgraph(LrtGetCompilerContext(), &graph.Get()));
  ASSERT_TRUE(built.HasValue()) << built.Error().Message();
  auto bytecode =
      PackTensorRtBytecode("decode", built->input_names, built->output_names,
                           built->engine.data(), built->engine.size());
  ASSERT_TRUE(bytecode.HasValue()) << bytecode.Error().Message();
  LiteRtMemBuffer bytecode_buffer{};
  bytecode_buffer.fd = -1;
  bytecode_buffer.base_addr = bytecode->data();
  bytecode_buffer.size = bytecode->size();
  ASSERT_EQ(interface_->invocation_context_create(
                &runtime_, device_, kLiteRtDispatchExecutableTypeMlModel,
                &bytecode_buffer, "decode", 10, 4, &context_),
            kLiteRtStatusOk);

  // The graph's inputs, with the model's tensor types.
  TestBuffer* k = AddBuffer(kLiteRtElementTypeFloat16, {1, kHeads, 1, kDepth});
  TestBuffer* v = AddBuffer(kLiteRtElementTypeFloat16, {1, 1, kHeads, kDepth});
  TestBuffer* q =
      AddBuffer(kLiteRtElementTypeFloat16, {1, kQueryHeads, 1, kDepth});
  TestBuffer* q2 =
      AddBuffer(kLiteRtElementTypeFloat16, {1, kQueryHeads, 1, kDepth});
  TestBuffer* k_cache =
      AddBuffer(kLiteRtElementTypeInt4, {1, kHeads, kCacheLen, kDepth});
  TestBuffer* v_cache =
      AddBuffer(kLiteRtElementTypeInt4, {1, kHeads, kDepth, kCacheLen});
  TestBuffer* mask = AddBuffer(kLiteRtElementTypeBool, {1, 1, 1, kCacheLen});
  TestBuffer* params = AddBuffer(kLiteRtElementTypeInt32, {1, 1, 1, 7});
  TestBuffer* k_start = AddBuffer(kLiteRtElementTypeInt32, {4});
  TestBuffer* v_start = AddBuffer(kLiteRtElementTypeInt32, {4});
  TestBuffer* attention[2] = {
      AddBuffer(kLiteRtElementTypeFloat16, {1, kQueryHeads, 1, kDepth}),
      AddBuffer(kLiteRtElementTypeFloat16, {1, kQueryHeads, 1, kDepth})};
  // An INT4 cache is half a byte per value.
  const size_t cache_bytes =
      static_cast<size_t>(kHeads) * kCacheLen * kDepth / 2;
  ASSERT_EQ(k_cache->bytes, cache_bytes);
  ASSERT_EQ(v_cache->bytes, cache_bytes);

  // The dispatch asks for a device buffer of exactly those bytes, for the
  // cache as an input and as an output.
  for (const bool output : {false, true}) {
    for (int i = 0; i < 2; ++i) {
      TestBuffer* cache = i == 0 ? k_cache : v_cache;
      LiteRtTensorBufferRequirements requirements = nullptr;
      ASSERT_EQ(output ? interface_->get_output_requirements(
                             context_, i, &cache->type, &requirements)
                       : interface_->get_input_requirements(
                             context_, 4 + i, &cache->type, &requirements),
                kLiteRtStatusOk);
      size_t size = 0;
      ASSERT_EQ(
          LiteRtGetTensorBufferRequirementsBufferSize(requirements, &size),
          kLiteRtStatusOk);
      EXPECT_EQ(size, cache_bytes);
      LiteRtTensorBufferType first_type;
      ASSERT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                    requirements, 0, &first_type),
                kLiteRtStatusOk);
      EXPECT_EQ(first_type, kCudaBufferType);
      LiteRtDestroyTensorBufferRequirements(requirements);
    }
  }

  for (auto& buffer : buffers_) {
    ASSERT_EQ(interface_->register_tensor_buffer(device_, buffer.buffer(),
                                                 &buffer.handle),
              kLiteRtStatusOk);
  }
  int input = 0;
  for (TestBuffer* buffer :
       {k, v, q, q2, k_cache, v_cache, mask, params, k_start, v_start}) {
    ASSERT_EQ(interface_->attach_input(context_, input++, buffer->handle),
              kLiteRtStatusOk);
  }
  // The caches' outputs are their inputs' buffers, as LiteRT-LM passes them.
  int output = 0;
  for (TestBuffer* buffer : {k_cache, v_cache, attention[0], attention[1]}) {
    ASSERT_EQ(interface_->attach_output(context_, output++, buffer->handle),
              kLiteRtStatusOk);
  }

  uint32_t state = 77;
  const auto next = [&]() {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>((state >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
  };
  Caches caches(kHeads, kCacheLen, kDepth);
  for (auto& value : caches.k) {
    value = static_cast<int8_t>(std::lround(next() * 7.4f));
  }
  for (auto& value : caches.v) {
    value = static_cast<int8_t>(std::lround(next() * 7.4f));
  }
  Upload(k_cache, PackInt4(caches.k));
  Upload(v_cache, PackInt4(caches.v));
  const auto random_bits = [&](size_t count, float range) {
    std::vector<float> values(count);
    for (auto& value : values) {
      value = next() * range;
    }
    return HalfBits(values);
  };
  const auto quantize = [](const std::vector<uint16_t>& bits, float scale,
                           int quant_bits) {
    std::vector<int8_t> values(bits.size());
    for (size_t i = 0; i < bits.size(); ++i) {
      values[i] = Quantize(HalfBitsToFloat(bits[i]), scale, quant_bits);
    }
    return values;
  };
  // The caches stay in their buffers from step to step.
  for (const int step : {0, 1, 200, kCacheLen - 1}) {
    SCOPED_TRACE(::testing::Message() << "step=" << step);
    const size_t row_count = static_cast<size_t>(kHeads) * kDepth;
    const size_t query_count = static_cast<size_t>(kQueryHeads) * kDepth;
    const auto k_bits = random_bits(row_count, 9.5f * scales.k);
    const auto v_bits = random_bits(row_count, 9.5f * scales.v);
    const std::vector<uint16_t> q_bits[2] = {
        random_bits(query_count, 110.0f * scales.q),
        random_bits(query_count, 110.0f * scales.q)};
    std::vector<uint8_t> mask_step(kCacheLen, 0);
    std::fill_n(mask_step.begin(), step + 1, 1);
    Upload(k, k_bits);
    Upload(v, v_bits);
    Upload(q, q_bits[0]);
    Upload(q2, q_bits[1]);
    Upload(mask, mask_step);
    Upload(params, std::vector<int32_t>{step, step + 1, step + 1, 0, 0, 0, 0});
    Upload(k_start, std::vector<int32_t>{0, 0, step, 0});
    Upload(v_start, std::vector<int32_t>{0, 0, 0, step});
    ASSERT_EQ(interface_->invoke(context_), kLiteRtStatusOk);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    quant_reference::CacheUpdate(&caches, quantize(k_bits, scales.k, 4),
                                 quantize(v_bits, scales.v, 4), 1, step, step);
    EXPECT_EQ(UnpackInt4(Download(k_cache), caches.k.size()), caches.k);
    EXPECT_EQ(UnpackInt4(Download(v_cache), caches.v.size()), caches.v);
    for (int reader = 0; reader < 2; ++reader) {
      const std::vector<int8_t> quantized =
          quantize(q_bits[reader], scales.q, 8);
      AttentionInputs reference;
      reference.q = &quantized;
      reference.rows = kGroup;
      reference.q_scale = scales.q;
      reference.k_scale = scales.k;
      reference.v_scale = scales.v;
      reference.caches = &caches;
      reference.mask = &mask_step;
      reference.mask_rows = 1;
      reference.fill = kSdpaFill;
      const auto expected = quant_reference::Attention(reference);
      const std::vector<uint8_t> bytes = Download(attention[reader]);
      std::vector<float> actual(expected.size());
      for (size_t i = 0; i < actual.size(); ++i) {
        uint16_t bits = 0;
        std::memcpy(&bits, bytes.data() + 2 * i, 2);
        actual[i] = HalfBitsToFloat(bits);
      }
      const auto errors = CompareRows(actual, expected, kDepth, 0.02f);
      EXPECT_TRUE(RowsMatch(errors))
          << "reader " << reader << ": " << errors.rows_off << " of "
          << errors.rows << " rows are off, the largest error is "
          << errors.max_error;
    }
  }
}

}  // namespace
}  // namespace litert::nvidia

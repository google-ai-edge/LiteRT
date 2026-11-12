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

#include "ml_drift_delegate/delegate/composite/add_values_to_cache_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/kernels/tests/kernel_test.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/testing_util.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"

namespace litert::ml_drift {
namespace {

using ::testing::Combine;
using ::testing::FloatEq;
using ::testing::Pointwise;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class AddValuesToCacheFloatTest : public ::ml_drift::FloatTest {
 public:
  void SetUp() override {
    if (!exec_env) {
      GTEST_SKIP() << "TestExecutionEnvironment not initialized.";
    }
  }
};

absl::Status RunAddValuesToCacheTest(
    ::ml_drift::TestExecutionEnvironment& env,
    ::ml_drift::CalculationsPrecision precision,
    ::ml_drift::TensorStorageType storage,
    ::ml_drift::DataType output_data_type, int k_update_ts_idx = 2,
    int v_update_ts_idx = 2) {
  constexpr int kCacheSize = 64;
  constexpr int kBatchSize = 2;
  constexpr int kInputWidth = 2;
  constexpr int kHeadSize = 8;
  constexpr int kKVCacheInputSliceSize = kBatchSize * kInputWidth * kHeadSize;
  constexpr int kKVCacheFullSize = kCacheSize * kInputWidth * kHeadSize;

  ::ml_drift::OperationDef op_def;
  ::ml_drift::DataType input_data_type =
      ::ml_drift::DeduceDataTypeFromPrecision(precision);

  // Updates are [T, D] (ts_idx = 2) or [D, T] (ts_idx = 3).
  const auto update_shape = [](int ts_idx) {
    return ts_idx == 3
               ? ::ml_drift::BHWC(1, kBatchSize, kHeadSize, kInputWidth)
               : ::ml_drift::BHWC(1, kBatchSize, kInputWidth, kHeadSize);
  };

  // src_k
  ::ml_drift::TensorDescriptor src_k_desc(input_data_type, storage,
                                          ::ml_drift::Layout::kHWC);
  src_k_desc.SetBHWCShape(update_shape(k_update_ts_idx));
  op_def.src_tensors.push_back(src_k_desc);

  // src_v
  ::ml_drift::TensorDescriptor src_v_desc(input_data_type, storage,
                                          ::ml_drift::Layout::kHWC);
  src_v_desc.SetBHWCShape(update_shape(v_update_ts_idx));
  op_def.src_tensors.push_back(src_v_desc);

  // params (token_index_offset, active_tokens)
  ::ml_drift::TensorDescriptor params_desc(
      ::ml_drift::DataType::kInt32, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kHWC);
  params_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, 2));
  op_def.src_tensors.push_back(params_desc);

  // cache_k
  ::ml_drift::TensorDescriptor cache_k_desc(output_data_type, storage,
                                            ::ml_drift::Layout::kLinear);
  cache_k_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, kKVCacheFullSize));
  op_def.dst_tensors.push_back(cache_k_desc);

  // cache_v
  ::ml_drift::TensorDescriptor cache_v_desc(output_data_type, storage,
                                            ::ml_drift::Layout::kLinear);
  cache_v_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, kKVCacheFullSize));
  op_def.dst_tensors.push_back(cache_v_desc);

  ::ml_drift::Node node = {1, {}};
  AddValuesToCacheAttributes attr;
  attr.cache_size = kCacheSize;
  attr.head_size = kHeadSize;
  attr.kv_cache_batch_size = kBatchSize;
  attr.k_update_ts_idx = k_update_ts_idx;
  attr.v_update_ts_idx = v_update_ts_idx;
  node.operation.attributes = attr;

  ABSL_ASSIGN_OR_RETURN(auto op, CreateAddValuesToCacheFromNode(op_def, node));

  // k_data and v_data are always indexed as [batch][token][channel] below.
  std::vector<float> k_data(kKVCacheInputSliceSize);
  for (int i = 0; i < kKVCacheInputSliceSize; ++i) {
    k_data[i] = static_cast<float>(i + 1);
  }
  std::vector<float> v_data(kKVCacheInputSliceSize);
  for (int i = 0; i < kKVCacheInputSliceSize; ++i) {
    v_data[i] = static_cast<float>(i + 1 + kKVCacheInputSliceSize);
  }
  // Rearranges [batch][token][channel] data into the upload layout.
  const auto to_upload = [](const std::vector<float>& data, int ts_idx) {
    if (ts_idx != 3) return data;
    std::vector<float> upload(data.size());
    for (int batch = 0; batch < kBatchSize; ++batch) {
      for (int x = 0; x < kInputWidth; ++x) {
        for (int c = 0; c < kHeadSize; ++c) {
          upload[(batch * kHeadSize + c) * kInputWidth + x] =
              data[(batch * kInputWidth + x) * kHeadSize + c];
        }
      }
    }
    return upload;
  };
  const std::vector<float> k_upload = to_upload(k_data, k_update_ts_idx);
  const std::vector<float> v_upload = to_upload(v_data, v_update_ts_idx);

  constexpr int kStartIndex = 10;
  std::vector<int32_t> params_data = {kStartIndex, kStartIndex + kBatchSize};

  std::vector<::ml_drift::TensorDescriptor*> src_cpu = {
      &src_k_desc, &src_v_desc, &params_desc};
  src_k_desc.UploadData(k_upload.data());
  src_v_desc.UploadData(v_upload.data());
  params_desc.UploadData(params_data.data());

  std::vector<::ml_drift::TensorDescriptor*> dst_cpu = {&cache_k_desc,
                                                        &cache_v_desc};

  // Pre-fill with zeros
  std::vector<float> zero_cache(kKVCacheFullSize, 0.0f);
  cache_k_desc.UploadData(zero_cache.data());
  cache_v_desc.UploadData(zero_cache.data());

  ABSL_RETURN_IF_ERROR(
      env.ExecuteGPUOperation(src_cpu, dst_cpu, std::move(op)));

  std::vector<float> cache_k_result(kKVCacheFullSize);
  cache_k_desc.DownloadData(cache_k_result.data());

  std::vector<float> cache_v_result(kKVCacheFullSize);
  cache_v_desc.DownloadData(cache_v_result.data());

  // CPU reference of the packed layouts the kernel writes:
  //   cache_k - kOSpatialIOGroupO4I4, O = cache_size, I = head_size
  //   cache_v - kOSpatialIOGroupI4O4, O = head_size,  I = cache_size
  // Slots that the update does not cover must stay zero, so comparing the full
  // cache also catches a grid that is dispatched too large or too small.
  constexpr int kHeadSizeSlices = (kHeadSize + 3) / 4;
  constexpr int kCacheSizeSlices = (kCacheSize + 3) / 4;
  const int active_tokens = params_data[1];
  std::vector<float> expected_cache_k(kKVCacheFullSize, 0.0f);
  std::vector<float> expected_cache_v(kKVCacheFullSize, 0.0f);
  for (int batch = 0; batch < kBatchSize; ++batch) {
    for (int x = 0; x < kInputWidth; ++x) {
      const int token_index = kStartIndex + x;
      if (token_index >= kCacheSize || token_index >= active_tokens) continue;
      const int token_slice = token_index / 4;
      const int token_in_slice = token_index % 4;
      for (int c = 0; c < kHeadSize; ++c) {
        const int src_index = (batch * kInputWidth + x) * kHeadSize + c;
        const int channel_slice = c / 4;
        const int channel_in_slice = c % 4;
        const int k_index =
            (((batch * kHeadSizeSlices + channel_slice) * kCacheSizeSlices +
              token_slice) *
                 4 +
             token_in_slice) *
                4 +
            channel_in_slice;
        expected_cache_k[k_index] = k_data[src_index];
        const int v_index =
            (((batch * kCacheSizeSlices + token_slice) * kHeadSizeSlices +
              channel_slice) *
                 4 +
             token_in_slice) *
                4 +
            channel_in_slice;
        expected_cache_v[v_index] = v_data[src_index];
      }
    }
  }

  EXPECT_THAT(cache_k_result, Pointwise(FloatEq(), expected_cache_k));
  EXPECT_THAT(cache_v_result, Pointwise(FloatEq(), expected_cache_v));
  return absl::OkStatus();
}

TEST_P(AddValuesToCacheFloatTest, Float32Cache) {
  if (!exec_env->IsStorageSupported(storage(),
                                    ::ml_drift::DataType::kFloat32)) {
    GTEST_SKIP() << "Unsupported storage type: "
                 << ::ml_drift::ToString(storage());
  }
  ASSERT_OK(RunAddValuesToCacheTest(*exec_env, precision(), storage(),
                                    ::ml_drift::DataType::kFloat32));
}

TEST_P(AddValuesToCacheFloatTest, Float32CacheVUpdateInCacheLayout) {
  if (!exec_env->IsStorageSupported(storage(),
                                    ::ml_drift::DataType::kFloat32)) {
    GTEST_SKIP() << "Unsupported storage type: "
                 << ::ml_drift::ToString(storage());
  }
  ASSERT_OK(RunAddValuesToCacheTest(*exec_env, precision(), storage(),
                                    ::ml_drift::DataType::kFloat32,
                                    /*k_update_ts_idx=*/2,
                                    /*v_update_ts_idx=*/3));
}

TEST_P(AddValuesToCacheFloatTest, Float32CacheKUpdateTransposed) {
  if (!exec_env->IsStorageSupported(storage(),
                                    ::ml_drift::DataType::kFloat32)) {
    GTEST_SKIP() << "Unsupported storage type: "
                 << ::ml_drift::ToString(storage());
  }
  ASSERT_OK(RunAddValuesToCacheTest(*exec_env, precision(), storage(),
                                    ::ml_drift::DataType::kFloat32,
                                    /*k_update_ts_idx=*/3,
                                    /*v_update_ts_idx=*/2));
}

TEST_P(AddValuesToCacheFloatTest, Float32CacheBothUpdatesTransposed) {
  if (!exec_env->IsStorageSupported(storage(),
                                    ::ml_drift::DataType::kFloat32)) {
    GTEST_SKIP() << "Unsupported storage type: "
                 << ::ml_drift::ToString(storage());
  }
  ASSERT_OK(RunAddValuesToCacheTest(*exec_env, precision(), storage(),
                                    ::ml_drift::DataType::kFloat32,
                                    /*k_update_ts_idx=*/3,
                                    /*v_update_ts_idx=*/3));
}

uint8_t QuantizeInt4Nibble(float val, float scale) {
  const float clamped = std::clamp(val, -8.0f * scale, 7.0f * scale);
  const float q =
      std::clamp(std::nearbyint(clamped * (1.0f / scale)) + 8.0f, 0.0f, 15.0f);
  return static_cast<uint8_t>(q);
}

uint8_t QuantizeInt8Byte(float val, float scale) {
  const float min_v = -128.0f * scale;
  const float max_v = 127.0f * scale;
  const float clamped = std::clamp(val, min_v, max_v);
  const float q = std::nearbyint((clamped - min_v) * (1.0f / scale));
  return static_cast<uint8_t>(std::clamp(q, 0.0f, 255.0f));
}

absl::Status RunAddValuesToCacheInt4Test(
    ::ml_drift::TestExecutionEnvironment& env,
    ::ml_drift::CalculationsPrecision precision, int k_update_ts_idx = 2,
    int v_update_ts_idx = 2, bool is_ring_buffer = false) {
  constexpr int kCacheSize = 16;
  constexpr int kBatchSize = 2;
  constexpr int kInputWidth = 3;
  constexpr int kHeadSize = 16;  // 2 packed uchar4 slices per token
  constexpr float kScaleK = 0.25f;
  constexpr float kScaleV = 0.5f;
  constexpr int kInputElements = kBatchSize * kInputWidth * kHeadSize;
  constexpr int kPackedCacheBytes = (kBatchSize * kCacheSize * kHeadSize) / 2;

  const ::ml_drift::DataType input_data_type =
      ::ml_drift::DeduceDataTypeFromPrecision(precision);
  const auto update_shape = [](int ts_idx) {
    return ts_idx == 3
               ? ::ml_drift::BHWC(1, kBatchSize, kHeadSize, kInputWidth)
               : ::ml_drift::BHWC(1, kBatchSize, kInputWidth, kHeadSize);
  };

  ::ml_drift::OperationDef op_def;
  ::ml_drift::TensorDescriptor src_k_desc(
      input_data_type, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kHWC);
  src_k_desc.SetBHWCShape(update_shape(k_update_ts_idx));
  op_def.src_tensors.push_back(src_k_desc);

  ::ml_drift::TensorDescriptor src_v_desc(
      input_data_type, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kHWC);
  src_v_desc.SetBHWCShape(update_shape(v_update_ts_idx));
  op_def.src_tensors.push_back(src_v_desc);

  ::ml_drift::TensorDescriptor params_desc(
      ::ml_drift::DataType::kInt32, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kHWC);
  params_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, 4));
  op_def.src_tensors.push_back(params_desc);

  const ::ml_drift::BHWC packed_k_shape = PackInt4KvCacheShape(
      ::ml_drift::BHWC(1, kBatchSize, kCacheSize, kHeadSize),
      /*cache_ts_idx=*/2);
  const ::ml_drift::BHWC packed_v_shape = PackInt4KvCacheShape(
      ::ml_drift::BHWC(1, kBatchSize, kHeadSize, kCacheSize),
      /*cache_ts_idx=*/3);

  ::ml_drift::TensorDescriptor cache_k_desc(
      ::ml_drift::DataType::kUint8, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kHWC);
  cache_k_desc.SetBHWCShape(packed_k_shape);
  EXPECT_EQ(cache_k_desc.GetMemorySizeInBytes(), kPackedCacheBytes);
  op_def.dst_tensors.push_back(cache_k_desc);

  ::ml_drift::TensorDescriptor cache_v_desc(
      ::ml_drift::DataType::kUint8, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kHWC);
  cache_v_desc.SetBHWCShape(packed_v_shape);
  EXPECT_EQ(cache_v_desc.GetMemorySizeInBytes(), kPackedCacheBytes);
  op_def.dst_tensors.push_back(cache_v_desc);

  ::ml_drift::Node node = {1, {}};
  AddValuesToCacheAttributes attr;
  attr.cache_size = kCacheSize;
  attr.head_size = kHeadSize;
  attr.kv_cache_batch_size = kBatchSize;
  attr.kv_quant_bits = 4;
  attr.scale_k = kScaleK;
  attr.scale_v = kScaleV;
  attr.is_ring_buffer = is_ring_buffer;
  attr.k_update_ts_idx = k_update_ts_idx;
  attr.v_update_ts_idx = v_update_ts_idx;
  node.operation.attributes = attr;

  ABSL_ASSIGN_OR_RETURN(auto op, CreateAddValuesToCacheFromNode(op_def, node));

  // Values span negative saturation (<-8*scale), exact integers, zero, and
  // positive saturation (>7*scale). Avoid exact half-integers so FP16 input
  // conversion matches FP32 deterministically.
  std::vector<float> k_data(kInputElements);
  std::vector<float> v_data(kInputElements);
  for (int i = 0; i < kInputElements; ++i) {
    const int centered = (i % 21) - 10;  // [-10, +10] -> clamps to [-8, +7]
    k_data[i] = static_cast<float>(centered) * kScaleK;
    v_data[i] = static_cast<float>(-centered) * kScaleV;
  }

  const auto to_upload = [](const std::vector<float>& data, int ts_idx) {
    if (ts_idx != 3) return data;
    std::vector<float> upload(data.size());
    for (int batch = 0; batch < kBatchSize; ++batch) {
      for (int x = 0; x < kInputWidth; ++x) {
        for (int c = 0; c < kHeadSize; ++c) {
          upload[(batch * kHeadSize + c) * kInputWidth + x] =
              data[(batch * kInputWidth + x) * kHeadSize + c];
        }
      }
    }
    return upload;
  };
  const std::vector<float> k_upload = to_upload(k_data, k_update_ts_idx);
  const std::vector<float> v_upload = to_upload(v_data, v_update_ts_idx);

  // Start at an odd index (or wrap boundary for ring buffer) and write only 2
  // of the 3 input tokens (via active_tokens / update_length) to verify both
  // odd-slot safety and active-bound masking.
  const int start_index = is_ring_buffer ? (kCacheSize - 1) : 5;
  const int valid_tokens = 2;
  std::vector<int32_t> params_data = {
      start_index, start_index + valid_tokens, start_index + valid_tokens,
      valid_tokens};

  src_k_desc.UploadData(k_upload.data());
  src_v_desc.UploadData(v_upload.data());
  params_desc.UploadData(params_data.data());

  // Seed cache bytes with a non-zero sentinel pattern so untouched token slots
  // (including neighbors sharing the same 4-token O4/I4 group!) are verified
  // not to be clobbered.
  std::vector<uint8_t> init_cache_k(kPackedCacheBytes);
  std::vector<uint8_t> init_cache_v(kPackedCacheBytes);
  for (int i = 0; i < kPackedCacheBytes; ++i) {
    init_cache_k[i] = static_cast<uint8_t>((i * 37 + 0x5A) & 0xFF);
    init_cache_v[i] = static_cast<uint8_t>((i * 53 + 0xA5) & 0xFF);
  }
  cache_k_desc.SetData(std::vector<uint8_t>(init_cache_k));
  cache_v_desc.SetData(std::vector<uint8_t>(init_cache_v));

  std::vector<::ml_drift::TensorDescriptor*> src_cpu = {
      &src_k_desc, &src_v_desc, &params_desc};
  std::vector<::ml_drift::TensorDescriptor*> dst_cpu = {&cache_k_desc,
                                                        &cache_v_desc};
  ABSL_RETURN_IF_ERROR(
      env.ExecuteGPUOperation(src_cpu, dst_cpu, std::move(op)));

  std::vector<uint8_t> expected_k = init_cache_k;
  std::vector<uint8_t> expected_v = init_cache_v;
  constexpr int kPackedHeadSlices = kHeadSize / 8;
  constexpr int kCacheSlices = kCacheSize / 4;
  for (int batch = 0; batch < kBatchSize; ++batch) {
    for (int x = 0; x < kInputWidth; ++x) {
      if (is_ring_buffer && x >= valid_tokens) continue;
      const int token_index = is_ring_buffer
                                  ? ((start_index + x) % kCacheSize)
                                  : (start_index + x);
      if (!is_ring_buffer &&
          (token_index >= kCacheSize || token_index >= params_data[1])) {
        continue;
      }
      const int token_slice = token_index / 4;
      const int token_in_slice = token_index % 4;
      for (int s = 0; s < kPackedHeadSlices; ++s) {
        const int k_vec4_idx =
            ((batch * kPackedHeadSlices + s) * kCacheSlices + token_slice) * 4 +
            token_in_slice;
        const int v_vec4_idx =
            ((batch * kCacheSlices + token_slice) * kPackedHeadSlices + s) * 4 +
            token_in_slice;
        for (int comp = 0; comp < 4; ++comp) {
          const int c_lo = s * 8 + comp;
          const int c_hi = s * 8 + 4 + comp;
          const int src_lo = (batch * kInputWidth + x) * kHeadSize + c_lo;
          const int src_hi = (batch * kInputWidth + x) * kHeadSize + c_hi;
          const uint8_t k_lo = QuantizeInt4Nibble(k_data[src_lo], kScaleK);
          const uint8_t k_hi = QuantizeInt4Nibble(k_data[src_hi], kScaleK);
          expected_k[k_vec4_idx * 4 + comp] =
              static_cast<uint8_t>(k_lo | (k_hi << 4));

          const uint8_t v_lo = QuantizeInt4Nibble(v_data[src_lo], kScaleV);
          const uint8_t v_hi = QuantizeInt4Nibble(v_data[src_hi], kScaleV);
          expected_v[v_vec4_idx * 4 + comp] =
              static_cast<uint8_t>(v_lo | (v_hi << 4));
        }
      }
    }
  }

  EXPECT_EQ(cache_k_desc.GetData(), absl::MakeConstSpan(expected_k));
  EXPECT_EQ(cache_v_desc.GetData(), absl::MakeConstSpan(expected_v));
  return absl::OkStatus();
}

absl::Status RunAddValuesToCacheInt8Test(
    ::ml_drift::TestExecutionEnvironment& env,
    ::ml_drift::CalculationsPrecision precision) {
  constexpr int kCacheSize = 16;
  constexpr int kBatchSize = 2;
  constexpr int kInputWidth = 2;
  constexpr int kHeadSize = 8;
  constexpr float kScaleK = 0.25f;
  constexpr float kScaleV = 0.5f;
  constexpr int kInputElements = kBatchSize * kInputWidth * kHeadSize;
  constexpr int kCacheBytes = kBatchSize * kCacheSize * kHeadSize;

  const ::ml_drift::DataType input_data_type =
      ::ml_drift::DeduceDataTypeFromPrecision(precision);
  ::ml_drift::OperationDef op_def;
  ::ml_drift::TensorDescriptor src_k_desc(
      input_data_type, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kHWC);
  src_k_desc.SetBHWCShape(
      ::ml_drift::BHWC(1, kBatchSize, kInputWidth, kHeadSize));
  op_def.src_tensors.push_back(src_k_desc);

  ::ml_drift::TensorDescriptor src_v_desc(
      input_data_type, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kHWC);
  src_v_desc.SetBHWCShape(
      ::ml_drift::BHWC(1, kBatchSize, kInputWidth, kHeadSize));
  op_def.src_tensors.push_back(src_v_desc);

  ::ml_drift::TensorDescriptor params_desc(
      ::ml_drift::DataType::kInt32, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kHWC);
  params_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, 2));
  op_def.src_tensors.push_back(params_desc);

  ::ml_drift::TensorDescriptor cache_k_desc(
      ::ml_drift::DataType::kUint8, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kHWC);
  cache_k_desc.SetBHWCShape(
      ::ml_drift::BHWC(1, kBatchSize, kCacheSize, kHeadSize));
  op_def.dst_tensors.push_back(cache_k_desc);

  ::ml_drift::TensorDescriptor cache_v_desc(
      ::ml_drift::DataType::kUint8, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kHWC);
  cache_v_desc.SetBHWCShape(
      ::ml_drift::BHWC(1, kBatchSize, kHeadSize, kCacheSize));
  op_def.dst_tensors.push_back(cache_v_desc);

  ::ml_drift::Node node = {1, {}};
  AddValuesToCacheAttributes attr;
  attr.cache_size = kCacheSize;
  attr.head_size = kHeadSize;
  attr.kv_cache_batch_size = kBatchSize;
  attr.kv_quant_bits = 8;
  attr.scale_k = kScaleK;
  attr.scale_v = kScaleV;
  node.operation.attributes = attr;

  ABSL_ASSIGN_OR_RETURN(auto op, CreateAddValuesToCacheFromNode(op_def, node));

  std::vector<float> k_data(kInputElements);
  std::vector<float> v_data(kInputElements);
  for (int i = 0; i < kInputElements; ++i) {
    k_data[i] = static_cast<float>((i * 11) - 150) * kScaleK;
    v_data[i] = static_cast<float>(150 - (i * 11)) * kScaleV;
  }
  constexpr int kStartIndex = 3;
  std::vector<int32_t> params_data = {kStartIndex, kStartIndex + kInputWidth};

  src_k_desc.UploadData(k_data.data());
  src_v_desc.UploadData(v_data.data());
  params_desc.UploadData(params_data.data());

  std::vector<uint8_t> init_cache(kCacheBytes, 0x80);
  cache_k_desc.SetData(std::vector<uint8_t>(init_cache));
  cache_v_desc.SetData(std::vector<uint8_t>(init_cache));

  std::vector<::ml_drift::TensorDescriptor*> src_cpu = {
      &src_k_desc, &src_v_desc, &params_desc};
  std::vector<::ml_drift::TensorDescriptor*> dst_cpu = {&cache_k_desc,
                                                        &cache_v_desc};
  ABSL_RETURN_IF_ERROR(
      env.ExecuteGPUOperation(src_cpu, dst_cpu, std::move(op)));

  std::vector<uint8_t> expected_k = init_cache;
  std::vector<uint8_t> expected_v = init_cache;
  constexpr int kHeadSlices = kHeadSize / 4;
  constexpr int kCacheSlices = kCacheSize / 4;
  for (int batch = 0; batch < kBatchSize; ++batch) {
    for (int x = 0; x < kInputWidth; ++x) {
      const int token_index = kStartIndex + x;
      const int token_slice = token_index / 4;
      const int token_in_slice = token_index % 4;
      for (int c = 0; c < kHeadSize; ++c) {
        const int src_idx = (batch * kInputWidth + x) * kHeadSize + c;
        const int c_slice = c / 4;
        const int c_in_slice = c % 4;
        const int k_idx =
            (((batch * kHeadSlices + c_slice) * kCacheSlices + token_slice) *
                 4 +
             token_in_slice) *
                4 +
            c_in_slice;
        const int v_idx =
            (((batch * kCacheSlices + token_slice) * kHeadSlices + c_slice) *
                 4 +
             token_in_slice) *
                4 +
            c_in_slice;
        expected_k[k_idx] = QuantizeInt8Byte(k_data[src_idx], kScaleK);
        expected_v[v_idx] = QuantizeInt8Byte(v_data[src_idx], kScaleV);
      }
    }
  }

  EXPECT_EQ(cache_k_desc.GetData(), absl::MakeConstSpan(expected_k));
  EXPECT_EQ(cache_v_desc.GetData(), absl::MakeConstSpan(expected_v));
  return absl::OkStatus();
}

TEST_F(AddValuesToCacheFloatTest, Int4CacheF32Linear) {
  ASSERT_OK(RunAddValuesToCacheInt4Test(
      *exec_env, ::ml_drift::CalculationsPrecision::kF32));
}

TEST_F(AddValuesToCacheFloatTest, Int4CacheF16Linear) {
  if (!exec_env->GetGpuInfo().SupportsFP16()) {
    GTEST_SKIP() << "FP16 not supported on this device.";
  }
  ASSERT_OK(RunAddValuesToCacheInt4Test(
      *exec_env, ::ml_drift::CalculationsPrecision::kF16));
}

TEST_F(AddValuesToCacheFloatTest, Int4CacheRingBufferWrapAround) {
  ASSERT_OK(RunAddValuesToCacheInt4Test(
      *exec_env, ::ml_drift::CalculationsPrecision::kF32,
      /*k_update_ts_idx=*/2, /*v_update_ts_idx=*/2, /*is_ring_buffer=*/true));
}

TEST_F(AddValuesToCacheFloatTest, Int4CacheTransposedUpdates) {
  ASSERT_OK(RunAddValuesToCacheInt4Test(
      *exec_env, ::ml_drift::CalculationsPrecision::kF32,
      /*k_update_ts_idx=*/3, /*v_update_ts_idx=*/3, /*is_ring_buffer=*/false));
}

TEST_F(AddValuesToCacheFloatTest, Int8CacheF32Linear) {
  ASSERT_OK(RunAddValuesToCacheInt8Test(
      *exec_env, ::ml_drift::CalculationsPrecision::kF32));
}

TEST_F(AddValuesToCacheFloatTest, RejectsInvalidQuantizedContracts) {
  using ::absl_testing::StatusIs;
  ::ml_drift::TensorDescriptor f32_src(::ml_drift::DataType::kFloat32,
                                       ::ml_drift::TensorStorageType::kBuffer,
                                       ::ml_drift::Layout::kHWC);
  f32_src.SetBHWCShape(::ml_drift::BHWC(1, 2, 1, 16));
  ::ml_drift::TensorDescriptor i32_src(::ml_drift::DataType::kInt32,
                                       ::ml_drift::TensorStorageType::kBuffer,
                                       ::ml_drift::Layout::kHWC);
  i32_src.SetBHWCShape(::ml_drift::BHWC(1, 2, 1, 16));
  ::ml_drift::TensorDescriptor params(::ml_drift::DataType::kInt32,
                                      ::ml_drift::TensorStorageType::kBuffer,
                                      ::ml_drift::Layout::kHWC);
  params.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, 4));
  ::ml_drift::TensorDescriptor u8_cache_k(
      ::ml_drift::DataType::kUint8, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kHWC);
  u8_cache_k.SetBHWCShape(::ml_drift::BHWC(1, 2, 16, 8));
  ::ml_drift::TensorDescriptor u8_cache_v(
      ::ml_drift::DataType::kUint8, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kHWC);
  u8_cache_v.SetBHWCShape(::ml_drift::BHWC(1, 2, 8, 16));

  AddValuesToCacheAttributes attr;
  attr.cache_size = 16;
  attr.head_size = 16;
  attr.kv_cache_batch_size = 2;
  attr.kv_quant_bits = 4;
  attr.scale_k = 0.25f;
  attr.scale_v = 0.5f;
  ::ml_drift::Node node = {1, {}};
  node.operation.attributes = attr;

  // Non-float src_v with INT4 cache must fail rather than falling back to 2x
  // unpacked writes.
  ::ml_drift::OperationDef bad_src_def;
  bad_src_def.src_tensors = {f32_src, i32_src, params};
  bad_src_def.dst_tensors = {u8_cache_k, u8_cache_v};
  EXPECT_THAT(CreateAddValuesToCacheFromNode(bad_src_def, node),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Non-kUint8 destination cache with kv_quant_bits == 4 must fail.
  ::ml_drift::OperationDef bad_dst_def;
  bad_dst_def.src_tensors = {f32_src, f32_src, params};
  bad_dst_def.dst_tensors = {f32_src, f32_src};
  EXPECT_THAT(CreateAddValuesToCacheFromNode(bad_dst_def, node),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Mismatched cache_k and cache_v dtypes must fail.
  ::ml_drift::OperationDef mismatch_dst_def;
  mismatch_dst_def.src_tensors = {f32_src, f32_src, params};
  mismatch_dst_def.dst_tensors = {u8_cache_k, f32_src};
  EXPECT_THAT(CreateAddValuesToCacheFromNode(mismatch_dst_def, node),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Non-kBuffer quantized cache storage must fail.
  ::ml_drift::TensorDescriptor tex_cache_k(
      ::ml_drift::DataType::kUint8, ::ml_drift::TensorStorageType::kTexture2D,
      ::ml_drift::Layout::kHWC);
  tex_cache_k.SetBHWCShape(::ml_drift::BHWC(1, 2, 16, 8));
  ::ml_drift::TensorDescriptor tex_cache_v(
      ::ml_drift::DataType::kUint8, ::ml_drift::TensorStorageType::kTexture2D,
      ::ml_drift::Layout::kHWC);
  tex_cache_v.SetBHWCShape(::ml_drift::BHWC(1, 2, 8, 16));
  ::ml_drift::OperationDef tex_dst_def;
  tex_dst_def.src_tensors = {f32_src, f32_src, params};
  tex_dst_def.dst_tensors = {tex_cache_k, tex_cache_v};
  EXPECT_THAT(CreateAddValuesToCacheFromNode(tex_dst_def, node),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Unsupported kv_quant_bits, unaligned head_size, non-finite scales, and
  // mismatched kv_cache_batch_size must fail.
  ::ml_drift::OperationDef valid_def;
  valid_def.src_tensors = {f32_src, f32_src, params};
  valid_def.dst_tensors = {u8_cache_k, u8_cache_v};

  AddValuesToCacheAttributes bad_bits = attr;
  bad_bits.kv_quant_bits = 2;
  node.operation.attributes = bad_bits;
  EXPECT_THAT(CreateAddValuesToCacheFromNode(valid_def, node),
              StatusIs(absl::StatusCode::kInvalidArgument));

  AddValuesToCacheAttributes bad_scale = attr;
  bad_scale.scale_k = std::numeric_limits<float>::quiet_NaN();
  node.operation.attributes = bad_scale;
  EXPECT_THAT(CreateAddValuesToCacheFromNode(valid_def, node),
              StatusIs(absl::StatusCode::kInvalidArgument));

  AddValuesToCacheAttributes bad_batch = attr;
  bad_batch.kv_cache_batch_size = 4;
  node.operation.attributes = bad_batch;
  EXPECT_THAT(CreateAddValuesToCacheFromNode(valid_def, node),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Mixed kFloat32 src_k and kFloat16 src_v in INT8 mode must succeed.
  ::ml_drift::TensorDescriptor f16_src_v(
      ::ml_drift::DataType::kFloat16, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kHWC);
  f16_src_v.SetBHWCShape(::ml_drift::BHWC(1, 2, 1, 16));
  ::ml_drift::TensorDescriptor i8_cache_k(
      ::ml_drift::DataType::kUint8, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kHWC);
  i8_cache_k.SetBHWCShape(::ml_drift::BHWC(1, 2, 16, 16));
  ::ml_drift::TensorDescriptor i8_cache_v(
      ::ml_drift::DataType::kUint8, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kHWC);
  i8_cache_v.SetBHWCShape(::ml_drift::BHWC(1, 2, 16, 16));
  ::ml_drift::OperationDef mixed_i8_def;
  mixed_i8_def.src_tensors = {f32_src, f16_src_v, params};
  mixed_i8_def.dst_tensors = {i8_cache_k, i8_cache_v};
  AddValuesToCacheAttributes i8_attr = attr;
  i8_attr.kv_quant_bits = 8;
  node.operation.attributes = i8_attr;
  EXPECT_OK(CreateAddValuesToCacheFromNode(mixed_i8_def, node));
}

INSTANTIATE_TEST_SUITE_P(
    AddValuesToCacheFloatTestSuite, AddValuesToCacheFloatTest,
    Combine(ValuesIn({::ml_drift::CalculationsPrecision::kF32}),
            ValuesIn({::ml_drift::TensorStorageType::kBuffer,
                      ::ml_drift::TensorStorageType::kTexture2D})),
    [](const TestParamInfo<AddValuesToCacheFloatTest::ParamType>& info) {
      return ::ml_drift::ToString(info.param);
    });

}  // namespace
}  // namespace litert::ml_drift

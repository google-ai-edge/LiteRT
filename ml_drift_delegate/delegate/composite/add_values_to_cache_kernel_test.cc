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

#include <cstdint>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/kernels/tests/kernel_test.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/testing_util.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"

namespace litert::ml_drift {
namespace {

using ::testing::Combine;
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
    ::ml_drift::DataType output_data_type) {
  constexpr int kCacheSize = 64;
  constexpr int kBatchSize = 2;
  constexpr int kInputWidth = 2;
  constexpr int kHeadSize = 8;
  constexpr int kKVCacheInputSliceSize = kBatchSize * kInputWidth * kHeadSize;
  constexpr int kKVCacheFullSize = kCacheSize * kInputWidth * kHeadSize;

  ::ml_drift::OperationDef op_def;
  ::ml_drift::DataType input_data_type =
      ::ml_drift::DeduceDataTypeFromPrecision(precision);

  // src_k
  ::ml_drift::TensorDescriptor src_k_desc(input_data_type, storage,
                                          ::ml_drift::Layout::HWC);
  src_k_desc.SetBHWCShape(
      ::ml_drift::BHWC(1, kBatchSize, kInputWidth, kHeadSize));
  op_def.src_tensors.push_back(src_k_desc);

  // src_v
  ::ml_drift::TensorDescriptor src_v_desc(input_data_type, storage,
                                          ::ml_drift::Layout::HWC);
  src_v_desc.SetBHWCShape(
      ::ml_drift::BHWC(1, kBatchSize, kInputWidth, kHeadSize));
  op_def.src_tensors.push_back(src_v_desc);

  // params (token_index_offset, active_tokens)
  ::ml_drift::TensorDescriptor params_desc(
      ::ml_drift::DataType::INT32, ::ml_drift::TensorStorageType::BUFFER,
      ::ml_drift::Layout::HWC);
  params_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, 2));
  op_def.src_tensors.push_back(params_desc);

  // cache_k
  ::ml_drift::TensorDescriptor cache_k_desc(output_data_type, storage,
                                            ::ml_drift::Layout::LINEAR);
  cache_k_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, kKVCacheFullSize));
  op_def.dst_tensors.push_back(cache_k_desc);

  // cache_v
  ::ml_drift::TensorDescriptor cache_v_desc(output_data_type, storage,
                                            ::ml_drift::Layout::LINEAR);
  cache_v_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, kKVCacheFullSize));
  op_def.dst_tensors.push_back(cache_v_desc);

  ::ml_drift::Node node = {1, {}};
  AddValuesToCacheAttributes attr;
  attr.cache_size = kCacheSize;
  attr.head_size = kHeadSize;
  attr.kv_cache_batch_size = kBatchSize;
  node.operation.attributes = attr;

  ABSL_ASSIGN_OR_RETURN(auto op, CreateAddValuesToCacheFromNode(op_def, node));

  std::vector<float> k_data(kKVCacheInputSliceSize);
  for (int i = 0; i < kKVCacheInputSliceSize; ++i) {
    k_data[i] = static_cast<float>(i + 1);
  }
  std::vector<float> v_data(kKVCacheInputSliceSize);
  for (int i = 0; i < kKVCacheInputSliceSize; ++i) {
    v_data[i] = static_cast<float>(i + 1 + kKVCacheInputSliceSize);
  }

  constexpr int kStartIndex = 10;
  std::vector<int32_t> params_data = {kStartIndex, kStartIndex + kBatchSize};

  std::vector<::ml_drift::TensorDescriptor*> src_cpu = {
      &src_k_desc, &src_v_desc, &params_desc};
  src_k_desc.UploadData(k_data.data());
  src_v_desc.UploadData(v_data.data());
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

  ABSL_LOG(ERROR) << "cache_k_result: " << absl::StrJoin(cache_k_result, ",");
  ABSL_LOG(ERROR) << "cache_v_result: " << absl::StrJoin(cache_v_result, ",");
  return absl::OkStatus();
}

TEST_P(AddValuesToCacheFloatTest, Float32Cache) {
  if (!exec_env->IsStorageSupported(storage(), ::ml_drift::DataType::FLOAT32)) {
    GTEST_SKIP() << "Unsupported storage type: "
                 << ::ml_drift::ToString(storage());
  }
  ASSERT_OK(RunAddValuesToCacheTest(*exec_env, precision(), storage(),
                                    ::ml_drift::DataType::FLOAT32));
}

INSTANTIATE_TEST_SUITE_P(
    AddValuesToCacheFloatTestSuite, AddValuesToCacheFloatTest,
    Combine(ValuesIn({::ml_drift::CalculationsPrecision::F32}),
            ValuesIn({::ml_drift::TensorStorageType::BUFFER,
                      ::ml_drift::TensorStorageType::TEXTURE_2D})),
    [](const TestParamInfo<AddValuesToCacheFloatTest::ParamType>& info) {
      return ::ml_drift::ToString(info.param);
    });

}  // namespace
}  // namespace litert::ml_drift

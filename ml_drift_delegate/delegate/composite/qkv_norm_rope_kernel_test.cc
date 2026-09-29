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

#include "ml_drift_delegate/delegate/composite/qkv_norm_rope_kernel.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/kernels/tests/kernel_test.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/testing_util.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/qkv_norm_rope_parser.h"

namespace litert::ml_drift {
namespace {

using ::testing::Combine;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

void ComputeQkvNormRopeReference(const std::vector<float>& qkv_data,
                                 const std::vector<int32_t>& pos_data,
                                 const std::vector<float>& q_weight_data,
                                 const std::vector<float>& k_weight_data,
                                 const QkvNormRopeAttributes& attr, int seq_len,
                                 std::vector<float>* expected_q,
                                 std::vector<float>* expected_k,
                                 std::vector<float>* expected_v) {
  const int num_heads = attr.num_heads;
  const int num_kv_heads = attr.num_kv_heads;
  const int head_dim = attr.head_dim;
  const int half_dim = head_dim / 2;
  const int total_heads = num_heads + 2 * num_kv_heads;
  const int total_dim = total_heads * head_dim;

  expected_q->assign(num_heads * seq_len * head_dim, 0.0f);
  expected_k->assign(num_kv_heads * seq_len * head_dim, 0.0f);
  expected_v->assign(num_kv_heads * seq_len * head_dim, 0.0f);

  const float inv_dst_ch = 1.0f / static_cast<float>(head_dim);

  for (int t = 0; t < seq_len; ++t) {
    const float pos_scalar = static_cast<float>(pos_data[t]);
    for (int y = 0; y < total_heads; ++y) {
      const float* head_ptr = &qkv_data[t * total_dim + y * head_dim];
      if (y < num_heads + num_kv_heads) {
        const bool is_query = (y < num_heads);
        const int dst_head = is_query ? y : (y - num_heads);
        const std::vector<float>& weight =
            is_query ? q_weight_data : k_weight_data;
        std::vector<float>* dst = is_query ? expected_q : expected_k;

        float sum_sq = 0.0f;
        for (int c = 0; c < head_dim; ++c) {
          sum_sq += head_ptr[c] * head_ptr[c];
        }
        const float inv_std =
            1.0f /
            std::sqrt(sum_sq / static_cast<float>(head_dim) + attr.epsilon);

        for (int i = 0; i < half_dim; ++i) {
          const float v0 = head_ptr[i] * inv_std * weight[i];
          const float v1 =
              head_ptr[i + half_dim] * inv_std * weight[i + half_dim];
          const float fraction = 2.0f * static_cast<float>(i) * inv_dst_ch;
          const float timescale =
              attr.min_timescale *
              std::pow(attr.max_timescale / attr.min_timescale, fraction);
          const float sinusoid = pos_scalar / timescale;
          const float cos_val = std::cos(sinusoid);
          const float sin_val = std::sin(sinusoid);

          const int out_base = (dst_head * seq_len + t) * head_dim;
          (*dst)[out_base + i] = v0 * cos_val - v1 * sin_val;
          (*dst)[out_base + i + half_dim] = v1 * cos_val + v0 * sin_val;
        }
      } else {
        const int kv_head = y - (num_heads + num_kv_heads);
        const int out_base = (kv_head * seq_len + t) * head_dim;
        for (int c = 0; c < head_dim; ++c) {
          (*expected_v)[out_base + c] = head_ptr[c];
        }
      }
    }
  }
}

absl::Status RunQkvNormRopeTest(::ml_drift::TestExecutionEnvironment& env,
                                ::ml_drift::CalculationsPrecision precision,
                                ::ml_drift::TensorStorageType storage,
                                int num_heads, int num_kv_heads, int seq_len,
                                int head_dim) {
  ::ml_drift::GpuModelBuilder builder(env.GetGpuInfo(), {}, precision, storage);
  const ::ml_drift::DataType data_type =
      ::ml_drift::DeduceDataTypeFromPrecision(precision);

  const int total_heads = num_heads + 2 * num_kv_heads;
  const int total_dim = total_heads * head_dim;

  const ::ml_drift::BHWC qkv_shape(1, 1, seq_len, total_dim);
  const ::ml_drift::BHWC pos_shape(1, 1, 1, seq_len);
  const ::ml_drift::BHWC weight_shape(head_dim, 1, 1, 1);
  const ::ml_drift::BHWC q_out_shape(1, num_heads, seq_len, head_dim);
  const ::ml_drift::BHWC kv_out_shape(1, num_kv_heads, seq_len, head_dim);

  auto qkv = builder.AddTensor(qkv_shape, data_type);
  auto q_weight = builder.AddTensor(weight_shape, data_type);
  auto k_weight = builder.AddTensor(weight_shape, data_type);

  std::vector<int32_t> pos_data(seq_len);
  for (int t = 0; t < seq_len; ++t) {
    pos_data[t] = t + 3;
  }
  ::ml_drift::Tensor<::ml_drift::StrongShape<::ml_drift::Layout::kBHWC>,
                     ::ml_drift::DataType::kInt32>
      pos_tensor_cpu;
  pos_tensor_cpu.shape = pos_shape;
  pos_tensor_cpu.data = pos_data;

  ::ml_drift::TensorDescriptor pos_desc(::ml_drift::DataType::kInt32,
                                        ::ml_drift::TensorStorageType::kBuffer,
                                        ::ml_drift::Layout::kHWC);
  pos_desc.UploadData(pos_tensor_cpu);
  auto pos = builder.AddConstantTensor(std::move(pos_desc));

  auto q_out = builder.AddTensor(q_out_shape, data_type);
  auto k_out = builder.AddTensor(kv_out_shape, data_type);
  auto v_out = builder.AddTensor(kv_out_shape, data_type);

  QkvNormRopeAttributes attr;
  attr.num_heads = num_heads;
  attr.num_kv_heads = num_kv_heads;
  attr.head_dim = head_dim;
  attr.min_timescale = 1.0f;
  attr.max_timescale = 10000.0f;
  attr.proportion = 1.0f;
  attr.epsilon = 1e-6f;

  ::ml_drift::Value qkv_val{qkv.id, {}, {}};
  ::ml_drift::Value pos_val{pos.id, {}, {}};
  ::ml_drift::Value q_weight_val{q_weight.id, {}, {}};
  ::ml_drift::Value k_weight_val{k_weight.id, {}, {}};
  ::ml_drift::Value q_out_val{q_out.id, {}, {}};
  ::ml_drift::Value k_out_val{k_out.id, {}, {}};
  ::ml_drift::Value v_out_val{v_out.id, {}, {}};

  ::ml_drift::Node node = {1, {}};
  node.operation.type = std::string(kQkvNormRopeType);
  node.operation.attributes = attr;

  ABSL_RETURN_IF_ERROR(CreateQkvNormRopeFromNode(
      {&qkv_val, &pos_val, &q_weight_val, &k_weight_val},
      {&q_out_val, &k_out_val, &v_out_val}, node, &builder));

  ::ml_drift::GpuModel gpu_model;
  ABSL_RETURN_IF_ERROR(builder.GetGpuModel(
      {{qkv.id, 0}, {q_weight.id, 1}, {k_weight.id, 2}},
      {{q_out.id, 0}, {k_out.id, 1}, {v_out.id, 2}}, &gpu_model));

  std::vector<float> qkv_data(seq_len * total_dim);
  for (size_t i = 0; i < qkv_data.size(); ++i) {
    qkv_data[i] = 0.1f * std::sin(static_cast<float>(i + 1) * 0.37f) +
                  0.05f * std::cos(static_cast<float>(i + 1) * 0.13f);
  }
  std::vector<float> q_weight_data(head_dim);
  std::vector<float> k_weight_data(head_dim);
  for (int c = 0; c < head_dim; ++c) {
    q_weight_data[c] = 0.75f + 0.5f * static_cast<float>(c % 8) / 8.0f;
    k_weight_data[c] = 0.85f + 0.3f * static_cast<float>((c + 3) % 8) / 8.0f;
  }

  std::vector<float> expected_q;
  std::vector<float> expected_k;
  std::vector<float> expected_v;
  ComputeQkvNormRopeReference(qkv_data, pos_data, q_weight_data, k_weight_data,
                              attr, seq_len, &expected_q, &expected_k,
                              &expected_v);

  ::ml_drift::TensorFloat32 qkv_tensor;
  qkv_tensor.shape = qkv_shape;
  qkv_tensor.data = qkv_data;

  ::ml_drift::TensorFloat32 q_weight_tensor;
  q_weight_tensor.shape = weight_shape;
  q_weight_tensor.data = q_weight_data;

  ::ml_drift::TensorFloat32 k_weight_tensor;
  k_weight_tensor.shape = weight_shape;
  k_weight_tensor.data = k_weight_data;

  ::ml_drift::TensorFloat32 q_out_cpu;
  q_out_cpu.shape = q_out_shape;
  q_out_cpu.data.resize(q_out_shape.DimensionsProduct());

  ::ml_drift::TensorFloat32 k_out_cpu;
  k_out_cpu.shape = kv_out_shape;
  k_out_cpu.data.resize(kv_out_shape.DimensionsProduct());

  ::ml_drift::TensorFloat32 v_out_cpu;
  v_out_cpu.shape = kv_out_shape;
  v_out_cpu.data.resize(kv_out_shape.DimensionsProduct());

  ABSL_RETURN_IF_ERROR(
      env.ExecuteGpuModel({qkv_tensor, q_weight_tensor, k_weight_tensor},
                          std::vector<::ml_drift::TensorFloat32*>{
                              &q_out_cpu, &k_out_cpu, &v_out_cpu},
                          &gpu_model));

  const float tolerance =
      (precision == ::ml_drift::CalculationsPrecision::kF16) ? 1e-2f : 1e-4f;
  EXPECT_THAT(q_out_cpu.data, Pointwise(FloatNear(tolerance), expected_q));
  EXPECT_THAT(k_out_cpu.data, Pointwise(FloatNear(tolerance), expected_k));
  EXPECT_THAT(v_out_cpu.data, Pointwise(FloatNear(tolerance), expected_v));
  return absl::OkStatus();
}

class QkvNormRopeKernelTest : public ::ml_drift::FloatTest {
 public:
  void SetUp() override {
    if (!exec_env) {
      GTEST_SKIP() << "TestExecutionEnvironment not initialized.";
    }
    const auto data_type = ::ml_drift::DeduceDataTypeFromPrecision(precision());
    if (!exec_env->IsStorageSupported(storage(), data_type)) {
      GTEST_SKIP() << "Unsupported data type: "
                   << ::ml_drift::ToString(data_type)
                   << " storage type: " << ::ml_drift::ToString(storage());
    }
  }
};

TEST_P(QkvNormRopeKernelTest, SingleTokenDecodeHeadDim64) {
  ASSERT_OK(RunQkvNormRopeTest(*exec_env, precision(), storage(),
                               /*num_heads=*/4, /*num_kv_heads=*/2,
                               /*seq_len=*/1, /*head_dim=*/64));
}

TEST_P(QkvNormRopeKernelTest, SingleTokenDecodeHeadDim128) {
  ASSERT_OK(RunQkvNormRopeTest(*exec_env, precision(), storage(),
                               /*num_heads=*/8, /*num_kv_heads=*/2,
                               /*seq_len=*/1, /*head_dim=*/128));
}

TEST_P(QkvNormRopeKernelTest, SingleTokenDecodeHeadDim256) {
  ASSERT_OK(RunQkvNormRopeTest(*exec_env, precision(), storage(),
                               /*num_heads=*/8, /*num_kv_heads=*/4,
                               /*seq_len=*/1, /*head_dim=*/256));
}

TEST_P(QkvNormRopeKernelTest, MultiTokenPrefillHeadDim128) {
  ASSERT_OK(RunQkvNormRopeTest(*exec_env, precision(), storage(),
                               /*num_heads=*/8, /*num_kv_heads=*/2,
                               /*seq_len=*/8, /*head_dim=*/128));
}

INSTANTIATE_TEST_SUITE_P(
    QkvNormRopeKernelTestSuite, QkvNormRopeKernelTest,
    Combine(ValuesIn({::ml_drift::CalculationsPrecision::kF32,
                      ::ml_drift::CalculationsPrecision::kF16}),
            ValuesIn({::ml_drift::TensorStorageType::kBuffer,
                      ::ml_drift::TensorStorageType::kTexture2D})),
    [](const TestParamInfo<QkvNormRopeKernelTest::ParamType>& info) {
      return ::ml_drift::ToString(info.param);
    });

}  // namespace
}  // namespace litert::ml_drift

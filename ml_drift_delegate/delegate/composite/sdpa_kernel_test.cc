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

#include "ml_drift_delegate/delegate/composite/sdpa_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/kernels/tests/kernel_test.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/testing_util.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift

namespace litert::ml_drift {
namespace {

using ::testing::Combine;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

enum class MaskMode { kBool, kFloatAdditive, kNone };

inline std::string ToString(MaskMode mask_mode) {
  switch (mask_mode) {
    case MaskMode::kBool:
      return "BoolMask";
    case MaskMode::kFloatAdditive:
      return "FloatAdditiveMask";
    case MaskMode::kNone:
      return "NoMask";
  }
  return "Unknown";
}

// Computes reference SDPA output on CPU for Q [B, T, N_q, H],
// K [B, S, N_kv, H], V [B, S, N_kv, H], and optional Mask
// [mask_B, mask_N, mask_T, S].
std::vector<float> ComputeSdpaReference(const std::vector<float>& q_data,
                                        const std::vector<float>& k_data,
                                        const std::vector<float>& v_data,
                                        const std::vector<float>& mask_data,
                                        int B, int T, int S, int N_q, int N_kv,
                                        int H, float scale, MaskMode mask_mode,
                                        int mask_B, int mask_N, int mask_T) {
  const int gqa_ratio = N_q / N_kv;
  std::vector<float> out_data(B * T * N_q * H, 0.0f);
  for (int b = 0; b < B; ++b) {
    const int mb = std::min(b, mask_B - 1);
    for (int nq = 0; nq < N_q; ++nq) {
      const int nkv = nq / gqa_ratio;
      const int mn = std::min(nq, mask_N - 1);
      for (int t = 0; t < T; ++t) {
        const int mt = std::min(t, mask_T - 1);
        std::vector<float> scores(S, -1.0e30f);
        float max_score = -1.0e30f;
        for (int s = 0; s < S; ++s) {
          float dot_val = 0.0f;
          for (int h = 0; h < H; ++h) {
            const int q_idx = ((b * T + t) * N_q + nq) * H + h;
            const int k_idx = ((b * S + s) * N_kv + nkv) * H + h;
            dot_val += (q_data[q_idx] * scale) * k_data[k_idx];
          }
          const int m_idx = ((mb * mask_N + mn) * mask_T + mt) * S + s;
          if (mask_mode == MaskMode::kNone) {
            scores[s] = dot_val;
          } else if (mask_mode == MaskMode::kBool) {
            if (mask_data[m_idx] != 0.0f) {
              scores[s] = dot_val;
            }
          } else if (mask_mode == MaskMode::kFloatAdditive) {
            scores[s] = (mask_data[m_idx] <= -1.0e29f)
                            ? -1.0e30f
                            : std::max(dot_val + mask_data[m_idx], -1.0e30f);
          }
          max_score = std::max(max_score, scores[s]);
        }

        float sum_exp = 0.0f;
        std::vector<float> probs(S, 0.0f);
        for (int s = 0; s < S; ++s) {
          if (scores[s] > -1.0e29f) {
            probs[s] = std::exp(scores[s] - max_score);
            sum_exp += probs[s];
          }
        }
        const float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
        for (int h = 0; h < H; ++h) {
          float acc = 0.0f;
          for (int s = 0; s < S; ++s) {
            const int v_idx = ((b * S + s) * N_kv + nkv) * H + h;
            acc += probs[s] * v_data[v_idx];
          }
          const int out_idx = ((b * T + t) * N_q + nq) * H + h;
          out_data[out_idx] = acc * inv_sum;
        }
      }
    }
  }
  return out_data;
}

MATCHER_P2(FloatNearWithRelativeTolerance, abs_tolerance, rel_tolerance,
           absl::StrCat("is within ", abs_tolerance, " + ", rel_tolerance,
                        " * |expected| of expected")) {
  const float actual = std::get<0>(arg);
  const float expected = std::get<1>(arg);
  const float diff = std::abs(actual - expected);
  *result_listener << "which is " << diff << " from " << expected;
  return diff <= abs_tolerance + rel_tolerance * std::abs(expected);
}

class SdpaKernelExecuteTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<
          std::tuple<::ml_drift::CalculationsPrecision,
                     ::ml_drift::TensorStorageType, MaskMode>> {
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

 protected:
  ::ml_drift::CalculationsPrecision precision() const {
    return std::get<0>(GetParam());
  }
  ::ml_drift::TensorStorageType storage() const {
    return std::get<1>(GetParam());
  }
  MaskMode mask_mode() const { return std::get<2>(GetParam()); }
};

absl::Status RunSdpaKernelTest(
    ::ml_drift::TestExecutionEnvironment& env,
    ::ml_drift::CalculationsPrecision precision,
    ::ml_drift::TensorStorageType storage, int B, int T, int S, int N_q,
    int N_kv, int H, MaskMode mask_mode, int mask_B = 1, int mask_N = 1,
    int mask_T = 0,
    std::optional<::ml_drift::GpuInfo> gpu_info_override = std::nullopt,
    std::optional<std::vector<float>> custom_mask_data = std::nullopt) {
  if (mask_T <= 0) mask_T = T;
  const ::ml_drift::GpuInfo gpu_info =
      gpu_info_override.value_or(env.GetGpuInfo());
  ::ml_drift::GpuModelBuilder builder(gpu_info, {}, precision, storage);
  const ::ml_drift::DataType datatype =
      ::ml_drift::DeduceDataTypeFromPrecision(precision);

  const ::ml_drift::BHWC q_shape(B, T, N_q, H);
  const ::ml_drift::BHWC k_shape(B, S, N_kv, H);
  const ::ml_drift::BHWC v_shape(B, S, N_kv, H);
  const ::ml_drift::BHWC mask_shape(mask_B, mask_N, mask_T, S);
  const ::ml_drift::BHWC out_shape(B, T, N_q, H);

  auto q = builder.AddTensor(q_shape, datatype);
  auto k = builder.AddTensor(k_shape, datatype);
  auto v = builder.AddTensor(v_shape, datatype);

  std::optional<::ml_drift::GpuModelBuilder::TensorHandle> mask_handle;
  std::optional<::ml_drift::GpuModelBuilder::TensorHandle> mask_feed_handle;
  if (mask_mode == MaskMode::kBool) {
    auto mask_float = builder.AddTensor(mask_shape, datatype);
    mask_feed_handle = mask_float;
    mask_handle = builder.Cast(mask_float, ::ml_drift::DataType::kBool);
  } else if (mask_mode == MaskMode::kFloatAdditive) {
    auto mask_float = builder.AddTensor(mask_shape, datatype);
    mask_feed_handle = mask_float;
    mask_handle = mask_float;
  }

  auto out_tensor = builder.AddTensor(out_shape, datatype);

  const float scale = 1.0f / std::sqrt(static_cast<float>(H));
  ::ml_drift::ScaledDotProductAttentionAttributes attr;
  attr.scale = scale;

  std::vector<uint32_t> graph_input_ids = {q.id, k.id, v.id};
  if (mask_handle.has_value()) {
    graph_input_ids.push_back(mask_handle->id);
  }

  ABSL_RETURN_IF_ERROR(
      BuildSdpaGpuGraph(graph_input_ids, out_tensor.id, attr, &builder));

  std::vector<std::pair<uint32_t, uint32_t>> model_inputs = {
      {q.id, 0}, {k.id, 1}, {v.id, 2}};
  if (mask_feed_handle.has_value()) {
    model_inputs.push_back({mask_feed_handle->id, 3});
  }

  ::ml_drift::GpuModel gpu_model;
  ABSL_RETURN_IF_ERROR(builder.GetGpuModel(
      model_inputs,
      std::vector<std::pair<uint32_t, uint32_t>>{{out_tensor.id, 0}},
      &gpu_model));

  std::vector<float> q_data(B * T * N_q * H);
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < T; ++t) {
      const float row_sign = (t % 2 == 0) ? 0.35f : -0.35f;
      for (int nq = 0; nq < N_q; ++nq) {
        for (int h = 0; h < H; ++h) {
          const int idx = ((b * T + t) * N_q + nq) * H + h;
          const float base =
              0.7f * std::sin(0.13f * static_cast<float>(idx + 1));
          q_data[idx] = (h % 4 == 0) ? (base + row_sign) : base;
        }
      }
    }
  }

  std::vector<float> k_data(B * S * N_kv * H);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S; ++s) {
      // Add a monotonic key-tile trend across `s / 16` so even query rows see
      // increasing tile maxima (`m_new > m_i`), actively exercising the
      // `alpha = exp2(m_i - m_new)` online softmax rescaling path.
      const float tile_trend = 0.6f * static_cast<float>(s / 16) /
                               static_cast<float>(std::max(1, (S - 1) / 16));
      for (int nkv = 0; nkv < N_kv; ++nkv) {
        for (int h = 0; h < H; ++h) {
          const int idx = ((b * S + s) * N_kv + nkv) * H + h;
          const float base =
              0.7f * std::cos(0.17f * static_cast<float>(idx + 1));
          k_data[idx] = (h % 4 == 0) ? (base + tile_trend) : base;
        }
      }
    }
  }

  std::vector<float> v_data(B * S * N_kv * H);
  for (int i = 0; i < static_cast<int>(v_data.size()); ++i) {
    v_data[i] = 0.8f * std::sin(0.07f * static_cast<float>(i + 1));
  }

  std::vector<float> mask_data(mask_B * mask_N * mask_T * S, 0.0f);
  if (custom_mask_data.has_value()) {
    mask_data = *std::move(custom_mask_data);
  } else if (mask_mode == MaskMode::kBool) {
    for (int mb = 0; mb < mask_B; ++mb) {
      for (int mn = 0; mn < mask_N; ++mn) {
        for (int mt = 0; mt < mask_T; ++mt) {
          for (int s = 0; s < S; ++s) {
            const bool unmasked = (s == 0) || ((mt + s + mn + mb) % 4 != 3);
            const int idx = ((mb * mask_N + mn) * mask_T + mt) * S + s;
            mask_data[idx] = unmasked ? 1.0f : 0.0f;
          }
        }
      }
    }
  } else if (mask_mode == MaskMode::kFloatAdditive) {
    for (int mb = 0; mb < mask_B; ++mb) {
      for (int mn = 0; mn < mask_N; ++mn) {
        for (int mt = 0; mt < mask_T; ++mt) {
          for (int s = 0; s < S; ++s) {
            const bool unmasked = (s == 0) || ((mt + s + mn + mb) % 4 != 3);
            const int idx = ((mb * mask_N + mn) * mask_T + mt) * S + s;
            mask_data[idx] = unmasked
                                 ? -0.25f * static_cast<float>((mt + s) % 4)
                                 : -10000.0f;
          }
        }
      }
    }
  }

  const std::vector<float> expected_out_data =
      ComputeSdpaReference(q_data, k_data, v_data, mask_data, B, T, S, N_q,
                           N_kv, H, scale, mask_mode, mask_B, mask_N, mask_T);

  ::ml_drift::TensorFloat32 q_tensor;
  q_tensor.shape = q_shape;
  q_tensor.data = q_data;

  ::ml_drift::TensorFloat32 k_tensor;
  k_tensor.shape = k_shape;
  k_tensor.data = k_data;

  ::ml_drift::TensorFloat32 v_tensor;
  v_tensor.shape = v_shape;
  v_tensor.data = v_data;

  std::vector<::ml_drift::TensorFloat32> src_cpu = {q_tensor, k_tensor,
                                                    v_tensor};
  if (mask_mode != MaskMode::kNone) {
    ::ml_drift::TensorFloat32 mask_tensor;
    mask_tensor.shape = mask_shape;
    mask_tensor.data = mask_data;
    src_cpu.push_back(mask_tensor);
  }

  ::ml_drift::TensorFloat32 out_tensor_cpu;
  out_tensor_cpu.shape = out_shape;
  out_tensor_cpu.data.resize(B * T * N_q * H);
  std::vector<::ml_drift::TensorFloat32*> dst_cpu = {&out_tensor_cpu};

  ABSL_RETURN_IF_ERROR(env.ExecuteGpuModel(src_cpu, dst_cpu, &gpu_model));

  float tol = 1e-4f;
  if (precision == ::ml_drift::CalculationsPrecision::kF32F16) {
    tol = 2e-3f;
  } else if (precision == ::ml_drift::CalculationsPrecision::kF16) {
    tol = 1.5e-2f;
  }
  EXPECT_THAT(out_tensor_cpu.data,
              testing::Pointwise(FloatNearWithRelativeTolerance(tol, tol),
                                 expected_out_data));
  return absl::OkStatus();
}

TEST_P(SdpaKernelExecuteTest, BuildAndExecuteSmall) {
  auto status = RunSdpaKernelTest(*exec_env, precision(), storage(), /*B=*/1,
                                  /*T=*/4, /*S=*/8, /*N_q=*/2, /*N_kv=*/2,
                                  /*H=*/8, mask_mode());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaKernelExecuteTest, Gemma4VisionHeadDim64) {
  auto status = RunSdpaKernelTest(*exec_env, precision(), storage(), /*B=*/1,
                                  /*T=*/20, /*S=*/20, /*N_q=*/12, /*N_kv=*/12,
                                  /*H=*/64, mask_mode());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaKernelExecuteTest, MultipleQueryAndKeyTilesHeadDim64) {
  auto status = RunSdpaKernelTest(*exec_env, precision(), storage(), /*B=*/1,
                                  /*T=*/130, /*S=*/140, /*N_q=*/4, /*N_kv=*/4,
                                  /*H=*/64, mask_mode());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaKernelExecuteTest, GroupedQueryAttentionHeadDim128) {
  auto status = RunSdpaKernelTest(*exec_env, precision(), storage(), /*B=*/1,
                                  /*T=*/24, /*S=*/48, /*N_q=*/8, /*N_kv=*/2,
                                  /*H=*/128, mask_mode());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaKernelExecuteTest, OddSliceSplitHeadDim76) {
  auto status = RunSdpaKernelTest(*exec_env, precision(), storage(), /*B=*/1,
                                  /*T=*/16, /*S=*/24, /*N_q=*/4, /*N_kv=*/2,
                                  /*H=*/76, mask_mode());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaKernelExecuteTest, FourWaySliceSplitMultiTailHeadDim68) {
  // H = 68 (17 slices, `5/4/4/4`, `rem_s = 1`) and H = 72 (18 slices,
  // `5/5/4/4`, `rem_s = 2`) exercise 4-way balanced slice splitting in FP32
  // (`s_splits = 4, block_s = 5`, `b_r = 32, threads = 128` when
  // `max_work_group_size >= 128`), skewed `k_local` / `v_local` indexing, and
  // double-buffered `red_local` ping-pong across multiple query tiles
  // (`T = 35`) and 16-key interior tiles (`S = 37`).
  for (int head_dim : {68, 72}) {
    auto status = RunSdpaKernelTest(*exec_env, precision(), storage(), /*B=*/1,
                                    /*T=*/35, /*S=*/37, /*N_q=*/4, /*N_kv=*/2,
                                    /*H=*/head_dim, mask_mode());
    EXPECT_TRUE(status.ok()) << "H=" << head_dim << ": " << status.message();
  }
}

TEST_P(SdpaKernelExecuteTest, RaggedQueryAndKeyLengths) {
  auto status = RunSdpaKernelTest(*exec_env, precision(), storage(), /*B=*/1,
                                  /*T=*/13, /*S=*/29, /*N_q=*/4, /*N_kv=*/2,
                                  /*H=*/64, mask_mode());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaKernelExecuteTest, MultiBatchAndPerHeadMask) {
  auto status = RunSdpaKernelTest(*exec_env, precision(), storage(), /*B=*/2,
                                  /*T=*/8, /*S=*/12, /*N_q=*/4, /*N_kv=*/2,
                                  /*H=*/32, mask_mode(), /*mask_B=*/2,
                                  /*mask_N=*/4, /*mask_T=*/8);
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaKernelExecuteTest, BroadcastRowMask) {
  auto status = RunSdpaKernelTest(*exec_env, precision(), storage(), /*B=*/2,
                                  /*T=*/10, /*S=*/16, /*N_q=*/4, /*N_kv=*/2,
                                  /*H=*/64, mask_mode(), /*mask_B=*/1,
                                  /*mask_N=*/1, /*mask_T=*/1);
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaKernelExecuteTest, LongKeySequenceHeadDim64) {
  auto status = RunSdpaKernelTest(*exec_env, precision(), storage(), /*B=*/1,
                                  /*T=*/4, /*S=*/2048, /*N_q=*/2, /*N_kv=*/2,
                                  /*H=*/64, mask_mode());
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaKernelExecuteTest, FullyMaskedRowAndFirstTileMasked) {
  if (mask_mode() == MaskMode::kNone) {
    GTEST_SKIP() << "Mask-specific edge case test.";
  }
  constexpr int B = 1, T = 4, S = 36, N_q = 2, N_kv = 2, H = 32;
  std::vector<float> custom_mask(T * S, 0.0f);
  for (int t = 0; t < T; ++t) {
    for (int s = 0; s < S; ++s) {
      bool unmasked = true;
      if (t == 0) {
        // Row 0: all keys masked out -> output must be 0.0f without NaN.
        unmasked = false;
      } else if (t == 1) {
        // First 18 keys (entire first 16-key tile + 2 keys of second tile)
        // masked out -> tests transition from m_i = -1.0e30f to finite keys.
        unmasked = (s >= 18);
      } else {
        unmasked = (s == S - 1) || ((t + s) % 3 == 0);
      }
      const float float_masked_val =
          (s % 2 == 0) ? -std::numeric_limits<float>::max()
                       : -std::numeric_limits<float>::infinity();
      custom_mask[t * S + s] =
          (mask_mode() == MaskMode::kBool)
              ? (unmasked ? 1.0f : 0.0f)
              : (unmasked ? -0.15f * static_cast<float>((t + s) % 4)
                          : float_masked_val);
    }
  }
  auto status = RunSdpaKernelTest(
      *exec_env, precision(), storage(), B, T, S, N_q, N_kv, H, mask_mode(),
      /*mask_B=*/1, /*mask_N=*/1, /*mask_T=*/T,
      /*gpu_info_override=*/std::nullopt, std::move(custom_mask));
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaKernelExecuteTest, MaliQInLocalCodegenAndExecute) {
  ::ml_drift::GpuInfo mali_gpu_info = exec_env->GetGpuInfo();
  mali_gpu_info.vendor = ::ml_drift::GpuVendor::kMali;
  auto status = RunSdpaKernelTest(
      *exec_env, precision(), storage(), /*B=*/1, /*T=*/16, /*S=*/24,
      /*N_q=*/4, /*N_kv=*/2, /*H=*/76, mask_mode(), /*mask_B=*/1, /*mask_N=*/1,
      /*mask_T=*/16, mali_gpu_info);
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_P(SdpaKernelExecuteTest, MaliGemma4VisionAndMultiTailSliceSplit) {
  ::ml_drift::GpuInfo mali_gpu_info = exec_env->GetGpuInfo();
  mali_gpu_info.vendor = ::ml_drift::GpuVendor::kMali;
  // Exercises Mali `q_in_local` with H = 64 (`s_splits = 2, b_r = 32` in FP32;
  // `s_splits = 1, b_r = 64` in FP16) across multiple query and key tiles.
  auto status_h64 = RunSdpaKernelTest(
      *exec_env, precision(), storage(), /*B=*/1, /*T=*/35, /*S=*/37,
      /*N_q=*/4, /*N_kv=*/4, /*H=*/64, mask_mode(), /*mask_B=*/1, /*mask_N=*/1,
      /*mask_T=*/35, mali_gpu_info);
  EXPECT_TRUE(status_h64.ok()) << status_h64.message();

  // Exercises Mali `q_in_local` with H = 68 (`s_splits = 4, block_s = 5,
  // b_r = 32` in FP32) and multiple query (`T = 35`) and key (`S = 37`) tiles.
  auto status_h68 = RunSdpaKernelTest(
      *exec_env, precision(), storage(), /*B=*/1, /*T=*/35, /*S=*/37,
      /*N_q=*/4, /*N_kv=*/2, /*H=*/68, mask_mode(), /*mask_B=*/1, /*mask_N=*/1,
      /*mask_T=*/35, mali_gpu_info);
  EXPECT_TRUE(status_h68.ok()) << status_h68.message();
}

TEST_P(SdpaKernelExecuteTest, Adreno8xxCodegenAndExecute) {
  ::ml_drift::GpuInfo adreno8xx_gpu_info = exec_env->GetGpuInfo();
  adreno8xx_gpu_info.vendor = ::ml_drift::GpuVendor::kQualcomm;
  adreno8xx_gpu_info.adreno_info.generation =
      ::ml_drift::AdrenoInfo::Generation::kGen8;
  // On non-Qualcomm OpenCL drivers (e.g. host x86 or Mali OpenCL), set
  // `is_clvk` so the Qualcomm-proprietary `qcom_wave_pair_mode(1)` attribute is
  // omitted while still executing the Adreno 8xx `max_block_s = 4`
  // (`s_splits = 4, b_r = 16`) tiling. On physical Qualcomm Adreno OpenCL
  // drivers, `is_clvk` remains false so `qcom_wave_pair_mode(1)` is also
  // compiled and executed on device.
  if (adreno8xx_gpu_info.IsApiOpenCl() && !exec_env->GetGpuInfo().IsAdreno()) {
    adreno8xx_gpu_info.opencl_info.platform_version = "clvk";
  }
  auto status_h64 = RunSdpaKernelTest(
      *exec_env, precision(), storage(), /*B=*/1, /*T=*/35, /*S=*/37,
      /*N_q=*/4, /*N_kv=*/4, /*H=*/64, mask_mode(), /*mask_B=*/1, /*mask_N=*/1,
      /*mask_T=*/35, adreno8xx_gpu_info);
  EXPECT_TRUE(status_h64.ok()) << status_h64.message();

  auto status_h68 = RunSdpaKernelTest(
      *exec_env, precision(), storage(), /*B=*/1, /*T=*/35, /*S=*/37,
      /*N_q=*/4, /*N_kv=*/2, /*H=*/68, mask_mode(), /*mask_B=*/1, /*mask_N=*/1,
      /*mask_T=*/35, adreno8xx_gpu_info);
  EXPECT_TRUE(status_h68.ok()) << status_h68.message();
}

INSTANTIATE_TEST_SUITE_P(
    SdpaKernelExecuteTestSuite, SdpaKernelExecuteTest,
    Combine(ValuesIn({::ml_drift::CalculationsPrecision::kF32,
                      ::ml_drift::CalculationsPrecision::kF32F16,
                      ::ml_drift::CalculationsPrecision::kF16}),
            ValuesIn({::ml_drift::TensorStorageType::kTexture2D,
                      ::ml_drift::TensorStorageType::kBuffer}),
            ValuesIn({MaskMode::kBool, MaskMode::kFloatAdditive,
                      MaskMode::kNone})),
    [](const TestParamInfo<SdpaKernelExecuteTest::ParamType>& info) {
      std::string name =
          absl::StrCat(::ml_drift::ToString(std::get<0>(info.param)), "_",
                       ::ml_drift::ToString(std::get<1>(info.param)), "_",
                       ToString(std::get<2>(info.param)));
      return absl::StrReplaceAll(name, {{":", ""}});
    });

TEST(IsSupportedFlashSdpaTest, SupportsGemma4VisionAndRejectsInvalidConfigs) {
  ::ml_drift::GpuInfo gpu_info;
  gpu_info.vendor = ::ml_drift::GpuVendor::kQualcomm;
  gpu_info.gpu_api = ::ml_drift::GpuApi::kOpenCl;
  gpu_info.opencl_info.extensions.push_back("ucl_wave_memory");

  ::ml_drift::TensorDescriptor q_desc(::ml_drift::DataType::kFloat16,
                                      ::ml_drift::TensorStorageType::kBuffer,
                                      ::ml_drift::Layout::kBHWC);
  q_desc.SetBHWCShape(::ml_drift::BHWC(1, 10080, 12, 64));
  ::ml_drift::TensorDescriptor k_desc = q_desc;
  ::ml_drift::TensorDescriptor v_desc = q_desc;
  ::ml_drift::TensorDescriptor dst_desc = q_desc;

  EXPECT_TRUE(IsSupportedFlashSdpa(gpu_info, q_desc, k_desc, v_desc,
                                   /*mask_desc=*/nullptr, dst_desc));
  EXPECT_TRUE(IsSupportedFlashSdpa(gpu_info, q_desc, k_desc, v_desc,
                                   /*mask_desc=*/nullptr, dst_desc,
                                   /*use_f32_accum=*/true));

  // Verify Adreno Gen 8 (e.g. S26 Ultra Adreno 840), Mali (e.g. Pixel 8 Pro),
  // Apple Metal, and Nvidia WebGPU support across FP16 and FP32.
  ::ml_drift::TensorDescriptor fp32_q_desc(
      ::ml_drift::DataType::kFloat32, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kBHWC);
  fp32_q_desc.SetBHWCShape(::ml_drift::BHWC(1, 10080, 12, 64));

  ::ml_drift::GpuInfo adreno8xx_info;
  ::ml_drift::GetGpuInfoFromDeviceDescription(
      "Adreno (TM) 840", ::ml_drift::GpuApi::kOpenCl, &adreno8xx_info);
  EXPECT_TRUE(adreno8xx_info.adreno_info.IsAdreno8xx());
  EXPECT_TRUE(IsSupportedFlashSdpa(adreno8xx_info, q_desc, k_desc, v_desc,
                                   /*mask_desc=*/nullptr, dst_desc));
  EXPECT_TRUE(IsSupportedFlashSdpa(adreno8xx_info, fp32_q_desc, fp32_q_desc,
                                   fp32_q_desc, /*mask_desc=*/nullptr,
                                   fp32_q_desc, /*use_f32_accum=*/true));

  ::ml_drift::GpuInfo metal_info;
  ::ml_drift::GetGpuInfoFromDeviceDescription(
      "Apple M3", ::ml_drift::GpuApi::kMetal, &metal_info);
  EXPECT_TRUE(IsSupportedFlashSdpa(metal_info, q_desc, k_desc, v_desc,
                                   /*mask_desc=*/nullptr, dst_desc));
  EXPECT_TRUE(IsSupportedFlashSdpa(metal_info, fp32_q_desc, fp32_q_desc,
                                   fp32_q_desc, /*mask_desc=*/nullptr,
                                   fp32_q_desc, /*use_f32_accum=*/true));

  ::ml_drift::GpuInfo webgpu_info;
  ::ml_drift::GetGpuInfoFromDeviceDescription(
      "NVIDIA GeForce RTX 4090", ::ml_drift::GpuApi::kWebGpu, &webgpu_info);
  EXPECT_TRUE(IsSupportedFlashSdpa(webgpu_info, q_desc, k_desc, v_desc,
                                   /*mask_desc=*/nullptr, dst_desc));
  EXPECT_TRUE(IsSupportedFlashSdpa(webgpu_info, fp32_q_desc, fp32_q_desc,
                                   fp32_q_desc, /*mask_desc=*/nullptr,
                                   fp32_q_desc, /*use_f32_accum=*/true));

  // Unsupported GPU APIs (e.g. Vulkan, OpenGL) should return false and fall
  // back to ML Drift's default SDPA lowering.
  ::ml_drift::GpuInfo vulkan_gpu_info = gpu_info;
  vulkan_gpu_info.gpu_api = ::ml_drift::GpuApi::kVulkan;
  EXPECT_FALSE(IsSupportedFlashSdpa(vulkan_gpu_info, q_desc, k_desc, v_desc,
                                    /*mask_desc=*/nullptr, dst_desc));

  // Single-token decode (`T == 1`) should return false so ML Drift's
  // decode-parallel path handles it.
  ::ml_drift::TensorDescriptor decode_q_desc = q_desc;
  decode_q_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 12, 64));
  EXPECT_FALSE(IsSupportedFlashSdpa(gpu_info, decode_q_desc, k_desc, v_desc,
                                    /*mask_desc=*/nullptr, decode_q_desc));

  // Unmasked MHA with `T >= 1024` and `H <= 40` (`ConvSoftmaxConv` regime)
  // should defer to ML Drift's `ConvSoftmaxConv` kernel.
  ::ml_drift::TensorDescriptor conv_sdpa_desc = q_desc;
  conv_sdpa_desc.SetBHWCShape(::ml_drift::BHWC(1, 1024, 12, 32));
  EXPECT_FALSE(IsSupportedFlashSdpa(gpu_info, conv_sdpa_desc, conv_sdpa_desc,
                                    conv_sdpa_desc, /*mask_desc=*/nullptr,
                                    conv_sdpa_desc));

  // Head dim 256 exceeds the 128 register/shared-memory cap and should fall
  // back.
  ::ml_drift::TensorDescriptor large_h_desc = q_desc;
  large_h_desc.SetBHWCShape(::ml_drift::BHWC(1, 64, 8, 256));
  EXPECT_FALSE(IsSupportedFlashSdpa(gpu_info, large_h_desc, large_h_desc,
                                    large_h_desc, /*mask_desc=*/nullptr,
                                    large_h_desc));

  // Non-multiple-of-4 head dimension (`H = 62`) should be rejected.
  ::ml_drift::TensorDescriptor unaligned_h_desc = q_desc;
  unaligned_h_desc.SetBHWCShape(::ml_drift::BHWC(1, 64, 8, 62));
  EXPECT_FALSE(IsSupportedFlashSdpa(gpu_info, unaligned_h_desc,
                                    unaligned_h_desc, unaligned_h_desc,
                                    /*mask_desc=*/nullptr, unaligned_h_desc));

  // Invalid GQA head ratio (`N_q % N_kv != 0`) should be rejected.
  ::ml_drift::TensorDescriptor bad_kv_desc = k_desc;
  bad_kv_desc.SetBHWCShape(::ml_drift::BHWC(1, 10080, 5, 64));
  EXPECT_FALSE(IsSupportedFlashSdpa(gpu_info, q_desc, bad_kv_desc, bad_kv_desc,
                                    /*mask_desc=*/nullptr, dst_desc));

  // Mismatched `Dst` shape should be rejected.
  ::ml_drift::TensorDescriptor bad_dst_desc = dst_desc;
  bad_dst_desc.SetBHWCShape(::ml_drift::BHWC(1, 5040, 12, 64));
  EXPECT_FALSE(IsSupportedFlashSdpa(gpu_info, q_desc, k_desc, v_desc,
                                    /*mask_desc=*/nullptr, bad_dst_desc));

  // Non-floating-point `Q` tensor should be rejected.
  ::ml_drift::TensorDescriptor int_q_desc(
      ::ml_drift::DataType::kInt32, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kBHWC);
  int_q_desc.SetBHWCShape(::ml_drift::BHWC(1, 10080, 12, 64));
  EXPECT_FALSE(IsSupportedFlashSdpa(gpu_info, int_q_desc, k_desc, v_desc,
                                    /*mask_desc=*/nullptr, dst_desc));

  // Valid broadcast mask `[1, 1, 10080, 10080]` is supported, while mismatched
  // key length (`c != S`) or integer mask type is rejected.
  ::ml_drift::TensorDescriptor valid_mask_desc(
      ::ml_drift::DataType::kBool, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kBHWC);
  valid_mask_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 10080, 10080));
  EXPECT_TRUE(IsSupportedFlashSdpa(gpu_info, q_desc, k_desc, v_desc,
                                   &valid_mask_desc, dst_desc));

  ::ml_drift::TensorDescriptor bad_mask_shape_desc = valid_mask_desc;
  bad_mask_shape_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 10080, 512));
  EXPECT_FALSE(IsSupportedFlashSdpa(gpu_info, q_desc, k_desc, v_desc,
                                    &bad_mask_shape_desc, dst_desc));

  ::ml_drift::TensorDescriptor int_mask_desc(
      ::ml_drift::DataType::kInt32, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kBHWC);
  int_mask_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 10080, 10080));
  EXPECT_FALSE(IsSupportedFlashSdpa(gpu_info, q_desc, k_desc, v_desc,
                                    &int_mask_desc, dst_desc));
}

TEST(IsSupportedFlashSdpaTest, AssemblesCrossBackendAndVendorCode) {
  auto assemble_flash_sdpa = [](const ::ml_drift::GpuInfo& gpu_info,
                                ::ml_drift::CalculationsPrecision precision,
                                MaskMode mask_mode,
                                std::string* out_code) -> absl::Status {
    const ::ml_drift::DataType datatype =
        ::ml_drift::DeduceDataTypeFromPrecision(precision);
    ::ml_drift::GpuModelBuilder builder(gpu_info, {}, precision,
                                        ::ml_drift::TensorStorageType::kBuffer);
    const ::ml_drift::BHWC qkv_shape(1, 10080, 12, 64);
    const ::ml_drift::BHWC mask_shape(1, 1, 10080, 10080);

    auto q = builder.AddTensor(qkv_shape, datatype);
    auto k = builder.AddTensor(qkv_shape, datatype);
    auto v = builder.AddTensor(qkv_shape, datatype);
    auto out = builder.AddTensor(qkv_shape, datatype);

    std::vector<uint32_t> input_ids = {q.id, k.id, v.id};
    std::vector<std::pair<uint32_t, uint32_t>> model_inputs = {
        {q.id, 0}, {k.id, 1}, {v.id, 2}};
    if (mask_mode == MaskMode::kBool) {
      auto mask = builder.AddTensor(mask_shape, ::ml_drift::DataType::kBool);
      input_ids.push_back(mask.id);
      model_inputs.push_back({mask.id, 3});
    } else if (mask_mode == MaskMode::kFloatAdditive) {
      auto mask = builder.AddTensor(mask_shape, datatype);
      input_ids.push_back(mask.id);
      model_inputs.push_back({mask.id, 3});
    }

    ::ml_drift::ScaledDotProductAttentionAttributes attr;
    attr.scale = 0.125f;
    ABSL_RETURN_IF_ERROR(BuildSdpaGpuGraph(input_ids, out.id, attr, &builder));

    ::ml_drift::GpuModel gpu_model;
    ABSL_RETURN_IF_ERROR(
        builder.GetGpuModel(model_inputs, {{out.id, 0}}, &gpu_model));
    for (const auto& node : gpu_model.nodes) {
      if (node.name == "flash_sdpa" && node.gpu_operation != nullptr) {
        *out_code = node.gpu_operation->code_;
        return absl::OkStatus();
      }
    }
    return absl::NotFoundError("flash_sdpa node not found in GpuModel");
  };

  // 1. S26 Ultra (Adreno 840, OpenCL): FP32 uses `max_block_s = 4`
  // (`s_splits = 4, block_s = 4, b_r = 16`, `q_0..q_3` + `acc_0..acc_3` = 32
  // registers in GPRs) and emits `__attribute__((qcom_wave_pair_mode(1)))`;
  // FP16 uses `s_splits = 1, block_s = 16, b_r = 64` and does not emit
  // `qcom_wave_pair_mode`.
  ::ml_drift::GpuInfo s26u_info;
  ::ml_drift::GetGpuInfoFromDeviceDescription(
      "Adreno (TM) 840", ::ml_drift::GpuApi::kOpenCl, &s26u_info);
  s26u_info.opencl_info.supports_fp16 = true;
  std::string s26u_fp32;
  ASSERT_OK(assemble_flash_sdpa(s26u_info,
                                ::ml_drift::CalculationsPrecision::kF32,
                                MaskMode::kNone, &s26u_fp32));
  EXPECT_THAT(s26u_fp32, HasSubstr("__attribute__((qcom_wave_pair_mode(1)))"));
  EXPECT_THAT(s26u_fp32, HasSubstr("int s_grp = tid % 4;"));
  EXPECT_THAT(s26u_fp32, HasSubstr("int q_idx = q_tile * 16 + q_grp;"));
  EXPECT_THAT(s26u_fp32, HasSubstr("float4 acc_3 ="));
  EXPECT_THAT(s26u_fp32, Not(HasSubstr("acc_4")));
  EXPECT_THAT(s26u_fp32, Not(HasSubstr("q_local[")));

  std::string s26u_fp16;
  ASSERT_OK(assemble_flash_sdpa(s26u_info,
                                ::ml_drift::CalculationsPrecision::kF16,
                                MaskMode::kNone, &s26u_fp16));
  EXPECT_THAT(s26u_fp16,
              Not(HasSubstr("__attribute__((qcom_wave_pair_mode(1)))")));
  EXPECT_THAT(s26u_fp16, Not(HasSubstr("q_local[")));

  // 2. S24 Ultra (Adreno 750, OpenCL): FP32 uses `max_block_s = 8`
  // (`s_splits = 2, block_s = 8, b_r = 32`) with `Q` in registers and without
  // `qcom_wave_pair_mode`; FP16 uses `s_splits = 1, block_s = 16, b_r = 64`.
  ::ml_drift::GpuInfo s24u_info;
  ::ml_drift::GetGpuInfoFromDeviceDescription(
      "Adreno (TM) 750", ::ml_drift::GpuApi::kOpenCl, &s24u_info);
  s24u_info.opencl_info.supports_fp16 = true;
  std::string s24u_fp32;
  ASSERT_OK(assemble_flash_sdpa(s24u_info,
                                ::ml_drift::CalculationsPrecision::kF32,
                                MaskMode::kBool, &s24u_fp32));
  EXPECT_THAT(s24u_fp32,
              Not(HasSubstr("__attribute__((qcom_wave_pair_mode(1)))")));
  EXPECT_THAT(s24u_fp32, HasSubstr("int s_grp = tid % 2;"));
  EXPECT_THAT(s24u_fp32, HasSubstr("int q_idx = q_tile * 32 + q_grp;"));
  EXPECT_THAT(s24u_fp32, HasSubstr("float4 acc_7 ="));
  EXPECT_THAT(s24u_fp32, Not(HasSubstr("q_local[")));

  std::string s24u_fp16;
  ASSERT_OK(assemble_flash_sdpa(s24u_info,
                                ::ml_drift::CalculationsPrecision::kF16,
                                MaskMode::kFloatAdditive, &s24u_fp16));
  EXPECT_THAT(s24u_fp16, Not(HasSubstr("q_local[")));

  // 3. Pixel 8 Pro (Mali-G715, OpenCL): stages `Q` in `__local` in both FP32
  // and FP16.
  ::ml_drift::GpuInfo p8pro_info;
  ::ml_drift::GetGpuInfoFromDeviceDescription(
      "Mali-G715", ::ml_drift::GpuApi::kOpenCl, &p8pro_info);
  p8pro_info.opencl_info.supports_fp16 = true;
  std::string p8pro_fp32;
  ASSERT_OK(assemble_flash_sdpa(p8pro_info,
                                ::ml_drift::CalculationsPrecision::kF32,
                                MaskMode::kNone, &p8pro_fp32));
  EXPECT_THAT(p8pro_fp32, HasSubstr("__local float4 q_local[512];"));

  std::string p8pro_fp16;
  ASSERT_OK(assemble_flash_sdpa(p8pro_info,
                                ::ml_drift::CalculationsPrecision::kF16,
                                MaskMode::kNone, &p8pro_fp16));
  EXPECT_THAT(p8pro_fp16, HasSubstr("__local half4 q_local[1024];"));

  // 4. Apple Metal (`GpuApi::kMetal`): assembles cleanly in FP32 and FP16 with
  // boolean and float masks.
  ::ml_drift::GpuInfo metal_info;
  ::ml_drift::GetGpuInfoFromDeviceDescription(
      "Apple M3", ::ml_drift::GpuApi::kMetal, &metal_info);
  for (auto prec : {::ml_drift::CalculationsPrecision::kF32,
                    ::ml_drift::CalculationsPrecision::kF16}) {
    for (auto mask :
         {MaskMode::kNone, MaskMode::kBool, MaskMode::kFloatAdditive}) {
      std::string metal_code;
      ASSERT_OK(assemble_flash_sdpa(metal_info, prec, mask, &metal_code));
      EXPECT_THAT(metal_code,
                  HasSubstr("threadgroup_barrier(mem_flags::mem_threadgroup)"));
    }
  }

  // 5. Nvidia WebGPU (`GpuApi::kWebGpu`): assembles cleanly in FP32 and FP16
  // with boolean and float masks within the 16 KB workgroup storage cap.
  ::ml_drift::GpuInfo webgpu_info;
  ::ml_drift::GetGpuInfoFromDeviceDescription(
      "NVIDIA GeForce RTX 4090", ::ml_drift::GpuApi::kWebGpu, &webgpu_info);
  for (auto prec : {::ml_drift::CalculationsPrecision::kF32,
                    ::ml_drift::CalculationsPrecision::kF16}) {
    for (auto mask :
         {MaskMode::kNone, MaskMode::kBool, MaskMode::kFloatAdditive}) {
      std::string webgpu_code;
      ASSERT_OK(assemble_flash_sdpa(webgpu_info, prec, mask, &webgpu_code));
      EXPECT_THAT(webgpu_code, HasSubstr("workgroupBarrier()"));
    }
  }
}

}  // namespace
}  // namespace litert::ml_drift

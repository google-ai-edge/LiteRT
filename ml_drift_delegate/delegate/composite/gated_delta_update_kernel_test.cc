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

#include "ml_drift_delegate/delegate/composite/gated_delta_update_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/kernels/tests/kernel_test.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/testing_util.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/gated_delta_update_parser.h"

namespace litert::ml_drift {
namespace {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

constexpr int kNumKHeads = 2;
constexpr int kHeadKDim = 128;

class GatedDeltaNetTest : public ::ml_drift::FloatTest {
 public:
  void SetUp() override {
    if (!exec_env) {
      GTEST_SKIP() << "TestExecutionEnvironment not initialized.";
    }
    const auto data_type = ::ml_drift::DeduceDataTypeFromPrecision(precision());
    if (!exec_env->IsStorageSupported(storage(), data_type) ||
        !exec_env->IsStorageSupported(::ml_drift::TensorStorageType::kBuffer,
                                      ::ml_drift::DataType::kFloat32)) {
      GTEST_SKIP() << "Unsupported data type: "
                   << ::ml_drift::ToString(data_type)
                   << " storage type: " << ::ml_drift::ToString(storage());
    }
  }
};

// The kernels that run `gated_delta_update` in GatedDeltaNet block mode.
enum class GdnKernels {
  // One token in one kernel (32-lane wave shuffles).
  kWaveDecode,
  // One token in one kernel (workgroup-memory reductions).
  kSharedMemoryDecode,
  // Parallel prep + subgroup recurrence (32-lane wave shuffles) + norm.
  kWaveRecurrence,
  // Parallel prep + workgroup-memory recurrence + norm.
  kSharedMemoryRecurrence,
};

struct GdnTestCase {
  int num_v_heads = 4;
  int head_v_dim = 128;
  int seq_len = 1;
  // Valid mask of the tokens, or empty for no mask input.
  std::vector<float> mask;
};

std::string Describe(const GdnTestCase& test_case) {
  return absl::StrCat("H_v=", test_case.num_v_heads,
                      " D_v=", test_case.head_v_dim, " T=", test_case.seq_len,
                      " mask=[", absl::StrJoin(test_case.mask, ","), "]");
}

// Deterministic value in [lo, hi) for element `i` of stream `stream`.
float PseudoRandom(int stream, int i, float lo, float hi) {
  uint32_t x = (static_cast<uint32_t>(i) * 2654435761u) ^
               ((static_cast<uint32_t>(stream) + 1u) * 2246822519u);
  x ^= x >> 15;
  x *= 0x2c1b3c6du;
  x ^= x >> 12;
  x *= 0x297a2d39u;
  x ^= x >> 15;
  return lo + (hi - lo) * static_cast<float>(x >> 8) / 16777216.0f;
}

// Rounds to the nearest float16 value (ties to even), like the float16 upload.
// Ignores float16 subnormals, whose error is far below the test tolerances.
float RoundToHalf(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  bits += 0xFFFu + ((bits >> 13) & 1u);
  bits &= ~0x1FFFu;
  float rounded;
  std::memcpy(&rounded, &bits, sizeof(rounded));
  return rounded;
}

struct GdnInputs {
  std::vector<float> conv_out;     // [T, 2 * H_k * D_k + H_v * D_v]
  std::vector<float> zba;          // [T, H_v * D_v + 2 * H_v]
  std::vector<float> state;        // [H_v, D_k, D_v]
  std::vector<float> a_log;        // [H_v]
  std::vector<float> dt_bias;      // [H_v]
  std::vector<float> norm_weight;  // [D_v]
  std::vector<float> mask;         // [T] or empty
};

struct GdnOutputs {
  std::vector<float> output;  // [T, H_v * D_v]
  std::vector<float> state;   // [H_v, D_k, D_v]
};

// Double-precision reference of the composite decomposition: masked q, beta
// and g, L2-normalized q (scaled by 1 / sqrt(D_k)) and k, the gated delta rule
// and the gated RMSNorm x SiLU(z).
GdnOutputs ComputeReference(const GatedDeltaNetAttributes& attr, int seq_len,
                            const GdnInputs& in) {
  const int hk = attr.num_k_heads;
  const int hv = attr.num_v_heads;
  const int dk = attr.head_k_dim;
  const int dv = attr.head_v_dim;
  const int key_dim = hk * dk;
  const int value_dim = hv * dv;
  const int conv_dim = 2 * key_dim + value_dim;
  const int zba_dim = value_dim + 2 * hv;
  std::vector<double> s(in.state.begin(), in.state.end());
  GdnOutputs ref;
  ref.output.assign(seq_len * value_dim, 0.0f);
  std::vector<double> delta(dv);
  std::vector<double> o(dv);
  for (int t = 0; t < seq_len; ++t) {
    const float* x = &in.conv_out[t * conv_dim];
    const float* zba = &in.zba[t * zba_dim];
    const double m = in.mask.empty() ? 1.0 : in.mask[t];
    for (int h = 0; h < hv; ++h) {
      const int h_k = h / (hv / hk);
      const float* q = x + h_k * dk;
      const float* k = x + key_dim + h_k * dk;
      const float* v = x + 2 * key_dim + h * dv;
      double qq = 0.0;
      double kk = 0.0;
      for (int i = 0; i < dk; ++i) {
        qq += m * q[i] * m * q[i];
        kk += static_cast<double>(k[i]) * k[i];
      }
      const double q_scale = m / std::sqrt(qq + attr.eps) / std::sqrt(dk);
      const double k_scale = 1.0 / std::sqrt(kk + attr.eps);
      const double beta = m / (1.0 + std::exp(-zba[value_dim + h]));
      const double gate_in =
          static_cast<double>(zba[value_dim + hv + h]) + in.dt_bias[h];
      const double softplus = std::log1p(std::exp(gate_in));
      const double decay =
          std::exp(-std::exp(static_cast<double>(in.a_log[h])) * softplus * m);
      double* sh = &s[h * dk * dv];
      for (int i = 0; i < dk * dv; ++i) {
        sh[i] *= decay;
      }
      for (int j = 0; j < dv; ++j) {
        double kv = 0.0;
        for (int i = 0; i < dk; ++i) {
          kv += sh[i * dv + j] * k[i] * k_scale;
        }
        delta[j] = (v[j] - kv) * beta;
      }
      for (int i = 0; i < dk; ++i) {
        for (int j = 0; j < dv; ++j) {
          sh[i * dv + j] += k[i] * k_scale * delta[j];
        }
      }
      double sumsq = 0.0;
      for (int j = 0; j < dv; ++j) {
        double acc = 0.0;
        for (int i = 0; i < dk; ++i) {
          acc += sh[i * dv + j] * q[i] * q_scale;
        }
        o[j] = acc;
        sumsq += acc * acc;
      }
      const double inv_rms = 1.0 / std::sqrt(sumsq / dv + attr.norm_eps);
      for (int j = 0; j < dv; ++j) {
        const double z = zba[h * dv + j];
        const double silu = z / (1.0 + std::exp(-z));
        ref.output[t * value_dim + h * dv + j] =
            static_cast<float>(o[j] * inv_rms * in.norm_weight[j] * silu);
      }
    }
  }
  ref.state.assign(s.begin(), s.end());
  return ref;
}

::ml_drift::TensorDescriptor MakeDesc(::ml_drift::DataType data_type,
                                      ::ml_drift::TensorStorageType storage,
                                      const ::ml_drift::BHWC& shape) {
  ::ml_drift::TensorDescriptor desc(data_type, storage,
                                    ::ml_drift::Layout::kBHWC);
  desc.SetBHWCShape(shape);
  return desc;
}

absl::Status CheckAllNear(absl::string_view name,
                          const std::vector<float>& expected,
                          const std::vector<float>& actual, float tol) {
  int num_mismatches = 0;
  float max_diff = 0.0f;
  std::string first_mismatches;
  for (size_t i = 0; i < expected.size(); ++i) {
    const float diff = std::abs(actual[i] - expected[i]);
    if (!(diff <= tol * std::max(1.0f, std::abs(expected[i])))) {
      if (num_mismatches < 4) {
        absl::StrAppend(&first_mismatches, " [", i, "] expected ", expected[i],
                        " got ", actual[i], ";");
      }
      ++num_mismatches;
      max_diff = std::isnan(diff) ? diff : std::max(max_diff, diff);
    }
  }
  if (num_mismatches > 0) {
    return absl::InternalError(absl::StrCat(
        name, ": ", num_mismatches, " of ", expected.size(),
        " elements mismatch, max diff ", max_diff, ":", first_mismatches));
  }
  return absl::OkStatus();
}

// Runs `kernels` on `test_case` and compares the output and the new state with
// the reference. Returns an Unimplemented error if the GPU lacks the features
// of `kernels`.
absl::Status RunGatedDeltaNetTest(::ml_drift::TestExecutionEnvironment& env,
                                  ::ml_drift::CalculationsPrecision precision,
                                  ::ml_drift::TensorStorageType storage,
                                  GdnKernels kernels,
                                  const GdnTestCase& test_case) {
  GatedDeltaNetAttributes attr;
  attr.num_k_heads = kNumKHeads;
  attr.num_v_heads = test_case.num_v_heads;
  attr.head_k_dim = kHeadKDim;
  attr.head_v_dim = test_case.head_v_dim;
  attr.eps = 1e-6f;
  attr.norm_eps = 1e-6f;
  const int seq_len = test_case.seq_len;
  const int hv = attr.num_v_heads;
  const int dk = attr.head_k_dim;
  const int dv = attr.head_v_dim;
  const int value_dim = hv * dv;
  const int conv_dim = 2 * attr.num_k_heads * dk + value_dim;
  const int zba_dim = value_dim + 2 * hv;
  const bool has_mask = !test_case.mask.empty();
  const bool is_decode = kernels == GdnKernels::kWaveDecode ||
                         kernels == GdnKernels::kSharedMemoryDecode;
  const ::ml_drift::DataType data_type =
      ::ml_drift::DeduceDataTypeFromPrecision(precision);
  const bool is_f16 = data_type == ::ml_drift::DataType::kFloat16;

  // float16 runs compare with the reference of the float16-rounded inputs. The
  // recurrent state is float32 in all runs.
  auto make_data = [&](int stream, int size, float lo, float hi,
                       bool round_to_half) {
    std::vector<float> data(size);
    for (int i = 0; i < size; ++i) {
      data[i] = PseudoRandom(stream, i, lo, hi);
      if (round_to_half) {
        data[i] = RoundToHalf(data[i]);
      }
    }
    return data;
  };
  GdnInputs in;
  in.conv_out = make_data(0, seq_len * conv_dim, -1.0f, 1.0f, is_f16);
  in.zba = make_data(1, seq_len * zba_dim, -2.0f, 2.0f, is_f16);
  in.a_log = make_data(2, hv, -1.0f, 1.0f, is_f16);
  in.dt_bias = make_data(3, hv, -1.0f, 1.0f, is_f16);
  in.norm_weight = make_data(4, dv, 0.5f, 1.5f, is_f16);
  in.state = make_data(5, hv * dk * dv, -0.5f, 0.5f, /*round_to_half=*/false);
  in.mask = test_case.mask;
  const GdnOutputs ref = ComputeReference(attr, seq_len, in);

  // The shapes follow how the delegate maps the composite operands to BHWC:
  // [1, T, C] -> (1, 1, T, C), [1, T] -> (1, 1, 1, T), [N] -> (N, 1, 1, 1) and
  // the state [1, H_v, D_k, D_v] -> (1, H_v, D_k, D_v).
  ::ml_drift::TensorDescriptor conv_out_desc =
      MakeDesc(data_type, storage, ::ml_drift::BHWC(1, 1, seq_len, conv_dim));
  ::ml_drift::TensorDescriptor zba_desc =
      MakeDesc(data_type, storage, ::ml_drift::BHWC(1, 1, seq_len, zba_dim));
  ::ml_drift::TensorDescriptor a_log_desc =
      MakeDesc(data_type, storage, ::ml_drift::BHWC(hv, 1, 1, 1));
  ::ml_drift::TensorDescriptor dt_bias_desc =
      MakeDesc(data_type, storage, ::ml_drift::BHWC(hv, 1, 1, 1));
  ::ml_drift::TensorDescriptor norm_weight_desc =
      MakeDesc(data_type, storage, ::ml_drift::BHWC(dv, 1, 1, 1));
  ::ml_drift::TensorDescriptor mask_desc =
      MakeDesc(data_type, storage, ::ml_drift::BHWC(1, 1, 1, seq_len));
  ::ml_drift::TensorDescriptor output_desc =
      MakeDesc(data_type, storage, ::ml_drift::BHWC(1, 1, seq_len, value_dim));
  ::ml_drift::TensorDescriptor state_desc = MakeDesc(
      ::ml_drift::DataType::kFloat32, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::BHWC(1, hv, dk, dv));
  const int key_dim = attr.num_k_heads * dk;
  ::ml_drift::TensorDescriptor qk_norm_desc = MakeDesc(
      ::ml_drift::DataType::kFloat32, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::BHWC(1, 1, seq_len, 2 * key_dim));
  ::ml_drift::TensorDescriptor gates_desc = MakeDesc(
      ::ml_drift::DataType::kFloat32, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::BHWC(1, 1, seq_len, 4 * hv));
  ::ml_drift::TensorDescriptor raw_desc = MakeDesc(
      ::ml_drift::DataType::kFloat32, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::BHWC(1, 1, seq_len, value_dim));

  // Create the operations before uploading the data, so that the operation
  // definitions do not copy it.
  std::vector<::ml_drift::TensorDescriptor*> srcs = {
      &conv_out_desc, &zba_desc, &a_log_desc, &dt_bias_desc};
  if (is_decode) {
    srcs.push_back(&norm_weight_desc);
  }
  if (has_mask) {
    srcs.push_back(&mask_desc);
  }
  ::ml_drift::OperationDef def;
  for (const ::ml_drift::TensorDescriptor* src : srcs) {
    def.src_tensors.push_back(*src);
  }
  std::unique_ptr<::ml_drift::GPUOperation> prep_op;
  std::unique_ptr<::ml_drift::GPUOperation> op;
  std::unique_ptr<::ml_drift::GPUOperation> norm_op;
  if (is_decode) {
    def.dst_tensors = {output_desc, state_desc};
    ABSL_ASSIGN_OR_RETURN(
        op, CreateGatedDeltaNetDecode(
                def, attr, env.GetGpuInfo(),
                /*use_wave_ops=*/kernels == GdnKernels::kWaveDecode));
  } else {
    def.dst_tensors = {qk_norm_desc, gates_desc};
    ABSL_ASSIGN_OR_RETURN(prep_op, CreateGatedDeltaNetPrep(def, attr));
    ::ml_drift::OperationDef rec_def;
    rec_def.src_tensors = {conv_out_desc, qk_norm_desc, gates_desc};
    rec_def.dst_tensors = {raw_desc, state_desc};
    ABSL_ASSIGN_OR_RETURN(
        op, CreateGatedDeltaNetRecurrence(
                rec_def, attr, env.GetGpuInfo(),
                /*use_wave_ops=*/kernels == GdnKernels::kWaveRecurrence));
    ::ml_drift::OperationDef norm_def;
    norm_def.src_tensors = {raw_desc, zba_desc, norm_weight_desc};
    norm_def.dst_tensors = {output_desc};
    ABSL_ASSIGN_OR_RETURN(norm_op, CreateGatedDeltaNetNorm(norm_def, attr));
  }

  conv_out_desc.UploadData(in.conv_out.data());
  zba_desc.UploadData(in.zba.data());
  a_log_desc.UploadData(in.a_log.data());
  dt_bias_desc.UploadData(in.dt_bias.data());
  norm_weight_desc.UploadData(in.norm_weight.data());
  if (has_mask) {
    mask_desc.UploadData(in.mask.data());
  }
  // Sentinels make elements that the kernels do not write fail the check. The
  // state is read and updated in place.
  const std::vector<float> qk_sentinel(seq_len * 2 * key_dim, -100.0f);
  const std::vector<float> gates_sentinel(seq_len * 4 * hv, -100.0f);
  const std::vector<float> sentinel(seq_len * value_dim, -100.0f);
  qk_norm_desc.UploadData(qk_sentinel.data());
  gates_desc.UploadData(gates_sentinel.data());
  output_desc.UploadData(sentinel.data());
  raw_desc.UploadData(sentinel.data());
  state_desc.UploadData(in.state.data());

  if (is_decode) {
    ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
        srcs, {&output_desc, &state_desc}, std::move(op)));
  } else {
    ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
        srcs, {&qk_norm_desc, &gates_desc}, std::move(prep_op)));
    ABSL_RETURN_IF_ERROR(
        env.ExecuteGPUOperation({&conv_out_desc, &qk_norm_desc, &gates_desc},
                                {&raw_desc, &state_desc}, std::move(op)));
    ABSL_RETURN_IF_ERROR(
        env.ExecuteGPUOperation({&raw_desc, &zba_desc, &norm_weight_desc},
                                {&output_desc}, std::move(norm_op)));
  }

  GdnOutputs gpu;
  gpu.output.resize(seq_len * value_dim);
  output_desc.DownloadData(gpu.output.data());
  gpu.state.resize(hv * dk * dv);
  state_desc.DownloadData(gpu.state.data());
  const float output_tol = is_f16 ? 5e-3f : 1e-3f;
  ABSL_RETURN_IF_ERROR(
      CheckAllNear("recurrent_state", ref.state, gpu.state, 1e-3f));
  ABSL_RETURN_IF_ERROR(
      CheckAllNear("output", ref.output, gpu.output, output_tol));
  return absl::OkStatus();
}

// Runs all test cases; the first failure names its case.
absl::Status RunTestCases(::ml_drift::CalculationsPrecision precision,
                          ::ml_drift::TensorStorageType storage,
                          GdnKernels kernels,
                          const std::vector<GdnTestCase>& test_cases) {
  for (const GdnTestCase& test_case : test_cases) {
    const absl::Status status =
        RunGatedDeltaNetTest(*exec_env, precision, storage, kernels, test_case);
    if (!status.ok()) {
      return absl::Status(status.code(), absl::StrCat(Describe(test_case), ": ",
                                                      status.message()));
    }
  }
  return absl::OkStatus();
}

std::vector<GdnTestCase> DecodeTestCases() {
  return {
      {.seq_len = 1},
      {.seq_len = 1, .mask = {1.0f}},
      {.seq_len = 1, .mask = {0.0f}},
      {.head_v_dim = 64, .seq_len = 1, .mask = {1.0f}},
      {.num_v_heads = 2, .head_v_dim = 16, .seq_len = 1},
  };
}

// Cases of the recurrence kernels: one or several tokens, with and without a
// mask, padded tails, D_v = 64 and 16 and GQA ratios 2 and 1.
std::vector<GdnTestCase> RecurrenceTestCases() {
  return {
      {.seq_len = 1},
      {.seq_len = 1, .mask = {1.0f}},
      {.seq_len = 1, .mask = {0.0f}},
      {.seq_len = 8},
      {.seq_len = 8, .mask = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f}},
      {.seq_len = 6, .mask = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
      {.head_v_dim = 64,
       .seq_len = 7,
       .mask = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f}},
      {.num_v_heads = 2, .head_v_dim = 16, .seq_len = 5},
  };
}

TEST_P(GatedDeltaNetTest, WaveDecodeKernel) {
  const absl::Status status = RunTestCases(
      precision(), storage(), GdnKernels::kWaveDecode, DecodeTestCases());
  if (absl::IsUnimplemented(status)) {
    GTEST_SKIP() << status.message();
  }
  ASSERT_OK(status);
}

TEST_P(GatedDeltaNetTest, SharedMemoryDecodeKernel) {
  ASSERT_OK(RunTestCases(precision(), storage(),
                         GdnKernels::kSharedMemoryDecode, DecodeTestCases()));
}

TEST_P(GatedDeltaNetTest, WaveRecurrenceAndNorm) {
  const absl::Status status =
      RunTestCases(precision(), storage(), GdnKernels::kWaveRecurrence,
                   RecurrenceTestCases());
  if (absl::IsUnimplemented(status)) {
    GTEST_SKIP() << status.message();
  }
  ASSERT_OK(status);
}

TEST_P(GatedDeltaNetTest, SharedMemoryRecurrenceAndNorm) {
  ASSERT_OK(RunTestCases(precision(), storage(),
                         GdnKernels::kSharedMemoryRecurrence,
                         RecurrenceTestCases()));
}

INSTANTIATE_TEST_SUITE_P(
    GatedDeltaNetTestSuite, GatedDeltaNetTest,
    Combine(ValuesIn({::ml_drift::CalculationsPrecision::kF32,
                      ::ml_drift::CalculationsPrecision::kF16}),
            ValuesIn({::ml_drift::TensorStorageType::kBuffer})),
    [](const TestParamInfo<GatedDeltaNetTest::ParamType>& info) {
      return ::ml_drift::ToString(info.param);
    });

}  // namespace
}  // namespace litert::ml_drift

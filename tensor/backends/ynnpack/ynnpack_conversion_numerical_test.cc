/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/testing/numerical_test_bridge.h"
#include "tensor/backends/testing/numerical_test_suite.h"
#include "tensor/backends/ynnpack/arithmetic.h"
#include "tensor/backends/ynnpack/conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/runners/ynnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {
namespace {

using ::absl_testing::StatusIs;

template <typename T>
absl::Span<const std::byte> AsByteSpan(const std::vector<T>& data) {
  return absl::MakeConstSpan(reinterpret_cast<const std::byte*>(data.data()),
                             data.size() * sizeof(T));
}

class YnnpackTestBackendBridge : public TestBackendBridge {
 public:
  ~YnnpackTestBackendBridge() override = default;

  absl::Status Initialize() override {
    // YNNPACK has no global initialization entry point.
    return absl::OkStatus();
  }

  absl::Status BuildGraph(absl::Span<const TensorHandle> inputs,
                          absl::Span<const TensorHandle> outputs) override {
    std::vector<TensorHandle> output_handles(outputs.begin(), outputs.end());
    auto runner_or = YnnpackRunner::Create(output_handles);
    if (!runner_or.ok()) {
      absl::Status s = runner_or.status();
      // Operations without a YNNPACK lowering are reported as invalid
      // arguments by the conversion.
      if (s.code() == absl::StatusCode::kInvalidArgument &&
          absl::StrContains(s.message(), "does not implement")) {
        return absl::UnimplementedError(s.message());
      }
      return s;
    }
    runner_ = std::make_unique<YnnpackRunner>(std::move(*runner_or));
    return absl::OkStatus();
  }

  absl::Status SetInput(const TensorHandle& tensor,
                        absl::Span<const std::byte> data) override {
    if (runner_ == nullptr) {
      return absl::FailedPreconditionError("Runner is not initialized");
    }
    return runner_->SetInput(tensor, data);
  }

  absl::Status Execute() override {
    if (runner_ == nullptr) {
      return absl::FailedPreconditionError("Runner is not initialized");
    }
    return runner_->Run();
  }

  absl::Status GetOutput(const TensorHandle& tensor,
                         absl::Span<std::byte> data) override {
    if (runner_ == nullptr) {
      return absl::FailedPreconditionError("Runner is not initialized");
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(auto output_bytes, runner_->ReadOutput(tensor));
    if (output_bytes.size() != data.size()) {
      return absl::InvalidArgumentError("Output buffer size mismatch");
    }
    std::memcpy(data.data(), output_bytes.data(), data.size());
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<YnnpackRunner> runner_;
};

struct YnnpackTraits {
  using Tag = YnnpackMixinTag;
  static std::unique_ptr<TestBackendBridge> CreateBridge() {
    return std::make_unique<YnnpackTestBackendBridge>();
  }
};

INSTANTIATE_TYPED_TEST_SUITE_P(Ynnpack, NumericalTestSuite, YnnpackTraits);

using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::status::IsOkAndHolds;
using YnnTensor = Tensor<YnnpackMixinTag>;

std::shared_ptr<OwningCpuBuffer> PackSignedInt4(
    absl::Span<const int8_t> values) {
  std::vector<int4_t> packed((values.size() + 1) / 2);
  for (size_t i = 0; i < values.size(); ++i) {
    if ((i & 1) == 0) {
      packed[i >> 1].a = values[i];
    } else {
      packed[i >> 1].b = values[i];
    }
  }
  return OwningCpuBuffer::Copy<Type::kI4>(packed);
}

std::shared_ptr<OwningCpuBuffer> PackSignedInt2(
    absl::Span<const int8_t> values) {
  std::vector<int2_t> packed((values.size() + 3) / 4);
  for (size_t i = 0; i < values.size(); ++i) {
    switch (i & 3) {
      case 0:
        packed[i >> 2].a = values[i];
        break;
      case 1:
        packed[i >> 2].b = values[i];
        break;
      case 2:
        packed[i >> 2].c = values[i];
        break;
      case 3:
        packed[i >> 2].d = values[i];
        break;
    }
  }
  return OwningCpuBuffer::Copy<Type::kI2>(packed);
}

std::vector<float> ReferenceMatMulNk(absl::Span<const float> lhs, size_t m,
                                     size_t k, absl::Span<const float> rhs_nk,
                                     size_t n,
                                     absl::Span<const float> bias = {}) {
  std::vector<float> out(m * n, 0.0f);
  for (size_t row = 0; row < m; ++row) {
    for (size_t col = 0; col < n; ++col) {
      double acc = bias.empty() ? 0.0 : bias[col];
      for (size_t i = 0; i < k; ++i) {
        acc += static_cast<double>(lhs[row * k + i]) *
               static_cast<double>(rhs_nk[col * k + i]);
      }
      out[row * n + col] = static_cast<float>(acc);
    }
  }
  return out;
}

TEST(YnnpackQuantizationTest, BlockwiseDequantizeFloatZeroPointsInt4AndInt2) {
  constexpr int kRows = 4;
  constexpr int kCols = 32;
  constexpr int kBlockSize = 16;
  constexpr int kNumBlocks = kCols / kBlockSize;

  std::vector<int8_t> q4_vals(kRows * kCols);
  std::vector<int8_t> q2_vals(kRows * kCols);
  for (int i = 0; i < kRows * kCols; ++i) {
    q4_vals[i] = static_cast<int8_t>((i % 15) - 7);
    q2_vals[i] = static_cast<int8_t>((i % 4) - 2);
  }
  std::vector<float> scales(kRows * kNumBlocks);
  std::vector<float> float_zps(kRows * kNumBlocks);
  for (int b = 0; b < kRows * kNumBlocks; ++b) {
    scales[b] = 0.05f + 0.02f * static_cast<float>(b);
    float_zps[b] = -1.35f + 0.45f * static_cast<float>(b);
  }

  std::vector<float> expected_q4(kRows * kCols);
  std::vector<float> expected_q2(kRows * kCols);
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols; ++c) {
      const int blk = r * kNumBlocks + (c / kBlockSize);
      expected_q4[r * kCols + c] =
          scales[blk] *
          (static_cast<float>(q4_vals[r * kCols + c]) - float_zps[blk]);
      expected_q2[r * kCols + c] =
          scales[blk] *
          (static_cast<float>(q2_vals[r * kCols + c]) - float_zps[blk]);
    }
  }

  auto lazy_q4 = BlockwiseQuantization::CreateLazily(
      kBlockSize, /*quantized_dimension=*/0, scales.size(),
      [scales, float_zps]() -> BlockwiseQuantization::
                                   MaterializedBlockwiseParams {
        return {.scales = scales, .float_zero_points = float_zps};
      },
      BlockwiseQuantization::LazyOptions{
          .zero_points =
              BlockwiseQuantization::ZeroPointPresence::kMayHaveFloat,
      });
  YnnTensor w_q4({.type = Type::kI4,
                  .shape = {kRows, kCols},
                  .buffer = PackSignedInt4(q4_vals),
                  .quantization = lazy_q4});
  YnnTensor w_q2({.type = Type::kI2,
                  .shape = {kRows, kCols},
                  .buffer = PackSignedInt2(q2_vals),
                  .quantization = std::make_shared<BlockwiseQuantization>(
                      scales, ZeroPoints::Float(float_zps), kBlockSize,
                      /*quantized_dimension=*/0)});

  YnnTensor out_q4 = Dequantize(w_q4);
  YnnTensor out_q2 = Dequantize(w_q2);

  ASSERT_OK_AND_ASSIGN(YnnpackRunner runner,
                       YnnpackRunner::Create({out_q4, out_q2}));
  ASSERT_OK(runner.Run());

  EXPECT_THAT(runner.ReadOutputAs<float>(out_q4),
              IsOkAndHolds(Pointwise(FloatNear(1e-5f), expected_q4)));
  EXPECT_THAT(runner.ReadOutputAs<float>(out_q2),
              IsOkAndHolds(Pointwise(FloatNear(1e-5f), expected_q2)));
}

TEST(YnnpackQuantizationTest,
     FullyConnectedAllNativeGgufSchemesQ8_Q6Block16_Q5K_Q4K_Q2K) {
  constexpr int kBatch = 2;
  constexpr int kSeq = 3;
  constexpr int kM = kBatch * kSeq;
  constexpr int kK = 64;
  constexpr int kN = 8;

  std::vector<float> input_data(kM * kK);
  for (int i = 0; i < kM * kK; ++i) {
    input_data[i] = 0.25f * std::sin(0.17f * static_cast<float>(i)) - 0.1f;
  }
  std::vector<float> bias_data(kN);
  for (int n = 0; n < kN; ++n) {
    bias_data[n] = -0.3f + 0.1f * static_cast<float>(n);
  }

  auto make_dequant_nk = [](absl::Span<const int8_t> q, int block_size,
                            absl::Span<const float> scales,
                            absl::Span<const float> float_zps) {
    const int num_blocks = kK / block_size;
    std::vector<float> w_f32(kN * kK);
    for (int n = 0; n < kN; ++n) {
      for (int k = 0; k < kK; ++k) {
        const int idx = n * num_blocks + (k / block_size);
        const float zp = float_zps.empty() ? 0.0f : float_zps[idx];
        w_f32[n * kK + k] =
            scales[idx] * (static_cast<float>(q[n * kK + k]) - zp);
      }
    }
    return w_f32;
  };

  std::vector<int8_t> q8_vals(kN * kK);
  std::vector<int8_t> q4_vals(kN * kK);
  std::vector<int8_t> q2_vals(kN * kK);
  for (int i = 0; i < kN * kK; ++i) {
    q8_vals[i] = static_cast<int8_t>(((i * 7) % 121) - 60);
    q4_vals[i] = static_cast<int8_t>(((i * 5) % 15) - 7);
    q2_vals[i] = static_cast<int8_t>(((i * 3) % 4) - 2);
  }
  // Exercise extreme int8 endpoints (-128 and 127) as well.
  q8_vals[0] = -128;
  q8_vals[1] = 127;

  // Block-32 parameters (Q8_0, Q5_K, Q4_K).
  const int kBlocks32 = kK / 32;
  std::vector<float> scales32(kN * kBlocks32);
  std::vector<float> zps32(kN * kBlocks32);
  for (int i = 0; i < kN * kBlocks32; ++i) {
    scales32[i] = 0.015f + 0.003f * static_cast<float>(i % 7);
    zps32[i] = -2.25f + 0.65f * static_cast<float>(i % 5);
  }

  // Block-16 parameters (Q6_K native block-16, Q2_K native int2 block-16).
  const int kBlocks16 = kK / 16;
  std::vector<float> scales16(kN * kBlocks16);
  std::vector<float> zps16(kN * kBlocks16);
  std::vector<int64_t> int_zps16(kN * kBlocks16);
  for (int i = 0; i < kN * kBlocks16; ++i) {
    scales16[i] = 0.04f + 0.005f * static_cast<float>(i % 9);
    zps16[i] = -0.75f + 0.35f * static_cast<float>(i % 6);
    int_zps16[i] = (i % 5) - 2;
  }

  YnnTensor x({.type = Type::kFP32, .shape = {kBatch, kSeq, kK}});
  YnnTensor bias({.type = Type::kFP32,
                  .shape = {kN},
                  .buffer = OwningCpuBuffer::Copy<Type::kFP32>(bias_data)});

  // 1. Q8_0: int8 symmetric block-32.
  YnnTensor w_q8_0({.type = Type::kI8,
                    .shape = {kN, kK},
                    .buffer = OwningCpuBuffer::Copy<Type::kI8>(q8_vals),
                    .quantization = std::make_shared<BlockwiseQuantization>(
                        scales32, std::vector<int64_t>{}, 32, 0)});
  // 2. Q6_K: int8 symmetric block-16.
  YnnTensor w_q6_k({.type = Type::kI8,
                    .shape = {kN, kK},
                    .buffer = OwningCpuBuffer::Copy<Type::kI8>(q8_vals),
                    .quantization = std::make_shared<BlockwiseQuantization>(
                        scales16, std::vector<int64_t>{}, 16, 0)});
  // 3. Q5_K: int8 affine float-zp block-32.
  YnnTensor w_q5_k({.type = Type::kI8,
                    .shape = {kN, kK},
                    .buffer = OwningCpuBuffer::Copy<Type::kI8>(q8_vals),
                    .quantization = std::make_shared<BlockwiseQuantization>(
                        scales32, ZeroPoints::Float(zps32), 32, 0)});
  // 4. Q4_K / Q4_1: int4 affine float-zp block-32.
  YnnTensor w_q4_k({.type = Type::kI4,
                    .shape = {kN, kK},
                    .buffer = PackSignedInt4(q4_vals),
                    .quantization = std::make_shared<BlockwiseQuantization>(
                        scales32, ZeroPoints::Float(zps32), 32, 0)});
  // 5. Q2_K: int2 affine float-zp block-16.
  YnnTensor w_q2_k({.type = Type::kI2,
                    .shape = {kN, kK},
                    .buffer = PackSignedInt2(q2_vals),
                    .quantization = std::make_shared<BlockwiseQuantization>(
                        scales16, ZeroPoints::Float(zps16), 16, 0)});
  // 6. Asymmetric integer zero-point block-16 int4.
  YnnTensor w_q4_intzp({.type = Type::kI4,
                        .shape = {kN, kK},
                        .buffer = PackSignedInt4(q4_vals),
                        .quantization = std::make_shared<BlockwiseQuantization>(
                            scales16, int_zps16, 16, 0)});

  YnnTensor out_q8_0 = FullyConnected(x, w_q8_0, bias);
  YnnTensor out_q6_k = FullyConnected(x, w_q6_k, bias);
  YnnTensor out_q5_k = FullyConnected(x, w_q5_k, bias);
  YnnTensor out_q4_k = FullyConnected(x, w_q4_k, bias);
  YnnTensor out_q2_k = FullyConnected(x, w_q2_k, bias);
  YnnTensor out_q4_intzp = FullyConnected(x, w_q4_intzp);

  ASSERT_OK_AND_ASSIGN(
      YnnpackRunner runner,
      YnnpackRunner::Create(
          {out_q8_0, out_q6_k, out_q5_k, out_q4_k, out_q2_k, out_q4_intzp}));
  ASSERT_OK(runner.SetInput(x, input_data));
  ASSERT_OK(runner.Run());

  std::vector<float> int_zps16_f32(int_zps16.begin(), int_zps16.end());
  EXPECT_THAT(runner.ReadOutputAs<float>(out_q8_0),
              IsOkAndHolds(Pointwise(
                  FloatNear(0.05f),
                  ReferenceMatMulNk(input_data, kM, kK,
                                    make_dequant_nk(q8_vals, 32, scales32, {}),
                                    kN, bias_data))));
  EXPECT_THAT(runner.ReadOutputAs<float>(out_q6_k),
              IsOkAndHolds(Pointwise(
                  FloatNear(0.12f),
                  ReferenceMatMulNk(input_data, kM, kK,
                                    make_dequant_nk(q8_vals, 16, scales16, {}),
                                    kN, bias_data))));
  EXPECT_THAT(
      runner.ReadOutputAs<float>(out_q5_k),
      IsOkAndHolds(Pointwise(
          FloatNear(0.06f),
          ReferenceMatMulNk(input_data, kM, kK,
                            make_dequant_nk(q8_vals, 32, scales32, zps32), kN,
                            bias_data))));
  EXPECT_THAT(
      runner.ReadOutputAs<float>(out_q4_k),
      IsOkAndHolds(Pointwise(
          FloatNear(0.03f),
          ReferenceMatMulNk(input_data, kM, kK,
                            make_dequant_nk(q4_vals, 32, scales32, zps32), kN,
                            bias_data))));
  EXPECT_THAT(
      runner.ReadOutputAs<float>(out_q2_k),
      IsOkAndHolds(Pointwise(
          FloatNear(0.03f),
          ReferenceMatMulNk(input_data, kM, kK,
                            make_dequant_nk(q2_vals, 16, scales16, zps16), kN,
                            bias_data))));
  EXPECT_THAT(
      runner.ReadOutputAs<float>(out_q4_intzp),
      IsOkAndHolds(Pointwise(
          FloatNear(0.05f),
          ReferenceMatMulNk(
              input_data, kM, kK,
              make_dequant_nk(q4_vals, 16, scales16, int_zps16_f32), kN))));
}

TEST(YnnpackQuantizationTest,
     NativeBlock32AndBlock64PreservedAndSequentialMixedBlock16) {
  constexpr int kM = 5;
  constexpr int kK = 64;
  constexpr int kH = 64;
  constexpr int kOut = 12;

  std::vector<float> input_data(kM * kK);
  for (int i = 0; i < kM * kK; ++i) {
    input_data[i] = 0.3f * std::sin(0.19f * static_cast<float>(i)) + 0.05f;
  }

  auto make_q4 = [](int rows, int cols, int seed) {
    std::vector<int8_t> vals(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
      vals[i] = static_cast<int8_t>(((i * seed + 3) % 15) - 7);
    }
    return vals;
  };
  auto make_q8 = [](int rows, int cols, int seed) {
    std::vector<int8_t> vals(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
      vals[i] = static_cast<int8_t>(((i * seed + 11) % 121) - 60);
    }
    return vals;
  };
  auto dequant_nk = [](absl::Span<const int8_t> q, int rows, int cols,
                       int block_size, absl::Span<const float> scales,
                       absl::Span<const float> zps) {
    const int num_blocks = cols / block_size;
    std::vector<float> out(rows * cols);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        const int idx = r * num_blocks + (c / block_size);
        const float zp = zps.empty() ? 0.0f : zps[idx];
        out[r * cols + c] =
            scales[idx] * (static_cast<float>(q[r * cols + c]) - zp);
      }
    }
    return out;
  };

  // Pure block-32 gate + up projections sharing `x` (no block-16 sibling on
  // `x`, so native `eff_block_size = 32` is preserved), followed sequentially
  // by a block-16 `down_proj` (`Q4_K_M` MLP pattern).
  const std::vector<int8_t> gate_q4 = make_q4(kH, kK, 5);
  const std::vector<int8_t> up_q8 = make_q8(kH, kK, 7);
  std::vector<float> gate_scales(kH * (kK / 32));
  std::vector<float> gate_zps(kH * (kK / 32));
  std::vector<float> up_scales(kH * (kK / 32));
  for (size_t i = 0; i < gate_scales.size(); ++i) {
    gate_scales[i] = 0.02f + 0.004f * static_cast<float>(i % 6);
    gate_zps[i] = -1.5f + 0.5f * static_cast<float>(i % 5);
    up_scales[i] = 0.01f + 0.002f * static_cast<float>(i % 7);
  }

  const std::vector<int8_t> down_q8 = make_q8(kOut, kH, 13);
  std::vector<float> down_scales(kOut * (kH / 16));
  for (size_t i = 0; i < down_scales.size(); ++i) {
    down_scales[i] = 0.015f + 0.003f * static_cast<float>(i % 5);
  }

  // Also test native block-64 affine int4 on its own input tensor `x64`.
  const std::vector<int8_t> b64_q4 = make_q4(kOut, kK, 9);
  std::vector<float> b64_scales(kOut * (kK / 64));
  std::vector<float> b64_zps(kOut * (kK / 64));
  for (size_t i = 0; i < b64_scales.size(); ++i) {
    b64_scales[i] = 0.03f + 0.005f * static_cast<float>(i % 4);
    b64_zps[i] = -2.0f + 0.75f * static_cast<float>(i % 5);
  }

  YnnTensor x({.type = Type::kFP32, .shape = {1, kM, kK}});
  YnnTensor x64({.type = Type::kFP32, .shape = {1, kM, kK}});

  YnnTensor w_gate({.type = Type::kI4,
                    .shape = {kH, kK},
                    .buffer = PackSignedInt4(gate_q4),
                    .quantization = std::make_shared<BlockwiseQuantization>(
                        gate_scales, ZeroPoints::Float(gate_zps), 32, 0)});
  YnnTensor w_up({.type = Type::kI8,
                  .shape = {kH, kK},
                  .buffer = OwningCpuBuffer::Copy<Type::kI8>(up_q8),
                  .quantization = std::make_shared<BlockwiseQuantization>(
                      up_scales, std::vector<int64_t>{}, 32, 0)});
  YnnTensor w_down({.type = Type::kI8,
                    .shape = {kOut, kH},
                    .buffer = OwningCpuBuffer::Copy<Type::kI8>(down_q8),
                    .quantization = std::make_shared<BlockwiseQuantization>(
                        down_scales, std::vector<int64_t>{}, 16, 0)});
  YnnTensor w_b64({.type = Type::kI4,
                   .shape = {kOut, kK},
                   .buffer = PackSignedInt4(b64_q4),
                   .quantization = std::make_shared<BlockwiseQuantization>(
                       b64_scales, ZeroPoints::Float(b64_zps), 64, 0)});

  YnnTensor gate_out = FullyConnected(x, w_gate);
  YnnTensor up_out = FullyConnected(x, w_up);
  YnnTensor hidden = Mul(gate_out, up_out);
  YnnTensor mlp_out = FullyConnected(hidden, w_down);
  YnnTensor out_b64 = FullyConnected(x64, w_b64);

  ASSERT_OK_AND_ASSIGN(
      YnnpackRunner runner,
      YnnpackRunner::Create({gate_out, up_out, mlp_out, out_b64}));
  ASSERT_OK(runner.SetInput(x, input_data));
  ASSERT_OK(runner.SetInput(x64, input_data));
  ASSERT_OK(runner.Run());

  const std::vector<float> deq_gate =
      dequant_nk(gate_q4, kH, kK, 32, gate_scales, gate_zps);
  const std::vector<float> deq_up =
      dequant_nk(up_q8, kH, kK, 32, up_scales, {});
  const std::vector<float> deq_down =
      dequant_nk(down_q8, kOut, kH, 16, down_scales, {});
  const std::vector<float> deq_b64 =
      dequant_nk(b64_q4, kOut, kK, 64, b64_scales, b64_zps);

  const std::vector<float> ref_gate =
      ReferenceMatMulNk(input_data, kM, kK, deq_gate, kH);
  const std::vector<float> ref_up =
      ReferenceMatMulNk(input_data, kM, kK, deq_up, kH);
  std::vector<float> ref_hidden(kM * kH);
  for (int i = 0; i < kM * kH; ++i) {
    ref_hidden[i] = ref_gate[i] * ref_up[i];
  }
  const std::vector<float> ref_mlp =
      ReferenceMatMulNk(ref_hidden, kM, kH, deq_down, kOut);
  const std::vector<float> ref_b64 =
      ReferenceMatMulNk(input_data, kM, kK, deq_b64, kOut);

  EXPECT_THAT(runner.ReadOutputAs<float>(gate_out),
              IsOkAndHolds(Pointwise(FloatNear(0.04f), ref_gate)));
  EXPECT_THAT(runner.ReadOutputAs<float>(up_out),
              IsOkAndHolds(Pointwise(FloatNear(0.05f), ref_up)));
  EXPECT_THAT(runner.ReadOutputAs<float>(mlp_out),
              IsOkAndHolds(Pointwise(FloatNear(0.15f), ref_mlp)));
  EXPECT_THAT(runner.ReadOutputAs<float>(out_b64),
              IsOkAndHolds(Pointwise(FloatNear(0.05f), ref_b64)));

  // Verify dynamic reshape to M = 1 decode on native block-32 + block-64.
  const std::vector<float> dec_in(input_data.begin(), input_data.begin() + kK);
  ASSERT_OK(runner.ReshapeInput(x, {1, 1, kK}));
  ASSERT_OK(runner.ReshapeInput(x64, {1, 1, kK}));
  ASSERT_OK(runner.SetInput(x, dec_in));
  ASSERT_OK(runner.SetInput(x64, dec_in));
  ASSERT_OK(runner.Run());

  EXPECT_THAT(
      runner.ReadOutputAs<float>(gate_out),
      IsOkAndHolds(Pointwise(
          FloatNear(0.04f),
          std::vector<float>(ref_gate.begin(), ref_gate.begin() + kH))));
  EXPECT_THAT(
      runner.ReadOutputAs<float>(out_b64),
      IsOkAndHolds(Pointwise(
          FloatNear(0.05f),
          std::vector<float>(ref_b64.begin(), ref_b64.begin() + kOut))));
}

TEST(YnnpackQuantizationTest, QuantizedBatchMatMulKnAndNkWithFloatZeroPoints) {
  constexpr int kB = 2;
  constexpr int kM = 3;
  constexpr int kK = 32;
  constexpr int kN = 8;
  constexpr int kBlockSize = 16;
  constexpr int kNumBlocks = kK / kBlockSize;

  std::vector<float> lhs_data(kB * kM * kK);
  for (size_t i = 0; i < lhs_data.size(); ++i) {
    lhs_data[i] = 0.3f * std::cos(0.13f * static_cast<float>(i));
  }
  std::vector<int8_t> q_kn(kK * kN);
  std::vector<int8_t> q_nk(kN * kK);
  for (int k = 0; k < kK; ++k) {
    for (int n = 0; n < kN; ++n) {
      const int8_t v = static_cast<int8_t>(((k * 3 + n) % 15) - 7);
      q_kn[k * kN + n] = v;
      q_nk[n * kK + k] = v;
    }
  }
  std::vector<float> scales(kN * kNumBlocks);
  std::vector<float> zps(kN * kNumBlocks);
  for (int i = 0; i < kN * kNumBlocks; ++i) {
    scales[i] = 0.03f + 0.01f * static_cast<float>(i % 4);
    zps[i] = -1.2f + 0.5f * static_cast<float>(i % 5);
  }
  std::vector<float> deq_nk(kN * kK);
  for (int n = 0; n < kN; ++n) {
    for (int k = 0; k < kK; ++k) {
      const int idx = n * kNumBlocks + (k / kBlockSize);
      deq_nk[n * kK + k] =
          scales[idx] * (static_cast<float>(q_nk[n * kK + k]) - zps[idx]);
    }
  }

  YnnTensor lhs({.type = Type::kFP32, .shape = {kB, kM, kK}});
  YnnTensor rhs_kn({.type = Type::kI4,
                    .shape = {kK, kN},
                    .buffer = PackSignedInt4(q_kn),
                    .quantization = std::make_shared<BlockwiseQuantization>(
                        scales, ZeroPoints::Float(zps), kBlockSize,
                        /*quantized_dimension=*/1)});
  YnnTensor rhs_nk({.type = Type::kI4,
                    .shape = {kN, kK},
                    .buffer = PackSignedInt4(q_nk),
                    .quantization = std::make_shared<BlockwiseQuantization>(
                        scales, ZeroPoints::Float(zps), kBlockSize,
                        /*quantized_dimension=*/0)});

  YnnTensor out_kn = BatchMatMul(lhs, rhs_kn, /*adj_x=*/false, /*adj_y=*/false);
  YnnTensor out_nk = BatchMatMul(lhs, rhs_nk, /*adj_x=*/false, /*adj_y=*/true);

  ASSERT_OK_AND_ASSIGN(YnnpackRunner runner,
                       YnnpackRunner::Create({out_kn, out_nk}));
  ASSERT_OK(runner.SetInput(lhs, lhs_data));
  ASSERT_OK(runner.Run());

  const std::vector<float> expected =
      ReferenceMatMulNk(lhs_data, kB * kM, kK, deq_nk, kN);
  EXPECT_THAT(runner.ReadOutputAs<float>(out_kn),
              IsOkAndHolds(Pointwise(FloatNear(0.03f), expected)));
  EXPECT_THAT(runner.ReadOutputAs<float>(out_nk),
              IsOkAndHolds(Pointwise(FloatNear(0.03f), expected)));
}

TEST(YnnpackQuantizationTest,
     RowOnlyQuantizedEmbeddingLookupAndTiedLmHeadWithDynamicReshape) {
  constexpr int kVocab = 16;
  constexpr int kDim = 32;
  constexpr int kBlockSize = 16;
  constexpr int kNumBlocks = kDim / kBlockSize;

  std::vector<int8_t> table_q4(kVocab * kDim);
  std::vector<int8_t> table_q2(kVocab * kDim);
  std::vector<int8_t> table_q8(kVocab * kDim);
  for (int i = 0; i < kVocab * kDim; ++i) {
    table_q4[i] = static_cast<int8_t>(((i * 7) % 15) - 7);
    table_q2[i] = static_cast<int8_t>(((i * 3) % 4) - 2);
    table_q8[i] = static_cast<int8_t>(((i * 11) % 200) - 100);
  }
  std::vector<float> bw_scales(kVocab * kNumBlocks);
  std::vector<float> bw_zps(kVocab * kNumBlocks);
  for (int i = 0; i < kVocab * kNumBlocks; ++i) {
    bw_scales[i] = 0.02f + 0.005f * static_cast<float>(i % 7);
    bw_zps[i] = -0.9f + 0.4f * static_cast<float>(i % 5);
  }
  std::vector<float> pc_scales(kVocab);
  std::vector<int64_t> pc_zps(kVocab);
  for (int v = 0; v < kVocab; ++v) {
    pc_scales[v] = 0.01f * static_cast<float>(v + 1);
    pc_zps[v] = (v % 5) - 2;
  }

  std::vector<float> deq_q4(kVocab * kDim);
  std::vector<float> deq_q2(kVocab * kDim);
  std::vector<float> deq_q8(kVocab * kDim);
  for (int v = 0; v < kVocab; ++v) {
    for (int d = 0; d < kDim; ++d) {
      const int idx = v * kNumBlocks + (d / kBlockSize);
      deq_q4[v * kDim + d] =
          bw_scales[idx] *
          (static_cast<float>(table_q4[v * kDim + d]) - bw_zps[idx]);
      deq_q2[v * kDim + d] =
          bw_scales[idx] *
          (static_cast<float>(table_q2[v * kDim + d]) - bw_zps[idx]);
      deq_q8[v * kDim + d] =
          pc_scales[v] * (static_cast<float>(table_q8[v * kDim + d]) -
                          static_cast<float>(pc_zps[v]));
    }
  }

  // Tied Q4_K weight shared simultaneously by EmbeddingLookup AND
  // FullyConnected (lm_head), plus Gather(Dequantize(...)) peephole on Q2_K and
  // per-channel INT8 EmbeddingLookup.
  YnnTensor tied_q4({.type = Type::kI4,
                     .shape = {kVocab, kDim},
                     .buffer = PackSignedInt4(table_q4),
                     .quantization = std::make_shared<BlockwiseQuantization>(
                         bw_scales, ZeroPoints::Float(bw_zps), kBlockSize, 0)});
  YnnTensor w_q2({.type = Type::kI2,
                  .shape = {kVocab, kDim},
                  .buffer = PackSignedInt2(table_q2),
                  .quantization = std::make_shared<BlockwiseQuantization>(
                      bw_scales, ZeroPoints::Float(bw_zps), kBlockSize, 0)});
  YnnTensor w_pc_q8(
      {.type = Type::kI8,
       .shape = {kVocab, kDim},
       .buffer = OwningCpuBuffer::Copy<Type::kI8>(table_q8),
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           pc_scales, pc_zps, /*quantized_dimension=*/0)});

  YnnTensor token_ids({.type = Type::kI32, .shape = {1, 4}});
  YnnTensor emb_q4 = EmbeddingLookup(token_ids, tied_q4);
  YnnTensor logits = FullyConnected(emb_q4, tied_q4);
  YnnTensor emb_q2_gather = Gather(Dequantize(w_q2), token_ids, /*axis=*/0);
  YnnTensor emb_pc_q8 = EmbeddingLookup(token_ids, w_pc_q8);

  ASSERT_OK_AND_ASSIGN(
      YnnpackRunner runner,
      YnnpackRunner::Create({emb_q4, logits, emb_q2_gather, emb_pc_q8}));

  // Prefill shape [1, 4].
  const std::vector<int32_t> prefill_ids = {3, 0, 15, 7};
  ASSERT_OK(runner.SetInput(token_ids, prefill_ids));
  ASSERT_OK(runner.Run());

  auto gather_rows = [&](absl::Span<const float> table,
                         absl::Span<const int32_t> ids) {
    std::vector<float> out(ids.size() * kDim);
    for (size_t i = 0; i < ids.size(); ++i) {
      std::memcpy(out.data() + i * kDim, table.data() + ids[i] * kDim,
                  kDim * sizeof(float));
    }
    return out;
  };

  const std::vector<float> expected_emb_q4 = gather_rows(deq_q4, prefill_ids);
  EXPECT_THAT(runner.ReadOutputAs<float>(emb_q4),
              IsOkAndHolds(Pointwise(FloatNear(1e-5f), expected_emb_q4)));
  EXPECT_THAT(runner.ReadOutputAs<float>(emb_q2_gather),
              IsOkAndHolds(Pointwise(FloatNear(1e-5f),
                                     gather_rows(deq_q2, prefill_ids))));
  EXPECT_THAT(runner.ReadOutputAs<float>(emb_pc_q8),
              IsOkAndHolds(Pointwise(FloatNear(1e-5f),
                                     gather_rows(deq_q8, prefill_ids))));
  EXPECT_THAT(runner.ReadOutputAs<float>(logits),
              IsOkAndHolds(Pointwise(
                  FloatNear(0.05f),
                  ReferenceMatMulNk(expected_emb_q4, prefill_ids.size(), kDim,
                                    deq_q4, kVocab))));

  // Dynamic reshape to decode shape [1, 1] without rebuilding the subgraph.
  const std::vector<int32_t> decode_ids = {11};
  ASSERT_OK(runner.ReshapeInput(token_ids, {1, 1}));
  ASSERT_OK(runner.SetInput(token_ids, decode_ids));
  ASSERT_OK(runner.Run());

  const std::vector<float> expected_dec_q4 = gather_rows(deq_q4, decode_ids);
  EXPECT_THAT(runner.ReadOutputAs<float>(emb_q4),
              IsOkAndHolds(Pointwise(FloatNear(1e-5f), expected_dec_q4)));
  EXPECT_THAT(runner.ReadOutputAs<float>(logits),
              IsOkAndHolds(Pointwise(FloatNear(0.05f),
                                     ReferenceMatMulNk(expected_dec_q4, 1, kDim,
                                                       deq_q4, kVocab))));
}

TEST(YnnpackQuantizationTest,
     RejectsZeroChannelBlockwiseDotAndInvalidSubByteRawGather) {
  {
    YnnTensor x({.type = Type::kFP32, .shape = {1, 32}});
    YnnTensor zero_channel_w(
        {.type = Type::kI4,
         .shape = {0, 32},
         .quantization = std::make_shared<BlockwiseQuantization>(
             std::vector<float>{}, ZeroPoints::Float(std::vector<float>{}),
             /*block_size=*/32, /*quantized_dimension=*/0)});
    YnnTensor out = FullyConnected(x, zero_channel_w);
    EXPECT_THAT(YnnpackRunner::Create({out}),
                StatusIs(absl::StatusCode::kUnimplemented));
  }

  {
    std::vector<int8_t> table_vals(4 * 32, 1);
    YnnTensor table(
        {.type = Type::kI4,
         .shape = {4, 32},
         .buffer = PackSignedInt4(table_vals),
         .quantization = std::make_shared<BlockwiseQuantization>(
             std::vector<float>(4, 0.25f), ZeroPoints::Int({0, 0, 0, 0}),
             /*block_size=*/32, /*quantized_dimension=*/0)});
    YnnTensor idx({.type = Type::kI32, .shape = {2}});
    YnnTensor gathered = Gather(table, idx, /*axis=*/1);
    EXPECT_THAT(YnnpackRunner::Create({gathered}),
                StatusIs(absl::StatusCode::kUnimplemented));
  }

  {
    std::vector<int8_t> table_vals(32, 1);
    YnnTensor vec(
        {.type = Type::kI4,
         .shape = {32},
         .buffer = PackSignedInt4(table_vals),
         .quantization = std::make_shared<PerChannelAffineQuantization>(
             std::vector<float>{0.25f}, ZeroPoints::Int({0}),
             /*quantized_dimension=*/0)});
    YnnTensor scalar_idx({.type = Type::kI32, .shape = {}});
    YnnTensor gathered = Gather(vec, scalar_idx, /*axis=*/0);
    EXPECT_THAT(YnnpackRunner::Create({gathered}),
                StatusIs(absl::StatusCode::kUnimplemented));
  }
}

TEST(YnnpackQuantizationTest,
     IsSymmetricMaterializesLazyBlockwiseQuantization) {
  std::shared_ptr<BlockwiseQuantization> lazy_bwq =
      BlockwiseQuantization::CreateLazily(
          /*block_size=*/32, /*quantized_dimension=*/0, /*num_blocks=*/2,
          []() -> BlockwiseQuantization::MaterializedBlockwiseParams {
            return {.scales = {0.25f, 0.5f},
                    .float_zero_points = {0.0f, -2.5f}};
          },
          BlockwiseQuantization::LazyOptions{
              .zero_points =
                  BlockwiseQuantization::ZeroPointPresence::kMayHaveFloat,
          });
  EXPECT_FALSE(YnnpackBuildContext::IsSymmetric(*lazy_bwq));
}

TEST(YnnpackConvPoolSelectTest,
     GroupedConv2DAndDepthwiseCausalConvWithDynamicReshape) {
  // Grouped Conv2D: input `[1, 1, 3, 4]`, 2 groups (`group_ci = 2`,
  // `group_co = 2`, filter `[4, 1, 2, 2]`), VALID padding, bias + Relu.
  YnnTensor x_grouped({.type = Type::kFP32, .shape = {1, 1, 3, 4}});
  const std::vector<float> grouped_filter = {
      // group 0, out 0: [[1, -1], [0.5, 2]]
      1.0f, -1.0f, 0.5f, 2.0f,
      // group 0, out 1: [[-0.5, 1], [1, -1]]
      -0.5f, 1.0f, 1.0f, -1.0f,
      // group 1, out 2: [[2, 0], [-1, 1]]
      2.0f, 0.0f, -1.0f, 1.0f,
      // group 1, out 3: [[0, 1], [1, 1]]
      0.0f, 1.0f, 1.0f, 1.0f};
  YnnTensor w_grouped(
      {.type = Type::kFP32,
       .shape = {4, 1, 2, 2},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>(grouped_filter)});
  YnnTensor b_grouped({.type = Type::kFP32,
                       .shape = {4},
                       .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                           {0.5f, -0.5f, 0.0f, 1.0f})});
  YnnTensor y_grouped = Conv2D(x_grouped, w_grouped, b_grouped,
                               /*stride_h=*/1, /*stride_w=*/1, kPaddingValid,
                               /*dilation_h=*/1, /*dilation_w=*/1, kActRelu);

  // Causal 1D depthwise conv (`[1, 1, T, C]` with filter `[1, 1, 3, C]`) used
  // in Qwen3.5 Gated DeltaNet and LFM2 LivConv, plus `depth_multiplier = 2`.
  YnnTensor x_dw({.type = Type::kFP32, .shape = {1, 1, 5, 2}});
  YnnTensor w_dw_causal({.type = Type::kFP32,
                         .shape = {1, 1, 3, 2},
                         .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                             {0.25f, -0.5f, 0.5f, 1.0f, 1.0f, 0.5f})});
  YnnTensor y_dw_causal =
      DepthwiseConv2D(x_dw, w_dw_causal,
                      /*stride_h=*/1, /*stride_w=*/1, kPaddingValid);

  YnnTensor w_dw_mult2(
      {.type = Type::kFP32,
       .shape = {1, 1, 2, 4},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
           {1.0f, -1.0f, 2.0f, 0.5f, 0.5f, 1.0f, -1.0f, 2.0f})});
  YnnTensor y_dw_mult2 =
      DepthwiseConv2D(x_dw, w_dw_mult2,
                      /*stride_h=*/1, /*stride_w=*/1, kPaddingSame);

  ASSERT_OK_AND_ASSIGN(
      YnnpackRunner runner,
      YnnpackRunner::Create({y_grouped, y_dw_causal, y_dw_mult2}));

  const std::vector<float> in_grouped = {1.0f,  2.0f,  3.0f,  4.0f,   // w = 0
                                         2.0f,  -1.0f, 0.5f,  1.0f,   // w = 1
                                         -1.0f, 3.0f,  -2.0f, 2.0f};  // w = 2
  const std::vector<float> in_dw_5 = {0.0f, 0.0f, 0.0f,  0.0f, 1.0f,
                                      2.0f, 3.0f, -1.0f, 2.0f, 4.0f};
  ASSERT_OK(runner.SetInput(x_grouped, in_grouped));
  ASSERT_OK(runner.SetInput(x_dw, in_dw_5));
  ASSERT_OK(runner.Run());

  // w=0:
  // g0,o0: (1*1 + 2*-1) + (2*0.5 + -1*2) + 0.5 = -1 - 1 + 0.5 = -1.5 -> 0
  // g0,o1: (1*-0.5 + 2*1) + (2*1 + -1*-1) - 0.5 = 1.5 + 3 - 0.5 = 4.0
  // g1,o2: (3*2 + 4*0) + (0.5*-1 + 1*1) = 6 + 0.5 = 6.5
  // g1,o3: (3*0 + 4*1) + (0.5*1 + 1*1) + 1 = 4 + 1.5 + 1 = 6.5
  // w=1:
  // g0,o0: (2*1 + -1*-1) + (-1*0.5 + 3*2) + 0.5 = 3 + 5.5 + 0.5 = 9.0
  // g0,o1: (2*-0.5 + -1*1) + (-1*1 + 3*-1) - 0.5 = -2 - 4 - 0.5 = -6.5 -> 0
  // g1,o2: (0.5*2 + 1*0) + (-2*-1 + 2*1) = 1 + 4 = 5.0
  // g1,o3: (0.5*0 + 1*1) + (-2*1 + 2*1) + 1 = 1 + 0 + 1 = 2.0
  EXPECT_THAT(
      runner.ReadOutputAs<float>(y_grouped),
      IsOkAndHolds(Pointwise(
          FloatNear(1e-5f), {0.0f, 4.0f, 6.5f, 6.5f, 9.0f, 0.0f, 5.0f, 2.0f})));

  // Causal VALID 1x3 on [0,0 | 0,0 | 1,2 | 3,-1 | 2,4]:
  // t=0: [0,0]*[0.25,-0.5] + [0,0]*[0.5,1] + [1,2]*[1,0.5] = [1.0, 1.0]
  // t=1: [0,0]*[0.25,-0.5] + [1,2]*[0.5,1] + [3,-1]*[1,0.5] = [3.5, 1.5]
  // t=2: [1,2]*[0.25,-0.5] + [3,-1]*[0.5,1] + [2,4]*[1,0.5] = [3.75, 0.0]
  EXPECT_THAT(runner.ReadOutputAs<float>(y_dw_causal),
              IsOkAndHolds(Pointwise(FloatNear(1e-5f),
                                     {1.0f, 1.0f, 3.5f, 1.5f, 3.75f, 0.0f})));

  // Dynamically reshape `x_dw` to decode length `[1, 1, 3, 2]` (single output
  // step) without rebuilding the subgraph.
  const std::vector<float> in_dw_3 = {1.0f, 2.0f, 3.0f, -1.0f, 2.0f, 4.0f};
  ASSERT_OK(runner.ReshapeInput(x_dw, {1, 1, 3, 2}));
  ASSERT_OK(runner.SetInput(x_dw, in_dw_3));
  ASSERT_OK(runner.Run());
  EXPECT_THAT(runner.ReadOutputAs<float>(y_dw_causal),
              IsOkAndHolds(Pointwise(FloatNear(1e-5f), {3.75f, 0.0f})));
  // `y_dw_mult2` SAME 1x2 on `[1,2 | 3,-1 | 2,4]`:
  // t=0: ch0=[1*1+3*0.5, 1*-1+3*1]=[2.5, 2], ch1=[2*2+-1*-1, 2*0.5+-1*2]=[5,-1]
  // t=1: ch0=[3*1+2*0.5, 3*-1+2*1]=[4, -1],
  //      ch1=[-1*2+4*-1, -1*0.5+4*2]=[-6, 7.5]
  // t=2 (right pad 0): ch0=[2, -2], ch1=[8, 2]
  EXPECT_THAT(runner.ReadOutputAs<float>(y_dw_mult2),
              IsOkAndHolds(Pointwise(FloatNear(1e-5f),
                                     {2.5f, 2.0f, 5.0f, -1.0f, 4.0f, -1.0f,
                                      -6.0f, 7.5f, 2.0f, -2.0f, 8.0f, 2.0f})));
}

TEST(YnnpackConvPoolSelectTest,
     MaxPool2DTopKAndSelectBroadcastWithDynamicReshape) {
  // MoE top-k pattern: `[1, T, E, 1]` -> `MaxPool2D` with `1xE` filter ->
  // broadcast comparison -> `SelectV2(mask, logits, -inf)` and 1-D `Select`.
  constexpr int kExperts = 4;
  YnnTensor logits({.type = Type::kFP32, .shape = {1, 2, kExperts, 1}});
  YnnTensor max_per_token =
      MaxPool2D(logits, /*filter_height=*/1, /*filter_width=*/kExperts,
                /*stride_h=*/1, /*stride_w=*/kExperts, kPaddingValid);
  YnnTensor clamped_pool =
      MaxPool2D(logits, /*filter_height=*/1, /*filter_width=*/2,
                /*stride_h=*/1, /*stride_w=*/2, kPaddingSame, kActRelu6);

  YnnTensor neg_inf({.type = Type::kFP32,
                     .shape = {},
                     .buffer = OwningCpuBuffer::Copy<Type::kFP32>({-1e9f})});
  YnnTensor cond_mask({.type = Type::kBOOL, .shape = {1, 2, kExperts, 1}});
  YnnTensor masked_logits = SelectV2(cond_mask, logits, neg_inf);

  // TFLite `Select` with 1-D `[2]` condition selecting rows of `[2, 3]` I32.
  YnnTensor row_cond({.type = Type::kBOOL, .shape = {2}});
  YnnTensor rows_a(
      {.type = Type::kI32,
       .shape = {2, 3},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>({10, 20, 30, 40, 50, 60})});
  YnnTensor rows_b(
      {.type = Type::kI32,
       .shape = {2, 3},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>({-1, -2, -3, -4, -5, -6})});
  YnnTensor selected_rows = Select(row_cond, rows_a, rows_b);

  ASSERT_OK_AND_ASSIGN(YnnpackRunner runner,
                       YnnpackRunner::Create({max_per_token, clamped_pool,
                                              masked_logits, selected_rows}));

  const std::vector<float> logits_2 = {
      -2.0f, 7.5f,  3.0f,  -0.5f,  // token 0 -> max = 7.5
      -5.0f, -1.0f, -9.0f, -3.0f   // token 1 -> max = -1.0
  };
  // Include a non-1 true byte (`255`) to verify boolean normalization.
  const std::vector<uint8_t> mask_2 = {0, 1, 255, 0, 0, 1, 0, 0};
  const std::vector<uint8_t> row_cond_data = {1, 0};
  ASSERT_OK(runner.SetInput(logits, logits_2));
  ASSERT_OK(runner.SetInput(cond_mask, AsByteSpan(mask_2)));
  ASSERT_OK(runner.SetInput(row_cond, AsByteSpan(row_cond_data)));
  ASSERT_OK(runner.Run());

  EXPECT_THAT(runner.ReadOutputAs<float>(max_per_token),
              IsOkAndHolds(Pointwise(FloatNear(1e-5f), {7.5f, -1.0f})));
  // Relu6 clamps 7.5 -> 6.0 and negative maxima (-1.0, -3.0) -> 0.0.
  EXPECT_THAT(
      runner.ReadOutputAs<float>(clamped_pool),
      IsOkAndHolds(Pointwise(FloatNear(1e-5f), {6.0f, 3.0f, 0.0f, 0.0f})));
  EXPECT_THAT(
      runner.ReadOutputAs<float>(masked_logits),
      IsOkAndHolds(Pointwise(FloatNear(1e-5f), {-1e9f, 7.5f, 3.0f, -1e9f, -1e9f,
                                                -1.0f, -1e9f, -1e9f})));

  ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> i32_bytes,
                       runner.ReadOutput(selected_rows));
  std::vector<int32_t> got_rows(6);
  std::memcpy(got_rows.data(), i32_bytes.data(), i32_bytes.size());
  EXPECT_EQ(got_rows, (std::vector<int32_t>{10, 20, 30, -4, -5, -6}));

  // Dynamic reshape to 1 token `[1, 1, kExperts, 1]` for decode.
  const std::vector<float> logits_1 = {0.5f, 2.0f, 9.0f, -4.0f};
  const std::vector<uint8_t> mask_1 = {1, 0, 1, 0};
  ASSERT_OK(runner.ReshapeInput(logits, {1, 1, kExperts, 1}));
  ASSERT_OK(runner.ReshapeInput(cond_mask, {1, 1, kExperts, 1}));
  ASSERT_OK(runner.SetInput(logits, logits_1));
  ASSERT_OK(runner.SetInput(cond_mask, AsByteSpan(mask_1)));
  ASSERT_OK(runner.Run());
  EXPECT_THAT(runner.ReadOutputAs<float>(max_per_token),
              IsOkAndHolds(Pointwise(FloatNear(1e-5f), {9.0f})));
  EXPECT_THAT(
      runner.ReadOutputAs<float>(masked_logits),
      IsOkAndHolds(Pointwise(FloatNear(1e-5f), {0.5f, -1e9f, 9.0f, -1e9f})));
}

TEST(YnnpackConvPoolSelectTest,
     ScalarSelectConv2DNoBiasSamePaddingAndValidationErrors) {
  // 1. Scalar (`output_rank == 0`) `Select` and `SelectV2`, plus `Conv2D`
  //    without bias and with `kPaddingSame`.
  YnnTensor scalar_cond({.type = Type::kBOOL, .shape = {}});
  YnnTensor scalar_a({.type = Type::kFP32,
                      .shape = {},
                      .buffer = OwningCpuBuffer::Copy<Type::kFP32>({3.5f})});
  YnnTensor scalar_b({.type = Type::kFP32,
                      .shape = {},
                      .buffer = OwningCpuBuffer::Copy<Type::kFP32>({-7.0f})});
  YnnTensor scalar_sel = Select(scalar_cond, scalar_a, scalar_b);
  YnnTensor scalar_sel_v2 = SelectV2(scalar_cond, scalar_b, scalar_a);

  YnnTensor x_conv({.type = Type::kFP32, .shape = {1, 1, 3, 2}});
  YnnTensor w_conv({.type = Type::kFP32,
                    .shape = {2, 1, 2, 2},
                    .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                        {1.0f, -1.0f, 0.5f, 2.0f, -0.5f, 1.0f, 1.0f, 1.0f})});
  YnnTensor y_conv_same = Conv2D(x_conv, w_conv, /*stride_h=*/1,
                                 /*stride_w=*/1, kPaddingSame);

  ASSERT_OK_AND_ASSIGN(
      YnnpackRunner runner,
      YnnpackRunner::Create({scalar_sel, scalar_sel_v2, y_conv_same}));

  const std::vector<uint8_t> cond_true = {1};
  const std::vector<float> in_conv = {1.0f, 2.0f, 3.0f, -1.0f, 2.0f, 0.5f};
  ASSERT_OK(runner.SetInput(scalar_cond, AsByteSpan(cond_true)));
  ASSERT_OK(runner.SetInput(x_conv, in_conv));
  ASSERT_OK(runner.Run());
  EXPECT_THAT(runner.ReadOutputAs<float>(scalar_sel),
              IsOkAndHolds(Pointwise(FloatNear(1e-5f), {3.5f})));
  EXPECT_THAT(runner.ReadOutputAs<float>(scalar_sel_v2),
              IsOkAndHolds(Pointwise(FloatNear(1e-5f), {-7.0f})));
  // w=0: o0 = (1*1 + 2*-1) + (3*0.5 + -1*2) = -1 - 0.5 = -1.5
  //      o1 = (1*-0.5 + 2*1) + (3*1 + -1*1) = 1.5 + 2 = 3.5
  // w=1: o0 = (3*1 + -1*-1) + (2*0.5 + 0.5*2) = 4 + 2 = 6.0
  //      o1 = (3*-0.5 + -1*1) + (2*1 + 0.5*1) = -2.5 + 2.5 = 0.0
  // w=2 (right pad 0): o0 = 2*1 + 0.5*-1 = 1.5, o1 = 2*-0.5 + 0.5*1 = -0.5
  EXPECT_THAT(runner.ReadOutputAs<float>(y_conv_same),
              IsOkAndHolds(Pointwise(FloatNear(1e-5f),
                                     {-1.5f, 3.5f, 6.0f, 0.0f, 1.5f, -0.5f})));

  const std::vector<uint8_t> cond_false = {0};
  ASSERT_OK(runner.SetInput(scalar_cond, AsByteSpan(cond_false)));
  ASSERT_OK(runner.Run());
  EXPECT_THAT(runner.ReadOutputAs<float>(scalar_sel),
              IsOkAndHolds(Pointwise(FloatNear(1e-5f), {-7.0f})));
  EXPECT_THAT(runner.ReadOutputAs<float>(scalar_sel_v2),
              IsOkAndHolds(Pointwise(FloatNear(1e-5f), {3.5f})));

  // 2. Lowering validation error paths: non-positive stride, indivisible
  //    grouped channels, and integer MaxPool2D input.
  YnnTensor bad_stride = Conv2D(x_conv, w_conv, /*stride_h=*/0,
                                /*stride_w=*/1, kPaddingValid);
  EXPECT_FALSE(YnnpackRunner::Create({bad_stride}).ok());

  YnnTensor w_bad_groups({.type = Type::kFP32,
                          .shape = {3, 1, 2, 1},
                          .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                              {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f})});
  YnnTensor bad_groups = Conv2D(x_conv, w_bad_groups, /*stride_h=*/1,
                                /*stride_w=*/1, kPaddingValid);
  EXPECT_FALSE(YnnpackRunner::Create({bad_groups}).ok());

  YnnTensor int_pool_in({.type = Type::kI32, .shape = {1, 2, 2, 1}});
  YnnTensor bad_pool = MaxPool2D(int_pool_in, /*filter_height=*/1,
                                 /*filter_width=*/2, /*stride_h=*/1,
                                 /*stride_w=*/2, kPaddingSame);
  EXPECT_FALSE(YnnpackRunner::Create({bad_pool}).ok());
}

}  // namespace
}  // namespace litert::tensor

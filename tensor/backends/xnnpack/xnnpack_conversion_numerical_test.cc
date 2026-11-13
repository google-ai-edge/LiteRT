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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/testing/numerical_test_bridge.h"
#include "tensor/backends/testing/numerical_test_suite.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

class XnnpackTestBackendBridge : public TestBackendBridge {
 public:
  ~XnnpackTestBackendBridge() override = default;

  absl::Status Initialize() override {
    if (xnn_initialize(nullptr) != xnn_status_success) {
      return absl::InternalError("Failed to initialize XNNPACK");
    }
    return absl::OkStatus();
  }

  absl::Status BuildGraph(absl::Span<const TensorHandle> inputs,
                          absl::Span<const TensorHandle> outputs) override {
    std::vector<TensorHandle> output_handles(outputs.begin(), outputs.end());
    auto runner_or = XnnpackRunner::Create(output_handles);
    if (!runner_or.ok()) {
      absl::Status s = runner_or.status();
      if (s.code() == absl::StatusCode::kInvalidArgument &&
          absl::StrContains(s.message(), "does not implement")) {
        return absl::UnimplementedError(s.message());
      }
      return s;
    }
    runner_ = std::make_unique<XnnpackRunner>(std::move(*runner_or));
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
  std::unique_ptr<XnnpackRunner> runner_;
};

struct XnnpackTraits {
  using Tag = XnnpackMixinTag;
  // TODO(b/574438217): Enable these tests once the XNNPACK backend is fixed.
  static constexpr absl::string_view kDisabledTests[] = {
      // Blocks of 16 fail when the runtime is created.
      "FullyConnectedQuantizedWeightsOp",
      // Fully connected ops reject per-tensor int8 weights.
      "FullyConnectedPerTensorAndAsymmetricOp",
      // Batch matmuls reject int4 right-hand sides.
      "BatchMatMulQuantizedRhsOp",
      // Transposes reject int4 and blockwise values.
      "TransposedQuantizedWeightsIntoBatchMatMulOp",
      // Transposes reject int4 values, and asymmetric int4 weights fail to
      // dequantize.
      "TransposedQuantizedWeightsIntoAddOp",
      // Dequantizing int4 values aborts in an XNNPACK assertion.
      "QuantizedWeightsSharedByDotAndAddOp",
      // Per-channel zero points of tensors that aren't constants are
      // dequantized as if they were constants, which fails.
      "DequantizePerChannelOp",
  };
  static std::unique_ptr<TestBackendBridge> CreateBridge() {
    return std::make_unique<XnnpackTestBackendBridge>();
  }
};

INSTANTIATE_TYPED_TEST_SUITE_P(Xnnpack, NumericalTestSuite, XnnpackTraits);

// Runs `FullyConnected` on XNNPACK with `[4, 32]` weights holding `levels`
// quantized with `scales` and `float_zero_point`, and compares it against a
// float reference.
template <Type kWeightType>
void ExpectQuantizedFullyConnectedMatchesReference(
    const int8_t (&levels)[4], const std::vector<float>& scales,
    std::vector<float> float_zero_points) {
  using XnnTensor = Tensor<XnnpackMixinTag>;
  constexpr int kBatch = 2;
  constexpr int kInCols = 32;
  constexpr int kOutCols = 4;

  // Input activation with range [-127, 128] (span 255) and integer values, so
  // qdint8 dynamic quantization has unit scale (255 / 255 = 1) and zero
  // rounding error.
  std::vector<float> x_vals(kBatch * kInCols);
  for (int b = 0; b < kBatch; ++b) {
    for (int i = 0; i < kInCols; ++i) {
      constexpr float kInputs[4] = {-127.0f, -32.0f, 32.0f, 128.0f};
      x_vals[b * kInCols + i] = kInputs[(b + i) % 4];
    }
  }

  std::vector<int8_t> q_vals(kOutCols * kInCols);
  for (int o = 0; o < kOutCols; ++o) {
    for (int i = 0; i < kInCols; ++i) {
      q_vals[o * kInCols + i] = levels[(o + i) % 4];
    }
  }
  const float zero_point =
      float_zero_points.empty() ? 0.0f : float_zero_points[0];
  auto w_buf = OwningCpuBuffer::Copy<kWeightType>(q_vals);

  XnnTensor x(TensorInit{
      .name = "x", .type = Type::kFP32, .shape = {kBatch, kInCols}});
  XnnTensor w(
      TensorInit{.name = "w",
                 .type = kWeightType,
                 .shape = {kOutCols, kInCols},
                 .buffer = w_buf,
                 .quantization = std::make_shared<PerChannelAffineQuantization>(
                     scales, /*zero_points=*/std::vector<int64_t>{},
                     /*quantized_dimension=*/0, std::move(float_zero_points))});
  XnnTensor y = FullyConnected(x, w);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(XnnpackRunner runner,
                                  XnnpackRunner::Create({y}));
  ASSERT_TRUE(runner.SetInput(x, x_vals).ok());
  ASSERT_TRUE(runner.Run().ok());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const float> actual,
                                  runner.ReadOutputAs<float>(y));

  for (int b = 0; b < kBatch; ++b) {
    for (int o = 0; o < kOutCols; ++o) {
      const float scale = scales.size() == 1 ? scales[0] : scales[o];
      float expected = 0.0f;
      for (int i = 0; i < kInCols; ++i) {
        const float w_deq =
            (static_cast<float>(q_vals[o * kInCols + i]) - zero_point) * scale;
        expected += x_vals[b * kInCols + i] * w_deq;
      }
      EXPECT_NEAR(actual.data()[b * kOutCols + o], expected, 5e-2f)
          << "batch=" << b << " out=" << o;
    }
  }
}

TEST(XnnpackCenteredInt2Test, FullyConnectedWithFloatZeroPoint) {
  // Signed 2-bit values in {-2, -1, 0, +1}; with zero_point = -0.5f,
  // dequantized weight is (q + 0.5f) * scale in {-1.5, -0.5, +0.5, +1.5} * s.
  ExpectQuantizedFullyConnectedMatchesReference<Type::kI2>(
      {-2, -1, 0, 1}, /*scales=*/{0.25f, 0.5f, 0.125f, 1.0f},
      /*float_zero_points=*/{-0.5f});
}

TEST(XnnpackCenteredInt2Test, FullyConnectedWithSingleScale) {
  ExpectQuantizedFullyConnectedMatchesReference<Type::kI2>(
      {-2, -1, 0, 1}, /*scales=*/{0.25f}, /*float_zero_points=*/{-0.5f});
}

TEST(XnnpackInt2Test, FullyConnectedWithSingleScaleAndNoZeroPoint) {
  ExpectQuantizedFullyConnectedMatchesReference<Type::kI2>(
      {-2, -1, 0, 1}, /*scales=*/{0.25f}, /*float_zero_points=*/{});
}

TEST(XnnpackInt4Test, FullyConnectedWithSingleScale) {
  ExpectQuantizedFullyConnectedMatchesReference<Type::kI4>(
      {-8, -1, 0, 7}, /*scales=*/{0.125f}, /*float_zero_points=*/{});
}

}  // namespace
}  // namespace litert::tensor

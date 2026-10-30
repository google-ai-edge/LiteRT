// Copyright 2026 The LiteRT Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/vendors/arm_vulkan_ml/tools/devtool/tool.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "xnnpack.h"

namespace litert::arm_vulkan_ml::tool {
namespace {
Error Invalid(const char* message) {
  return Error(kLiteRtStatusErrorInvalidArgument, message);
}
}  // namespace

Expected<std::vector<int8_t>> SoftmaxReference(
    const std::vector<int8_t>& input, size_t channels, double input_scale,
    double beta, const std::string& reference) {
  if (!channels || input.empty() || input.size() % channels ||
      !std::isfinite(input_scale) || input_scale <= 0 || !std::isfinite(beta) ||
      beta <= 0 || !std::isfinite(input_scale * beta)) {
    return Invalid("Invalid Softmax reference shape/scale/beta");
  }
  std::vector<int8_t> result(input.size());
  if (reference == "fp64") {
    // Independent stable mathematical oracle. No production LUT, fixed-point
    // multiplier, or graph-lowering helper is shared with this calculation.
    std::vector<double> exponential(channels);
    for (size_t row = 0; row < input.size(); row += channels) {
      const int maximum = *std::max_element(
          input.begin() + row, input.begin() + row + channels);
      double sum = 0, correction = 0;
      for (size_t i = 0; i < channels; ++i) {
        exponential[i] = std::exp(
            (static_cast<int>(input[row + i]) - maximum) * input_scale * beta);
        const double adjusted = exponential[i] - correction;
        const double next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
      }
      for (size_t i = 0; i < channels; ++i) {
        const double code = std::floor(256 * exponential[i] / sum + .5);
        result[row + i] = static_cast<int8_t>(std::min(code, 255.0) - 128);
      }
    }
    return result;
  }
  if (reference != "xnnpack") return Invalid("Unknown Softmax reference");
  const float scale = static_cast<float>(input_scale * beta);
  if (!std::isnormal(scale) || scale <= 0) {
    return Invalid("XNNPACK requires a positive normal effective input scale");
  }
  // XNNPACK QU8 accumulates the rounded table in UINT32. Rounding qscale can
  // exceed the nominal UINT32_MAX/channels bound for tied rows. Check the
  // actual inputs in UINT64 before calling it, rather than accepting a wrapped
  // reference or triggering its zero-sum assertion/division by zero. FP64
  // remains available for these cases; this is not a backend support limit.
  const double qscale =
      std::min(static_cast<double>(UINT32_MAX) / channels, 8388607.0);
  for (size_t row = 0; row < input.size(); row += channels) {
    const int maximum = *std::max_element(
        input.begin() + row, input.begin() + row + channels);
    uint64_t sum = 0;
    for (size_t i = 0; i < channels; ++i) {
      sum += static_cast<uint64_t>(std::lrint(qscale * std::exp(
          (static_cast<int>(input[row + i]) - maximum) *
          static_cast<double>(scale))));
    }
    if (!sum || sum > UINT32_MAX) {
      return Error(kLiteRtStatusErrorUnsupported,
          "XNNPACK QU8 reference's rounded exponential sum exceeds UINT32; "
          "use fp64 for this case");
    }
  }
  if (xnn_initialize(nullptr) != xnn_status_success) {
    return Invalid("XNNPACK initialization failed");
  }
  xnn_operator_t raw = nullptr;
  if (xnn_create_softmax_nc_qu8(scale, 0, 1.0f / 256, 0, &raw) !=
      xnn_status_success) {
    return Invalid("Cannot create XNNPACK QU8 Softmax");
  }
  struct Delete {
    void operator()(xnn_operator_t op) const { xnn_delete_operator(op); }
  };
  std::unique_ptr<xnn_operator, Delete> op(raw);
  std::vector<uint8_t> rebased(input.size() + XNN_EXTRA_BYTES);
  std::vector<uint8_t> output(input.size() + XNN_EXTRA_BYTES);
  for (size_t i = 0; i < input.size(); ++i) {
    rebased[i] = static_cast<uint8_t>(static_cast<int>(input[i]) + 128);
  }
  if (xnn_reshape_softmax_nc_qu8(op.get(), channels, channels, channels,
          input.size() / channels, nullptr) != xnn_status_success ||
      xnn_setup_softmax_nc_qu8(op.get(), rebased.data(), output.data()) !=
          xnn_status_success ||
      xnn_run_operator(op.get(), nullptr) != xnn_status_success) {
    return Invalid("XNNPACK QU8 Softmax failed");
  }
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = static_cast<int8_t>(static_cast<int>(output[i]) - 128);
  }
  return result;
}

}  // namespace litert::arm_vulkan_ml::tool

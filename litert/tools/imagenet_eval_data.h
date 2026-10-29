// Copyright 2026 Google LLC.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LITERT_TOOLS_IMAGENET_EVAL_DATA_H_
#define LITERT_TOOLS_IMAGENET_EVAL_DATA_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"

namespace litert::imagenet {
using Json = nlohmann::json;

// Streaming SHA-256; file and input hashing never retain an entire model.
class Sha256 {
 public:
  void Update(const void* data, size_t size);
  std::string Hex() const;

 private:
  void Block(const uint8_t* data);
  std::array<uint32_t, 8> state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                    0xa54ff53a, 0x510e527f, 0x9b05688c,
                                    0x1f83d9ab, 0x5be0cd19};
  std::array<uint8_t, 64> pending_{};
  size_t used_ = 0;
  uint64_t bytes_ = 0;
};
Expected<std::string> HashFile(const std::string& path);
std::string HashBytes(const void* data, size_t size);
Expected<Json> ReadJson(const std::string& path);
Expected<void> WriteJson(const std::string& path, const Json& value);

struct Sample {
  int id;
  int label;
  std::string path;
};
// Empty image_root resolves relative paths against the manifest directory.
// A zero limit selects all remaining rows; a nonzero limit selects at most N.
Expected<std::vector<Sample>> ReadSamples(const std::string& manifest,
                                        const std::string& image_root,
                                        int start, int limit);

// Callers validate finite inputs and a finite, positive per-tensor scale.
template <typename T>
std::vector<T> QuantizePerTensor(
    const std::vector<float>& values,
    const LiteRtQuantizationPerTensor& quantization) {
  const int64_t q_min = std::numeric_limits<T>::min();
  const int64_t q_max = std::numeric_limits<T>::max();
  const float reciprocal = 1.0f / quantization.scale;
  std::vector<T> out(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    // Match XNNPACK: FP32 reciprocal multiplication and nearest-even rounding
    // before adding the zero point. TFLite's reference AffineQuantize rounds
    // ties away from zero and can disagree with XNNPACK at half steps.
    const float scaled = std::clamp(
        values[i] * reciprocal,
        static_cast<float>(q_min - quantization.zero_point),
        static_cast<float>(q_max - quantization.zero_point));
    const int64_t lower = static_cast<int64_t>(std::floor(scaled));
    const float fraction = scaled - static_cast<float>(lower);
    const int64_t rounded = lower +
        (fraction > 0.5f || (fraction == 0.5f && lower % 2 != 0));
    out[i] = static_cast<T>(rounded + quantization.zero_point);
  }
  return out;
}

struct Preprocessing {
  std::string interpolation = "bilinear";
  int resize = 232;
  int crop_h = 224;
  int crop_w = 224;
  bool antialias = true;
  std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};
  std::array<float, 3> stddev = {0.229f, 0.224f, 0.225f};
};
Expected<Preprocessing> ParsePreprocessing(const Json& value);
Json DescribePreprocessing(const Preprocessing& config);
Expected<void> ValidatePreprocessing(const Preprocessing& config);
Expected<std::array<int, 2>> ResizedDimensions(int width, int height,
                                             const Preprocessing& config);
Expected<int> CropOffset(int difference);
Expected<std::vector<uint8_t>> ResizeRgb(const uint8_t* input, int width,
                                         int height, int out_w, int out_h,
                                         const Preprocessing& config);
Expected<std::vector<float>> PreprocessRgb(const uint8_t* input, int width,
                                          int height,
                                          const Preprocessing& config, bool nchw);
Expected<std::vector<float>> DecodePreprocess(const std::string& path,
                                             const Preprocessing& config,
                                             bool nchw);

struct Comparison {
  bool allclose = true;
  bool finite = true;
  double max_abs = 0;
  double mean_abs = 0;
};
Expected<Comparison> CompareTensorValues(const void* actual, const void* reference,
                                        size_t count, bool floating,
                                        double atol, double rtol,
                                        double max_code, double mean_code);
Expected<Comparison> CompareQuantizedScores(
    const std::vector<float>& actual, const std::vector<float>& reference,
    const LiteRtQuantizationPerTensor& quantization,
    double max_code, double mean_code);
Expected<Comparison> CompareScores(const std::vector<float>& actual,
                                   const std::vector<float>& reference,
                                   double atol, double rtol);
Expected<std::vector<int>> TopK(const std::vector<float>& scores, int k);
}  // namespace litert::imagenet
#endif  // LITERT_TOOLS_IMAGENET_EVAL_DATA_H_

// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LITERT_TOOLS_VISION_EVAL_IMAGE_H_
#define LITERT_TOOLS_VISION_EVAL_IMAGE_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <limits>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"

namespace litert::vision {

// Converts a UTF-8 string to `std::filesystem::path` (preserving Unicode wide
// paths and long-path support on Windows without legacy ANSI conversion).
inline std::filesystem::path Utf8ToPath(absl::string_view utf8) {
  if (utf8.empty()) return {};
#if defined(__cpp_lib_char8_t)
  return std::filesystem::path(std::u8string_view(
      reinterpret_cast<const char8_t*>(utf8.data()), utf8.size()));
#else
  return std::filesystem::u8path(utf8.begin(), utf8.end());
#endif
}

// Converts `path` to a UTF-8 string preserving native separators.
inline std::string PathToUtf8(const std::filesystem::path& path) {
  const auto u8 = path.u8string();
  return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
}

// Single row from a classification TSV manifest.
struct ImageSample {
  int id;
  int label;
  std::filesystem::path path;
};

// Parses a tab-separated `id\tlabel\tpath` manifest at `manifest` (skipping
// blank lines, `#` comments, and an optional `id\tlabel\tpath` header),
// resolves paths relative to `image_root` (or `manifest`'s parent directory if
// `image_root` is empty), and returns rows starting at `start` (all remaining
// rows when `limit` is 0, or at most `limit` rows otherwise).
Expected<std::vector<ImageSample>> ReadSamples(
    const std::filesystem::path& manifest,
    const std::filesystem::path& image_root, int start, int limit);

// Quantizes finite `values` to integral type `T` using `quantization`
// (`quantization.scale` must be finite and positive) with FP32 reciprocal
// multiplication, clamping, and ties-to-even rounding before adding
// `quantization.zero_point`.
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
    const float scaled =
        std::clamp(values[i] * reciprocal,
                   static_cast<float>(q_min - quantization.zero_point),
                   static_cast<float>(q_max - quantization.zero_point));
    const int64_t lower = static_cast<int64_t>(std::floor(scaled));
    const float fraction = scaled - static_cast<float>(lower);
    const int64_t rounded =
        lower + (fraction > 0.5f || (fraction == 0.5f && lower % 2 != 0));
    out[i] = static_cast<T>(rounded + quantization.zero_point);
  }
  return out;
}

// Parameters for aspect-preserving shorter-side resize, center crop/pad, and
// per-channel RGB normalization.
struct ImagePreprocessing {
  std::string interpolation = "bilinear";
  int resize = 232;
  int crop_h = 224;
  int crop_w = 224;
  bool antialias = true;
  std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};
  std::array<float, 3> stddev = {0.229f, 0.224f, 0.225f};
};

// Validates `c` (`"bilinear"` or `"bicubic"` interpolation, positive resize
// and crop dimensions within the 256 MiB image limit, and finite per-channel
// `mean` and positive `stddev`).
Expected<void> ValidatePreprocessing(const ImagePreprocessing& c);

// Validates `c` and input dimensions `w` x `h`, and returns
// `{resized_w, resized_h}` scaling the shorter side to `c.resize` while
// preserving aspect ratio.
Expected<std::array<int, 2>> ResizedDimensions(int w, int h,
                                               const ImagePreprocessing& c);

// Returns the centered crop start offset for nonnegative excess dimension `d`
// using ties-to-even rounding (`round(d / 2)`).
Expected<int> CropOffset(int d);

// Resizes packed 3-channel RGB8 buffer `src` (`w` x `h`) to `ow` x `oh` using
// separable PIL-compatible bilinear or bicubic filtering configured by `c`.
Expected<std::vector<uint8_t>> ResizeRgb(const uint8_t* src, int w, int h,
                                         int ow, int oh,
                                         const ImagePreprocessing& c);

// Resizes packed 3-channel RGB8 buffer `src` (`w` x `h`), center-crops or
// zero-pads to `c.crop_w` x `c.crop_h`, normalizes pixels by
// `(pixel / 255 - c.mean) / c.stddev`, and returns `float` values in NCHW
// (`nchw` true) or NHWC (`nchw` false) layout.
Expected<std::vector<float>> PreprocessRgb(const uint8_t* src, int w, int h,
                                           const ImagePreprocessing& c,
                                           bool nchw);

// Decodes the image file at `path` as 3-channel RGB8 and applies
// `PreprocessRgb` with `c` to return normalized `float` values in NCHW (`nchw`
// true) or NHWC (`nchw` false) layout.
Expected<std::vector<float>> DecodePreprocess(
    const std::filesystem::path& path, const ImagePreprocessing& c, bool nchw);

// Numerical parity summary between candidate and reference tensor values.
struct Comparison {
  bool allclose = true;
  bool finite = true;
  double max_abs = 0;
  double mean_abs = 0;
};

// Compares `count` elements of `actual` and `reference` as `float` buffers
// when `floating` is true (gated by `atol` and `rtol`) or `int8_t` buffers
// when `floating` is false (gated by `max_code` and `mean_code`), and returns
// their `Comparison` summary.
Expected<Comparison> CompareTensorValues(const void* actual,
                                         const void* reference, size_t count,
                                         bool floating, double atol,
                                         double rtol, double max_code,
                                         double mean_code);

// Reconstructs rounded integer quantization codes from equal-length
// dequantized `actual` and `reference` score vectors using `quantization`, and
// returns their `Comparison` summary gated by `max_code` and `mean_code`.
Expected<Comparison> CompareQuantizedScores(
    const std::vector<float>& actual, const std::vector<float>& reference,
    const LiteRtQuantizationPerTensor& quantization, double max_code,
    double mean_code);

// Compares equal-length `actual` and `reference` float score vectors within
// `atol` and `rtol`, returning an error if any score is non-finite.
Expected<Comparison> CompareScores(const std::vector<float>& actual,
                                   const std::vector<float>& reference,
                                   double atol, double rtol);

// Returns up to `k` (`k >= 0`) class indices sorted by descending finite score
// in `scores`, breaking ties by smaller index.
Expected<std::vector<int>> TopK(const std::vector<float>& scores, int k);

}  // namespace litert::vision

#endif  // LITERT_TOOLS_VISION_EVAL_IMAGE_H_

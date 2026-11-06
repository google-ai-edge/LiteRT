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

#include "litert/tools/vision_eval/image.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <ios>
#include <iosfwd>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"  // from @stblib

namespace litert::vision {
namespace {
constexpr uint64_t kImageLimit = 256ULL * 1024 * 1024;
Expected<void> Require(bool condition, const std::string& message) {
  if (!condition) return Error(kLiteRtStatusErrorInvalidArgument, message);
  return {};
}
}  // namespace

Expected<std::vector<ImageSample>> ReadSamples(
    const std::filesystem::path& manifest,
    const std::filesystem::path& image_root, int start, int limit) {
  LITERT_RETURN_IF_ERROR(
      Require(start >= 0 && limit >= 0, "Negative sample range"));
  std::ifstream file(manifest);
  LITERT_RETURN_IF_ERROR(
      Require(static_cast<bool>(file),
              "Cannot open manifest: " + PathToUtf8(manifest)));
  std::vector<ImageSample> rows;
  std::set<int> ids;
  std::string line;
  bool first = true;
  const std::filesystem::path manifest_root = manifest.parent_path();
  const std::filesystem::path& root =
      image_root.empty() ? manifest_root : image_root;
  auto integer = [](const std::string& text, int& value) {
    const auto parsed =
        std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
  };
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    if (first && line == "id\tlabel\tpath") {
      first = false;
      continue;
    }
    first = false;
    const auto tab = line.find('\t');
    const auto second =
        tab == std::string::npos ? tab : line.find('\t', tab + 1);
    LITERT_RETURN_IF_ERROR(
        Require(second != std::string::npos &&
                    line.find('\t', second + 1) == std::string::npos,
                "Bad manifest row: " + line));
    ImageSample row{};
    LITERT_RETURN_IF_ERROR(
        Require(integer(line.substr(0, tab), row.id) &&
                    integer(line.substr(tab + 1, second - tab - 1), row.label),
                "Invalid manifest ID or label: " + line));
    const std::string raw_path = line.substr(second + 1);
    LITERT_RETURN_IF_ERROR(
        Require(!raw_path.empty() && ids.insert(row.id).second,
                "Empty image path or duplicate sample ID"));
    const std::filesystem::path sample_path = Utf8ToPath(raw_path);
    row.path = root.empty() ? sample_path : (root / sample_path);
    rows.push_back(std::move(row));
  }
  LITERT_RETURN_IF_ERROR(
      Require(file.eof(), "Failed to read manifest: " + PathToUtf8(manifest)));
  LITERT_RETURN_IF_ERROR(
      Require(static_cast<size_t>(start) < rows.size(), "No selected samples"));
  rows.erase(rows.begin(), rows.begin() + start);
  if (limit && static_cast<size_t>(limit) < rows.size()) rows.resize(limit);
  return rows;
}

namespace {
template <typename Actual, typename Reference>
Expected<Comparison> CompareElements(size_t count, Actual actual,
                                     Reference reference, bool floating,
                                     double atol, double rtol, double max_code,
                                     double mean_code) {
  LITERT_RETURN_IF_ERROR(Require(count > 0, "Cannot compare empty tensor"));
  for (double tolerance : {atol, rtol, max_code, mean_code}) {
    LITERT_RETURN_IF_ERROR(
        Require(std::isfinite(tolerance) && tolerance >= 0,
                "Tolerances must be finite and nonnegative"));
  }
  Comparison out;
  for (size_t i = 0; i < count; ++i) {
    const double a = actual(i), b = reference(i);
    const double delta = std::abs(a - b);
    out.finite &= std::isfinite(a) && std::isfinite(b);
    out.max_abs = std::max(out.max_abs, delta);
    out.mean_abs += delta;
    out.allclose &= delta <= atol + rtol * std::abs(b);
  }
  out.mean_abs /= count;
  out.allclose = out.finite && (floating ? out.allclose
                                         : out.max_abs <= max_code &&
                                               out.mean_abs <= mean_code);
  return out;
}
}  // namespace

Expected<Comparison> CompareTensorValues(const void* actual,
                                         const void* reference, size_t count,
                                         bool floating, double atol,
                                         double rtol, double max_code,
                                         double mean_code) {
  LITERT_RETURN_IF_ERROR(Require(actual && reference, "Missing tensor data"));
  return CompareElements(
      count,
      [&](size_t i) -> double {
        return floating ? static_cast<const float*>(actual)[i]
                        : static_cast<const int8_t*>(actual)[i];
      },
      [&](size_t i) -> double {
        return floating ? static_cast<const float*>(reference)[i]
                        : static_cast<const int8_t*>(reference)[i];
      },
      floating, atol, rtol, max_code, mean_code);
}

Expected<Comparison> CompareScores(const std::vector<float>& actual,
                                   const std::vector<float>& reference,
                                   double atol, double rtol) {
  LITERT_RETURN_IF_ERROR(
      Require(actual.size() == reference.size(), "Output size mismatch"));
  LITERT_ASSIGN_OR_RETURN(
      auto result, CompareTensorValues(actual.data(), reference.data(),
                                       actual.size(), true, atol, rtol, 0, 0));
  LITERT_RETURN_IF_ERROR(Require(result.finite, "Nonfinite output score"));
  return result;
}

Expected<Comparison> CompareQuantizedScores(
    const std::vector<float>& actual, const std::vector<float>& reference,
    const LiteRtQuantizationPerTensor& quantization, double max_code,
    double mean_code) {
  LITERT_RETURN_IF_ERROR(
      Require(actual.size() == reference.size(), "Output size mismatch"));
  LITERT_RETURN_IF_ERROR(
      Require(std::isfinite(quantization.scale) && quantization.scale > 0,
              "Invalid quantization scale"));
  // Recover integer output codes from dequantized scores. This is distinct
  // from quantizing FP32 model inputs; these values already represent codes.
  return CompareElements(
      actual.size(),
      [&](size_t i) {
        return std::round(actual[i] / quantization.scale +
                          quantization.zero_point);
      },
      [&](size_t i) {
        return std::round(reference[i] / quantization.scale +
                          quantization.zero_point);
      },
      false, 0, 0, max_code, mean_code);
}
Expected<std::vector<int>> TopK(const std::vector<float>& scores, int k) {
  LITERT_RETURN_IF_ERROR(
      Require(k >= 0 && std::all_of(scores.begin(), scores.end(),
                                    [](float f) { return std::isfinite(f); }),
              "Invalid top-k scores"));
  std::vector<int> indices(scores.size());
  std::iota(indices.begin(), indices.end(), 0);
  absl::c_partial_sort(
      indices,
      indices.begin() + std::min(static_cast<size_t>(k), indices.size()),
      [&](int a, int b) {
        return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
      });
  indices.resize(std::min(static_cast<size_t>(k), indices.size()));
  return indices;
}

Expected<void> ValidatePreprocessing(const ImagePreprocessing& c) {
  LITERT_RETURN_IF_ERROR(
      Require(c.interpolation == "bilinear" || c.interpolation == "bicubic",
              "Unknown interpolation"));
  LITERT_RETURN_IF_ERROR(Require(
      c.resize > 0 && c.resize <= 65536 && c.crop_h > 0 && c.crop_w > 0 &&
          static_cast<uint64_t>(c.crop_h) * c.crop_w * 3 * sizeof(float) <=
              kImageLimit,
      "Invalid resize/crop dimensions"));
  for (int i = 0; i < 3; ++i)
    LITERT_RETURN_IF_ERROR(Require(std::isfinite(c.mean[i]) &&
                                       std::isfinite(c.stddev[i]) &&
                                       c.stddev[i] > 0,
                                   "Invalid normalization parameters"));
  return {};
}
Expected<std::array<int, 2>> ResizedDimensions(int w, int h,
                                               const ImagePreprocessing& c) {
  LITERT_RETURN_IF_ERROR(ValidatePreprocessing(c));
  LITERT_RETURN_IF_ERROR(
      Require(w > 0 && h > 0 && static_cast<uint64_t>(w) * h * 3 <= kImageLimit,
              "Invalid decoded image dimensions"));
  double longer =
      static_cast<double>(std::max(w, h)) * c.resize / std::min(w, h);
  LITERT_RETURN_IF_ERROR(Require(longer >= 1 && longer <= 65536,
                                 "Resized image dimension exceeds limit"));
  return w < h ? std::array<int, 2>{c.resize, static_cast<int>(longer)}
               : std::array<int, 2>{static_cast<int>(longer), c.resize};
}
Expected<int> CropOffset(int d) {
  LITERT_RETURN_IF_ERROR(Require(d >= 0, "Crop exceeds resized image"));
  // Python round() uses ties-to-even, independent of the process rounding mode.
  return d / 2 + ((d % 2) && ((d / 2) % 2));
}
Expected<std::vector<uint8_t>> ResizeRgb(const uint8_t* src, int w, int h,
                                         int ow, int oh,
                                         const ImagePreprocessing& c) {
  LITERT_RETURN_IF_ERROR(ValidatePreprocessing(c));
  LITERT_RETURN_IF_ERROR(Require(
      src && w > 0 && h > 0 && ow > 0 && oh > 0 &&
          static_cast<uint64_t>(w) * h * 3 <= kImageLimit &&
          static_cast<uint64_t>(ow) * h * 3 * sizeof(float) <= kImageLimit &&
          static_cast<uint64_t>(ow) * oh * 3 <= kImageLimit,
      "Resize exceeds image allocation limit"));
  struct Coefficients {
    int first;
    std::vector<double> weights;
  };
  auto coefficients = [&](int input,
                          int output) -> Expected<std::vector<Coefficients>> {
    std::vector<Coefficients> result(output);
    double scale = static_cast<double>(input) / output;
    double filter_scale = c.antialias ? std::max(1.0, scale) : 1.0;
    double support = (c.interpolation == "bicubic" ? 2.0 : 1.0) * filter_scale;
    for (int x = 0; x < output; ++x) {
      double center = (x + 0.5) * scale;
      int first = std::max(0, static_cast<int>(center - support + 0.5));
      int last = std::min(input, static_cast<int>(center + support + 0.5));
      Coefficients entry{first, {}};
      double total = 0;
      for (int i = first; i < last; ++i) {
        double v = std::abs((i - center + 0.5) / filter_scale);
        double weight;
        if (c.interpolation == "bilinear")
          weight = std::max(0.0, 1.0 - v);
        else if (v < 1)
          weight = ((1.5 * v - 2.5) * v) * v + 1;
        else if (v < 2)
          weight = ((-0.5 * v + 2.5) * v - 4) * v + 2;
        else
          weight = 0;
        entry.weights.push_back(weight);
        total += weight;
      }
      LITERT_RETURN_IF_ERROR(Require(total != 0, "Empty resize filter"));
      for (auto& weight : entry.weights) weight /= total;
      result[x] = std::move(entry);
    }
    return result;
  };
  LITERT_ASSIGN_OR_RETURN(auto xs, coefficients(w, ow));
  LITERT_ASSIGN_OR_RETURN(auto ys, coefficients(h, oh));
  auto pixel = [](double x) {
    return static_cast<uint8_t>(std::clamp(std::floor(x + 0.5), 0.0, 255.0));
  };
  // PIL's RGB image pipeline rounds/clamps the horizontal pass to uint8.
  std::vector<uint8_t> tmp(static_cast<size_t>(ow) * h * 3),
      out(static_cast<size_t>(ow) * oh * 3);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < ow; ++x)
      for (int ch = 0; ch < 3; ++ch) {
        double v = 0;
        const auto& k = xs[x];
        for (size_t i = 0; i < k.weights.size(); ++i)
          v += k.weights[i] *
               src[(static_cast<size_t>(y) * w + k.first + i) * 3 + ch];
        tmp[(static_cast<size_t>(y) * ow + x) * 3 + ch] = pixel(v);
      }
  for (int y = 0; y < oh; ++y)
    for (int x = 0; x < ow; ++x)
      for (int ch = 0; ch < 3; ++ch) {
        double v = 0;
        const auto& k = ys[y];
        for (size_t i = 0; i < k.weights.size(); ++i)
          v += k.weights[i] * tmp[((k.first + i) * ow + x) * 3 + ch];
        out[(static_cast<size_t>(y) * ow + x) * 3 + ch] = pixel(v);
      }
  return out;
}
Expected<std::vector<float>> PreprocessRgb(const uint8_t* src, int w, int h,
                                           const ImagePreprocessing& c, bool nchw) {
  LITERT_ASSIGN_OR_RETURN(auto dims, ResizedDimensions(w, h, c));
  LITERT_ASSIGN_OR_RETURN(auto rgb, ResizeRgb(src, w, h, dims[0], dims[1], c));
  int padded_w = std::max(dims[0], c.crop_w),
      padded_h = std::max(dims[1], c.crop_h);
  int pad_left = (padded_w - dims[0]) / 2, pad_top = (padded_h - dims[1]) / 2;
  LITERT_ASSIGN_OR_RETURN(int left, CropOffset(padded_w - c.crop_w));
  LITERT_ASSIGN_OR_RETURN(int top, CropOffset(padded_h - c.crop_h));
  left -= pad_left;
  top -= pad_top;
  std::vector<float> out(static_cast<size_t>(c.crop_w) * c.crop_h * 3);
  for (int y = 0; y < c.crop_h; ++y)
    for (int x = 0; x < c.crop_w; ++x)
      for (int ch = 0; ch < 3; ++ch) {
        int sx = x + left, sy = y + top;
        float p = (sx >= 0 && sx < dims[0] && sy >= 0 && sy < dims[1])
                      ? rgb[(static_cast<size_t>(sy) * dims[0] + sx) * 3 + ch]
                      : 0;
        float v = (p / 255.0f - c.mean[ch]) / c.stddev[ch];
        size_t index =
            nchw ? (static_cast<size_t>(ch) * c.crop_h + y) * c.crop_w + x
                 : (static_cast<size_t>(y) * c.crop_w + x) * 3 + ch;
        out[index] = v;
      }
  return out;
}
Expected<std::vector<float>> DecodePreprocess(
    const std::filesystem::path& path, const ImagePreprocessing& c, bool nchw) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  const auto end = input ? input.tellg() : std::streampos(-1);
  LITERT_RETURN_IF_ERROR(Require(
      input && end > 0 && static_cast<uint64_t>(end) <= kImageLimit,
      "Cannot read image: " + PathToUtf8(path)));
  std::vector<stbi_uc> encoded(static_cast<size_t>(end));
  input.seekg(0);
  LITERT_RETURN_IF_ERROR(Require(
      static_cast<bool>(
          input.read(reinterpret_cast<char*>(encoded.data()),
                     static_cast<std::streamsize>(encoded.size()))),
      "Cannot read image: " + PathToUtf8(path)));
  int w = 0;
  int h = 0;
  int channels = 0;
  auto* data = stbi_load_from_memory(
      encoded.data(), static_cast<int>(encoded.size()), &w, &h, &channels, 3);
  LITERT_RETURN_IF_ERROR(
      Require(data != nullptr, "Cannot decode image: " + PathToUtf8(path)));
  std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> image(data,
                                                             stbi_image_free);
  return PreprocessRgb(image.get(), w, h, c, nchw);
}
}  // namespace litert::vision

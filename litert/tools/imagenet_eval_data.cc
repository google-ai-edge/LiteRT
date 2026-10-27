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

#include "litert/tools/imagenet_eval_data.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace litert::imagenet {
namespace {
uint32_t Ror(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
constexpr uint64_t kImageLimit = 256ULL * 1024 * 1024;
void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

void Sha256::Block(const uint8_t* p) {
  static constexpr uint32_t k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
      0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  uint32_t w[64];
  for (int i = 0; i < 16; ++i)
    w[i] = uint32_t(p[4 * i]) << 24 | uint32_t(p[4 * i + 1]) << 16 |
           uint32_t(p[4 * i + 2]) << 8 | uint32_t(p[4 * i + 3]);
  for (int i = 16; i < 64; ++i) {
    auto x = w[i - 15], y = w[i - 2];
    w[i] = w[i - 16] + (Ror(x, 7) ^ Ror(x, 18) ^ (x >> 3)) + w[i - 7] +
           (Ror(y, 17) ^ Ror(y, 19) ^ (y >> 10));
  }
  auto a = state_[0], b = state_[1], c = state_[2], d = state_[3];
  auto e = state_[4], f = state_[5], g = state_[6], h = state_[7];
  for (int i = 0; i < 64; ++i) {
    uint32_t t1 = h + (Ror(e, 6) ^ Ror(e, 11) ^ Ror(e, 25)) +
                  ((e & f) ^ (~e & g)) + k[i] + w[i];
    uint32_t t2 =
        (Ror(a, 2) ^ Ror(a, 13) ^ Ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}
void Sha256::Update(const void* raw, size_t size) {
  const auto* data = static_cast<const uint8_t*>(raw);
  Require(size <= std::numeric_limits<uint64_t>::max() / 8 - bytes_,
          "SHA256 input too large");
  bytes_ += size;
  while (size) {
    size_t n = std::min(size, pending_.size() - used_);
    std::memcpy(pending_.data() + used_, data, n);
    used_ += n;
    data += n;
    size -= n;
    if (used_ == 64) {
      Block(pending_.data());
      used_ = 0;
    }
  }
}
std::string Sha256::Hex() const {
  Sha256 copy = *this;
  uint64_t bits = bytes_ * 8;
  uint8_t tail[128]{};
  tail[0] = 0x80;
  size_t padding = used_ < 56 ? 56 - used_ : 120 - used_;
  for (int i = 0; i < 8; ++i) tail[padding + i] = bits >> (56 - i * 8);
  copy.Update(tail, padding + 8);
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (auto word : copy.state_) out << std::setw(8) << word;
  return out.str();
}
std::string HashBytes(const void* data, size_t size) {
  Sha256 hash;
  hash.Update(data, size);
  return hash.Hex();
}
std::string HashFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  Require(bool(input), "Cannot open file for hashing: " + path);
  std::array<char, 65536> buffer;
  Sha256 hash;
  while (input) {
    input.read(buffer.data(), buffer.size());
    hash.Update(buffer.data(), input.gcount());
  }
  Require(input.eof(), "Failed to read file: " + path);
  return hash.Hex();
}
Json ReadJson(const std::string& path) {
  std::ifstream in(path);
  Require(bool(in), "Cannot open JSON: " + path);
  return Json::parse(in);
}
void WriteJson(const std::string& path, const Json& value) {
  std::ofstream out(path);
  Require(bool(out), "Cannot open JSON output: " + path);
  out << value.dump(2) << '\n';
  out.flush();
  Require(bool(out), "Failed to write JSON: " + path);
}

Comparison CompareScores(const std::vector<float>& actual,
                         const std::vector<float>& reference, double atol,
                         double rtol) {
  Require(actual.size() == reference.size() && !actual.empty(),
          "Output size mismatch");
  Require(std::isfinite(atol) && std::isfinite(rtol) && atol >= 0 && rtol >= 0,
          "Tolerances must be finite and nonnegative");
  Comparison out;
  for (size_t i = 0; i < actual.size(); ++i) {
    Require(std::isfinite(actual[i]) && std::isfinite(reference[i]),
            "Nonfinite output score");
    double delta = std::abs(double(actual[i]) - reference[i]);
    out.max_abs = std::max(out.max_abs, delta);
    out.mean_abs += delta;
    out.allclose &= delta <= atol + rtol * std::abs(double(reference[i]));
  }
  out.mean_abs /= actual.size();
  return out;
}
std::vector<int> TopK(const std::vector<float>& scores, int k) {
  Require(k >= 0 && std::all_of(scores.begin(), scores.end(),
                                [](float f) { return std::isfinite(f); }),
          "Invalid top-k scores");
  std::vector<int> indices(scores.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(
      indices.begin(), indices.begin() + std::min(size_t(k), indices.size()),
      indices.end(), [&](int a, int b) {
        return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
      });
  indices.resize(std::min(size_t(k), indices.size()));
  return indices;
}

void ValidatePreprocessing(const Preprocessing& c) {
  Require(c.interpolation == "bilinear" || c.interpolation == "bicubic",
          "Unknown interpolation");
  Require(c.resize > 0 && c.resize <= 65536 && c.crop_h > 0 && c.crop_w > 0 &&
              uint64_t(c.crop_h) * c.crop_w * 3 * sizeof(float) <= kImageLimit,
          "Invalid resize/crop dimensions");
  for (int i = 0; i < 3; ++i)
    Require(std::isfinite(c.mean[i]) && std::isfinite(c.stddev[i]) &&
                c.stddev[i] > 0,
            "Invalid normalization parameters");
}
Preprocessing ParsePreprocessing(const Json& value) {
  const auto& j =
      value.contains("preprocessing") ? value.at("preprocessing") : value;
  Preprocessing c;
  Require(j.value("profile", "torchvision") == "torchvision",
          "Only native TorchVision preprocessing is supported");
  c.interpolation = j.at("interpolation").get<std::string>();
  c.resize = j.at("resize").get<int>();
  c.crop_h = j.at("crop_h").get<int>();
  c.crop_w = j.at("crop_w").get<int>();
  c.antialias = j.at("antialias").get<bool>();
  c.mean = j.at("mean").get<std::array<float, 3>>();
  c.stddev = j.at("std").get<std::array<float, 3>>();
  ValidatePreprocessing(c);
  return c;
}
Json DescribePreprocessing(const Preprocessing& c) {
  return {{"profile", "torchvision"}, {"interpolation", c.interpolation},
          {"resize", c.resize},       {"crop_h", c.crop_h},
          {"crop_w", c.crop_w},       {"antialias", c.antialias},
          {"mean", c.mean},           {"std", c.stddev}};
}
std::array<int, 2> ResizedDimensions(int w, int h, const Preprocessing& c) {
  ValidatePreprocessing(c);
  Require(w > 0 && h > 0 && uint64_t(w) * h * 3 <= kImageLimit,
          "Invalid decoded image dimensions");
  const auto longer = [&](int a, int b) {
    double result = double(a) * c.resize / b;
    Require(result >= 1 && result <= 65536,
            "Resized image dimension exceeds limit");
    return static_cast<int>(result);
  };
  return w < h ? std::array<int, 2>{c.resize, longer(h, w)}
               : std::array<int, 2>{longer(w, h), c.resize};
}
int CropOffset(int d) {
  Require(d >= 0, "Crop exceeds resized image");
  // Python round() uses ties-to-even, independent of the process rounding mode.
  return d / 2 + ((d % 2) && ((d / 2) % 2));
}
std::vector<uint8_t> ResizeRgb(const uint8_t* src, int w, int h, int ow, int oh,
                               const Preprocessing& c) {
  ValidatePreprocessing(c);
  Require(src && w > 0 && h > 0 && ow > 0 && oh > 0 &&
              uint64_t(w) * h * 3 <= kImageLimit &&
              uint64_t(ow) * h * 3 * sizeof(float) <= kImageLimit &&
              uint64_t(ow) * oh * 3 <= kImageLimit,
          "Resize exceeds image allocation limit");
  struct Coefficients {
    int first;
    std::vector<double> weights;
  };
  auto coefficients = [&](int input, int output) {
    std::vector<Coefficients> result(output);
    double scale = double(input) / output;
    double filter_scale = c.antialias ? std::max(1.0, scale) : 1.0;
    double support = (c.interpolation == "bicubic" ? 2.0 : 1.0) * filter_scale;
    for (int x = 0; x < output; ++x) {
      double center = (x + 0.5) * scale;
      int first = std::max(0, int(center - support + 0.5));
      int last = std::min(input, int(center + support + 0.5));
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
      Require(total != 0, "Empty resize filter");
      for (auto& weight : entry.weights) weight /= total;
      result[x] = std::move(entry);
    }
    return result;
  };
  auto xs = coefficients(w, ow), ys = coefficients(h, oh);
  auto pixel = [](double x) {
    return static_cast<uint8_t>(std::clamp(std::floor(x + 0.5), 0.0, 255.0));
  };
  // PIL's RGB image pipeline rounds/clamps the horizontal pass to uint8.
  std::vector<uint8_t> tmp(size_t(ow) * h * 3), out(size_t(ow) * oh * 3);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < ow; ++x)
      for (int ch = 0; ch < 3; ++ch) {
        double v = 0;
        const auto& k = xs[x];
        for (size_t i = 0; i < k.weights.size(); ++i)
          v += k.weights[i] * src[(size_t(y) * w + k.first + i) * 3 + ch];
        tmp[(size_t(y) * ow + x) * 3 + ch] = pixel(v);
      }
  for (int y = 0; y < oh; ++y)
    for (int x = 0; x < ow; ++x)
      for (int ch = 0; ch < 3; ++ch) {
        double v = 0;
        const auto& k = ys[y];
        for (size_t i = 0; i < k.weights.size(); ++i)
          v += k.weights[i] * tmp[((k.first + i) * ow + x) * 3 + ch];
        out[(size_t(y) * ow + x) * 3 + ch] = pixel(v);
      }
  return out;
}
std::vector<float> PreprocessRgb(const uint8_t* src, int w, int h,
                                 const Preprocessing& c, bool nchw) {
  auto dims = ResizedDimensions(w, h, c);
  auto rgb = ResizeRgb(src, w, h, dims[0], dims[1], c);
  int padded_w = std::max(dims[0], c.crop_w),
      padded_h = std::max(dims[1], c.crop_h);
  int pad_left = (padded_w - dims[0]) / 2, pad_top = (padded_h - dims[1]) / 2;
  int left = CropOffset(padded_w - c.crop_w) - pad_left;
  int top = CropOffset(padded_h - c.crop_h) - pad_top;
  std::vector<float> out(size_t(c.crop_w) * c.crop_h * 3);
  for (int y = 0; y < c.crop_h; ++y)
    for (int x = 0; x < c.crop_w; ++x)
      for (int ch = 0; ch < 3; ++ch) {
        int sx = x + left, sy = y + top;
        float p = (sx >= 0 && sx < dims[0] && sy >= 0 && sy < dims[1])
                      ? rgb[(size_t(sy) * dims[0] + sx) * 3 + ch]
                      : 0;
        float v = (p / 255.0f - c.mean[ch]) / c.stddev[ch];
        size_t index = nchw ? (size_t(ch) * c.crop_h + y) * c.crop_w + x
                            : (size_t(y) * c.crop_w + x) * 3 + ch;
        out[index] = v;
      }
  return out;
}
std::vector<float> DecodePreprocess(const std::string& path,
                                    const Preprocessing& c, bool nchw) {
  int w, h, channels;
  auto* data = stbi_load(path.c_str(), &w, &h, &channels, 3);
  Require(data != nullptr, "Cannot decode image: " + path);
  try {
    auto out = PreprocessRgb(data, w, h, c, nchw);
    stbi_image_free(data);
    return out;
  } catch (...) {
    stbi_image_free(data);
    throw;
  }
}
}  // namespace litert::imagenet

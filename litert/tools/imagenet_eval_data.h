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

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

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
std::string HashFile(const std::string& path);
std::string HashBytes(const void* data, size_t size);
Json ReadJson(const std::string& path);
void WriteJson(const std::string& path, const Json& value);

struct Preprocessing {
  std::string interpolation = "bilinear";
  int resize = 232;
  int crop_h = 224;
  int crop_w = 224;
  bool antialias = true;
  std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};
  std::array<float, 3> stddev = {0.229f, 0.224f, 0.225f};
};
Preprocessing ParsePreprocessing(const Json& value);
Json DescribePreprocessing(const Preprocessing& config);
void ValidatePreprocessing(const Preprocessing& config);
std::array<int, 2> ResizedDimensions(int width, int height,
                                     const Preprocessing& config);
int CropOffset(int difference);
std::vector<uint8_t> ResizeRgb(const uint8_t* input, int width, int height,
                               int out_w, int out_h,
                               const Preprocessing& config);
std::vector<float> PreprocessRgb(const uint8_t* input, int width, int height,
                                 const Preprocessing& config, bool nchw);
std::vector<float> DecodePreprocess(const std::string& path,
                                    const Preprocessing& config, bool nchw);

struct Comparison {
  bool allclose = true;
  double max_abs = 0;
  double mean_abs = 0;
};
Comparison CompareScores(const std::vector<float>& actual,
                         const std::vector<float>& reference, double atol,
                         double rtol);
std::vector<int> TopK(const std::vector<float>& scores, int k);
}  // namespace litert::imagenet
#endif  // LITERT_TOOLS_IMAGENET_EVAL_DATA_H_

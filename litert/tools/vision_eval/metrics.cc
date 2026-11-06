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

#include "litert/tools/vision_eval/metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <functional>
#include <ios>
#include <limits>
#include <queue>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/tools/vision_eval/data.h"
#include "litert/tools/vision_eval/image.h"

namespace litert::vision {
namespace {
Expected<std::vector<double>> Targets(const TargetFile& target,
                                      const std::filesystem::path& root) {
  const auto& spec = target.tensor();
  if (spec.type() != FLOAT32 && spec.type() != UINT8 && spec.type() != INT32) {
    return Invalid("Targets require FLOAT32, UINT8 or INT32");
  }
  LITERT_ASSIGN_OR_RETURN(
      auto bytes,
      ReadTensor(Resolve(root, Utf8ToPath(target.file().path())), spec));
  const size_t width = spec.type() == UINT8 ? 1 : 4;
  std::vector<double> result(bytes.size() / width);
  for (size_t i = 0; i < result.size(); ++i) {
    if (spec.type() == UINT8) {
      result[i] = bytes[i];
    } else if (spec.type() == FLOAT32) {
      float value;
      std::memcpy(&value, bytes.data() + i * sizeof(value), sizeof(value));
      result[i] = value;
    } else {
      int32_t value;
      std::memcpy(&value, bytes.data() + i * sizeof(value), sizeof(value));
      result[i] = value;
    }
  }
  return result;
}

Expected<std::vector<float>> Planes(const TensorSpec& spec,
                                    const std::vector<float>& input) {
  if (spec.shape_size() != 4 || spec.shape(0) != 1) {
    return Invalid("Dense predictions require static batch-one rank-four tensors");
  }
  if (spec.layout() == NCHW) return input;
  const int h = spec.shape(1), w = spec.shape(2), c = spec.shape(3);
  std::vector<float> result(input.size());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int k = 0; k < c; ++k) {
        result[(static_cast<size_t>(k) * h + y) * w + x] =
            input[(static_cast<size_t>(y) * w + x) * c + k];
      }
    }
  }
  return result;
}

void SegmentationResult(SegmentationMetrics& result) {
  double total = 0;
  uint32_t valid = 0;
  for (uint32_t c = 0; c < result.classes(); ++c) {
    uint64_t row = 0, column = 0;
    for (uint32_t k = 0; k < result.classes(); ++k) {
      row += result.confusion(c * result.classes() + k);
      column += result.confusion(k * result.classes() + c);
    }
    const auto intersection = result.confusion(c * result.classes() + c);
    const auto united = row + column - intersection;
    if (united) {
      total += static_cast<double>(intersection) / united;
      ++valid;
    }
  }
  result.set_evaluated_classes(valid);
  if (valid) result.set_miou(total / valid);
  else result.clear_miou();
}

bool NonnegativeFinite(double value) {
  return std::isfinite(value) && value >= 0;
}

Expected<void> ValidateMetric(const OutputMetrics& row) {
  if (!SafeName(row.name()) || row.metrics_case() == OutputMetrics::METRICS_NOT_SET) {
    return Invalid("Missing task metric contract");
  }
  if (row.has_classification()) {
    const auto& r = row.classification();
    if (!r.images() || r.correct_top1() > r.correct_top5() ||
        r.correct_top5() > r.images()) return Invalid("Corrupt classification metrics");
  } else if (row.has_segmentation()) {
    const auto& r = row.segmentation();
    if (!r.images() || !r.classes() || r.classes() > 4096 ||
        r.confusion_size() != static_cast<int64_t>(r.classes()) * r.classes()) {
      return Invalid("Corrupt confusion matrix");
    }
    uint64_t count = 0;
    for (auto value : r.confusion()) {
      if (count > std::numeric_limits<uint64_t>::max() - value) {
        return Invalid("Confusion matrix count overflow");
      }
      count += value;
    }
    if (count != r.valid_pixels()) return Invalid("Confusion matrix pixel count mismatch");
  } else if (row.has_depth()) {
    const auto& r = row.depth();
    if ((r.images() == 0) != (r.valid_pixels() == 0) ||
        (r.images() && (!r.has_abs_rel() || !r.has_rmse() || !r.has_delta1() ||
                        !NonnegativeFinite(r.abs_rel()) ||
                        !NonnegativeFinite(r.rmse()) ||
                        !NonnegativeFinite(r.delta1()) || r.delta1() > 1))) {
      return Invalid("Corrupt depth metrics");
    }
  } else if (row.has_normals()) {
    const auto& r = row.normals();
    if ((r.images() == 0) != (r.valid_pixels() == 0) ||
        (r.valid_pixels() &&
         (!r.has_mean_degrees() || !NonnegativeFinite(r.mean_degrees()) ||
          r.mean_degrees() > 180 || !NonnegativeFinite(r.within_11_25_degrees()) ||
          !NonnegativeFinite(r.within_22_5_degrees()) ||
          !NonnegativeFinite(r.within_30_degrees()) ||
          r.within_11_25_degrees() > r.within_22_5_degrees() ||
          r.within_22_5_degrees() > r.within_30_degrees() ||
          r.within_30_degrees() > 1))) {
      return Invalid("Corrupt normals metrics");
    }
    if (r.valid_pixels()) {
      std::error_code ec;
      const auto bytes = std::filesystem::file_size(
          Utf8ToPath(row.normal_errors().path()), ec);
      if (ec || bytes / sizeof(float) != r.valid_pixels() ||
          bytes % sizeof(float)) {
        return Invalid("Normals spool pixel count mismatch");
      }
    }
  }
  return {};
}

// A single merge keeps at most 64 input streams open. Stop early for median
// selection; full merges use the default limit.
template <class Consume>
Expected<uint64_t> MergeRuns(
    const std::vector<std::filesystem::path>& paths, Consume consume,
    uint64_t limit = std::numeric_limits<uint64_t>::max()) {
  using Entry = std::pair<float, size_t>;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
  std::vector<std::ifstream> inputs(paths.size());
  auto advance = [&](size_t index) -> Expected<void> {
    char raw[sizeof(float)];
    if (inputs[index].read(raw, sizeof(raw))) {
      float value;
      std::memcpy(&value, raw, sizeof(value));
      heap.push({value, index});
    } else if (!inputs[index].eof() || inputs[index].gcount()) {
      return Invalid("Median merge read failed");
    }
    return {};
  };
  for (size_t i = 0; i < paths.size(); ++i) {
    inputs[i].open(paths[i], std::ios::binary);
    if (!inputs[i]) return Invalid("Cannot open median run");
    LITERT_RETURN_IF_ERROR(advance(i));
  }
  uint64_t count = 0;
  while (!heap.empty() && count < limit) {
    const auto [value, index] = heap.top();
    heap.pop();
    consume(value, count++);
    LITERT_RETURN_IF_ERROR(advance(index));
  }
  return count;
}
}  // namespace

Expected<std::vector<float>> ResizePlane(const float* source, int height,
                                         int width, int output_height,
                                         int output_width) {
  if (!source || height <= 0 || width <= 0 || output_height <= 0 ||
      output_width <= 0 ||
      static_cast<uint64_t>(output_height) * output_width >
          128u * 1024u * 1024u) {
    return Invalid("Invalid metric resize");
  }
  std::vector<float> out(static_cast<size_t>(output_height) * output_width);
  for (int y = 0; y < output_height; ++y) {
    const double fy = std::max((y + .5) * height / output_height - .5, 0.0);
    const int y0 = static_cast<int>(fy), y1 = std::min(y0 + 1, height - 1);
    const double dy = fy - y0;
    for (int x = 0; x < output_width; ++x) {
      const double fx = std::max((x + .5) * width / output_width - .5, 0.0);
      const int x0 = static_cast<int>(fx), x1 = std::min(x0 + 1, width - 1);
      const double dx = fx - x0;
      const double top = source[static_cast<size_t>(y0) * width + x0] * (1 - dx) +
                         source[static_cast<size_t>(y0) * width + x1] * dx;
      const double bottom =
          source[static_cast<size_t>(y1) * width + x0] * (1 - dx) +
          source[static_cast<size_t>(y1) * width + x1] * dx;
      out[static_cast<size_t>(y) * output_width + x] =
          static_cast<float>(top * (1 - dy) + bottom * dy);
    }
  }
  return out;
}

Expected<OutputMetrics> Score(const OutputSpec& spec,
                              const std::vector<float>& prediction,
                              const TargetFile* target, int label,
                              const std::filesystem::path& root,
                              const std::filesystem::path& checkpoint) {
  LITERT_ASSIGN_OR_RETURN(auto bytes, TensorBytes(spec.tensor()));
  const size_t width = spec.tensor().type() == FLOAT32 ? sizeof(float) : 1;
  if (prediction.size() != bytes / width ||
      !std::all_of(prediction.begin(), prediction.end(),
                   [](float value) { return std::isfinite(value); })) {
    return Invalid("Invalid/nonfinite prediction tensor");
  }
  OutputMetrics out;
  out.set_name(spec.tensor().name());
  const auto& options = spec.scoring();
  if (options.task() == CLASSIFICATION) {
    if (label < 0 || static_cast<size_t>(label) >= prediction.size()) {
      return Invalid("Classification label is out of range");
    }
    LITERT_ASSIGN_OR_RETURN(auto top, TopK(prediction, std::min<size_t>(5, prediction.size())));
    auto* m = out.mutable_classification();
    m->set_images(1);
    m->set_correct_top1(top[0] == label);
    m->set_correct_top5(std::find(top.begin(), top.end(), label) != top.end());
    for (auto index : top) m->add_ranked_classes(index);
    m->set_top1(m->correct_top1());
    m->set_top5(m->correct_top5());
    return out;
  }
  if (!target) return Invalid("Dense scoring requires a target");
  const bool normals = options.task() == NORMALS;
  const auto& ts = target->tensor();
  if (ts.shape_size() != (normals ? 3 : 2) || (normals && ts.shape(0) != 3)) {
    return Invalid("Targets must have HW or normals CHW shape");
  }
  const int th = ts.shape(ts.shape_size() - 2), tw = ts.shape(ts.shape_size() - 1);
  const size_t pixels = static_cast<size_t>(th) * tw;
  LITERT_ASSIGN_OR_RETURN(auto truth, Targets(*target, root));
  LITERT_ASSIGN_OR_RETURN(auto planes, Planes(spec.tensor(), prediction));
  const auto& shape = spec.tensor().shape();
  const bool nchw = spec.tensor().layout() == NCHW;
  const int channels = shape.Get(nchw ? 1 : 3);
  const int h = shape.Get(nchw ? 2 : 1), w = shape.Get(nchw ? 3 : 2);
  std::vector<uint8_t> mask(pixels, 1);
  if (target->has_mask()) {
    TensorSpec ms;
    ms.set_type(BOOL);
    ms.add_shape(th);
    ms.add_shape(tw);
    LITERT_ASSIGN_OR_RETURN(
        mask, ReadTensor(Resolve(root, Utf8ToPath(target->mask().path())), ms));
  }
  if (options.task() == SEGMENTATION) {
    if (channels != options.classes() || (ts.type() != UINT8 && ts.type() != INT32)) {
      return Invalid("Segmentation classes/target type mismatch");
    }
    // Resize one logit plane at a time, before argmax. Strict comparison keeps
    // the lowest channel index when logits tie.
    std::vector<float> best(pixels, -std::numeric_limits<float>::infinity());
    std::vector<int> labels(pixels, 0);
    for (int c = 0; c < channels; ++c) {
      LITERT_ASSIGN_OR_RETURN(auto resized,
          ResizePlane(planes.data() + static_cast<size_t>(c) * h * w, h, w, th, tw));
      for (size_t p = 0; p < pixels; ++p) {
        if (resized[p] > best[p]) {
          best[p] = resized[p];
          labels[p] = c;
        }
      }
    }
    auto* m = out.mutable_segmentation();
    m->set_images(1);
    m->set_classes(channels);
    m->mutable_confusion()->Resize(channels * channels, 0);
    uint64_t valid = 0;
    for (size_t p = 0; p < pixels; ++p) {
      if (!mask[p] || truth[p] == options.ignore_label()) continue;
      if (truth[p] < 0 || truth[p] >= channels) {
        return Invalid("Segmentation target label out of range");
      }
      const size_t index = static_cast<size_t>(truth[p]) * channels + labels[p];
      m->set_confusion(index, m->confusion(index) + 1);
      ++valid;
    }
    m->set_valid_pixels(valid);
    SegmentationResult(*m);
    return out;
  }
  if (ts.type() != FLOAT32 || channels != (normals ? 3 : 1)) {
    return Invalid("Dense channel count or target type mismatch");
  }
  std::vector<std::vector<float>> resized;
  for (int c = 0; c < channels; ++c) {
    LITERT_ASSIGN_OR_RETURN(auto plane,
        ResizePlane(planes.data() + static_cast<size_t>(c) * h * w, h, w, th, tw));
    resized.push_back(std::move(plane));
  }
  if (options.task() == DEPTH) {
    auto* m = out.mutable_depth();
    double rel = 0, squared = 0;
    uint64_t valid = 0, delta = 0;
    for (size_t p = 0; p < pixels; ++p) {
      const double t = truth[p];
      if (!mask[p] || !std::isfinite(t) || t <= options.min_depth() ||
          t > options.max_depth()) continue;
      const double v = std::clamp(static_cast<double>(resized[0][p]),
                                  options.min_depth(), options.max_depth());
      rel += std::abs(v - t) / t;
      squared += (v - t) * (v - t);
      delta += std::max(v / t, t / v) < 1.25;
      ++valid;
    }
    m->set_valid_pixels(valid);
    if (valid) {
      m->set_images(1);
      m->set_abs_rel(rel / valid);
      m->set_rmse(std::sqrt(squared / valid));
      m->set_delta1(static_cast<double>(delta) / valid);
    } else {
      m->set_empty_images(1);
    }
    return out;
  }
  if (!normals) return Invalid("Unknown dense task");
  std::vector<float> errors;
  double sum = 0;
  uint64_t counts[3]{};
  for (size_t p = 0; p < pixels; ++p) {
    double dot = 0, tn = 0, pn = 0;
    bool finite = true;
    for (int c = 0; c < 3; ++c) {
      const double t = truth[c * pixels + p], v = resized[c][p];
      finite &= std::isfinite(t);
      dot += t * v;
      tn += t * t;
      pn += v * v;
    }
    if (!mask[p] || !finite || std::sqrt(tn) <= 1e-12) continue;
    const double cosine = dot / std::max(std::sqrt(pn) * std::sqrt(tn), 1e-30);
    const float degrees = static_cast<float>(
        std::acos(std::clamp(cosine, -1.0, 1.0)) * (180 / std::acos(-1.0)));
    errors.push_back(degrees);
    sum += degrees;
    counts[0] += degrees < 11.25;
    counts[1] += degrees < 22.5;
    counts[2] += degrees < 30;
  }
  auto* m = out.mutable_normals();
  m->set_valid_pixels(errors.size());
  if (errors.empty()) {
    m->set_empty_images(1);
    return out;
  }
  m->set_images(1);
  m->set_mean_degrees(sum / errors.size());
  m->set_within_11_25_degrees(static_cast<double>(counts[0]) / errors.size());
  m->set_within_22_5_degrees(static_cast<double>(counts[1]) / errors.size());
  m->set_within_30_degrees(static_cast<double>(counts[2]) / errors.size());
  const auto path = checkpoint / (spec.tensor().name() + ".angles.f32");
  LITERT_RETURN_IF_ERROR(WriteBytes(path, errors.data(), errors.size() * sizeof(float)));
  LITERT_ASSIGN_OR_RETURN(auto identity, Identify(path));
  *out.mutable_normal_errors() = std::move(identity);
  return out;
}

Expected<double> ExactMedian(const std::vector<FileIdentity>& files,
                             const std::filesystem::path& scratch,
                             size_t chunk_values) {
  if (!chunk_values || chunk_values > 64u * 1024u * 1024u / sizeof(float)) {
    return Invalid("Median sorting buffer must be 1..64 MiB");
  }
  if (scratch.empty()) return Invalid("Empty median scratch directory");
  std::error_code ec;
  std::filesystem::create_directories(scratch, ec);
  if (ec || !std::filesystem::is_directory(scratch, ec) || ec) {
    return Invalid("Cannot create median scratch directory: " +
                   PathToUtf8(scratch));
  }
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code cleanup_ec;
      std::filesystem::remove_all(path, cleanup_ec);
    }
  } cleanup{scratch};
  std::vector<float> chunk;
  chunk.reserve(chunk_values);
  std::vector<std::filesystem::path> runs;
  uint64_t count = 0;
  auto flush = [&]() -> Expected<void> {
    if (chunk.empty()) return {};
    std::sort(chunk.begin(), chunk.end());
    const auto path = scratch / (std::to_string(runs.size()) + ".raw");
    LITERT_RETURN_IF_ERROR(
        WriteBytes(path, chunk.data(), chunk.size() * sizeof(float)));
    runs.push_back(path);
    chunk.clear();
    return {};
  };
  for (const auto& file : files) {
    LITERT_RETURN_IF_ERROR(VerifyFile(file));
    std::ifstream input(Utf8ToPath(file.path()), std::ios::binary);
    if (!input) return Invalid("Cannot open normals error spool");
    char raw[sizeof(float)];
    while (input.read(raw, sizeof(raw))) {
      float value;
      std::memcpy(&value, raw, sizeof(value));
      if (!std::isfinite(value) || value < 0 || value > 180) {
        return Invalid("Corrupt normals error spool");
      }
      chunk.push_back(value);
      ++count;
      if (chunk.size() == chunk_values) LITERT_RETURN_IF_ERROR(flush());
    }
    if (!input.eof() || input.gcount()) return Invalid("Truncated normals error spool");
  }
  LITERT_RETURN_IF_ERROR(flush());
  if (!count) return Invalid("Median of empty input");
  while (runs.size() > 64) {
    std::vector<std::filesystem::path> next;
    for (size_t start = 0; start < runs.size(); start += 64) {
      const size_t n = std::min<size_t>(64, runs.size() - start);
      const std::vector<std::filesystem::path> group(runs.begin() + start,
                                                     runs.begin() + start + n);
      const auto filename =
          "merge_" + std::to_string(runs.size()) + "_" + std::to_string(start);
      const auto path = scratch / filename;
      std::ofstream output(path, std::ios::binary);
      if (!output) return Invalid("Cannot create median merge run");
      LITERT_RETURN_IF_ERROR(MergeRuns(group, [&](float v, uint64_t) {
        char raw[sizeof(float)];
        std::memcpy(raw, &v, sizeof(raw));
        output.write(raw, sizeof(raw));
      }));
      output.flush();
      if (!output) return Invalid("Median merge write failed");
      output.close();
      for (const auto& old : group) {
        std::filesystem::remove(old, ec);
        if (ec) return Invalid("Cannot remove temporary median run");
      }
      next.push_back(path);
    }
    runs = std::move(next);
  }
  double lower = 0, upper = 0;
  LITERT_ASSIGN_OR_RETURN(auto consumed, MergeRuns(runs, [&](float v, uint64_t i) {
    if (i == (count - 1) / 2) lower = v;
    if (i == count / 2) upper = v;
  }, count / 2 + 1));
  if (consumed != count / 2 + 1) return Invalid("Incomplete median merge");
  return (lower + upper) / 2;
}

Expected<void> MetricAccumulator::Add(const OutputMetrics& row) {
  LITERT_RETURN_IF_ERROR(ValidateMetric(row));
  if (total_.metrics_case() != OutputMetrics::METRICS_NOT_SET &&
      (total_.name() != row.name() || total_.metrics_case() != row.metrics_case())) {
    return Invalid("Checkpoint metric contract changed");
  }
  auto& out = total_;
  out.set_name(row.name());
  if (row.has_classification()) {
    auto* m = out.mutable_classification();
    const auto& r = row.classification();
    m->set_images(m->images() + r.images());
    m->set_correct_top1(m->correct_top1() + r.correct_top1());
    m->set_correct_top5(m->correct_top5() + r.correct_top5());
  } else if (row.has_segmentation()) {
    auto* m = out.mutable_segmentation();
    const auto& r = row.segmentation();
    if (!m->classes()) {
      m->set_classes(r.classes());
      m->mutable_confusion()->Resize(r.confusion_size(), 0);
    }
    if (r.classes() != m->classes()) return Invalid("Confusion matrix class count changed");
    for (int i = 0; i < r.confusion_size(); ++i) {
      m->set_confusion(i, m->confusion(i) + r.confusion(i));
    }
    m->set_images(m->images() + r.images());
    m->set_valid_pixels(m->valid_pixels() + r.valid_pixels());
  } else if (row.has_depth()) {
    auto* m = out.mutable_depth();
    const auto& r = row.depth();
    m->set_images(m->images() + r.images());
    m->set_empty_images(m->empty_images() + r.empty_images());
    m->set_valid_pixels(m->valid_pixels() + r.valid_pixels());
    if (r.images()) {
      m->set_abs_rel(m->abs_rel() + r.abs_rel() * r.images());
      m->set_rmse(m->rmse() + r.rmse() * r.images());
      m->set_delta1(m->delta1() + r.delta1() * r.images());
    }
  } else if (row.has_normals()) {
    auto* m = out.mutable_normals();
    const auto& r = row.normals();
    m->set_images(m->images() + r.images());
    m->set_empty_images(m->empty_images() + r.empty_images());
    m->set_valid_pixels(m->valid_pixels() + r.valid_pixels());
    if (r.valid_pixels()) {
      m->set_mean_degrees(m->mean_degrees() + r.mean_degrees() * r.valid_pixels());
      m->set_within_11_25_degrees(m->within_11_25_degrees() + r.within_11_25_degrees() * r.valid_pixels());
      m->set_within_22_5_degrees(m->within_22_5_degrees() + r.within_22_5_degrees() * r.valid_pixels());
      m->set_within_30_degrees(m->within_30_degrees() + r.within_30_degrees() * r.valid_pixels());
      if (!row.has_normal_errors()) return Invalid("Missing normals checkpoint state");
      normal_errors_.push_back(row.normal_errors());
    }
  }
  return {};
}

Expected<OutputMetrics> MetricAccumulator::Result(
    const std::filesystem::path& scratch) const {
  if (total_.metrics_case() == OutputMetrics::METRICS_NOT_SET) return Invalid("No metric records");
  auto out = total_;
  if (out.has_classification() && out.classification().images()) {
    auto* m = out.mutable_classification();
    m->set_top1(static_cast<double>(m->correct_top1()) / m->images());
    m->set_top5(static_cast<double>(m->correct_top5()) / m->images());
  }
  if (out.has_segmentation()) SegmentationResult(*out.mutable_segmentation());
  if (out.has_depth() && out.depth().images()) {
    auto* m = out.mutable_depth();
    m->set_abs_rel(m->abs_rel() / m->images());
    m->set_rmse(m->rmse() / m->images());
    m->set_delta1(m->delta1() / m->images());
  }
  if (out.has_normals() && out.normals().valid_pixels()) {
    auto* m = out.mutable_normals();
    const double n = m->valid_pixels();
    m->set_mean_degrees(m->mean_degrees() / n);
    m->set_within_11_25_degrees(m->within_11_25_degrees() / n);
    m->set_within_22_5_degrees(m->within_22_5_degrees() / n);
    m->set_within_30_degrees(m->within_30_degrees() / n);
    LITERT_ASSIGN_OR_RETURN(auto median, ExactMedian(normal_errors_, scratch));
    m->set_median_degrees(median);
  }
  return out;
}

Expected<OutputMetrics> Aggregate(const std::vector<OutputMetrics>& rows,
                                  const std::filesystem::path& scratch) {
  MetricAccumulator accumulator;
  for (const auto& row : rows) LITERT_RETURN_IF_ERROR(accumulator.Add(row));
  return accumulator.Result(scratch);
}
}  // namespace litert::vision

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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"         // from @com_google_absl
#include "absl/strings/str_split.h"    // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"           // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/cc/options/litert_intel_openvino_options.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace {

struct Config {
  std::string model;
  std::string manifest;
  std::string image_root;
  std::string out;
  std::string accelerator = "cpu";
  std::string dispatch_library_dir;
  std::string openvino_device;
  std::string gpu_backend = "automatic";
  std::string gpu_precision = "default";
  int gpu_benchmark_mode = 0;
  int resize = 232;
  int crop_h = 224;
  int crop_w = 224;
  int antialias = 1;
  std::string mean = "0.485,0.456,0.406";
  std::string stddev = "0.229,0.224,0.225";
  int max_samples = 0;
};

struct Row {
  int id;
  int label;
  std::string path;
};

void PrintUsage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " --model=MODEL --manifest=MANIFEST --image_root=DIR [options]\n"
            << "Options:\n"
            << "  --out=PATH              Output JSONL path\n"
            << "  --accelerator=cpu       Comma-delimited cpu,gpu,npu\n"
            << "  --dispatch_library_dir=DIR\n"
            << "  --openvino_device=cpu|gpu|npu|auto\n"
            << "  --gpu_backend=automatic|webgpu|opengl|opencl\n"
            << "  --gpu_precision=default|fp32|fp16\n"
            << "  --gpu_benchmark_mode=0|1\n"
            << "  --resize=232            Resize shorter side\n"
            << "  --crop_h=224 --crop_w=224\n"
            << "  --antialias=1           Antialias bilinear downsampling\n"
            << "  --mean=0.485,0.456,0.406 --std=0.229,0.224,0.225\n"
            << "  --max_samples=0         0 means all rows\n";
}

Config ParseArgs(int argc, char** argv) {
  Config cfg;
  std::unordered_map<std::string, std::string*> strings = {
      {"model", &cfg.model},
      {"manifest", &cfg.manifest},
      {"image_root", &cfg.image_root},
      {"out", &cfg.out},
      {"accelerator", &cfg.accelerator},
      {"dispatch_library_dir", &cfg.dispatch_library_dir},
      {"openvino_device", &cfg.openvino_device},
      {"gpu_backend", &cfg.gpu_backend},
      {"gpu_precision", &cfg.gpu_precision},
      {"mean", &cfg.mean},
      {"std", &cfg.stddev},
  };
  std::unordered_map<std::string, int*> ints = {
      {"resize", &cfg.resize},
      {"crop_h", &cfg.crop_h},
      {"crop_w", &cfg.crop_w},
      {"antialias", &cfg.antialias},
      {"gpu_benchmark_mode", &cfg.gpu_benchmark_mode},
      {"max_samples", &cfg.max_samples},
  };
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (arg.rfind("--", 0) != 0) {
      std::cerr << "Unexpected positional argument: " << arg << "\n";
      std::exit(EXIT_FAILURE);
    }
    size_t eq = arg.find('=');
    if (eq == std::string::npos) {
      std::cerr << "Expected --name=value argument: " << arg << "\n";
      std::exit(EXIT_FAILURE);
    }
    std::string name = arg.substr(2, eq - 2);
    std::string value = arg.substr(eq + 1);
    if (auto it = strings.find(name); it != strings.end()) {
      *it->second = value;
    } else if (auto int_it = ints.find(name); int_it != ints.end()) {
      *int_it->second = std::stoi(value);
    } else {
      std::cerr << "Unknown argument: --" << name << "\n";
      std::exit(EXIT_FAILURE);
    }
  }
  if (cfg.model.empty() || cfg.manifest.empty() || cfg.image_root.empty()) {
    PrintUsage(argv[0]);
    std::exit(EXIT_FAILURE);
  }
  return cfg;
}

bool RequestsGpu(const Config& cfg) {
  for (absl::string_view accelerator : absl::StrSplit(cfg.accelerator, ',')) {
    if (accelerator == "gpu") return true;
  }
  return false;
}

std::vector<float> ParseTriple(const std::string& text) {
  std::vector<float> out;
  for (absl::string_view part : absl::StrSplit(text, ',')) {
    out.push_back(std::stof(std::string(part)));
  }
  if (out.size() != 3) {
    std::cerr << "Expected 3 comma-delimited values: " << text << "\n";
    std::exit(EXIT_FAILURE);
  }
  return out;
}

litert::HwAcceleratorSet GetAccelerators(const Config& cfg) {
  litert::HwAcceleratorSet accelerators(litert::HwAccelerators::kNone);
  for (absl::string_view accelerator : absl::StrSplit(cfg.accelerator, ',')) {
    if (accelerator == "cpu") {
      accelerators |= litert::HwAccelerators::kCpu;
    } else if (accelerator == "gpu") {
      accelerators |= litert::HwAccelerators::kGpu;
    } else if (accelerator == "npu") {
      accelerators |= litert::HwAccelerators::kNpu;
    }
  }
  return accelerators;
}

litert::Expected<void> SetOpenVinoDevice(litert::Options& options,
                                         const std::string& device) {
  if (device.empty()) {
    return {};
  }
  LITERT_ASSIGN_OR_RETURN(auto& openvino_options,
                          options.GetIntelOpenVinoOptions());
  if (device == "cpu") {
    openvino_options.SetDeviceType(kLiteRtIntelOpenVinoDeviceTypeCPU);
  } else if (device == "gpu") {
    openvino_options.SetDeviceType(kLiteRtIntelOpenVinoDeviceTypeGPU);
  } else if (device == "npu") {
    openvino_options.SetDeviceType(kLiteRtIntelOpenVinoDeviceTypeNPU);
  } else if (device == "auto") {
    openvino_options.SetDeviceType(kLiteRtIntelOpenVinoDeviceTypeAUTO);
  } else {
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Unknown openvino_device");
  }
  return {};
}

std::vector<Row> ReadManifest(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    std::cerr << "Failed to open manifest: " << path << "\n";
    std::exit(EXIT_FAILURE);
  }
  std::vector<Row> rows;
  std::string line;
  bool first = true;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    if (first && line == "id\tlabel\tpath") {
      first = false;
      continue;
    }
    first = false;
    std::vector<std::string> cols;
    for (absl::string_view col : absl::StrSplit(line, '\t')) {
      cols.emplace_back(col);
    }
    if (cols.size() != 3) {
      std::cerr << "Bad manifest row: " << line << "\n";
      std::exit(EXIT_FAILURE);
    }
    rows.push_back({std::stoi(cols[0]), std::stoi(cols[1]), cols[2]});
  }
  return rows;
}

struct Contributor {
  int first = 0;
  std::vector<float> weights;
};

std::vector<Contributor> BuildBilinearContributors(int src_size, int dst_size,
                                                   bool antialias) {
  std::vector<Contributor> contributors(dst_size);
  const float scale = static_cast<float>(src_size) / dst_size;
  const float filter_scale = antialias && scale > 1.0f ? scale : 1.0f;
  const float support = filter_scale;
  for (int out = 0; out < dst_size; ++out) {
    const float center = (out + 0.5f) * scale;
    const int first =
        std::max(0, static_cast<int>(std::floor(center - support - 0.5f)) + 1);
    const int last =
        std::min(src_size - 1,
                 static_cast<int>(std::floor(center + support - 0.5f)));
    Contributor contributor;
    contributor.first = first;
    contributor.weights.reserve(last - first + 1);
    float sum = 0.0f;
    for (int in = first; in <= last; ++in) {
      const float input_center = in + 0.5f;
      const float weight =
          std::max(0.0f, 1.0f - std::abs(center - input_center) / filter_scale);
      contributor.weights.push_back(weight);
      sum += weight;
    }
    if (sum > 0.0f) {
      for (float& weight : contributor.weights) weight /= sum;
    }
    contributors[out] = std::move(contributor);
  }
  return contributors;
}

std::vector<uint8_t> ResizeBilinear(const uint8_t* src, int src_w, int src_h,
                                    int dst_w, int dst_h, bool antialias) {
  const std::vector<Contributor> x_contributors =
      BuildBilinearContributors(src_w, dst_w, antialias);
  const std::vector<Contributor> y_contributors =
      BuildBilinearContributors(src_h, dst_h, antialias);

  std::vector<float> tmp(static_cast<size_t>(src_h) * dst_w * 3);
  for (int y = 0; y < src_h; ++y) {
    for (int x = 0; x < dst_w; ++x) {
      const Contributor& contributor = x_contributors[x];
      for (int c = 0; c < 3; ++c) {
        float value = 0.0f;
        for (int i = 0; i < static_cast<int>(contributor.weights.size());
             ++i) {
          value += contributor.weights[i] *
                   src[(static_cast<size_t>(y) * src_w +
                        contributor.first + i) *
                           3 +
                       c];
        }
        tmp[(static_cast<size_t>(y) * dst_w + x) * 3 + c] = value;
      }
    }
  }

  std::vector<uint8_t> dst(static_cast<size_t>(dst_w) * dst_h * 3);
  for (int y = 0; y < dst_h; ++y) {
    const Contributor& contributor = y_contributors[y];
    for (int x = 0; x < dst_w; ++x) {
      for (int c = 0; c < 3; ++c) {
        float value = 0.0f;
        for (int i = 0; i < static_cast<int>(contributor.weights.size());
             ++i) {
          value += contributor.weights[i] *
                   tmp[(static_cast<size_t>(contributor.first + i) * dst_w +
                        x) *
                           3 +
                       c];
        }
        value = std::clamp(std::round(value), 0.0f, 255.0f);
        dst[(static_cast<size_t>(y) * dst_w + x) * 3 + c] =
            static_cast<uint8_t>(value);
      }
    }
  }
  return dst;
}

std::vector<float> DecodePreprocess(const std::string& path, int resize,
                                    int crop_h, int crop_w,
                                    const std::vector<float>& mean,
                                    const std::vector<float>& stddev,
                                    bool nchw, bool antialias) {
  int width = 0;
  int height = 0;
  int channels = 0;
  stbi_uc* decoded = stbi_load(path.c_str(), &width, &height, &channels, 3);
  if (decoded == nullptr) {
    std::cerr << "Failed to decode image: " << path << "\n";
    std::exit(EXIT_FAILURE);
  }
  int resized_w;
  int resized_h;
  if (width < height) {
    resized_w = resize;
    resized_h = static_cast<int>(
        std::round(height * resize / static_cast<float>(width)));
  } else {
    resized_h = resize;
    resized_w = static_cast<int>(
        std::round(width * resize / static_cast<float>(height)));
  }
  std::vector<uint8_t> resized =
      ResizeBilinear(decoded, width, height, resized_w, resized_h, antialias);
  stbi_image_free(decoded);

  int left = static_cast<int>(std::round((resized_w - crop_w) / 2.0f));
  int top = static_cast<int>(std::round((resized_h - crop_h) / 2.0f));
  std::vector<float> out(static_cast<size_t>(crop_h) * crop_w * 3);
  for (int y = 0; y < crop_h; ++y) {
    for (int x = 0; x < crop_w; ++x) {
      for (int c = 0; c < 3; ++c) {
        const float pixel =
            resized[(static_cast<size_t>(y + top) * resized_w + (x + left)) *
                        3 +
                    c] /
            255.0f;
        const float normalized = (pixel - mean[c]) / stddev[c];
        if (nchw) {
          out[(static_cast<size_t>(c) * crop_h + y) * crop_w + x] = normalized;
        } else {
          out[(static_cast<size_t>(y) * crop_w + x) * 3 + c] = normalized;
        }
      }
    }
  }
  return out;
}

std::vector<int> TopK(const std::vector<float>& scores, int k) {
  std::vector<int> indices(scores.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(
      indices.begin(), indices.begin() + std::min(k, (int)indices.size()),
      indices.end(), [&](int a, int b) { return scores[a] > scores[b]; });
  indices.resize(std::min(k, static_cast<int>(indices.size())));
  return indices;
}

litert::Expected<void> Run(const Config& cfg) {
  std::vector<litert::EnvironmentOptions::Option> env_options;
  if (!cfg.dispatch_library_dir.empty()) {
    env_options.push_back(litert::EnvironmentOptions::Option{
        litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
        cfg.dispatch_library_dir.c_str()});
  }
  LITERT_ASSIGN_OR_RETURN(
      auto env,
      litert::Environment::Create(litert::EnvironmentOptions(env_options)));
  LITERT_ASSIGN_OR_RETURN(auto options, litert::Options::Create());
  options.SetHardwareAccelerators(GetAccelerators(cfg));
  LITERT_RETURN_IF_ERROR(SetOpenVinoDevice(options, cfg.openvino_device));
  if (RequestsGpu(cfg)) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
    if (cfg.gpu_backend == "webgpu") {
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetBackend(litert::GpuOptions::Backend::kWebGpu));
    } else if (cfg.gpu_backend == "opengl" || cfg.gpu_backend == "gl") {
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetBackend(litert::GpuOptions::Backend::kOpenGl));
    } else if (cfg.gpu_backend == "opencl" || cfg.gpu_backend == "cl") {
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetBackend(litert::GpuOptions::Backend::kOpenCl));
    } else if (cfg.gpu_backend != "automatic") {
      return litert::Error(kLiteRtStatusErrorInvalidArgument,
                           "Unknown gpu_backend");
    }
    if (cfg.gpu_precision == "fp32") {
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetPrecision(litert::GpuOptions::Precision::kFp32));
    } else if (cfg.gpu_precision == "fp16") {
      LITERT_RETURN_IF_ERROR(
          gpu_options.SetPrecision(litert::GpuOptions::Precision::kFp16));
    } else if (cfg.gpu_precision != "default") {
      return litert::Error(kLiteRtStatusErrorInvalidArgument,
                           "Unknown gpu_precision");
    }
    LITERT_RETURN_IF_ERROR(
        gpu_options.EnableBenchmarkMode(cfg.gpu_benchmark_mode != 0));
  }
  LITERT_ASSIGN_OR_RETURN(
      auto model, litert::CompiledModel::Create(env, cfg.model, options));
  LITERT_ASSIGN_OR_RETURN(auto input_buffers, model.CreateInputBuffers(0));
  LITERT_ASSIGN_OR_RETURN(auto output_buffers, model.CreateOutputBuffers(0));
  LITERT_ASSIGN_OR_RETURN(auto input_type, input_buffers[0].TensorType());
  LITERT_ASSIGN_OR_RETURN(auto output_type, output_buffers[0].TensorType());
  if (input_type.ElementType() != litert::ElementType::Float32 ||
      output_type.ElementType() != litert::ElementType::Float32) {
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "This prototype supports float32 input/output only");
  }

  const auto dims = input_type.Layout().Dimensions();
  if (dims.size() != 4) {
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Expected rank-4 image input");
  }
  bool nchw = dims[1] == 3;
  int input_h = nchw ? dims[2] : dims[1];
  int input_w = nchw ? dims[3] : dims[2];
  int crop_h = cfg.crop_h > 0 ? cfg.crop_h : input_h;
  int crop_w = cfg.crop_w > 0 ? cfg.crop_w : input_w;

  std::vector<Row> rows = ReadManifest(cfg.manifest);
  int max_samples = cfg.max_samples;
  if (max_samples > 0 && max_samples < static_cast<int>(rows.size())) {
    rows.resize(max_samples);
  }
  std::ofstream out_file;
  if (!cfg.out.empty()) {
    out_file.open(cfg.out);
  }
  std::ostream& out = out_file.is_open() ? out_file : std::cout;
  std::vector<float> mean = ParseTriple(cfg.mean);
  std::vector<float> stddev = ParseTriple(cfg.stddev);
  const auto output_dims = output_type.Layout().Dimensions();
  size_t output_size = 1;
  for (int32_t dim : output_dims) output_size *= dim;
  std::vector<float> scores(output_size);

  int top1_correct = 0;
  int top5_correct = 0;
  for (const Row& row : rows) {
    std::string image_path = cfg.image_root + "/" + row.path;
    std::vector<float> input =
        DecodePreprocess(image_path, cfg.resize, crop_h, crop_w, mean, stddev,
                         nchw, cfg.antialias != 0);
    LITERT_RETURN_IF_ERROR(
        input_buffers[0].Write<float>(absl::MakeSpan(input)));
    LITERT_RETURN_IF_ERROR(
        model.Run(static_cast<size_t>(0), input_buffers, output_buffers));
    LITERT_RETURN_IF_ERROR(
        output_buffers[0].Read<float>(absl::MakeSpan(scores)));
    std::vector<int> top5 = TopK(scores, 5);
    int pred = top5.empty() ? -1 : top5[0];
    top1_correct += pred == row.label;
    top5_correct +=
        std::find(top5.begin(), top5.end(), row.label) != top5.end();
    out << "{\"id\":" << row.id << ",\"label\":" << row.label
        << ",\"top1\":" << pred << ",\"top5\":[";
    for (size_t i = 0; i < top5.size(); ++i) {
      if (i) out << ",";
      out << top5[i];
    }
    out << "]}\n";
  }
  std::cerr << "samples=" << rows.size() << " top1="
            << (rows.empty() ? 0.0 : (double)top1_correct / rows.size())
            << " top5="
            << (rows.empty() ? 0.0 : (double)top5_correct / rows.size())
            << "\n";
  return {};
}

}  // namespace

int main(int argc, char** argv) {
  Config cfg = ParseArgs(argc, argv);
  auto status = Run(cfg);
  if (!status) {
    ABSL_LOG(ERROR) << status.Error().Message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

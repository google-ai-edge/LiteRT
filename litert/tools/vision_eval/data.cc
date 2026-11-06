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

#include "litert/tools/vision_eval/data.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <ios>
#include <iostream>
#include <set>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/tools/vision_eval/image.h"
#include "google/protobuf/descriptor.h"  // from @com_google_protobuf
#include "google/protobuf/message.h"  // from @com_google_protobuf
#include "google/protobuf/text_format.h"  // from @com_google_protobuf

namespace litert::vision {
namespace {
constexpr size_t kMaxTensorBytes = 512u * 1024u * 1024u;

bool ModelType(ElementType type) {
  return type == FLOAT32 || type == INT8 || type == UINT8;
}

Expected<void> SetFlag(google::protobuf::Message& message, const std::string& key,
                       const std::string& value) {
  using Field = google::protobuf::FieldDescriptor;
  const auto* field = message.GetDescriptor()->FindFieldByName(key);
  if (!field || field->is_repeated() ||
      field->cpp_type() == Field::CPPTYPE_MESSAGE) {
    return Invalid("Unknown scalar flag: --" + key);
  }
  const auto* reflection = message.GetReflection();
  if (field->cpp_type() == Field::CPPTYPE_STRING) {
    reflection->SetString(&message, field, value);
    return {};
  }
  if (field->cpp_type() == Field::CPPTYPE_BOOL) {
    if (value != "true" && value != "false" && value != "1" && value != "0") {
      return Invalid("Expected boolean: --" + key);
    }
    reflection->SetBool(&message, field, value == "true" || value == "1");
    return {};
  }
  if (field->cpp_type() == Field::CPPTYPE_ENUM) {
    auto name = value;
    absl::c_transform(name, name.begin(), [](unsigned char c) {
      return static_cast<char>(std::toupper(c));
    });
    if (key == "summary_format") {
      if (name == "TEXTPROTO" || name == "TEXT") name = "TEXT_PROTO";
      if (name == "BINARYPROTO" || name == "BINARY") name = "BINARY_PROTO";
    }
    const auto* enumeration = field->enum_type()->FindValueByName(name);
    if (!enumeration) return Invalid("Invalid enum: --" + key + "=" + value);
    reflection->SetEnum(&message, field, enumeration);
    return {};
  }
  if (!google::protobuf::TextFormat::ParseFieldValueFromString(value, field,
                                                              &message)) {
    return Invalid("Invalid value: --" + key);
  }
  return {};
}
}  // namespace

Error Invalid(const std::string& message) {
  return Error(kLiteRtStatusErrorInvalidArgument, message);
}

Expected<void> ReadProto(const std::filesystem::path& path,
                         google::protobuf::Message& message) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input || input.tellg() < 0 || input.tellg() > 64 * 1024 * 1024) {
    return Invalid("Cannot read proto or file exceeds 64 MiB: " +
                   PathToUtf8(path));
  }
  std::string bytes(static_cast<size_t>(input.tellg()), '\0');
  input.seekg(0);
  if (!input.read(bytes.data(), bytes.size())) {
    return Invalid("Cannot read proto: " + PathToUtf8(path));
  }
  bool parsed = false;
  const auto ext = path.extension();
  if (ext == ".binpb") {
    parsed = message.ParseFromString(bytes);
  } else if (ext == ".txtpb") {
    parsed = google::protobuf::TextFormat::ParseFromString(bytes, &message);
  } else {
    return Invalid("Proto filenames must end in .txtpb or .binpb: " +
                   PathToUtf8(path));
  }
  if (!parsed || !message.IsInitialized()) {
    return Invalid("Invalid protobuf: " + PathToUtf8(path));
  }
  return {};
}

Expected<void> WriteBytes(const std::filesystem::path& path, const void* data,
                          size_t size) {
  if (path.empty()) return Invalid("Cannot write empty path");
  if (size > 0 && !data) return Invalid("Null data for non-empty write");
  const auto parent = path.parent_path();
  std::error_code ec;
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec || !std::filesystem::is_directory(parent, ec) || ec) {
      return Invalid("Cannot create directory: " + PathToUtf8(parent));
    }
  }
  // The run lock serializes writers. Readers see only complete files.
  auto temporary = path;
  temporary += ".tmp";
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (size > 0) {
      out.write(static_cast<const char*>(data),
                static_cast<std::streamsize>(size));
    }
    out.flush();
    if (!out) return Invalid("Cannot write: " + PathToUtf8(temporary));
  }
  std::filesystem::rename(temporary, path, ec);
  if (ec) {
    std::error_code remove_ec;
    std::filesystem::remove(temporary, remove_ec);
    return Invalid("Cannot rename temporary file to: " + PathToUtf8(path));
  }
  return {};
}

Expected<void> WriteProto(const std::filesystem::path& path,
                          const google::protobuf::Message& message,
                          ProtoFormat format) {
  std::string bytes;
  const bool ok = format == TEXT_PROTO
                      ? google::protobuf::TextFormat::PrintToString(message, &bytes)
                      : message.SerializeToString(&bytes);
  if (!ok) return Invalid("Cannot serialize protobuf");
  return WriteBytes(path, bytes.data(), bytes.size());
}

Expected<size_t> TensorBytes(const TensorSpec& spec) {
  size_t size;
  switch (spec.type()) {
    case FLOAT32:
    case INT32:
      size = 4;
      break;
    case INT8:
    case UINT8:
    case BOOL:
      size = 1;
      break;
    default:
      return Invalid("Unknown tensor element type");
  }
  if (spec.has_scale()) {
    const int low = spec.type() == INT8 ? -128 : 0;
    const int high = spec.type() == INT8 ? 127 : 255;
    if ((spec.type() != INT8 && spec.type() != UINT8) ||
        !std::isfinite(spec.scale()) || spec.scale() <= 0 ||
        !std::isfinite(1.0f / spec.scale()) || spec.zero_point() < low ||
        spec.zero_point() > high) {
      return Invalid("Invalid tensor quantization");
    }
  }
  if (spec.shape_size() > 8) return Invalid("Tensor rank exceeds eight");
  for (auto dim : spec.shape()) {
    if (dim <= 0 || size > kMaxTensorBytes / static_cast<size_t>(dim)) {
      return Invalid("Dynamic/empty/oversized tensor shape");
    }
    size *= dim;
  }
  return size;
}

Expected<std::vector<uint8_t>> ReadTensor(const std::filesystem::path& path,
                                          const TensorSpec& spec) {
  LITERT_ASSIGN_OR_RETURN(auto count, TensorBytes(spec));
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file || file.tellg() != static_cast<std::streamoff>(count)) {
    return Invalid("Tensor byte count mismatch: " + PathToUtf8(path));
  }
  std::string raw(count, '\0');
  file.seekg(0);
  file.read(raw.data(), static_cast<std::streamsize>(count));
  if (!file) return Invalid("Cannot read tensor: " + PathToUtf8(path));
  std::vector<uint8_t> bytes(count);
  if (count > 0) {
    std::memcpy(bytes.data(), raw.data(), count);
  }
  return bytes;
}

Expected<std::vector<float>> ReadFloats(const std::filesystem::path& path,
                                        const TensorSpec& spec) {
  if (spec.type() != FLOAT32) return Invalid("Expected FLOAT32 predictions");
  LITERT_ASSIGN_OR_RETURN(auto bytes, ReadTensor(path, spec));
  std::vector<float> result(bytes.size() / sizeof(float));
  std::memcpy(result.data(), bytes.data(), bytes.size());
  for (auto f : result) {
    if (!std::isfinite(f)) {
      return Invalid("Tensor contains NaN or infinity: " + PathToUtf8(path));
    }
  }
  return result;
}

Expected<FileIdentity> Identify(const std::filesystem::path& path) {
  if (path.empty()) return Invalid("Empty file path");
  std::error_code ec;
  const auto absolute = std::filesystem::absolute(path, ec);
  if (ec) {
    return Invalid("Cannot resolve path: " + PathToUtf8(path));
  }
  const bool regular = std::filesystem::is_regular_file(absolute, ec);
  if (ec || !regular) {
    return Invalid("Missing regular file: " + PathToUtf8(absolute));
  }
  FileIdentity file;
  file.set_path(PathToUtf8(absolute));
  return file;
}

Expected<void> VerifyFile(const FileIdentity& file) {
  if (file.path().empty()) {
    return Invalid("Incomplete file identity");
  }
  std::error_code ec;
  const bool regular =
      std::filesystem::is_regular_file(Utf8ToPath(file.path()), ec);
  if (ec || !regular) {
    return Invalid("Missing file: " + file.path());
  }
  return {};
}

bool SafeName(const std::string& value) {
  return !value.empty() && value.size() <= 128 &&
         std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '_' || c == '-';
         });
}

std::filesystem::path Resolve(const std::filesystem::path& root,
                              const std::filesystem::path& path) {
  if (root.empty() || path.empty()) return root.empty() ? path : root;
  return root / path;
}

Expected<ImagePreprocessing> ImageOptions(const Preprocessing& p) {
  ImagePreprocessing out;
  out.interpolation = p.interpolation();
  out.resize = p.resize();
  out.crop_h = p.crop_h();
  out.crop_w = p.crop_w();
  out.antialias = p.antialias();
  if (p.mean_size() && p.mean_size() != 3) {
    return Invalid("Expected three channel means");
  }
  if (p.stddev_size() && p.stddev_size() != 3) {
    return Invalid("Expected three channel standard deviations");
  }
  for (int i = 0; i < 3; ++i) {
    if (p.mean_size()) out.mean[i] = p.mean(i);
    if (p.stddev_size()) out.stddev[i] = p.stddev(i);
  }
  LITERT_RETURN_IF_ERROR(ValidatePreprocessing(out));
  return out;
}

Expected<void> ValidateManifest(const DatasetManifest& manifest, Task task) {
  if (manifest.version() != 1 || manifest.samples_size() == 0) {
    return Invalid("Expected version-1 manifest with samples");
  }
  std::set<std::string> ids, input_names, output_names;
  for (const auto& input : manifest.inputs()) {
    if (!SafeName(input.name()) || !input_names.insert(input.name()).second) {
      return Invalid("Duplicate or invalid input name");
    }
    if (!ModelType(input.type())) {
      return Invalid("Model inputs require FLOAT32, INT8 or UINT8");
    }
    LITERT_RETURN_IF_ERROR(TensorBytes(input));
  }
  for (const auto& output : manifest.outputs()) {
    const auto& tensor = output.tensor();
    const auto& scoring = output.scoring();
    if (!SafeName(tensor.name()) || !output_names.insert(tensor.name()).second) {
      return Invalid("Duplicate or invalid output name");
    }
    if (!ModelType(tensor.type())) {
      return Invalid("Model outputs require FLOAT32, INT8 or UINT8");
    }
    LITERT_RETURN_IF_ERROR(TensorBytes(tensor));
    const auto selected = task == AUTO ? scoring.task() : task;
    if (selected == AUTO) return Invalid("An output needs an evaluation task");
    if (selected == SEGMENTATION &&
        (scoring.classes() <= 0 || scoring.classes() > 4096)) {
      return Invalid("Segmentation class count must be 1..4096");
    }
    if (selected == DEPTH &&
        (!std::isfinite(scoring.min_depth()) ||
         !std::isfinite(scoring.max_depth()) ||
         !(0 < scoring.min_depth() && scoring.min_depth() < scoring.max_depth()))) {
      return Invalid("Invalid depth bounds");
    }
  }
  for (const auto& sample : manifest.samples()) {
    if (!SafeName(sample.id()) || !ids.insert(sample.id()).second) {
      return Invalid("Sample IDs must be unique and filename safe");
    }
    if (sample.has_image()) {
      if (sample.image().empty() || sample.inputs_size() || !sample.has_label() ||
          sample.label() < 0 || (task != AUTO && task != CLASSIFICATION)) {
        return Invalid("Image samples require classification labels and no raw inputs");
      }
    } else {
      std::set<std::string> names;
      for (const auto& input : sample.inputs()) {
        if (!input_names.count(input.name()) || input.file().path().empty() ||
            !names.insert(input.name()).second) {
          return Invalid("Unknown, duplicate or missing prepared input");
        }
      }
      if (names != input_names || names.empty()) {
        return Invalid("Prepared sample does not bind every input");
      }
    }
    std::set<std::string> targets;
    for (const auto& target : sample.targets()) {
      if (!output_names.count(target.name()) ||
          !targets.insert(target.name()).second || target.file().path().empty()) {
        return Invalid("Invalid target binding");
      }
      LITERT_RETURN_IF_ERROR(TensorBytes(target.tensor()));
    }
    if (!sample.has_label() && targets.empty()) {
      return Invalid("Sample has no labels or targets");
    }
  }
  return {};
}

Expected<DatasetManifest> LoadManifest(const EvaluationConfig& config) {
  DatasetManifest result;
  const auto manifest_path = Utf8ToPath(config.manifest());
  if (manifest_path.extension() == ".tsv") {
    LITERT_ASSIGN_OR_RETURN(
        auto rows,
        ReadSamples(manifest_path, Utf8ToPath(config.image_root()), 0, 0));
    for (const auto& row : rows) {
      std::error_code ec;
      const auto image_path = std::filesystem::absolute(row.path, ec);
      if (ec) {
        return Invalid("Cannot resolve image path: " + PathToUtf8(row.path));
      }
      auto* sample = result.add_samples();
      sample->set_id(std::to_string(row.id));
      sample->set_label(row.label);
      sample->set_image(PathToUtf8(image_path));
    }
  } else {
    LITERT_RETURN_IF_ERROR(ReadProto(manifest_path, result));
  }
  LITERT_RETURN_IF_ERROR(ValidateManifest(result, config.task()));
  const size_t begin = static_cast<size_t>(config.start_sample());
  if (begin >= static_cast<size_t>(result.samples_size())) {
    return Invalid("Sample start is out of range");
  }
  if (config.max_samples() &&
      begin + config.max_samples() < static_cast<size_t>(result.samples_size())) {
    result.mutable_samples()->DeleteSubrange(
        begin + config.max_samples(),
        result.samples_size() - begin - config.max_samples());
  }
  if (begin) result.mutable_samples()->DeleteSubrange(0, begin);
  return result;
}

Expected<void> ValidateConfig(const EvaluationConfig& config) {
  if (config.manifest().empty() || config.output_dir().empty() ||
      (!config.score_only() && config.model().empty())) {
    return Invalid("manifest, output_dir and an inference model are required");
  }
  if (config.score_only() &&
      (config.predictions_dir().empty() || !config.baseline_model().empty())) {
    return Invalid("Offline scoring requires predictions_dir and no baseline_model");
  }
  if (config.signature_index() < 0 || config.warmup_runs() < 0 ||
      config.start_sample() < 0 || config.max_samples() < 0) {
    return Invalid("Negative signature, warmup or sample selection");
  }
  if (!std::isfinite(config.atol()) || !std::isfinite(config.rtol()) ||
      config.atol() < 0 || config.rtol() < 0 ||
      !std::isfinite(config.max_code_error()) ||
      !std::isfinite(config.mean_code_error()) ||
      config.max_code_error() < 0 || config.mean_code_error() < 0) {
    return Invalid("Invalid numerical tolerance");
  }
  if ((config.require_output_parity() || config.require_prediction_agreement()) &&
      config.baseline_model().empty() && config.reference_dir().empty()) {
    return Invalid("Comparison gates require a baseline or reference directory");
  }
  if (config.baseline_accelerator() != "inherit" &&
      config.baseline_accelerator() != "cpu") {
    return Invalid("baseline_accelerator must be inherit or cpu");
  }
  if (config.baseline_accelerator() == "cpu" && config.baseline_model().empty()) {
    return Invalid("CPU comparison requires baseline_model");
  }
  if (config.baseline_num_threads() <= 0) {
    return Invalid("baseline_num_threads must be positive");
  }
  if (!config.reference_dir().empty() && !config.baseline_model().empty()) {
    return Invalid("Select baseline_model or reference_dir, not both");
  }
  LITERT_RETURN_IF_ERROR(ImageOptions(config.preprocessing()));
  return {};
}

Expected<EvaluationConfig> ParseArgs(int argc, char** argv) {
  EvaluationConfig config;
  std::set<std::string> seen;
  // Load the whole config before applying explicit scalar flags, regardless of
  // where --config appears in the command line.
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg.rfind("--config=", 0) == 0) {
      if (!seen.insert("config").second) return Invalid("Repeated --config");
      LITERT_RETURN_IF_ERROR(ReadProto(Utf8ToPath(arg.substr(9)), config));
    }
  }
  seen.clear();
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") continue;
    const auto equal = arg.find('=');
    if (arg.rfind("--", 0) != 0 || equal == std::string::npos) {
      return Invalid("Expected --name=value: " + arg);
    }
    const auto name = arg.substr(2, equal - 2);
    const auto value = arg.substr(equal + 1);
    if (name == "config") continue;
    if (!seen.insert(name).second) return Invalid("Repeated flag: --" + name);
    auto* message = static_cast<google::protobuf::Message*>(&config);
    if (!config.GetDescriptor()->FindFieldByName(name)) {
      if (config.backend().GetDescriptor()->FindFieldByName(name)) {
        message = config.mutable_backend();
      } else if (config.preprocessing().GetDescriptor()->FindFieldByName(name)) {
        message = config.mutable_preprocessing();
      }
    }
    LITERT_RETURN_IF_ERROR(SetFlag(*message, name, value));
  }
  LITERT_RETURN_IF_ERROR(ValidateConfig(config));
  return config;
}

void PrintUsage() {
  std::cerr
      << "litert_vision_eval --model=model.tflite --manifest=manifest.txtpb "
         "--output_dir=results [options]\n"
      << "  --config=run.txtpb|run.binpb  CLI values override config fields\n"
      << "  --task=auto|classification|segmentation|depth|normals\n"
      << "  --summary_format=textproto|binaryproto\n"
      << "  --dump_predictions=true --dump_inputs=true\n"
      << "  --accelerator=cpu|gpu|npu --require_full_acceleration=true\n"
      << "  --baseline_model=reference.tflite --baseline_accelerator=inherit|cpu\n"
      << "  --score_only=true --predictions_dir=saved --reference_dir=reference\n"
      << "  --start_sample=0 --max_samples=3 --warmup_runs=0\n"
      << "  Backend and preprocessing scalar fields are also CLI flags.\n"
      << "  See litert/tools/vision_eval/README.md for tensor contracts.\n";
}
}  // namespace litert::vision

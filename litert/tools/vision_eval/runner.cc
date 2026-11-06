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

#include "litert/tools/vision_eval/runner.h"

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifdef ERROR
#undef ERROR
#endif
#ifdef FAILED
#undef FAILED
#endif
#endif

#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/tools/vision_eval/data.h"
#include "litert/tools/vision_eval/execution.h"
#include "litert/tools/vision_eval/image.h"
#include "litert/tools/vision_eval/metrics.h"
#include "litert/tools/vision_eval/run_lock.h"
#include "google/protobuf/repeated_ptr_field.h"  // from @com_google_protobuf

namespace litert::vision {
namespace {
volatile std::sig_atomic_t interrupted = 0;
void Interrupt(int) { interrupted = 1; }

Expected<bool> PathExists(const std::filesystem::path& path) {
  std::error_code ec;
  const bool found = std::filesystem::exists(path, ec);
  if (ec) return Invalid("Cannot stat path: " + PathToUtf8(path));
  return found;
}

std::string Binding(const TensorSpec& tensor) {
  return tensor.binding_name().empty() ? tensor.name() : tensor.binding_name();
}

Expected<size_t> Find(const std::vector<TensorSpec>& tensors,
                      const TensorSpec& requested) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].name() == Binding(requested)) return i;
  }
  return Invalid("Model has no signature tensor named " + Binding(requested));
}

Expected<void> Match(const TensorSpec& expected, const TensorSpec& actual,
                     bool check_type = true) {
  if ((check_type && expected.type() != actual.type()) ||
      expected.shape_size() != actual.shape_size() ||
      !std::equal(expected.shape().begin(), expected.shape().end(), actual.shape().begin())) {
    return Invalid("Tensor contract mismatch: " + expected.name());
  }
  if (check_type && expected.has_scale() &&
      (expected.scale() != actual.scale() || expected.zero_point() != actual.zero_point())) {
    return Invalid("Tensor quantization mismatch: " + expected.name());
  }
  return {};
}

void SupplyQuantization(TensorSpec& requested, const TensorSpec& actual) {
  if (!requested.has_scale() && actual.has_scale()) {
    requested.set_scale(actual.scale());
    requested.set_zero_point(actual.zero_point());
  }
}

Expected<void> BindManifest(DatasetManifest& manifest, Task task,
                            const Execution* execution) {
  if (manifest.outputs_size() == 0) {
    if (!execution || execution->outputs().size() != 1 ||
        (task != AUTO && task != CLASSIFICATION)) {
      return Invalid("Dense/offline evaluation requires output contracts");
    }
    auto* output = manifest.add_outputs();
    *output->mutable_tensor() = execution->outputs()[0];
    output->mutable_tensor()->set_binding_name(execution->outputs()[0].name());
    output->mutable_tensor()->set_name("output");
    output->mutable_scoring()->set_task(CLASSIFICATION);
  }
  std::set<std::string> outputs;
  for (auto& output : *manifest.mutable_outputs()) {
    if (!outputs.insert(Binding(output.tensor())).second) return Invalid("Duplicate output binding");
    if (task != AUTO) output.mutable_scoring()->set_task(task);
    if (execution) {
      LITERT_ASSIGN_OR_RETURN(auto slot, Find(execution->outputs(), output.tensor()));
      const auto& actual = execution->outputs()[slot];
      LITERT_RETURN_IF_ERROR(Match(output.tensor(), actual));
      SupplyQuantization(*output.mutable_tensor(), actual);
    }
  }
  if (execution && manifest.inputs_size() == 0 && execution->inputs().size() == 1) {
    auto* input = manifest.add_inputs();
    *input = execution->inputs()[0];
    input->set_binding_name(input->name());
    input->set_name("image");
    if (input->shape_size() == 4) input->set_layout(input->shape(1) == 3 ? NCHW : NHWC);
  }
  if (execution && manifest.inputs_size()) {
    if (static_cast<size_t>(manifest.inputs_size()) != execution->inputs().size()) {
      return Invalid("Manifest must describe every model input");
    }
    std::set<std::string> inputs;
    for (auto& input : *manifest.mutable_inputs()) {
      if (!inputs.insert(Binding(input)).second) return Invalid("Duplicate input binding");
      LITERT_ASSIGN_OR_RETURN(auto slot, Find(execution->inputs(), input));
      const auto& actual = execution->inputs()[slot];
      LITERT_RETURN_IF_ERROR(Match(input, actual));
      SupplyQuantization(input, actual);
    }
  }
  return ValidateManifest(manifest, task);
}

Expected<std::vector<FileIdentity>> SampleFiles(
    const Sample& sample, const std::filesystem::path& root) {
  std::vector<FileIdentity> files;
  auto add = [&](const FileIdentity& source) -> Expected<void> {
    LITERT_ASSIGN_OR_RETURN(
        auto identity, Identify(Resolve(root, Utf8ToPath(source.path()))));
    files.push_back(std::move(identity));
    return {};
  };
  if (sample.has_image()) {
    FileIdentity image;
    image.set_path(sample.image());
    LITERT_RETURN_IF_ERROR(add(image));
  }
  for (const auto& input : sample.inputs()) LITERT_RETURN_IF_ERROR(add(input.file()));
  for (const auto& target : sample.targets()) {
    LITERT_RETURN_IF_ERROR(add(target.file()));
    if (target.has_mask()) LITERT_RETURN_IF_ERROR(add(target.mask()));
  }
  return files;
}

Expected<std::vector<std::vector<uint8_t>>> Inputs(
    const EvaluationConfig& config, const DatasetManifest& manifest,
    const Sample& sample, const std::filesystem::path& root,
    const Execution& execution) {
  std::vector<std::vector<uint8_t>> values(execution.inputs().size());
  if (sample.has_image()) {
    if (execution.inputs().size() != 1) return Invalid("Encoded images require one model input");
    const auto& spec = execution.inputs()[0];
    if (spec.shape_size() != 4 || spec.shape(0) != 1 ||
        ((spec.shape(1) == 3) == (spec.shape(3) == 3))) {
      return Invalid("Image input must be unambiguous batch-one NCHW/NHWC RGB");
    }
    LITERT_ASSIGN_OR_RETURN(auto preprocessing, ImageOptions(config.preprocessing()));
    const bool nchw = spec.shape(1) == 3;
    if (spec.shape(nchw ? 2 : 1) != preprocessing.crop_h ||
        spec.shape(nchw ? 3 : 2) != preprocessing.crop_w) {
      return Invalid("Preprocessing crop does not match the model input shape");
    }
    LITERT_ASSIGN_OR_RETURN(
        auto pixels,
        DecodePreprocess(Utf8ToPath(sample.image()), preprocessing, nchw));
    LITERT_ASSIGN_OR_RETURN(values[0], EncodeImage(pixels, spec));
    return values;
  }
  for (const auto& source : sample.inputs()) {
    const TensorSpec* spec = nullptr;
    for (const auto& input : manifest.inputs()) {
      if (input.name() == source.name()) spec = &input;
    }
    if (!spec) return Invalid("Missing input tensor description");
    LITERT_ASSIGN_OR_RETURN(auto index, Find(execution.inputs(), *spec));
    const auto& actual = execution.inputs()[index];
    LITERT_RETURN_IF_ERROR(Match(*spec, actual, false));
    LITERT_ASSIGN_OR_RETURN(
        auto bytes,
        ReadTensor(Resolve(root, Utf8ToPath(source.file().path())), *spec));
    if (spec->type() == FLOAT32) {
      for (size_t i = 0; i < bytes.size(); i += sizeof(float)) {
        float value;
        std::memcpy(&value, bytes.data() + i, sizeof(value));
        if (!std::isfinite(value)) return Invalid("Nonfinite prepared input: " + source.name());
      }
    }
    if (spec->type() != actual.type() ||
        (spec->has_scale() && (spec->scale() != actual.scale() ||
                               spec->zero_point() != actual.zero_point()))) {
      // Both sides receive the same real inputs, with their own IO quantization.
      const size_t width = spec->type() == FLOAT32 ? sizeof(float) : 1;
      std::vector<float> real(bytes.size() / width);
      if (spec->type() != FLOAT32 && !spec->has_scale()) {
        return Invalid("Quantized baseline conversion needs manifest scale/zero_point");
      }
      for (size_t i = 0; i < real.size(); ++i) {
        if (spec->type() == FLOAT32) {
          std::memcpy(&real[i], bytes.data() + i * sizeof(float), sizeof(float));
        } else {
          int code = bytes[i];
          if (spec->type() == INT8 && code >= 128) code -= 256;
          real[i] = (code - spec->zero_point()) * spec->scale();
        }
      }
      LITERT_ASSIGN_OR_RETURN(bytes, EncodeImage(real, actual));
    }
    values[index] = std::move(bytes);
  }
  return values;
}

const TargetFile* Target(const Sample& sample, const std::string& name) {
  for (const auto& target : sample.targets()) {
    if (target.name() == name) return &target;
  }
  return nullptr;
}

Expected<void> ScoreSample(const DatasetManifest& manifest,
                           const Sample& sample,
                           const std::vector<std::vector<float>>& predictions,
                           const std::filesystem::path& root,
                           const std::filesystem::path& checkpoint,
                           bool baseline, SampleResult& row) {
  for (int i = 0; i < manifest.outputs_size(); ++i) {
    const auto& spec = manifest.outputs(i);
    const auto* target = Target(sample, spec.tensor().name());
    if (!target && (spec.scoring().task() != CLASSIFICATION || !sample.has_label())) continue;
    LITERT_ASSIGN_OR_RETURN(auto metrics,
        Score(spec, predictions[i], target, sample.has_label() ? sample.label() : -1,
              root, checkpoint));
    *(baseline ? row.add_baseline_metrics() : row.add_metrics()) = std::move(metrics);

  }
  if (!baseline && row.metrics_size() == 0) return Invalid("Sample has no applicable output targets");
  return {};
}

Expected<std::vector<std::vector<float>>> SelectOutputs(
    const Execution& execution, const DatasetManifest& manifest,
    std::vector<std::vector<float>> all) {
  std::vector<std::vector<float>> selected;
  for (const auto& output : manifest.outputs()) {
    LITERT_ASSIGN_OR_RETURN(auto index, Find(execution.outputs(), output.tensor()));
    LITERT_RETURN_IF_ERROR(Match(output.tensor(), execution.outputs()[index], false));
    selected.push_back(std::move(all[index]));
  }
  return selected;
}

Expected<TensorArtifact> Artifact(const TensorSpec& spec, const void* data,
                                  size_t bytes,
                                  const std::filesystem::path& path = {}) {
  TensorArtifact artifact;
  *artifact.mutable_tensor() = spec;
  if (!path.empty()) {
    LITERT_RETURN_IF_ERROR(WriteBytes(path, data, bytes));
    std::error_code ec;
    const auto absolute = std::filesystem::absolute(path, ec);
    if (ec) {
      return Invalid("Cannot resolve artifact path: " + PathToUtf8(path));
    }
    artifact.mutable_file()->set_path(PathToUtf8(absolute));
  }
  return artifact;
}

Expected<std::vector<std::vector<float>>> Offline(
    const std::filesystem::path& directory, const DatasetManifest& manifest,
    const Sample& sample, google::protobuf::RepeatedPtrField<TensorArtifact>& artifacts) {
  std::vector<std::vector<float>> outputs;
  for (const auto& spec : manifest.outputs()) {
    const auto path =
        directory / Utf8ToPath(sample.id()) / (spec.tensor().name() + ".raw");
    auto tensor = spec.tensor();
    tensor.set_type(FLOAT32);
    tensor.clear_scale();
    tensor.clear_zero_point();
    LITERT_ASSIGN_OR_RETURN(auto values, ReadFloats(path, tensor));
    LITERT_ASSIGN_OR_RETURN(
        auto artifact,
        Artifact(tensor, values.data(), values.size() * sizeof(float)));
    std::error_code ec;
    const auto absolute = std::filesystem::absolute(path, ec);
    if (ec) return Invalid("Cannot resolve offline path: " + PathToUtf8(path));
    artifact.mutable_file()->set_path(PathToUtf8(absolute));
    *artifacts.Add() = std::move(artifact);
    outputs.push_back(std::move(values));
  }
  return outputs;
}

Expected<void> DumpOutputs(const DatasetManifest& manifest,
                           const std::vector<std::vector<float>>& values,
                           const std::filesystem::path& directory, bool dump,
                           google::protobuf::RepeatedPtrField<TensorArtifact>& artifacts) {
  if (!artifacts.empty()) return {};  // Offline files already have identities.
  for (size_t i = 0; i < values.size(); ++i) {
    auto tensor = manifest.outputs(i).tensor();
    tensor.set_type(FLOAT32);
    tensor.clear_scale();
    tensor.clear_zero_point();
    const auto path = dump ? (directory / (tensor.name() + ".raw"))
                           : std::filesystem::path();
    LITERT_ASSIGN_OR_RETURN(auto artifact,
        Artifact(tensor, values[i].data(), values[i].size() * sizeof(float), path));
    *artifacts.Add() = std::move(artifact);
  }
  return {};
}

Expected<void> EvaluateSample(const EvaluationConfig& config,
                              const DatasetManifest& manifest,
                              const Sample& sample,
                              const std::filesystem::path& root,
                              const std::filesystem::path& checkpoint,
                              Execution* candidate, Execution* baseline,
                              int warmups, SampleResult& row) {
  std::vector<std::vector<float>> predictions, references;
  if (candidate) {
    LITERT_ASSIGN_OR_RETURN(auto inputs, Inputs(config, manifest, sample, root, *candidate));
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto path = config.dump_inputs()
                            ? (checkpoint / "inputs" /
                               ("input_" + std::to_string(i) + ".raw"))
                            : std::filesystem::path();
      auto tensor = candidate->inputs()[i];
      for (const auto& contract : manifest.inputs()) {
        if (Binding(contract) == tensor.name()) tensor.set_layout(contract.layout());
      }
      LITERT_ASSIGN_OR_RETURN(auto artifact,
          Artifact(tensor, inputs[i].data(), inputs[i].size(), path));
      *row.add_model_inputs() = std::move(artifact);
    }
    double ms;
    LITERT_ASSIGN_OR_RETURN(auto all, candidate->Invoke(inputs, warmups, ms));
    row.set_inference_ms(ms);
    LITERT_ASSIGN_OR_RETURN(predictions, SelectOutputs(*candidate, manifest, std::move(all)));
    if (baseline) {
      LITERT_ASSIGN_OR_RETURN(auto base_inputs, Inputs(config, manifest, sample, root, *baseline));
      LITERT_ASSIGN_OR_RETURN(auto all_base, baseline->Invoke(base_inputs, warmups, ms));
      row.set_baseline_inference_ms(ms);
      LITERT_ASSIGN_OR_RETURN(references, SelectOutputs(*baseline, manifest, std::move(all_base)));
    }
  } else {
    LITERT_ASSIGN_OR_RETURN(
        predictions, Offline(Utf8ToPath(config.predictions_dir()), manifest,
                             sample, *row.mutable_predictions()));
  }
  if (!config.reference_dir().empty()) {
    LITERT_ASSIGN_OR_RETURN(
        references, Offline(Utf8ToPath(config.reference_dir()), manifest,
                            sample, *row.mutable_baseline_predictions()));
  }
  LITERT_RETURN_IF_ERROR(DumpOutputs(manifest, predictions, checkpoint,
                                    config.dump_predictions(), *row.mutable_predictions()));
  LITERT_RETURN_IF_ERROR(
      DumpOutputs(manifest, references, checkpoint / "baseline",
                  config.dump_predictions(), *row.mutable_baseline_predictions()));
  LITERT_RETURN_IF_ERROR(ScoreSample(manifest, sample, predictions, root, checkpoint, false, row));
  if (references.empty()) return {};
  LITERT_RETURN_IF_ERROR(
      ScoreSample(manifest, sample, references, root, checkpoint / "baseline",
                  true, row));
  for (size_t i = 0; i < predictions.size(); ++i) {
    const auto& spec = manifest.outputs(i).tensor();
    LITERT_ASSIGN_OR_RETURN(auto comparison,
        CompareScores(predictions[i], references[i], config.atol(), config.rtol()));
    auto* result = row.add_comparisons();
    result->set_name(spec.name());
    result->set_elements(predictions[i].size());
    result->set_allclose(comparison.allclose);
    result->set_max_abs(comparison.max_abs);
    result->set_mean_abs(comparison.mean_abs);
    if (config.compare_int8_codes() && spec.type() == INT8) {
      LITERT_ASSIGN_OR_RETURN(auto codes,
          CompareQuantizedScores(predictions[i], references[i],
                                 {spec.scale(), spec.zero_point()},
                                 config.max_code_error(), config.mean_code_error()));
      result->set_allclose(codes.allclose);
      result->set_max_code_error(codes.max_abs);
      result->set_mean_code_error(codes.mean_abs);
    }
    if (manifest.outputs(i).scoring().task() == CLASSIFICATION) {
      const auto a = std::max_element(predictions[i].begin(), predictions[i].end());
      const auto b = std::max_element(references[i].begin(), references[i].end());
      result->set_prediction_agreement(a - predictions[i].begin() == b - references[i].begin());
    }
  }
  return {};
}

Expected<void> Checkpoint(const SampleResult& row, const Sample& sample,
                          const DatasetManifest& manifest, bool has_baseline,
                          const std::vector<FileIdentity>& files) {
  if (!row.complete() || row.version() != 1 || row.id() != sample.id() ||
      row.sources_size() != static_cast<int>(files.size()) || !row.metrics_size()) {
    return Invalid("Checkpoint identity or completion mismatch");
  }
  for (size_t i = 0; i < files.size(); ++i) {
    if (files[i].SerializeAsString() != row.sources(i).SerializeAsString()) {
      return Invalid("Completed sample data changed");
    }
  }
  for (bool base : {false, true}) {
    std::map<std::string, Task> expected;
    for (const auto& output : manifest.outputs()) {
      if (base && !has_baseline) continue;
      if ((output.scoring().task() == CLASSIFICATION && sample.has_label()) ||
          Target(sample, output.tensor().name())) {
        expected[output.tensor().name()] = output.scoring().task();
      }
    }
    const auto& records = base ? row.baseline_metrics() : row.metrics();
    if (records.size() != static_cast<int>(expected.size())) {
      return Invalid("Checkpoint metric count mismatch");
    }
    for (const auto& record : records) {
      const auto it = expected.find(record.name());
      if (it == expected.end()) return Invalid("Checkpoint metric output mismatch");
      const auto task = it->second;
      if ((task == CLASSIFICATION && !record.has_classification()) ||
          (task == SEGMENTATION && !record.has_segmentation()) ||
          (task == DEPTH && !record.has_depth()) ||
          (task == NORMALS && !record.has_normals())) {
        return Invalid("Checkpoint task mismatch");
      }
      if ((record.has_classification() && record.classification().images() != 1) ||
          (record.has_segmentation() && record.segmentation().images() != 1) ||
          (record.has_depth() && record.depth().images() + record.depth().empty_images() != 1) ||
          (record.has_normals() && record.normals().images() + record.normals().empty_images() != 1)) {
        return Invalid("Invalid per-sample observation count");
      }
      expected.erase(it);
    }
    const auto& artifacts = base ? row.baseline_predictions() : row.predictions();
    if (artifacts.size() != (base && !has_baseline ? 0 : manifest.outputs_size())) {
      return Invalid("Checkpoint prediction count mismatch");
    }
    for (int i = 0; i < artifacts.size(); ++i) {
      const auto& tensor = artifacts.Get(i).tensor();
      if (tensor.name() != manifest.outputs(i).tensor().name() || tensor.type() != FLOAT32) {
        return Invalid("Checkpoint tensor contract mismatch");
      }
      LITERT_RETURN_IF_ERROR(Match(tensor, manifest.outputs(i).tensor(), false));
    }
  }
  if (row.comparisons_size() != (has_baseline ? manifest.outputs_size() : 0)) {
    return Invalid("Checkpoint comparison count mismatch");
  }
  std::set<std::string> names;
  for (const auto& comparison : row.comparisons()) {
    if (!names.insert(comparison.name()).second || !comparison.elements() ||
        !std::isfinite(comparison.max_abs()) || !std::isfinite(comparison.mean_abs()) ||
        comparison.max_abs() < 0 || comparison.mean_abs() < 0) {
      return Invalid("Invalid checkpoint comparison");
    }
  }
  if (!std::isfinite(row.inference_ms()) || row.inference_ms() < 0 ||
      !std::isfinite(row.baseline_inference_ms()) || row.baseline_inference_ms() < 0) {
    return Invalid("Invalid checkpoint timing");
  }
  for (const auto* artifacts : {&row.predictions(), &row.baseline_predictions(), &row.model_inputs()}) {
    for (const auto& artifact : *artifacts) {
      LITERT_ASSIGN_OR_RETURN(auto expected_bytes, TensorBytes(artifact.tensor()));
      if (!artifact.file().path().empty()) {
        LITERT_RETURN_IF_ERROR(VerifyFile(artifact.file()));
        std::error_code ec;
        const auto actual_bytes =
            std::filesystem::file_size(Utf8ToPath(artifact.file().path()), ec);
        if (ec || actual_bytes != expected_bytes) {
          return Invalid("Checkpoint tensor file size mismatch: " +
                         artifact.file().path());
        }
      }
    }
  }
  for (const auto* records : {&row.metrics(), &row.baseline_metrics()}) {
    for (const auto& metric : *records) {
      if (metric.has_normal_errors()) LITERT_RETURN_IF_ERROR(VerifyFile(metric.normal_errors()));
    }
  }
  return {};
}

class RunMetrics {
 public:
  Expected<void> Add(const SampleResult& row) {
    for (const auto& metric : row.metrics()) {
      LITERT_RETURN_IF_ERROR(candidate_[metric.name()].Add(metric));
    }
    for (const auto& metric : row.baseline_metrics()) {
      LITERT_RETURN_IF_ERROR(baseline_[metric.name()].Add(metric));
    }
    for (const auto& comparison : row.comparisons()) {
      auto& total = comparisons_[comparison.name()];
      if (!total.has_name()) {
        total.set_name(comparison.name());
        total.set_allclose(true);
      }
      total.set_allclose(total.allclose() && comparison.allclose());
      total.set_max_abs(std::max(total.max_abs(), comparison.max_abs()));
      total.set_mean_abs(total.mean_abs() + comparison.mean_abs() * comparison.elements());
      if (comparison.has_max_code_error()) {
        total.set_max_code_error(std::max(total.max_code_error(), comparison.max_code_error()));
        total.set_mean_code_error(
            total.mean_code_error() +
            comparison.mean_code_error() * comparison.elements());
      }
      total.set_elements(total.elements() + comparison.elements());
      if (comparison.has_prediction_agreement()) {
        total.set_prediction_agreement(
            (!total.has_prediction_agreement() || total.prediction_agreement()) &&
            comparison.prediction_agreement());
      }
    }
    return {};
  }

  Expected<void> Finish(const std::filesystem::path& root,
                        RunSummary& summary) const {
    summary.clear_metrics();
    summary.clear_baseline_metrics();
    summary.clear_comparisons();
    for (const auto* groups : {&candidate_, &baseline_}) {
      for (const auto& group : *groups) {
        const auto suffix = groups == &baseline_ ? "_baseline" : "";
        const auto scratch = root / (group.first + suffix);
        LITERT_ASSIGN_OR_RETURN(auto value,
                               group.second.Result(scratch));
        auto* result = groups == &baseline_ ? summary.add_baseline_metrics()
                                           : summary.add_metrics();
        *result = std::move(value);
      }
    }
    for (const auto& pair : comparisons_) {
      auto value = pair.second;
      if (value.elements()) {
        value.set_mean_abs(value.mean_abs() / value.elements());
        if (value.has_mean_code_error()) {
          value.set_mean_code_error(value.mean_code_error() / value.elements());
        }
      }
      *summary.add_comparisons() = std::move(value);
    }
    return {};
  }

 private:
  std::map<std::string, MetricAccumulator> candidate_, baseline_;
  std::map<std::string, OutputComparison> comparisons_;
};

void ApplyLayouts(ExecutionIdentity& identity, const DatasetManifest& manifest) {
  for (auto& input : *identity.mutable_inputs()) {
    for (const auto& contract : manifest.inputs()) {
      if (input.name() == Binding(contract)) input.set_layout(contract.layout());
    }
  }
  for (auto& output : *identity.mutable_outputs()) {
    for (const auto& contract : manifest.outputs()) {
      if (output.name() == Binding(contract.tensor())) output.set_layout(contract.tensor().layout());
    }
  }
}

void ClearSetupTimes(RunIdentity& identity) {
  if (identity.has_candidate()) identity.mutable_candidate()->clear_setup_ms();
  if (identity.has_baseline()) identity.mutable_baseline()->clear_setup_ms();
}

Expected<void> PublishManifest(const std::filesystem::path& output,
                               const DatasetManifest& manifest,
                               const std::filesystem::path& root) {
  auto resolved = manifest;
  for (auto& sample : *resolved.mutable_samples()) {
    for (auto& input : *sample.mutable_inputs()) {
      input.mutable_file()->set_path(
          PathToUtf8(Resolve(root, Utf8ToPath(input.file().path()))));
    }
    for (auto& target : *sample.mutable_targets()) {
      target.mutable_file()->set_path(
          PathToUtf8(Resolve(root, Utf8ToPath(target.file().path()))));
      if (target.has_mask()) {
        target.mutable_mask()->set_path(
            PathToUtf8(Resolve(root, Utf8ToPath(target.mask().path()))));
      }
    }
  }
  return WriteProto(output / "resolved_manifest.txtpb", resolved, TEXT_PROTO);
}

struct PreparedRun {
  DatasetManifest manifest;
  std::filesystem::path root;
  // Declared before `candidate` and `baseline` so `environment` outlives them.
  std::optional<Environment> environment;
  std::unique_ptr<Execution> candidate;
  std::unique_ptr<Execution> baseline;
};

Expected<Environment> CreateEnvironment(const BackendOptions& backend) {
  std::vector<EnvironmentOptions::Option> options;
  if (!backend.runtime_library_dir().empty()) {
    options.push_back({EnvironmentOptions::Tag::kRuntimeLibraryDir,
                       backend.runtime_library_dir()});
  }
  if (!backend.dispatch_library_dir().empty()) {
    options.push_back({EnvironmentOptions::Tag::kDispatchLibraryDir,
                       backend.dispatch_library_dir()});
  }
  if (!backend.compiler_plugin_library_dir().empty()) {
    options.push_back({EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
                       backend.compiler_plugin_library_dir()});
  }
  return Environment::Create(EnvironmentOptions(options));
}

Expected<PreparedRun> PrepareRun(const EvaluationConfig& config,
                                 const std::filesystem::path& binary_path,
                                 RunIdentity& identity) {
  LITERT_RETURN_IF_ERROR(ValidateBackendOptions(config.backend()));
  PreparedRun prepared;
  LITERT_ASSIGN_OR_RETURN(prepared.manifest, LoadManifest(config));
  *identity.mutable_config() = config;
  *identity.mutable_device() = DeviceInfo();
  LITERT_ASSIGN_OR_RETURN(auto manifest_file,
                          Identify(Utf8ToPath(config.manifest())));
  prepared.root = Utf8ToPath(manifest_file.path()).parent_path();
  *identity.mutable_manifest() = std::move(manifest_file);
  LITERT_ASSIGN_OR_RETURN(auto binary, Identify(binary_path));
  *identity.mutable_binary() = std::move(binary);

  if (!config.score_only()) {
    LITERT_ASSIGN_OR_RETURN(auto env, CreateEnvironment(config.backend()));
    prepared.environment.emplace(std::move(env));
    LITERT_ASSIGN_OR_RETURN(
        prepared.candidate,
        Execution::Create(*prepared.environment, config.backend(),
                          Utf8ToPath(config.model()), config.signature_index(),
                          /*reference_cpu=*/false,
                          *identity.mutable_candidate()));
    if (!config.baseline_model().empty()) {
      const auto base_options = BaselineOptions(config);
      LITERT_ASSIGN_OR_RETURN(
          prepared.baseline,
          Execution::Create(
              *prepared.environment, base_options,
              Utf8ToPath(config.baseline_model()), config.signature_index(),
              config.baseline_accelerator() == "cpu",
              *identity.mutable_baseline()));
    }
  }

  const std::filesystem::path image_root =
      config.image_root().empty() ? prepared.root
                                  : Utf8ToPath(config.image_root());
  for (auto& sample : *prepared.manifest.mutable_samples()) {
    if (sample.has_image()) {
      sample.set_image(
          PathToUtf8(Resolve(image_root, Utf8ToPath(sample.image()))));
    }
  }
  LITERT_RETURN_IF_ERROR(BindManifest(prepared.manifest, config.task(),
                                      prepared.candidate.get()));
  if (prepared.baseline) {
    for (const auto& spec : prepared.manifest.outputs()) {
      LITERT_ASSIGN_OR_RETURN(
          auto slot, Find(prepared.baseline->outputs(), spec.tensor()));
      LITERT_RETURN_IF_ERROR(
          Match(spec.tensor(), prepared.baseline->outputs()[slot], false));
    }
  }
  if (identity.has_candidate()) {
    ApplyLayouts(*identity.mutable_candidate(), prepared.manifest);
  }
  if (identity.has_baseline()) {
    ApplyLayouts(*identity.mutable_baseline(), prepared.manifest);
  }
  LITERT_ASSIGN_OR_RETURN(auto libraries, RuntimeLibraries(config));
  for (auto& file : libraries) *identity.add_libraries() = std::move(file);
  for (const auto& sample : prepared.manifest.samples()) {
    identity.add_sample_ids(sample.id());
  }
  return prepared;
}

Expected<void> VerifyOrSaveIdentity(const std::filesystem::path& output_dir,
                                    const RunIdentity& identity) {
  const auto identity_path = output_dir / "identity.binpb";
  LITERT_ASSIGN_OR_RETURN(auto has_identity, PathExists(identity_path));
  if (!has_identity) {
    return WriteProto(identity_path, identity);
  }
  RunIdentity stable = identity;
  ClearSetupTimes(stable);
  RunIdentity previous;
  LITERT_RETURN_IF_ERROR(ReadProto(identity_path, previous));
  ClearSetupTimes(previous);
  if (previous.SerializeAsString() != stable.SerializeAsString()) {
    return Invalid("Run provenance changed; use another output directory");
  }
  return {};
}

Expected<void> SaveSummary(const EvaluationConfig& config,
                           const std::filesystem::path& output_dir,
                           const RunSummary& summary) {
  const char* filename = config.summary_format() == TEXT_PROTO
                             ? "summary.txtpb"
                             : "summary.binpb";
  return WriteProto(output_dir / filename, summary, config.summary_format());
}

Expected<SampleResult> RunOrResumeSample(
    const EvaluationConfig& config, const PreparedRun& prepared,
    const Sample& sample, const std::filesystem::path& output_dir,
    bool& warmup_done, bool& resumed) {
  resumed = false;
  if (interrupted) {
    return Invalid("Evaluation interrupted; completed samples can be resumed");
  }
  const auto checkpoint = output_dir / "samples" / Utf8ToPath(sample.id());
  const auto completion = checkpoint / "result.binpb";
  LITERT_ASSIGN_OR_RETURN(auto files, SampleFiles(sample, prepared.root));
  SampleResult row;
  LITERT_ASSIGN_OR_RETURN(auto has_completion, PathExists(completion));
  if (has_completion) {
    LITERT_RETURN_IF_ERROR(ReadProto(completion, row));
    const bool has_baseline =
        prepared.baseline != nullptr || !config.reference_dir().empty();
    LITERT_RETURN_IF_ERROR(
        Checkpoint(row, sample, prepared.manifest, has_baseline, files));
    resumed = true;
    return row;
  }
  row.set_id(sample.id());
  for (const auto& file : files) *row.add_sources() = file;
  const int warmups = warmup_done ? 0 : config.warmup_runs();
  LITERT_RETURN_IF_ERROR(EvaluateSample(
      config, prepared.manifest, sample, prepared.root, checkpoint,
      prepared.candidate.get(), prepared.baseline.get(), warmups, row));
  warmup_done = true;
  row.set_complete(true);
  LITERT_RETURN_IF_ERROR(WriteProto(completion, row));
  return row;
}

Expected<void> CheckComparisonGates(const EvaluationConfig& config,
                                    const RunSummary& summary) {
  for (const auto& comparison : summary.comparisons()) {
    if (config.require_output_parity() && !comparison.allclose()) {
      return Invalid("Output parity tolerance exceeded");
    }
    if (config.require_prediction_agreement() &&
        comparison.has_prediction_agreement() &&
        !comparison.prediction_agreement()) {
      return Invalid("Prediction agreement gate failed");
    }
  }
  return {};
}

Expected<void> ExecuteRun(const EvaluationConfig& config,
                          const std::filesystem::path& output_dir,
                          const std::filesystem::path& binary_path,
                          bool& identity_accepted, RunSummary& summary) {
  LITERT_ASSIGN_OR_RETURN(
      auto prepared,
      PrepareRun(config, binary_path, *summary.mutable_identity()));
  LITERT_RETURN_IF_ERROR(VerifyOrSaveIdentity(output_dir, summary.identity()));
  LITERT_RETURN_IF_ERROR(
      PublishManifest(output_dir, prepared.manifest, prepared.root));
  identity_accepted = true;
  summary.set_requested(prepared.manifest.samples_size());
  LITERT_RETURN_IF_ERROR(SaveSummary(config, output_dir, summary));

  RunMetrics metrics;
  bool warmup_done = false;
  for (const auto& sample : prepared.manifest.samples()) {
    bool resumed = false;
    LITERT_ASSIGN_OR_RETURN(
        auto row, RunOrResumeSample(config, prepared, sample, output_dir,
                                    warmup_done, resumed));
    if (resumed) summary.set_resumed(summary.resumed() + 1);
    LITERT_RETURN_IF_ERROR(metrics.Add(row));
    summary.set_inference_ms(summary.inference_ms() + row.inference_ms());
    summary.set_baseline_inference_ms(summary.baseline_inference_ms() +
                                      row.baseline_inference_ms());
    summary.set_completed(summary.completed() + 1);
    LITERT_RETURN_IF_ERROR(SaveSummary(config, output_dir, summary));
    std::cerr << "Completed " << summary.completed() << "/"
              << prepared.manifest.samples_size() << '\n';
  }
  LITERT_RETURN_IF_ERROR(metrics.Finish(output_dir / "scratch", summary));
  return CheckComparisonGates(config, summary);
}
}  // namespace

Expected<void> Evaluate(const EvaluationConfig& config, RunSummary& summary,
                        const std::filesystem::path& binary_path) {
  summary.Clear();
  if (const auto valid = ValidateConfig(config); !valid) {
    summary.set_status(FAILED);
    summary.set_error(valid.Error().Message());
    return valid.Error();
  }
  std::error_code ec;
  const auto output_dir =
      std::filesystem::absolute(Utf8ToPath(config.output_dir()), ec);
  if (ec) {
    return Invalid("Cannot resolve output_dir: " + config.output_dir());
  }
  std::filesystem::create_directories(output_dir, ec);
  if (ec || !std::filesystem::is_directory(output_dir, ec) || ec) {
    return Invalid("Cannot create output_dir: " + PathToUtf8(output_dir));
  }
  RunLock lock;
  LITERT_RETURN_IF_ERROR(lock.Acquire(output_dir / "run.lock"));

  summary.set_status(RUNNING);
  bool identity_accepted = false;
  const auto started = std::chrono::steady_clock::now();
  const auto status =
      ExecuteRun(config, output_dir, binary_path, identity_accepted, summary);
  summary.set_elapsed_seconds(
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count());
  summary.set_status(status ? PASSED : FAILED);
  if (!status) summary.set_error(status.Error().Message());

  // A provenance mismatch must preserve the existing run's summary.
  LITERT_ASSIGN_OR_RETURN(auto has_identity,
                          PathExists(output_dir / "identity.binpb"));
  if (identity_accepted || !has_identity) {
    LITERT_RETURN_IF_ERROR(SaveSummary(config, output_dir, summary));
  }
  if (!status) return status.Error();
  return {};
}

int Main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
      PrintUsage();
      return 0;
    }
  }
  const uint16_t marker = 1;
  uint8_t marker_bytes[sizeof(marker)];
  std::memcpy(marker_bytes, &marker, sizeof(marker_bytes));
  if (marker_bytes[0] != 1) {
    std::cerr << "Raw tensor files require a little-endian host\n";
    return 1;
  }
  auto config = ParseArgs(argc, argv);
  if (!config) {
    std::cerr << config.Error().Message() << '\n';
    PrintUsage();
    return 1;
  }
  std::filesystem::path executable =
      argc > 0 && argv[0] ? Utf8ToPath(argv[0]) : std::filesystem::path();
#if defined(_WIN32)
  std::wstring module_path(32768, L'\0');
  const DWORD length =
      GetModuleFileNameW(nullptr, module_path.data(),
                         static_cast<DWORD>(module_path.size()));
  if (length > 0 && length < module_path.size()) {
    module_path.resize(length);
    executable = std::filesystem::path(std::move(module_path));
  }
#elif defined(__linux__)
  std::error_code ec;
  const auto actual = std::filesystem::canonical("/proc/self/exe", ec);
  if (!ec) executable = actual;
#endif
  interrupted = 0;
  const auto previous_int = std::signal(SIGINT, Interrupt);
  const auto previous_term = std::signal(SIGTERM, Interrupt);
  RunSummary summary;
  const auto status = Evaluate(*config, summary, executable);
  std::signal(SIGINT, previous_int);
  std::signal(SIGTERM, previous_term);
  if (!status) {
    std::cerr << status.Error().Message() << '\n';
    return 1;
  }
  return 0;
}
}  // namespace litert::vision

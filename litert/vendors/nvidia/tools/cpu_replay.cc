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

// cpu_replay: runs recorded inputs of one signature of a model on LiteRT's
// CPU backend with the settings LiteRT-LM gives its CPU executor, and writes
// chosen outputs. It is a CPU reference for steps that another backend ran:
// the NVIDIA dispatch records every step's inputs
// (LITERT_NVIDIA_DISPATCH_DUMP_IO=dir:PATH), and LiteRT-LM's own CPU run of
// a long context does not finish (its session ends after ten minutes, and
// its CPU prefill reads a few tokens a second).
//
// USAGE
//
//   cpu_replay --model=FILE.tflite --manifest=FILE [--signature=decode]
//              [--threads=4] [--weight_cache=FILE] [--outputs=logits]
//
// Each line of the manifest is one run of the signature: an output prefix,
// then NAME=FILE for every input of the signature, FILE holding the tensor's
// packed bytes (the size of the input exactly). Every output named in
// --outputs is written to PREFIX.NAME.raw. The inputs are not kept between
// lines: each line names all of them. The inputs have the model's shapes:
// LiteRT-LM grows a CPU run's caches and mask with the context, and a step of
// such a run gives the same logits, bit for bit, with its caches padded with
// zeros and its mask with false to the model's length (the positions past
// the mask weigh exactly nothing).
//
// The settings are those of LiteRT-LM's CPU backend (SetCpuOptions in
// runtime/executor/litert_compiled_model_executor_utils.cc, the CPU case of
// runtime/executor/llm_executor_settings_utils.cc): they decide which ops
// XNNPACK takes, and so the arithmetic. The weight cache only saves the
// packing of the weights at the next start.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/cc/options/litert_runtime_options.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"

ABSL_FLAG(std::string, model, "", "The TFLite model.");
ABSL_FLAG(std::string, manifest, "",
          "One run per line: PREFIX NAME=FILE for every input.");
ABSL_FLAG(std::string, signature, "decode", "The signature to run.");
ABSL_FLAG(int, threads, 4,
          "CPU threads (LiteRT-LM's --num_cpu_threads, 4 by default).");
ABSL_FLAG(std::string, weight_cache, "", "XNNPACK weight cache file.");
ABSL_FLAG(std::string, outputs, "logits", "Comma-separated outputs to write.");

namespace litert {
namespace {

Expected<std::string> ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    return Error(kLiteRtStatusErrorFileIO, "Cannot open " + path);
  }
  std::string bytes(static_cast<size_t>(in.tellg()), '\0');
  in.seekg(0);
  if (!in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
    return Error(kLiteRtStatusErrorFileIO, "Cannot read " + path);
  }
  return bytes;
}

Expected<void> WriteFile(const std::string& path, const void* data,
                         size_t size) {
  std::ofstream out(path, std::ios::binary);
  out.write(static_cast<const char*>(data), size);
  if (!out) {
    return Error(kLiteRtStatusErrorFileIO, "Cannot write " + path);
  }
  return {};
}

Expected<Options> LiteRtLmCpuOptions() {
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  options.SetHardwareAccelerators(HwAccelerators::kCpu);
  LITERT_ASSIGN_OR_RETURN(auto& cpu, options.GetCpuOptions());
  LITERT_RETURN_IF_ERROR(cpu.SetNumThreads(absl::GetFlag(FLAGS_threads)));
  LITERT_RETURN_IF_ERROR(cpu.SetXNNPackFlags(
      TfLiteXNNPackDelegateOptionsDefault().flags |
      TFLITE_XNNPACK_DELEGATE_FLAG_DYNAMIC_FULLY_CONNECTED |
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS));
  // LiteRT-LM's default (YnnpackMode::kOff).
  LITERT_RETURN_IF_ERROR(cpu.SetEnableYNNPack(false));
  const std::string weight_cache = absl::GetFlag(FLAGS_weight_cache);
  if (!weight_cache.empty()) {
    LITERT_RETURN_IF_ERROR(cpu.SetXNNPackWeightCachePath(weight_cache.c_str()));
  }
  LITERT_ASSIGN_OR_RETURN(auto& runtime, options.GetRuntimeOptions());
  LITERT_RETURN_IF_ERROR(runtime.SetCompressQuantizationZeroPoints(true));
  return options;
}

Expected<void> Replay() {
  LITERT_ASSIGN_OR_RETURN(auto env, Environment::Create({}));
  LITERT_ASSIGN_OR_RETURN(auto options, LiteRtLmCpuOptions());
  LITERT_ASSIGN_OR_RETURN(
      auto model,
      CompiledModel::Create(env, absl::GetFlag(FLAGS_model), options));
  LITERT_ASSIGN_OR_RETURN(
      const size_t signature,
      model.GetSignatureIndex(absl::GetFlag(FLAGS_signature)));
  LITERT_ASSIGN_OR_RETURN(auto input_names,
                          model.GetSignatureInputNames(signature));
  LITERT_ASSIGN_OR_RETURN(auto output_names,
                          model.GetSignatureOutputNames(signature));
  LITERT_ASSIGN_OR_RETURN(auto inputs, model.CreateInputBuffers(signature));
  LITERT_ASSIGN_OR_RETURN(auto outputs, model.CreateOutputBuffers(signature));
  std::vector<std::pair<std::string, size_t>> written;
  for (absl::string_view name :
       absl::StrSplit(absl::GetFlag(FLAGS_outputs), ',', absl::SkipEmpty())) {
    size_t index = 0;
    while (index < output_names.size() && output_names[index] != name) {
      ++index;
    }
    if (index == output_names.size()) {
      return Error(kLiteRtStatusErrorNotFound,
                   absl::StrCat("No output ", name));
    }
    written.emplace_back(std::string(name), index);
  }

  std::ifstream manifest(absl::GetFlag(FLAGS_manifest));
  if (!manifest) {
    return Error(kLiteRtStatusErrorFileIO, "Cannot open the manifest");
  }
  std::string line;
  while (std::getline(manifest, line)) {
    std::vector<std::string> fields =
        absl::StrSplit(line, ' ', absl::SkipEmpty());
    if (fields.empty()) continue;
    const std::string& prefix = fields[0];
    absl::flat_hash_map<std::string, std::string> files;
    for (size_t i = 1; i < fields.size(); ++i) {
      std::pair<std::string, std::string> entry =
          absl::StrSplit(fields[i], absl::MaxSplits('=', 1));
      files[entry.first] = entry.second;
    }
    for (size_t i = 0; i < input_names.size(); ++i) {
      const auto file = files.find(std::string(input_names[i]));
      if (file == files.end()) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     absl::StrCat(prefix, ": no file for ", input_names[i]));
      }
      LITERT_ASSIGN_OR_RETURN(const std::string bytes, ReadFile(file->second));
      LITERT_ASSIGN_OR_RETURN(const size_t size, inputs[i].PackedSize());
      if (bytes.size() != size) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     absl::StrCat(file->second, " holds ", bytes.size(),
                                  " bytes, ", input_names[i], " ", size));
      }
      LITERT_ASSIGN_OR_RETURN(auto lock,
                              TensorBufferScopedLock::Create<uint8_t>(
                                  inputs[i], TensorBuffer::LockMode::kWrite));
      std::memcpy(lock.second, bytes.data(), size);
    }
    const absl::Time start = absl::Now();
    LITERT_RETURN_IF_ERROR(model.Run(signature, inputs, outputs));
    const double seconds = absl::ToDoubleSeconds(absl::Now() - start);
    for (const auto& [name, index] : written) {
      LITERT_ASSIGN_OR_RETURN(const size_t size, outputs[index].PackedSize());
      LITERT_ASSIGN_OR_RETURN(
          auto lock, TensorBufferScopedLock::Create<const uint8_t>(
                         outputs[index], TensorBuffer::LockMode::kRead));
      LITERT_RETURN_IF_ERROR(WriteFile(absl::StrCat(prefix, ".", name, ".raw"),
                                       lock.second, size));
    }
    ABSL_LOG(INFO) << prefix << ": " << seconds << " s";
  }
  return {};
}

}  // namespace
}  // namespace litert

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  if (auto result = litert::Replay(); !result) {
    ABSL_LOG(ERROR) << result.Error().Message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

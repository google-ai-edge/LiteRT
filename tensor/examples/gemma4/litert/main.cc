/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <sys/resource.h>
#include <sys/stat.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/initialize.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "tensor/buffer.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/litert/kv_bank.h"
#include "tensor/examples/gemma4/litert/kv_cache.h"
#include "tensor/examples/gemma4/litert/runner.h"
#include "tensor/utils/macros.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"

ABSL_FLAG(std::string, bundle_dir, "",
          "Verified Gemma4 bundle (host embedding tables).");
ABSL_FLAG(litert::tensor::examples::gemma4::ModelVariant, model_variant,
          litert::tensor::examples::gemma4::ModelVariant::kE2B,
          "Gemma4 architecture: e2b or e4b.");
ABSL_FLAG(std::string, model_path, "", "Exported two-signature .tflite model.");
ABSL_FLAG(std::string, cases_file, "",
          "TSV: case ID, prompt CSV, forced decode CSV or -.");
ABSL_FLAG(std::string, output_dir, "",
          "New directory for timings and optional correctness dumps.");
ABSL_FLAG(std::string, weight_cache_path, "",
          "Packed cache path; defaults to model_path.xnnpack_cache.");
ABSL_FLAG(int, num_threads, 4, "CPU thread count.");
ABSL_FLAG(int, cache_capacity, 8448,
          "Persistent KV capacity, independent of attention width.");
ABSL_FLAG(int, warmup_runs, 1, "Warmup runs per case.");
ABSL_FLAG(int, measured_runs, 3, "Measured runs per case.");
ABSL_FLAG(bool, dump_outputs, false,
          "Dump complete logits and committed KV after each token; do not use "
          "for timing.");
ABSL_FLAG(bool, experimental_compact_int2, false,
          "Read host embeddings from the experimental compact INT2 bundle.");
ABSL_FLAG(bool, experimental_dynamic_prefill, false,
          "Resize prefill to valid query rows; requires the dynamic export.");
ABSL_FLAG(bool, experimental_explicit_dynamic_shapes, false,
          "Use strict input resizing; requires an export with explicit "
          "dynamic shape signatures.");

namespace litert::tensor::examples::gemma4::cpu {
namespace {
double Elapsed(absl::Time at) {
  return absl::ToDoubleMilliseconds(absl::Now() - at);
}
bool PathExists(const std::string& path) {
  struct stat info{};
  return stat(path.c_str(), &info) == 0;
}
absl::Status CreateDirectories(const std::string& path) {
  std::string current;
  if (!path.empty() && path.front() == '/') {
    current = "/";
  }
  for (absl::string_view part : absl::StrSplit(path, '/', absl::SkipEmpty())) {
    if (!current.empty() && current.back() != '/') {
      current.push_back('/');
    }
    absl::StrAppend(&current, part);
    if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
      return absl::InternalError(
          absl::StrCat("mkdir ", current, ": ", std::strerror(errno)));
    }
  }
  return absl::OkStatus();
}
struct Case {
  std::string name;
  std::vector<int32_t> prompt, forced;
};
absl::StatusOr<std::vector<int32_t>> Tokens(const std::string& csv) {
  std::vector<int32_t> result;
  for (absl::string_view field : absl::StrSplit(csv, ',')) {
    int32_t token;
    if (!absl::SimpleAtoi(field, &token) || token < 0 ||
        token >= Config::From(absl::GetFlag(FLAGS_model_variant)).vocab_size) {
      return absl::InvalidArgumentError("Invalid Gemma4 token ID");
    }
    result.push_back(token);
  }
  return result;
}
absl::StatusOr<std::vector<Case>> ReadCases(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    return absl::InvalidArgumentError("Cannot open cases file");
  }
  std::vector<Case> result;
  std::set<std::string> seen;
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> fields = absl::StrSplit(line, '\t');
    if (fields.size() != 3 || fields[0].empty() ||
        fields[0].find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP"
                                    "QRSTUVWXYZ0123456789_-") !=
            std::string::npos ||
        !seen.insert(fields[0]).second) {
      return absl::InvalidArgumentError(
          "Invalid or duplicate case ID / TSV row");
    }
    Case item;
    item.name = fields[0];
    LRT_TENSOR_ASSIGN_OR_RETURN(item.prompt, Tokens(fields[1]));
    if (fields[2] != "-") {
      LRT_TENSOR_ASSIGN_OR_RETURN(item.forced, Tokens(fields[2]));
    }
    if (item.prompt.empty() || item.prompt.size() + item.forced.size() >
                                   absl::GetFlag(FLAGS_cache_capacity)) {
      return absl::InvalidArgumentError(
          "Empty prompt or case exceeds KV capacity");
    }
    result.push_back(std::move(item));
  }
  if (result.empty() || !file.eof()) {
    return absl::InvalidArgumentError("Invalid cases file");
  }
  return result;
}
absl::Status DumpBytes(const std::string& path, const void* data,
                       size_t bytes) {
  std::ofstream file(path, std::ios::binary);
  if (!file.write(static_cast<const char*>(data), bytes)) {
    return absl::InternalError("Could not write " + path);
  }
  return absl::OkStatus();
}
absl::Status Dump(const std::string& path,
                  const LockedBufferSpan<const float>& logits,
                  const ActiveKvBank& bank) {
  LRT_TENSOR_RETURN_IF_ERROR(
      DumpBytes(path, logits.data(), logits.size() * sizeof(float)));
  for (const KvOwnerSpec& spec : bank.specs()) {
    for (int head = 0; head < spec.num_heads; ++head) {
      const std::string suffix =
          spec.num_heads > 1 ? absl::StrCat(".head", head) : "";
      LRT_TENSOR_ASSIGN_OR_RETURN(
          absl::Span<const int8_t> k,
          bank.Keys(spec.owner, 0, bank.length(), head));
      LRT_TENSOR_ASSIGN_OR_RETURN(
          absl::Span<const int8_t> v,
          bank.Values(spec.owner, 0, bank.length(), head));
      LRT_TENSOR_RETURN_IF_ERROR(
          DumpBytes(path + absl::StrCat(".owner", spec.owner, suffix, ".k.i8"),
                    k.data(), k.size()));
      LRT_TENSOR_RETURN_IF_ERROR(
          DumpBytes(path + absl::StrCat(".owner", spec.owner, suffix, ".v.i8"),
                    v.data(), v.size()));
    }
  }
  return absl::OkStatus();
}
absl::Status Main() {
  const std::string dir = absl::GetFlag(FLAGS_output_dir);
  if (dir.empty() || PathExists(dir) ||
      absl::GetFlag(FLAGS_num_threads) <= 0 ||
      absl::GetFlag(FLAGS_warmup_runs) < 0 ||
      absl::GetFlag(FLAGS_measured_runs) <= 0) {
    return absl::InvalidArgumentError(
        "Require new output directory, positive threads/runs, nonnegative "
        "warmup");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<Case> cases,
                              ReadCases(absl::GetFlag(FLAGS_cases_file)));
  LRT_TENSOR_RETURN_IF_ERROR(CreateDirectories(dir));
  LITERT_ASSIGN_OR_RETURN(Environment env, Environment::Create({}));
  LITERT_ASSIGN_OR_RETURN(Options options, Options::Create());
  LITERT_RETURN_IF_ERROR(options.SetHardwareAccelerators(HwAccelerators::kCpu));
  LITERT_ASSIGN_OR_RETURN(CpuOptions & cpu_options, options.GetCpuOptions());
  LITERT_RETURN_IF_ERROR(
      cpu_options.SetNumThreads(absl::GetFlag(FLAGS_num_threads)));
  LITERT_RETURN_IF_ERROR(cpu_options.SetXNNPackFlags(
      TFLITE_XNNPACK_DELEGATE_FLAG_QS8 |
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS));
  LITERT_RETURN_IF_ERROR(
      cpu_options.SetHintFullyDelegatedToSingleDelegate(true));
  std::string cache_path = absl::GetFlag(FLAGS_weight_cache_path);
  if (cache_path.empty()) {
    cache_path = absl::GetFlag(FLAGS_model_path) + ".xnnpack_cache";
  }
  LITERT_RETURN_IF_ERROR(
      cpu_options.SetXNNPackWeightCachePath(cache_path.c_str()));
  absl::Time at = absl::Now();
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::unique_ptr<Runner> runner,
      Runner::Create(env, options, absl::GetFlag(FLAGS_model_path),
                     absl::GetFlag(FLAGS_bundle_dir),
                     absl::GetFlag(FLAGS_cache_capacity), 32,
                     absl::GetFlag(FLAGS_experimental_compact_int2),
                     absl::GetFlag(FLAGS_experimental_dynamic_prefill),
                     absl::GetFlag(FLAGS_experimental_explicit_dynamic_shapes),
                     absl::GetFlag(FLAGS_model_variant)));
  const double setup_ms = Elapsed(at);
  std::ofstream results(dir + "/timings.jsonl");
  results << std::setprecision(12);
  for (const Case& item : cases) {
    for (int rep = -absl::GetFlag(FLAGS_warmup_runs);
         rep < absl::GetFlag(FLAGS_measured_runs); ++rep) {
      runner->Reset();
      at = absl::Now();
      LRT_TENSOR_RETURN_IF_ERROR(runner->Prefill(
          absl::MakeConstSpan(item.prompt).subspan(0, item.prompt.size() - 1)));
      double prefix_ms = Elapsed(at);
      std::vector<int32_t> decode{item.prompt.back()};
      decode.insert(decode.end(), item.forced.begin(), item.forced.end());
      for (size_t step = 0; step < decode.size(); ++step) {
        at = absl::Now();
        LRT_TENSOR_ASSIGN_OR_RETURN(LockedBufferSpan<const float> logits,
                                    runner->Decode(decode[step]));
        double elapsed_ms = Elapsed(at);
        if (logits.size() !=
                Config::From(absl::GetFlag(FLAGS_model_variant)).vocab_size ||
            !std::all_of(logits.data(), logits.data() + logits.size(),
                         [](float v) { return std::isfinite(v); })) {
          return absl::InternalError("Invalid logits");
        }
        const std::ptrdiff_t token =
            std::max_element(logits.data(), logits.data() + logits.size()) -
            logits.data();
        // Include validation and greedy selection when comparing token latency
        // with generation runners. Keep the model-only duration separately.
        const double token_ms = Elapsed(at);
        struct rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        results << "{\"case\":\"" << item.name << "\",\"repetition\":" << rep
                << ",\"step\":" << step << ",\"setup_ms\":" << setup_ms
                << ",\"prefix_ms\":" << prefix_ms
                << ",\"decode_ms\":" << elapsed_ms
                << ",\"token_ms\":" << token_ms
                << ",\"context\":" << runner->bank().length()
                << ",\"argmax\":" << token
                << ",\"peak_rss_kib\":" << usage.ru_maxrss << "}\n";
        if (rep == 0 && absl::GetFlag(FLAGS_dump_outputs)) {
          const std::string suffix =
              step == 0 ? ".prefill.f32"
                        : absl::StrFormat(".decode_%04d.f32", step);
          LRT_TENSOR_RETURN_IF_ERROR(
              Dump(dir + "/" + item.name + suffix, logits, runner->bank()));
        }
      }
      std::cerr << item.name << " run=" << rep << " prefix_ms=" << prefix_ms
                << "\n";
    }
  }
  if (!results) {
    return absl::InternalError("Could not write timings");
  }
  return absl::OkStatus();
}
}  // namespace
}  // namespace litert::tensor::examples::gemma4::cpu
int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  const absl::Status status = litert::tensor::examples::gemma4::cpu::Main();
  if (!status.ok()) {
    std::cerr << status << "\n";
    return 1;
  }
  return 0;
}

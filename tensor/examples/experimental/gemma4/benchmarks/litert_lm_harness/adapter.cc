// Copyright 2026 The ODML Authors.
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

// Raw-token benchmark client for the official LiteRT-LM CPU executor.
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "nlohmann/json.hpp"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_compiled_model_executor.h"
#include "runtime/executor/llm_litert_compiled_model_executor_factory.h"
#include "runtime/executor/magic_number_configs_helper.h"
#include "runtime/executor/speculative_decoding_mode.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using namespace litert::lm;

template <class T>
auto Take(T value) {
  if constexpr (requires { value.ok(); }) {
    if (!value.ok()) throw std::runtime_error(value.status().ToString());
  } else {
    if (!value.HasValue())
      throw std::runtime_error(std::string(value.Error().Message()));
  }
  return std::move(*value);
}

void Check(absl::Status status) {
  if (!status.ok()) throw std::runtime_error(status.ToString());
}

double Ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

long PeakRssKiB() {
  struct rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    throw std::runtime_error("getrusage failed");
  return usage.ru_maxrss;  // Linux/Android, including setup and warmup.
}

void WriteJson(const std::filesystem::path& path, const Json& value) {
  std::ofstream file(path);
  file << value.dump(2) << '\n';
  if (!file) throw std::runtime_error("Failed JSON write");
}

struct Case {
  std::string id;
  std::vector<int32_t> prompt, forced;
};
std::vector<int32_t> Ids(const std::string& s) {
  std::vector<int32_t> v;
  if (s == "-") return v;
  std::stringstream in(s);
  std::string part;
  while (std::getline(in, part, ',')) {
    size_t used = 0;
    long x = std::stol(part, &used);
    if (used != part.size() || x < 0 || x > INT32_MAX)
      throw std::runtime_error("Invalid token id");
    v.push_back(x);
  }
  return v;
}
std::vector<Case> Cases(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Cannot read cases");
  std::vector<Case> out;
  std::set<std::string> names;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto a = line.find('\t'),
         b = line.find('\t', a == std::string::npos ? 0 : a + 1);
    if (a == std::string::npos || b == std::string::npos ||
        line.find('\t', b + 1) != std::string::npos)
      throw std::runtime_error("Expected three TSV columns");
    Case c{line.substr(0, a), Ids(line.substr(a + 1, b - a - 1)),
           Ids(line.substr(b + 1))};
    if (c.id.empty() ||
        c.id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU"
                               "VWXYZ0123456789_-") != std::string::npos ||
        !names.insert(c.id).second || c.prompt.empty())
      throw std::runtime_error("Invalid case name or empty prompt");
    out.push_back(std::move(c));
  }
  if (out.empty()) throw std::runtime_error("No cases");
  return out;
}
ExecutorInputs Inputs(const std::vector<int32_t>& ids) {
  ExecutorInputs in;
  auto buffer = Take(CopyToTensorBuffer<int32_t>(
      absl::MakeConstSpan(ids), {1, static_cast<int>(ids.size())}));
  in.SetTextData(ExecutorTextData(std::move(buffer)));
  return in;
}

int Run(int argc, char** argv) {
  std::map<std::string, std::string> args;
  const std::set<std::string> allowed{
      "model_path",        "cases_file",      "output_dir",     "num_threads",
      "warmup_runs",       "measured_runs",   "max_num_tokens", "cache_dir",
      "prefill_signature", "trace_signatures"};
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help") {
      std::cout << "--model_path FILE --cases_file TSV --output_dir NEW_DIR "
                   "[--num_threads 4 --warmup_runs 1 --measured_runs 3 "
                   "--max_num_tokens 8448 --cache_dir :memory "
                   "--prefill_signature auto --trace_signatures false]\n";
      return 0;
    }
    if (!arg.starts_with("--")) throw std::runtime_error("Expected flag");
    const auto equal = arg.find('=');
    const std::string name =
        arg.substr(2, equal == std::string::npos ? equal : equal - 2);
    if (!allowed.contains(name) || args.contains(name))
      throw std::runtime_error("Unknown or repeated flag: " + name);
    if (equal == std::string::npos) {
      if (++i == argc) throw std::runtime_error("Missing flag value");
      args[name] = argv[i];
    } else {
      args[name] = arg.substr(equal + 1);
    }
  }
  const auto get = [&](const std::string& key, const std::string& fallback) {
    return args.contains(key) ? args.at(key) : fallback;
  };
  const auto integer = [&](const std::string& key, int fallback) {
    const std::string value = get(key, std::to_string(fallback));
    size_t used = 0;
    const int number = std::stoi(value, &used);
    if (used != value.size())
      throw std::runtime_error("Invalid integer: " + key);
    return number;
  };
  const std::string model = get("model_path", "");
  const std::filesystem::path output = get("output_dir", "");
  if (model.empty() || output.empty())
    throw std::runtime_error("Missing required paths");
  const auto cases = Cases(get("cases_file", ""));
  const int threads = integer("num_threads", 4);
  const int warmups = integer("warmup_runs", 1);
  const int runs = integer("measured_runs", 3);
  const int capacity = integer("max_num_tokens", 8448);
  const std::string signature = get("prefill_signature", "auto");
  const std::string trace_text = get("trace_signatures", "false");
  if (trace_text != "true" && trace_text != "false")
    throw std::runtime_error("trace_signatures must be true or false");
  const bool trace = trace_text == "true";
  if (threads < 1 || warmups < 0 || runs < 1 || capacity < 1 ||
      (signature != "auto" && signature != "prefill_128"))
    throw std::runtime_error("Invalid benchmark option");
  for (const auto& c : cases) {
    if (c.prompt.size() + c.forced.size() > static_cast<size_t>(capacity))
      throw std::runtime_error("Schedule exceeds capacity");
  }
  if (!std::filesystem::create_directory(output))
    throw std::runtime_error("Output directory must be new; parent must exist");

  auto start = Clock::now();
  auto file = Take(ScopedFile::Open(model));
  auto loader = Take(LitertLmLoader::Create(std::move(file)));
  auto resources = Take(ModelResourcesLitertLm::Create(std::move(loader)));
  auto assets = Take(ModelAssets::Create(model));
  auto engine = Take(EngineSettings::CreateDefault(assets, Backend::CPU));
  auto& settings = engine.GetMutableMainExecutorSettings();
  settings.SetCacheDir(get("cache_dir", ":memory"));
  settings.SetMaxNumTokens(capacity);
  CpuConfig cpu;
  cpu.number_of_threads = threads;
  cpu.enable_ynnpack = YnnpackMode::kOff;
  settings.SetBackendConfig(cpu);
  engine.SetSpeculativeDecodingConfig({.mode = SpeculativeDecodingMode::kOff});
  const auto* metadata = Take(resources->GetLlmMetadata());
  Check(engine.MaybeUpdateAndValidate(
      nullptr, metadata, "",
      resources->GetTFLiteModelBackendConstraint(
          "tf_lite_prefill_decode"),
      std::nullopt, std::nullopt,
      resources->GetTFLiteModelPreferActivationType(
          "tf_lite_prefill_decode")));
  if (Take(settings.GetBackendConfig<CpuConfig>()).enable_ynnpack !=
          YnnpackMode::kOff ||
      settings.GetEffectiveSpeculativeDecodingMode() !=
          SpeculativeDecodingMode::kOff)
    throw std::runtime_error(
        "Metadata changed the requested CPU/MTP configuration");
  if (signature != "auto")
    settings.SetSelectedSignatures({"decode", signature});

  // Use the same metadata-driven context rewriting as the official engine.
  // Options borrow from helper, which must outlive environment and executor.
  MagicNumberConfigsHelper helper;
  auto options = helper.GetLiteRtEnvOptions(*resources, settings);
  Json rewrites = Json::array();
  if (const auto* configs = helper.magic_number_configs()) {
    for (int64_t i = 0; i < configs->num_configs; ++i) {
      const auto& c = configs->configs[i];
      rewrites.push_back(
          {{"magic_number", c.magic_number},
           {"target_number", c.target_number},
           {"signature_prefix",
            c.signature_prefix ? Json(c.signature_prefix) : Json(nullptr)}});
    }
  }
  auto env =
      Take(litert::Environment::Create(litert::EnvironmentOptions(options)));
  auto executor =
      Take(CreateLlmLiteRtCompiledModelExecutor(settings, env, *resources));
  const double setup_ms = Ms(start);
  if (trace) {
    auto* compiled =
        dynamic_cast<LlmLiteRtCompiledModelExecutorBase*>(executor.get());
    if (!compiled)
      throw std::runtime_error("Executor does not support signature tracing");
    compiled->UpdatePreGraphRunCallback(
        [](absl::string_view name, int step, auto&) {
          std::cerr << "SIGNATURE " << name << " step=" << step << '\n';
        });
  }
  Json manifest{
      {"schema_version", 1},
      {"runner", "litert_lm"},
      {"backend", "cpu_xnnpack"},
      {"speculative_decoding", false},
      {"prefill_signature", signature},
      {"threads", threads},
      {"capacity", capacity},
      {"warmups", warmups},
      {"runs", runs},
      {"timings_valid_for_benchmark", !trace},
      {"cache_policy",
       get("cache_dir", ":memory") == ":memory" ? "memory" : "persistent"},
      {"environment_rewrites", rewrites},
      {"setup_ms", setup_ms},
      {"status", "running"}};
  WriteJson(output / "manifest.json", manifest);
  std::ofstream timings(output / "timings.jsonl");
  const int vocab = Take(executor->GetVocabSize());
  for (const auto& c : cases) {
    for (const auto& ids : {c.prompt, c.forced}) {
      for (int32_t id : ids) {
        if (id >= vocab)
          throw std::runtime_error("Token outside model vocabulary");
      }
    }
    for (int run = -warmups; run < runs; ++run) {
      // Match the Tensor API timer: reset and runtime setup are excluded.
      Check(executor->Reset());
      std::vector<int32_t> history = c.prompt;
      for (size_t step = 0; step <= c.forced.size(); ++step) {
        start = Clock::now();
        auto input = Inputs(
            step == 0 ? c.prompt : std::vector<int32_t>{c.forced[step - 1]});
        if (step == 0) Check(executor->Prefill(input));
        auto logits =
            Take(step == 0 ? executor->DecodeLogits(ExecutorInputs())
                           : executor->DecodeLogits(input));
        auto type = Take(logits.TensorType());
        if (type.ElementType() != litert::ElementType::Float32 ||
            Take(type.Layout().NumElements()) != vocab)
          throw std::runtime_error("Expected one full FP32 vocabulary row");
        auto lock = Take(litert::TensorBufferScopedLock::Create(
            logits, litert::TensorBuffer::LockMode::kRead));
        absl::Span<const float> values(static_cast<const float*>(lock.second),
                                       vocab);
        if (!std::all_of(values.begin(), values.end(),
                         [](float value) { return std::isfinite(value); }))
          throw std::runtime_error("Nonfinite logits");
        const int argmax =
            std::max_element(values.begin(), values.end()) - values.begin();
        const double elapsed = Ms(start);
        if (step) history.push_back(c.forced[step - 1]);
        const auto* processed = Take(executor->GetProcessedTokens());
        if (!processed->GetNextUnprocessedToken().token.empty() ||
            processed->GetTokensUnsafe() != history)
          throw std::runtime_error(
              "Consumed tokens differ from the forced fixture");
        timings << Json({{"case", c.id},
                         {"repetition", run},
                         {"step", step},
                         {"setup_ms", setup_ms},
                         {"prefix_ms", 0.0},
                         {"token_ms", elapsed},
                         {"context", history.size()},
                         {"argmax", argmax},
                         {"peak_rss_kib", PeakRssKiB()},
                         {"consumed_token_ids_verified", true}})
                       .dump()
                << '\n';
      }
    }
  }
  if (!timings) throw std::runtime_error("Failed timings write");
  manifest["status"] = "completed";
  manifest["peak_rss_kib"] = PeakRssKiB();
  WriteJson(output / "manifest.json", manifest);
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    return Run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}

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

// Local diagnostic driver using the unchanged LiteRT-LM CPU executor.
#include "litert/cc/litert_environment.h"
#include "nlohmann/json.hpp"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_litert_compiled_model_executor_factory.h"
#include "runtime/executor/llm_processed_context.h"
#include "runtime/executor/litert/state.h"
#include "runtime/executor/magic_number_configs_helper.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using namespace litert::lm;
template <class T> auto Take(T v) {
  if constexpr (requires { v.ok(); }) {
    if (!v.ok())
      throw std::runtime_error(v.status().ToString());
  } else {
    if (!v.HasValue())
      throw std::runtime_error(std::string(v.Error().Message()));
  }
  return std::move(*v);
}
void Check(absl::Status s) {
  if (!s.ok())
    throw std::runtime_error(s.ToString());
}
long PeakRssKiB() {
  struct rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    throw std::runtime_error("getrusage failed");
  return usage.ru_maxrss;
}
double Ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
struct Case {
  std::string id;
  std::vector<int32_t> prompt, forced;
};
std::vector<int32_t> Ids(const std::string &s) {
  std::vector<int32_t> v;
  if (s == "-")
    return v;
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
std::vector<Case> Cases(const std::string &path) {
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("Cannot read cases");
  std::vector<Case> out;
  std::set<std::string> names;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#')
      continue;
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
  if (out.empty())
    throw std::runtime_error("No cases");
  return out;
}
ExecutorInputs Inputs(const std::vector<int32_t> &ids) {
  ExecutorInputs in;
  auto buffer = Take(CopyToTensorBuffer<int32_t>(
      absl::MakeConstSpan(ids), {1, static_cast<int>(ids.size())}));
  in.SetTextData(ExecutorTextData(std::move(buffer)));
  return in;
}
void WriteFloats(const std::filesystem::path &p,
                 absl::Span<const float> values) {
  std::ofstream f(p, std::ios::binary);
  for (float x : values) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    char bytes[4] = {char(u), char(u >> 8), char(u >> 16), char(u >> 24)};
    f.write(bytes, 4);
  }
  if (!f)
    throw std::runtime_error("Failed float dump: " + p.string());
}
void WriteJson(const std::filesystem::path &p, const Json &j) {
  std::ofstream f(p);
  f << j.dump(2) << '\n';
  if (!f)
    throw std::runtime_error("Failed JSON write");
}
int main(int argc, char **argv) {
  try {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--help") {
        std::cout << "--model_path FILE --cases_file TSV --output_dir DIR "
                     "[--num_threads 2 --warmup_runs 0 --measured_runs 1 "
                     "--max_num_tokens 2048 --reuse_runtimes false "
                     "--dump_logits true --enable_ynnpack true "
                     "--enable_profiling false]\n";
        return 0;
      }
      if (a.rfind("--", 0) != 0)
        throw std::runtime_error("Expected flag");
      auto e = a.find('=');
      if (e == std::string::npos) {
        if (++i == argc)
          throw std::runtime_error("Missing flag value");
        args[a.substr(2)] = argv[i];
      } else
        args[a.substr(2, e - 2)] = a.substr(e + 1);
    }
    auto get = [&](std::string k, std::string d) {
      return args.contains(k) ? args.at(k) : d;
    };
    const std::set<std::string> allowed = {
        "model_path",  "cases_file",      "output_dir",     "num_threads",
        "warmup_runs", "measured_runs",   "max_num_tokens", "reuse_runtimes",
        "dump_logits", "dump_full_logits", "enable_ynnpack",
        "enable_profiling"};
    for (const auto &[k, v] : args)
      if (!allowed.contains(k))
        throw std::runtime_error("Unknown flag " + k);
    auto boolean = [&](std::string k, std::string d) {
      auto s = get(k, d);
      if (s != "true" && s != "false")
        throw std::runtime_error("Boolean flag must be true/false");
      return s == "true";
    };
    const std::string model = get("model_path", "");
    const auto cases = Cases(get("cases_file", ""));
    const std::filesystem::path output = get("output_dir", "");
    if (model.empty() || output.empty())
      throw std::runtime_error("Missing paths");
    const int threads = std::stoi(get("num_threads", "2")),
              warmups = std::stoi(get("warmup_runs", "0")),
              runs = std::stoi(get("measured_runs", "1")),
              capacity = std::stoi(get("max_num_tokens", "2048"));
    const bool reuse = boolean("reuse_runtimes", "false"),
               dump = boolean("dump_full_logits", get("dump_logits", "true")),
               enable_ynnpack = boolean("enable_ynnpack", "true"),
               enable_profiling = boolean("enable_profiling", "false");
    if (threads < 1 || warmups < 0 || runs < 1 || capacity < 1)
      throw std::runtime_error("Invalid numerical option");
    if (std::filesystem::exists(output))
      throw std::runtime_error("Output directory already exists");
    std::filesystem::create_directories(output);
    auto startup = Clock::now();
    auto file = Take(ScopedFile::Open(model));
    auto loader = Take(LitertLmLoader::Create(std::move(file)));
    auto resources = Take(ModelResourcesLitertLm::Create(std::move(loader)));
    auto assets = Take(ModelAssets::Create(model));
    auto engine_settings =
        Take(EngineSettings::CreateDefault(assets, Backend::CPU));
    auto &settings = engine_settings.GetMutableMainExecutorSettings();
    settings.SetCacheDir(":memory");
    settings.SetMaxNumTokens(capacity);
    CpuConfig cpu;
    cpu.number_of_threads = threads;
    cpu.enable_ynnpack = enable_ynnpack;
    settings.SetBackendConfig(cpu);
    settings.SetAdvancedSettings(
        AdvancedSettings{.clear_kv_cache_before_prefill = true,
                         .enable_profiling = enable_profiling});
    const auto *metadata = Take(resources->GetLlmMetadata());
    Check(engine_settings.MaybeUpdateAndValidate(
        /*tokenizer=*/nullptr, metadata, /*input_prompt_as_hint=*/"",
        resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLitePrefillDecode),
        /*vision_backend_constraint=*/std::nullopt,
        /*audio_backend_constraint=*/std::nullopt,
        resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLitePrefillDecode)));
    if (!settings.GetAdvancedSettings()->disable_delegate_clustering)
      throw std::runtime_error(
          "Expected Gemma4 Engine default disabling delegate clustering");
    if (Take(settings.GetBackendConfig<CpuConfig>()).enable_ynnpack !=
            enable_ynnpack ||
        settings.GetAdvancedSettings()->enable_profiling != enable_profiling)
      throw std::runtime_error("Requested CPU backend/profiling options changed");
    std::cerr << "LiteRT-LM CPU request: enable_ynnpack=" << enable_ynnpack
              << " enable_profiling=" << enable_profiling << '\n';
    std::ostringstream effective_settings;
    effective_settings << settings;
    // Same CPU model-shape configuration as Engine's CreateEnvironment.
    // Its environment options point into this helper; declaration order keeps
    // the helper alive until all executors and the environment are destroyed.
    MagicNumberConfigsHelper magic_number_helper;
    auto environment_options =
        magic_number_helper.GetLiteRtEnvOptions(*resources, settings);
    Json magic_number_configs = Json::array();
    bool configured_e2b_context = false;
    if (const auto *configs = magic_number_helper.magic_number_configs()) {
      for (int64_t i = 0; i < configs->num_configs; ++i) {
        const auto &config = configs->configs[i];
        magic_number_configs.push_back(
            {{"magic_number", config.magic_number},
             {"target_number", config.target_number},
             {"signature_prefix", config.signature_prefix
                                      ? Json(config.signature_prefix)
                                      : Json(nullptr)}});
        std::cerr << "Environment magic-number rewrite " << config.magic_number
                  << " -> " << config.target_number << " signature_prefix="
                  << (config.signature_prefix ? config.signature_prefix : "all")
                  << '\n';
        if (config.magic_number == 32003 &&
            config.signature_prefix == nullptr) {
          if (config.target_number != capacity)
            throw std::runtime_error(
                "E2B context rewrite differs from requested capacity");
          configured_e2b_context = true;
        }
      }
    }
    if (!configured_e2b_context)
      throw std::runtime_error(
          "Expected published E2B context magic-number 32003 rewrite");
    auto env = Take(litert::Environment::Create(
        litert::EnvironmentOptions(environment_options)));
    double startup_ms = Ms(startup);
    Json index = {
        {"adapter", "LiteRT-LM CPU LlmExecutor"},
        {"model_path", model},
        {"cases_file", get("cases_file", "")},
        {"num_threads", threads},
        {"enable_ynnpack", enable_ynnpack},
        {"enable_profiling", enable_profiling},
        {"timings_valid_for_benchmark", !enable_profiling},
        {"model_resources_startup_ms", startup_ms},
        {"reuse_runtimes", reuse},
        {"warmup_runs", warmups},
        {"measured_runs", runs},
        {"max_num_tokens", capacity},
        {"environment_magic_number_configs", magic_number_configs},
        {"context_magic_number_configuration_verified", configured_e2b_context},
        {"engine_metadata_settings_applied", true},
        {"effective_executor_settings", effective_settings.str()},
        {"disable_delegate_clustering",
         settings.GetAdvancedSettings()->disable_delegate_clustering},
        {"activation_data_type", "executor/bundle default"},
        {"kv_cache_dtype", "bundle defined"},
        {"timing_scope",
         "Prefill pass includes Prefill plus empty-input DecodeLogits for "
         "pending last prompt token; decode passes include DecodeLogits. Input "
         "tensor construction, locked output access/argmax included; dump and "
         "report I/O excluded."},
        {"warmup_scope",
         reuse ? "Same executor reused with public Reset; warmups warm "
                 "executor and OS/model pages."
               : "Fresh executor each repetition: warmup only warms OS and "
                 "model-resource pages, not the executor runtime."},
        {"runs", Json::array()}};
    std::unique_ptr<LlmExecutor> reused_executor;
    for (const auto &c : cases) {
      if (c.prompt.size() + c.forced.size() > static_cast<size_t>(capacity))
        throw std::runtime_error("Schedule exceeds configured capacity");
      for (int r = -warmups; r < runs; ++r) {
        bool warmup = r < 0;
        auto init = Clock::now();
        std::unique_ptr<LlmExecutor> fresh_executor;
        LlmExecutor *exec = nullptr;
        double reset_ms = 0;
        bool created = false;
        if (reuse) {
          if (!reused_executor) {
            reused_executor = Take(CreateLlmLiteRtCompiledModelExecutor(
                settings, env, *resources));
            created = true;
          } else {
            auto reset_begin = Clock::now();
            Check(reused_executor->Reset());
            reset_ms = Ms(reset_begin);
          }
          exec = reused_executor.get();
        } else {
          fresh_executor = Take(
              CreateLlmLiteRtCompiledModelExecutor(settings, env, *resources));
          exec = fresh_executor.get();
          created = true;
        }
        double executor_ms = created ? Ms(init) : 0;
        if (Take(exec->GetCurrentStep()) != 0)
          throw std::runtime_error(
              "Executor not at initial step after create/reset");
        int vocab = Take(exec->GetVocabSize());
        for (int32_t x : c.prompt)
          if (x >= vocab)
            throw std::runtime_error("Prompt token outside vocabulary");
        for (int32_t x : c.forced)
          if (x >= vocab)
            throw std::runtime_error("Forced token outside vocabulary");
        Json run = {
            {"schema_version", 1},
            {"runner", "litert_lm_cpu"},
            {"case_id", c.id},
            {"prompt_token_ids", c.prompt},
            {"forced_decode_token_ids", c.forced},
            {"run_index", r},
            {"warmup", warmup},
            {"num_threads", threads},
            {"enable_ynnpack", enable_ynnpack},
            {"enable_profiling", enable_profiling},
            {"timings_valid_for_benchmark", !enable_profiling},
            {"vocab_size", vocab},
            {"logits_dtype", "float32-little-endian"},
            {"reuse_runtimes", reuse},
            {"reused_runtimes", reuse},
            {"cold_process_model_load_and_prepare_ms", startup_ms},
            {"runtime_reused", !created},
            {"executor_startup_ms", executor_ms},
            {"runtime_setup_ms", executor_ms},
            {"session_reset_ms", reset_ms},
            {"session_reset_included_in_prefill_elapsed", true},
            {"cache_policy", ":memory"},
            {"clear_kv_cache_before_prefill", true},
            {"max_num_tokens", capacity},
            {"environment_magic_number_configs", magic_number_configs},
            {"context_magic_number_configuration_verified",
             configured_e2b_context},
            {"engine_metadata_settings_applied", true},
            {"effective_executor_settings", effective_settings.str()},
            {"disable_delegate_clustering",
             settings.GetAdvancedSettings()->disable_delegate_clustering},
            {"passes", Json::array()}};
        double inference_work_ms = 0;
        std::vector<int32_t> history = c.prompt;
        for (size_t i = 0; i <= c.forced.size(); ++i) {
          auto outer_begin = Clock::now();
          auto in =
              Inputs(i == 0 ? c.prompt : std::vector<int32_t>{c.forced[i - 1]});
          auto begin = Clock::now();
          if (i == 0)
            Check(exec->Prefill(in));
          auto logits = Take((i == 0 ? exec->DecodeLogits(ExecutorInputs())
                                     : exec->DecodeLogits(in)));
          double forward_ms = Ms(begin);
          auto type = Take(logits.TensorType());
          if (type.ElementType() != litert::ElementType::Float32 ||
              Take(type.Layout().NumElements()) != vocab)
            throw std::runtime_error("Expected one full FP32 vocabulary row");
          auto lock = Take(litert::TensorBufferScopedLock::Create(
              logits, litert::TensorBuffer::LockMode::kRead));
          absl::Span<const float> values(
              static_cast<const float *>(lock.second), vocab);
          int argmax =
              std::max_element(values.begin(), values.end()) - values.begin();
          if (!std::all_of(values.begin(), values.end(),
                           [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("Nonfinite logits");
          double elapsed = Ms(outer_begin) + (i == 0 ? reset_ms : 0);
          inference_work_ms += elapsed;
          if (i)
            history.push_back(c.forced[i - 1]);
          int actual_step = Take(exec->GetCurrentStep());
          if (actual_step != static_cast<int>(history.size() + 1))
            throw std::runtime_error(
                "Unexpected executor current-step convention");
          auto *processed = Take(exec->GetProcessedTokens());
          if (!processed->GetNextUnprocessedToken().token.empty() ||
              processed->GetTokensUnsafe() != history)
            throw std::runtime_error(
                "Executor consumed-token history differs from forced fixture");
          std::ostringstream suffix;
          suffix << (i == 0 ? "prefill" : "decode_");
          if (i)
            suffix << std::setw(4) << std::setfill('0') << i;
          std::string filename = c.id + "." + suffix.str() + ".f32";
          Json file_json = nullptr;
          if (dump && !warmup && r == 0) {
            WriteFloats(output / filename, values);
            file_json = filename;
          }
          run["passes"].push_back(
              {{"kind", i == 0 ? "prefill" : "decode"},
               {"decode_index", i},
               {"input_ids",
                i == 0 ? c.prompt : std::vector<int32_t>{c.forced[i - 1]}},
               {"context_length_after", c.prompt.size() + i},
               {"logits_position", c.prompt.size() + i - 1},
               {"executor_current_step_after", Take(exec->GetCurrentStep())},
               {"argmax_id", argmax},
               {"elapsed_ms", elapsed},
               {"forward_ms", forward_ms},
               {"logits_file", file_json},
               {"consumed_token_ids_verified", true},
               {"logits_dtype", "float32-le"}});
          std::cerr << c.id << " run=" << r << " " << suffix.str()
                    << " token=" << argmax << " ms=" << elapsed << '\n';
        }
        // CloneContext copies the currently bound cache tensors with their
        // actual TensorTypes. Audit once after timed correctness passes only;
        // performance captures perform no cache clone.
        if (dump && !warmup && r == 0 && &c == &cases.front()) {
          auto clone = Take(exec->CloneContext());
          auto &cloned_state =
              static_cast<LlmProcessedContext &>(clone->processed_context())
                  .state();
          auto *litert_state = dynamic_cast<LitertState *>(cloned_state.get());
          if (!litert_state || litert_state->GetNumEntries() != capacity ||
              litert_state->GetBatchSize() != 1)
            throw std::runtime_error("Unexpected cloned LiteRT KV cache state");
          // The diagnostic accessor only reads the deep-copied primary bank.
          // CPU allocation uses one in-place bank; no live state is changed.
          const auto &cloned_buffers =
              litert_state->PrimaryBankForLocalAudit();
          if (cloned_buffers.empty())
            throw std::runtime_error("Bound KV cache audit unexpectedly empty");
          Json bound_cache = Json::object();
          if (cloned_buffers.size() != 30)
            throw std::runtime_error("Expected 30 published E2B KV tensors");
          for (const auto &[name, state_buffer] : cloned_buffers) {
            const auto &buffer = state_buffer.buffer;
            auto type = Take(buffer.TensorType());
            if (type.ElementType() != litert::ElementType::Int8)
              throw std::runtime_error("Bound E2B KV cache is not INT8");
            const auto dims = type.Layout().Dimensions();
            std::vector<int> shape(dims.begin(), dims.end());
            if (std::find(shape.begin(), shape.end(), capacity) ==
                    shape.end() ||
                std::find(shape.begin(), shape.end(), 32003) != shape.end())
              throw std::runtime_error("Actual bound KV cache shape does not "
                                       "use configured capacity");
            bound_cache[std::string(name)] = {
                {"shape", shape},
                {"element_type", static_cast<int>(type.ElementType())},
                {"dtype", type.ElementType() == litert::ElementType::Int8
                              ? "INT8"
                              : "other"},
                {"packed_bytes", Take(buffer.PackedSize())}};
          }
          run["bound_kv_cache_shapes"] = bound_cache;
          run["bound_kv_cache_capacity_verified"] = capacity;
          index["bound_kv_cache_shapes"] = bound_cache;
          index["bound_kv_cache_capacity_verified"] = capacity;
          std::cerr << "Verified actual bound KV cache capacity " << capacity
                    << " across " << cloned_buffers.size() << " tensors\n";
        }
        run["inference_work_ms"] = inference_work_ms;
        run["peak_process_rss_kib"] = PeakRssKiB();
        run["status"] = "completed";
        std::ostringstream rn;
        rn << c.id << (warmup ? ".warmup_" : ".run_") << std::setw(3)
           << std::setfill('0') << (warmup ? r + warmups : r) << ".json";
        std::string runfile = rn.str();
        if (enable_profiling) {
          // Summary collection is outside inference timers. The executor starts
          // profiling at creation; reused executors retain cumulative events.
          std::string profile_file = runfile + ".profile.txt";
          std::ofstream profile(output / profile_file);
          profile << Take(exec->GetProfileSummary());
          if (!profile)
            throw std::runtime_error("Failed profile summary write");
          run["profile_file"] = profile_file;
          run["profile_scope"] = "Cumulative compiled-model events since executor "
                                 "creation; use one run and no warmup for an "
                                 "isolated diagnostic capture.";
        }
        WriteJson(output / runfile, run);
        index["runs"].push_back(runfile);
        WriteJson(output / "manifest.json", index);
      }
    }
    index["status"] = "completed";
    index["completed_case_count"] = cases.size();
    index["cases"] = cases.size();
    index["peak_process_rss_kib"] = PeakRssKiB();
    WriteJson(output / "manifest.json", index);
    WriteJson(output / "run.json", index);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}

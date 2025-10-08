// Copyright 2025 Google LLC.
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

#include "litert/ats/configure.h"

#include <chrono>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <optional>
#include <regex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/ats/common.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

ABSL_FLAG(std::optional<int>, data_seed, std::nullopt,
          "Seed for the buffer data generation.");

ABSL_FLAG(std::vector<std::string>, seeds, std::vector<std::string>({}),
          "Comma-separated test-generator/seed pairings in the form "
          "<generator_name>:<seed>. This seed will be "
          "used to generator the randomized parameters for all invocations of "
          "the respective test-generator.");

ABSL_FLAG(bool, quiet, false, "Minimize logging.");

ABSL_FLAG(std::string, backend, "cpu",
          "Which backend to use as the \"actual\".");

ABSL_FLAG(std::string, dispatch_dir, "",
          "Path to directory containing the dispatch library. Only relevant "
          "for NPU.");

ABSL_FLAG(std::string, plugin_dir, "",
          "Path to directory containing the compiler plugin library. Only "
          "relevant for NPU.");

ABSL_FLAG(
    std::string, dont_register, "^$",
    "Regex for test selection. This is a negative search match, if the pattern "
    "can be found anywhere in the test name, it will be skipped.");

ABSL_FLAG(std::string, do_register, ".*",
          "Regex for test selection. This is a positive search match, if the "
          "pattern can be found anywhere in the test name, it will be run. "
          "This has lower priority over the dont_register filter.");

ABSL_FLAG(
    bool, f16_range_for_f32, false,
    "If true, will generate values f16 values stored as f32 for f32 tensors.");

ABSL_FLAG(std::string, extra_models, "",
          "Optional directory containing models which to add to the test.");

ABSL_FLAG(size_t, iters_per_test, 1, "Number of iterations per test.");

ABSL_FLAG(int64_t, max_ms_per_test, -1,
          "Maximum time in milliseconds to run each test, -1 means no limit "
          "and a default will be provided.");

ABSL_FLAG(bool, fail_on_timeout, true,
          "Whether to fail a test if it times out.");

ABSL_FLAG(std::string, csv, "",
          "If specified, a CSV file will be written to this path containing "
          "the results of the test run.");

ABSL_FLAG(bool, dump_report, true,
          "Whether to dump the report to the user after completion.");

namespace litert::testing {

namespace {

Expected<AtsConf::SeedMap> ParseParamSeedMap() {
  const auto seed_flags = absl::GetFlag(FLAGS_seeds);
  AtsConf::SeedMap seeds;
  for (const auto& seed : seed_flags) {
    std::pair<std::string, std::string> seed_pair = absl::StrSplit(seed, ':');
    int seed_int;
    if (absl::SimpleAtoi(seed_pair.second, &seed_int)) {
      seeds.insert({seed_pair.first, seed_int});
    } else {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   absl::StrFormat("Failed to parse seed %s", seed.c_str()));
    }
  }
  return seeds;
}

Expected<ExecutionBackend> ParseBackend() {
  const auto backend_flag = absl::GetFlag(FLAGS_backend);
  if (backend_flag == "cpu") {
    return ExecutionBackend::kCpu;
  } else if (backend_flag == "gpu") {
    return ExecutionBackend::kGpu;
  } else if (backend_flag == "npu") {
    return ExecutionBackend::kNpu;
  } else {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 absl::StrFormat("Unknown backend: %s", backend_flag.c_str()));
  }
}

void Setup(const AtsConf& options) {
  if (options.Quiet()) {
    LiteRtSetMinLoggerSeverity(LiteRtGetDefaultLogger(), LITERT_SILENT);
  }
}

}  // namespace

Expected<AtsConf> AtsConf::ParseFlagsAndDoSetup() {
  LITERT_ASSIGN_OR_RETURN(auto seeds, ParseParamSeedMap());
  LITERT_ASSIGN_OR_RETURN(auto backend, ParseBackend());
  std::regex neg_re(absl::GetFlag(FLAGS_dont_register),
                    std::regex_constants::ECMAScript);
  std::regex pos_re(absl::GetFlag(FLAGS_do_register),
                    std::regex_constants::ECMAScript);
  auto extra_models = absl::GetFlag(FLAGS_extra_models);
  auto f16_range_for_f32 = absl::GetFlag(FLAGS_f16_range_for_f32);
  auto data_seed = absl::GetFlag(FLAGS_data_seed);
  auto dispatch_dir = absl::GetFlag(FLAGS_dispatch_dir);
  auto plugin_dir = absl::GetFlag(FLAGS_plugin_dir);
  auto quiet = absl::GetFlag(FLAGS_quiet);
  auto iters_per_test = absl::GetFlag(FLAGS_iters_per_test);
  auto max_ms_per_test = absl::GetFlag(FLAGS_max_ms_per_test);
  std::chrono::milliseconds max_ms_per_test_opt(std::chrono::seconds(10));
  if (max_ms_per_test > 0) {
    max_ms_per_test_opt = std::chrono::milliseconds(max_ms_per_test);
  }
  auto fail_on_timeout = absl::GetFlag(FLAGS_fail_on_timeout);
  auto csv = absl::GetFlag(FLAGS_csv);
  auto dump_report = absl::GetFlag(FLAGS_dump_report);
  AtsConf res(std::move(seeds), backend, quiet, dispatch_dir, plugin_dir,
              std::move(neg_re), std::move(pos_re), std::move(extra_models),
              f16_range_for_f32, data_seed, iters_per_test,
              std::move(max_ms_per_test_opt), fail_on_timeout, dump_report,
              std::move(csv));
  Setup(res);
  return res;
}

int AtsConf::GetSeedForParams(absl::string_view name) const {
  static constexpr int kDefaultSeed = 42;
  auto it = seeds_for_params_.find(name);
  if (it == seeds_for_params_.end()) {
    return kDefaultSeed;
  }
  return it->second;
}

bool AtsConf::ShouldRegister(const std::string& name) const {
  return std::regex_search(name, pos_re_) && !std::regex_search(name, neg_re_);
};

bool AtsConf::ShouldRegister(absl::string_view name) const {
  return ShouldRegister(std::string(name));
}

}  // namespace litert::testing

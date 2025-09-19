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
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_rng.h"

ABSL_FLAG(std::optional<int>, data_seed, std::nullopt,
          "Seed for the buffer data generation.");

ABSL_FLAG(std::vector<std::string>, seeds, std::vector<std::string>({}),
          "Comma-separated test-generator/seed pairings in the form "
          "<generator_name>:<seed>. This seed will be "
          "used to generator the randomized parameters for all invocations of "
          "the respective test-generator.");

ABSL_FLAG(bool, quiet, true, "Minimize logging.");

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

ABSL_FLAG(
    bool, f16_range_for_f32, false,
    "If true, will generate values f16 values stored as f32 for f32 tensors.");

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

Expected<AtsConf::ExecutionBackend> ParseBackend() {
  const auto backend_flag = absl::GetFlag(FLAGS_backend);
  if (backend_flag == "cpu") {
    return AtsConf::ExecutionBackend::kCpu;
  } else if (backend_flag == "gpu") {
    return AtsConf::ExecutionBackend::kGpu;
  } else if (backend_flag == "npu") {
    return AtsConf::ExecutionBackend::kNpu;
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
  AtsConf res(
      std::move(seeds), backend, absl::GetFlag(FLAGS_quiet),
      absl::GetFlag(FLAGS_dispatch_dir), absl::GetFlag(FLAGS_plugin_dir),
      std::regex(absl::GetFlag(FLAGS_dont_register),
                 std::regex_constants::ECMAScript),
      absl::GetFlag(FLAGS_f16_range_for_f32), absl::GetFlag(FLAGS_data_seed));
  Setup(res);
  return res;
}

RandomTensorDataBuilder AtsConf::CreateDataBuilder() const {
  RandomTensorDataBuilder builder;
  if (f16_range_for_f32_) {
    builder.SetF16InF32();
  }
  return builder;
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
  return !std::regex_search(name, re_);
};

bool AtsConf::ShouldRegister(absl::string_view name) const {
  return ShouldRegister(std::string(name));
}

}  // namespace litert::testing

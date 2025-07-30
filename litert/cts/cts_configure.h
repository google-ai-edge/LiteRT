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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CTS_CTS_CONFIGURE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CTS_CTS_CONFIGURE_H_

#include <regex>  // NOLINT
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/flags/declare.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"

// Which backend to use as the "actual".
ABSL_DECLARE_FLAG(std::string, backend);

// Comma-separated test-generator/seed pairings in the form
// <generator_name>:<seed>. This seed will be used to generator the randomized
// parameters for all invocations of the respective test-generator.
ABSL_DECLARE_FLAG(std::vector<std::string>, seeds);

// Minimize logging.
ABSL_DECLARE_FLAG(bool, quiet);

// Path to directory containing the dispatch library. Only relevant for NPU.
ABSL_DECLARE_FLAG(std::string, dispatch_dir);

// Path to directory containing the compiler plugin library. Only relevant for
// NPU.
ABSL_DECLARE_FLAG(std::string, plugin_dir);

// Regex to filter tests.
ABSL_DECLARE_FLAG(std::string, dont_register);

namespace litert::testing {

class CtsConf {
 public:
  using SeedMap = absl::flat_hash_map<std::string, int>;

  enum class ExecutionBackend { kCpu, kGpu, kNpu };

  // Parse flags into this class and do any global setup needed which depends
  // on said flags.
  static Expected<CtsConf> ParseFlagsAndDoSetup();

  // Get the user-specified seed for param generation for the test logic with
  // the given name. Default is provided if not specified.
  int GetSeedForParams(absl::string_view name) const;

  // The backend to use as the "actual".
  ExecutionBackend Backend() const { return backend_; }

  // Whether to minimize logging.
  bool Quiet() const { return quiet_; }

  // Given name of a potential test, determines if it should be run based on
  // the filter regex.
  bool ShouldRegister(const std::string& name) const;
  bool ShouldRegister(absl::string_view name) const;

  // Npu libraries.
  const std::string& DispatchDir() const { return dispatch_dir_; }
  const std::string& PluginDir() const { return plugin_dir_; }

  // TODO add printer.

 private:
  explicit CtsConf(SeedMap&& seeds_for_params, ExecutionBackend backend,
                   bool quiet, std::string dispatch_dir, std::string plugin_dir,
                   std::regex&& re)
      : seeds_for_params_(std::move(seeds_for_params)),
        backend_(backend),
        quiet_(quiet),
        dispatch_dir_(std::move(dispatch_dir)),
        plugin_dir_(std::move(plugin_dir)),
        re_(std::move(re)) {}

  SeedMap seeds_for_params_;
  ExecutionBackend backend_;
  bool quiet_;
  std::string dispatch_dir_;
  std::string plugin_dir_;
  std::regex re_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CTS_CTS_CONFIGURE_H_

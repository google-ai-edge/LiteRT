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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_ATS_CONFIGURE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_ATS_CONFIGURE_H_

#include <chrono>  // NOLINT
#include <fstream>
#include <optional>
#include <regex>  // NOLINT
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/flags/declare.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/ats/common.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_options.h"
#include "litert/compiler/plugin/compiler_plugin.h"
#include "litert/core/filesystem.h"
#include "litert/core/model/model_serialize.h"
#include "litert/tools/flags/vendors/mediatek_flags.h"  // IWYU pragma: export
#include "litert/tools/flags/vendors/qualcomm_flags.h"  // IWYU pragma: export

// Seed for the data generation.
ABSL_DECLARE_FLAG(std::optional<int>, data_seed);

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
ABSL_DECLARE_FLAG(std::vector<std::string>, dont_register);

// Regex for explicit inclusions.
ABSL_DECLARE_FLAG(std::vector<std::string>, do_register);

// Optional list of directories, or model files to add to the test.
ABSL_DECLARE_FLAG(std::vector<std::string>, extra_models);

// Number of iterations per test, each one will have different tensor data.
ABSL_DECLARE_FLAG(size_t, iters_per_test);

// Maximum time in milliseconds to run each test.
ABSL_DECLARE_FLAG(int64_t, max_ms_per_test);

// Whether to fail the test if the test times out.
ABSL_DECLARE_FLAG(bool, fail_on_timeout);

// Where to save report CSV.
ABSL_DECLARE_FLAG(std::string, csv);

// Whether to dump the report to the user.
ABSL_DECLARE_FLAG(bool, dump_report);

// Uses the same input generated models, but instead only runs AOT compilation.
ABSL_DECLARE_FLAG(bool, compile_mode);

// Where to put any side effect serialized models from the test.
ABSL_DECLARE_FLAG(std::string, models_out);

// Limit the number of tests registered.
ABSL_DECLARE_FLAG(int32_t, limit);

// The SOC manufacturer to target for compilation. Only relevant for NPU
// compilation.
ABSL_DECLARE_FLAG(std::string, soc_manufacturer);

// The SOC model to target for compilation. Only relevant for NPU
// compilation.
ABSL_DECLARE_FLAG(std::string, soc_model);

namespace litert::testing {

class AtsConf {
 public:
  // How to dump a summary of the run results after completion.
  enum class PrintOpt { kLatency, kAll, kNone };
  using SeedMap = absl::flat_hash_map<std::string, int>;


  // Parse flags into this class and do any global setup needed which depends
  // on said flags.
  static Expected<AtsConf> ParseFlagsAndDoSetup();

  // Get the user-specified seed for param generation for the test logic with
  // the given name. Default is provided if not specified.
  int GetSeedForParams(absl::string_view name) const;

  // The backend to use as the "actual".
  ExecutionBackend Backend() const { return backend_; }
  bool IsNpu() const { return backend_ == ExecutionBackend::kNpu; }
  bool IsGpu() const { return backend_ == ExecutionBackend::kGpu; }
  bool IsCpu() const { return backend_ == ExecutionBackend::kCpu; }

  // Whether to minimize logging.
  bool Quiet() const { return quiet_; }

  // Given name of a potential test, determines if it should be run based on
  // the filter regex.
  bool ShouldRegister(const std::string& name) const;
  bool ShouldRegister(absl::string_view name) const;

  // NPU library directories.
  const std::string& DispatchDir() const { return dispatch_dir_; }
  const std::string& PluginDir() const { return plugin_dir_; }

  // Create the object that encapsulates the tensor data generation configured
  // by the user.
  const RandomTensorDataBuilder& DataBuilder() const { return data_builder_; }

  // Seed for the data generation.
  std::optional<int> DataSeed() const { return data_seed_; }

  // List of models to add to the test.
  std::vector<std::string> ExtraModels() const {
    std::vector<std::string> res;
    for (const auto& model : extra_models_) {
      if (internal::IsDir(model)) {
        auto list = internal::ListDir(model);
        if (!list) {
          continue;
        }
        res.insert(res.end(), list->begin(), list->end());
      } else {
        res.push_back(model);
      }
    }
    return res;
  }

  // Number of iterations per test, each one will have different tensor data.
  size_t ItersPerTest() const { return iters_per_test_; }

  // Maximum time in milliseconds to run each test.
  std::chrono::milliseconds MaxMsPerTest() const { return max_ms_per_test_; }

  // Whether to fail the test if the test times out.
  bool FailOnTimeout() const { return fail_on_timeout_; }

  // Save the results of the test run to a CSV file if the user has requested.
  template <typename T>
  void Csv(const T& capture) const {
    if (csv_.empty() || compile_mode_) {
      return;
    }
    std::ofstream out(csv_);
    capture.Csv(out);
  }

  // Dump the results of the test to user.
  template <typename T>
  void Print(const T& capture) const {
    if (!dump_report_) {
      // Compile capture not implemented yet.
      return;
    }
    capture.Print(std::cerr);
  }

  // Whether to run the AOT compilation flow.
  bool CompileMode() const { return compile_mode_; }

  // Saves a model to the user provided directory.
  Expected<void> SaveModel(const std::string& name,
                           LiteRtModelT&& model) const {
    if (models_out_.empty()) {
      return {};
    }
    std::string file_name = name;
    if (!EndsWith(file_name, ".tflite")) {
      file_name += ".tflite";
    }
    LITERT_RETURN_IF_ERROR(internal::MkDir(models_out_));
    LITERT_ASSIGN_OR_RETURN(auto serialized,
                            internal::SerializeModel(std::move(model)));
    std::ofstream out(internal::Join({models_out_, file_name}));
    out.write(serialized.StrData(), serialized.Size());
    return {};
  }

  // Max tests to register.
  std::optional<uint32_t> Limit() const {
    return limit_ > 0 ? std::make_optional(limit_) : std::nullopt;
  }

  // Whether to stop registering tests because we've hit the limit.
  bool AtLimit(size_t test_id) const { return Limit() && test_id >= limit_; }

  // The compiler plugin to use for compilation. Only will be loaded if
  // CompileMode() is true and user requests NPU.
  std::optional<internal::CompilerPlugin::Ref> Plugin() const {
    if (!plugin_) {
      return std::nullopt;
    }
    return std::ref(const_cast<internal::CompilerPlugin&>(*plugin_));
  }

  // The SOC manufacturer to target for compilation. Only relevant for NPU
  // compilation.
  const std::string& SocManufacturer() const { return soc_manufacturer_; }

  // The SOC model to target for compilation. Only relevant for NPU
  // compilation.
  const std::string& SocModel() const { return soc_model_; }

  // Litert options to use for the target backend.
  const Options& TargetOptions() const { return target_options_; }

  // Litert options to use for the reference backend.
  const Options& ReferenceOptions() const { return reference_options_; }

  AtsConf(const AtsConf&) = delete;
  AtsConf& operator=(const AtsConf&) = delete;
  AtsConf(AtsConf&&) = default;
  AtsConf& operator=(AtsConf&&) = default;

 private:
  explicit AtsConf(SeedMap&& seeds_for_params, ExecutionBackend backend,
                   bool quiet, std::string dispatch_dir, std::string plugin_dir,
                   std::vector<std::regex> neg_re,
                   std::vector<std::regex> pos_re,
                   std::vector<std::string> extra_models,
                   std::optional<int> data_seed, size_t iters_per_test,
                   std::chrono::milliseconds max_ms_per_test,
                   bool fail_on_timeout, bool dump_report, std::string csv,
                   bool compile_mode, std::string models_out, int32_t limit,
                   std::optional<internal::CompilerPlugin> plugin,
                   std::string soc_manufacturer, std::string soc_model,
                   Options&& target_options, Options&& reference_options)
      : seeds_for_params_(std::move(seeds_for_params)),
        backend_(backend),
        quiet_(quiet),
        dispatch_dir_(std::move(dispatch_dir)),
        plugin_dir_(std::move(plugin_dir)),
        neg_re_(std::move(neg_re)),
        pos_re_(std::move(pos_re)),
        extra_models_(std::move(extra_models)),
        data_seed_(data_seed),
        iters_per_test_(iters_per_test),
        max_ms_per_test_(std::move(max_ms_per_test)),
        fail_on_timeout_(fail_on_timeout),
        dump_report_(dump_report),
        csv_(std::move(csv)),
        compile_mode_(compile_mode),
        models_out_(std::move(models_out)),
        limit_(limit),
        plugin_(std::move(plugin)),
        soc_manufacturer_(std::move(soc_manufacturer)),
        soc_model_(std::move(soc_model)),
        target_options_(std::move(target_options)),
        reference_options_(std::move(reference_options)) {
    // For now, we will provide default settings for data generation.
    // More configurability may be introduced later.
    data_builder_.SetSin();
  }

  SeedMap seeds_for_params_;

  ExecutionBackend backend_;
  bool quiet_;
  std::string dispatch_dir_;
  std::string plugin_dir_;
  std::vector<std::regex> neg_re_;
  std::vector<std::regex> pos_re_;
  std::vector<std::string> extra_models_;
  std::optional<int> data_seed_;
  size_t iters_per_test_;
  std::chrono::milliseconds max_ms_per_test_;
  bool fail_on_timeout_;
  bool dump_report_;
  std::string csv_;
  bool compile_mode_;
  std::string models_out_;
  int32_t limit_;
  std::optional<internal::CompilerPlugin> plugin_;
  std::string soc_manufacturer_;
  std::string soc_model_;
  Options target_options_;
  Options reference_options_;

  RandomTensorDataBuilder data_builder_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_ATS_CONFIGURE_H_

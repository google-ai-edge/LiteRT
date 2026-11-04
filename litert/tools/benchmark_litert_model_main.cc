/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdlib>
#include <iostream>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/flags/reflection.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "litert/tools/benchmark_litert_model.h"
#include "tflite/c/c_api_types.h"
#include "tflite/tools/command_line_flags.h"
#include "tflite/tools/logging.h"

namespace litert::benchmark {
namespace {

// Returns the name of a `--name[=value]` or `-name[=value]` argument, or an
// empty string if `arg` is not a flag.
absl::string_view FlagName(absl::string_view arg) {
  if (!absl::ConsumePrefix(&arg, "--") && !absl::ConsumePrefix(&arg, "-")) {
    return "";
  }
  return arg.substr(0, arg.find('='));
}

// Returns true if Abseil flag parsing consumes a flag argument named
// `flag_name`, using the same lookup as Abseil: `name` or, for `noname`, the
// negation of `name`.
bool IsAbseilFlag(absl::string_view flag_name) {
  return absl::FindCommandLineFlag(flag_name) != nullptr ||
         (absl::ConsumePrefix(&flag_name, "no") &&
          absl::FindCommandLineFlag(flag_name) != nullptr);
}

// Returns the arguments of `argv` that are meant for the TFLite benchmark flag
// parser, i.e. everything that Abseil flag parsing did not consume: flags that
// are not Abseil flags (whether or not they are listed in `--undefok`) and
// leftover positional arguments, which the TFLite parser reports as
// unconsumed.
std::vector<char*> GetTfLiteBenchmarkArgs(
    int argc, char** argv, const std::vector<char*>& positional_args) {
  // `positional_args` points into `argv`; its first element is the program
  // name.
  absl::flat_hash_set<const char*> positional(positional_args.begin() + 1,
                                              positional_args.end());
  std::vector<char*> tflite_argv = {argv[0]};
  for (int i = 1; i < argc; ++i) {
    const absl::string_view flag_name = FlagName(argv[i]);
    const bool is_tflite_flag = !flag_name.empty() && !IsAbseilFlag(flag_name);
    if (is_tflite_flag || positional.contains(argv[i])) {
      tflite_argv.push_back(argv[i]);
    }
  }
  return tflite_argv;
}

// Prints the usage of the NPU vendor flags and of the TFLite benchmark flags.
void PrintUsage(const char* program_name, BenchmarkLiteRtModel& benchmark) {
  // Called before the command line is parsed, so that the current values are
  // the effective defaults (library defaults overridden by the
  // benchmark-specific defaults) rather than values passed on the command
  // line.
  std::cout << "\n  NPU Vendor Flags from LiteRT:\n";
  for (const auto& [name, flag] : absl::GetAllFlags()) {
    if (absl::StrContains(flag->Filename(), "flags/vendors/")) {
      std::cout << "    --" << flag->Name() << " (" << flag->Help()
                << "); default: " << flag->CurrentValue() << ";\n";
    }
  }
  std::cout << "\n"
            << tflite::Flags::Usage(program_name, benchmark.GetFlags())
            << std::endl;
}

}  // namespace

int Main(int argc, char** argv) {
  // Benchmark-specific NPU defaults are set before parsing so that flags given
  // on the command line take precedence over them.
  SetBenchmarkNpuFlagDefaults();

  BenchmarkLiteRtModel benchmark;
  // `--help` is handled here rather than by Abseil, which would print only the
  // Abseil usage and exit.
  for (int i = 1; i < argc; ++i) {
    if (absl::string_view(argv[i]) == "--help" ||
        absl::string_view(argv[i]) == "-h") {
      PrintUsage(argv[0], benchmark);
      return EXIT_SUCCESS;
    }
  }

  // Abseil flags (e.g. the NPU vendor flags) are parsed first. Whatever Abseil
  // does not consume is handed to the TFLite benchmark flag parser, so no list
  // of vendor flag prefixes needs to be maintained here.
  std::vector<char*> positional_args;
  std::vector<absl::UnrecognizedFlag> unrecognized_flags;
  absl::ParseAbseilFlagsOnly(argc, argv, positional_args, unrecognized_flags);
  std::vector<char*> new_argv =
      GetTfLiteBenchmarkArgs(argc, argv, positional_args);
  int new_argc = static_cast<int>(new_argv.size());

  TFLITE_LOG(INFO) << "STARTING!";
  if (benchmark.Run(new_argc, new_argv.data()) != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Benchmarking failed.";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
}  // namespace litert::benchmark

#ifndef BENCHMARK_MODEL_NO_MAIN
int main(int argc, char** argv) { return litert::benchmark::Main(argc, argv); }
#endif

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

#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/flags/reflection.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "litert/tools/benchmark_litert_model.h"
#include "tflite/c/c_api_types.h"
#include "tflite/tools/logging.h"

namespace litert::benchmark {

int Main(int argc, char** argv) {
  // Parse Abseil flags first (vendor flags are defined as Abseil flags).
  std::vector<char*> positional_args;
  std::vector<absl::UnrecognizedFlag> unrecognized_flags;
  absl::ParseAbseilFlagsOnly(argc, argv, positional_args, unrecognized_flags);

  // We filter out known vendor flags to avoid unrecognized flag warnings
  // from the TFLite benchmark runner.
  std::vector<char*> new_argv;
  new_argv.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    absl::string_view arg(argv[i]);
    if (arg == "--" || arg == "-") {
      new_argv.push_back(argv[i]);
      continue;
    }
    if (absl::ConsumePrefix(&arg, "--") || absl::ConsumePrefix(&arg, "-")) {
      size_t eq_pos = arg.find('=');
      absl::string_view flag_name =
          (eq_pos == absl::string_view::npos) ? arg : arg.substr(0, eq_pos);
      if (absl::StartsWith(flag_name, "qualcomm_") ||
          absl::StartsWith(flag_name, "mediatek_") ||
          absl::StartsWith(flag_name, "google_tensor_") ||
          absl::StartsWith(flag_name, "intel_openvino_") ||
          absl::StartsWith(flag_name, "samsung_")) {
        continue;
      }
    }
    new_argv.push_back(argv[i]);
  }
  int new_argc = static_cast<int>(new_argv.size());

  bool has_help = false;
  for (int i = 1; i < argc; ++i) {
    if (absl::string_view(argv[i]) == "--help" ||
        absl::string_view(argv[i]) == "-h") {
      has_help = true;
      break;
    }
  }

  if (has_help) {
    std::cout << "\n  NPU Vendor Flags from LiteRT:\n";
    auto flags = absl::GetAllFlags();
    for (const auto& [name, flag] : flags) {
      if (absl::StrContains(flag->Filename(), "flags/vendors/")) {
        std::cout << "    --" << flag->Name() << " (" << flag->Help()
                  << "); default: " << flag->DefaultValue() << ";\n";
      }
    }
    std::cout << std::endl;
  }

  TFLITE_LOG(INFO) << "STARTING!";
  BenchmarkLiteRtModel benchmark;
  if (benchmark.Run(new_argc, new_argv.data()) != kTfLiteOk) {
    // If help was requested, tflite benchmark returns kTfLiteError, which is
    // fine
    if (has_help) {
      return EXIT_SUCCESS;
    }
    TFLITE_LOG(ERROR) << "Benchmarking failed.";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
}  // namespace litert::benchmark

#ifndef BENCHMARK_MODEL_NO_MAIN
int main(int argc, char** argv) { return litert::benchmark::Main(argc, argv); }
#endif

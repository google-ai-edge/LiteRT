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

#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/flags/reflection.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/tools/benchmark_litert_model.h"
#include "tflite/c/c_api_types.h"
#include "tflite/tools/logging.h"

namespace litert::benchmark {

int Main(int argc, char** argv) {
  bool has_help = false;
  for (int i = 1; i < argc; ++i) {
    if (absl::string_view(argv[i]) == "--help" ||
        absl::string_view(argv[i]) == "-h") {
      has_help = true;
      break;
    }
  }

  // Filter argv for absl flags (starting with vendor prefixes)
  std::vector<char*> absl_argv;
  absl_argv.push_back(argv[0]);
  std::vector<char*> remaining_argv;
  remaining_argv.push_back(argv[0]);

  for (int i = 1; i < argc; ++i) {
    absl::string_view arg(argv[i]);
    if (arg.starts_with("--qualcomm_") || arg.starts_with("--mediatek_") ||
        arg.starts_with("--google_tensor_")) {
      absl_argv.push_back(argv[i]);
    } else {
      remaining_argv.push_back(argv[i]);
    }
  }

  if (absl_argv.size() > 1) {
    int absl_argc = absl_argv.size();
    absl::ParseCommandLine(absl_argc, absl_argv.data());
  }

  int remaining_argc = remaining_argv.size();
  char** remaining_argv_ptr = remaining_argv.data();

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
  if (benchmark.Run(remaining_argc, remaining_argv_ptr) != kTfLiteOk) {
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

int main(int argc, char** argv) { return litert::benchmark::Main(argc, argv); }

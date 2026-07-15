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

  // Populate the LiteRT vendor (absl) flags -- e.g. --intel_openvino_configs_map
  // -- from the command line. Without this they keep their defaults and the
  // options parser registry (run during compile) never sees user-provided NPU
  // options.
  //
  // ParseAbseilFlagsOnly (not ParseCommandLine) is used so the TFLite benchmark
  // flags (--graph, --use_npu, ...) are NOT treated as errors: they come back as
  // `unrecognized_flags` and are ignored here. The ORIGINAL argc/argv is then
  // still forwarded to benchmark.Run() below, which runs TFLite's own parser
  // over them. (TFLite only warns about the vendor flags it does not know, which
  // is harmless.)
  std::vector<char*> positional_args;
  std::vector<absl::UnrecognizedFlag> unrecognized_flags;
  absl::ParseAbseilFlagsOnly(argc, argv, positional_args, unrecognized_flags);

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
  if (benchmark.Run(argc, argv) != kTfLiteOk) {
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

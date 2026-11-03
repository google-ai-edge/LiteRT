// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LITERT_TOOLS_VISION_EVAL_RUNNER_H_
#define LITERT_TOOLS_VISION_EVAL_RUNNER_H_

#include <filesystem>  // NOLINT(build/c++17)

#include "litert/cc/litert_expected.h"
#include "litert/tools/vision_eval/evaluation.pb.h"

namespace litert::vision {

// Runs dataset evaluation (or offline prediction scoring) described by
// `config`, recording run provenance (including `binary_path`), resumable
// per-sample checkpoints, aggregated task metrics, and baseline comparisons in
// `summary` and `config.output_dir()`.
Expected<void> Evaluate(const EvaluationConfig& config,
                        const std::filesystem::path& binary_path,
                        RunSummary& summary);

// Parses command-line flags from `argc` and `argv`, installs signal handlers
// for resumable interruption, runs `Evaluate`, and returns process exit code 0
// on success or 1 on error.
int Main(int argc, char** argv);

}  // namespace litert::vision

#endif  // LITERT_TOOLS_VISION_EVAL_RUNNER_H_

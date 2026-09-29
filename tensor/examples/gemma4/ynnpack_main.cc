/* Copyright 2026 Google LLC.

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
#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "tensor/backends/ynnpack/arithmetic.h"
#include "tensor/examples/gemma3/util.h"
#include "tensor/examples/gemma4/gemma4_runtime.h"
#include "tensor/examples/ops/transformer/transformer_ops_ynnpack.h"
#include "tensor/examples/utils/initialization.h"
#include "tensor/examples/utils/perfetto_session.h"
#include "tensor/examples/utils/tensor_mapping.h"
#include "tensor/runners/ynnpack/runner.h"
#include "tensor/utils/macros.h"
#include "perfetto/tracing/track_event.h"  // from @perfetto

ABSL_FLAG(std::string, weights, "",
          "Path to safetensor weights file or directory.");
ABSL_FLAG(std::string, tokenizer, "",
          "Path to SentencePiece tokenizer model file.");
ABSL_FLAG(std::string, prompt, "Write a short poem about coding.",
          "Prompt to run.");
ABSL_FLAG(int, max_tokens, 100, "Maximum number of tokens to generate.");
ABSL_FLAG(int, num_threads, 4, "Number of threads for YNNPack.");
ABSL_FLAG(bool, verbose, false, "Verbose logging.");
ABSL_FLAG(litert::tensor::examples::TokenPrinter::Kind, print,
          litert::tensor::examples::TokenPrinter::Kind::kTokens,
          "Output mode (tokens or progress).");
ABSL_FLAG(std::string, perfetto_output, "",
          "Path to output Perfetto trace file.");
ABSL_FLAG(bool, instruction_tuned, true,
          "Wraps the prompt with turn instruction markers. This is only useful "
          "for instruction tuned models.");

namespace litert::tensor::examples::gemma4 {
namespace {

absl::StatusOr<CompiledRunners<YnnpackRunner>> CompileRunners(
    BuiltGraphs& graphs, int num_threads) {
  TRACE_EVENT(kTensorApiCategory, "CompileRunners");
  LRT_TENSOR_ASSIGN_OR_RETURN(
      YnnpackRunner runner,
      YnnpackRunner::Create(graphs.prefill_outputs.GetAllHandles()));
  LRT_TENSOR_RETURN_IF_ERROR(runner.SetNumThreads(num_threads));

  LRT_TENSOR_ASSIGN_OR_RETURN(
      YnnpackRunner decode_runner,
      YnnpackRunner::Create(graphs.decode_outputs.GetAllHandles()));
  LRT_TENSOR_RETURN_IF_ERROR(decode_runner.SetNumThreads(num_threads));

  return CompiledRunners<YnnpackRunner>{std::move(runner),
                                        std::move(decode_runner)};
}

absl::Status Run() {
  // YNNPACK has no global initialization entry point.
  const std::string& perfetto_out = absl::GetFlag(FLAGS_perfetto_output);
  std::unique_ptr<PerfettoSession> perfetto_session;
  if (!perfetto_out.empty()) {
    LRT_TENSOR_ASSIGN_OR_RETURN(perfetto_session,
                                PerfettoSession::Create(perfetto_out));
  }

  const GenerateOptions options = {
      .weights_path = absl::GetFlag(FLAGS_weights),
      .tokenizer_path = absl::GetFlag(FLAGS_tokenizer),
      .prompt = absl::GetFlag(FLAGS_prompt),
      .max_tokens = absl::GetFlag(FLAGS_max_tokens),
      .instruction_tuned = absl::GetFlag(FLAGS_instruction_tuned),
      .verbose = absl::GetFlag(FLAGS_verbose),
      .print = absl::GetFlag(FLAGS_print),
  };

  const int num_threads = absl::GetFlag(FLAGS_num_threads);
  return Generate<YnnpackMixinTag, YnnpackRunner>(
      options, [](LazyTensorMapping&) {},
      [&](BuiltGraphs& graphs) { return CompileRunners(graphs, num_threads); });
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4

int main(int argc, char** argv) {
  litert::tensor::Initialize("gemma4", argc, argv, true);

  absl::Status status = litert::tensor::examples::gemma4::Run();

  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run Gemma4 model: " << status;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

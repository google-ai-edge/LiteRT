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

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/examples/gemma3/util.h"
#include "tensor/examples/gemma4/gemma4_runtime.h"
#include "tensor/examples/ops/transformer/transformer_ops_xnnpack.h"
#include "tensor/examples/utils/initialization.h"
#include "tensor/examples/utils/perfetto_session.h"
#include "tensor/examples/utils/tensor_mapping.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "perfetto/tracing/track_event.h"  // from @perfetto
#include "tflite/delegates/xnnpack/weight_cache.h"

namespace {
constexpr absl::string_view kAutoWeightCacheFlag = ":auto";
}

ABSL_FLAG(std::string, weights, "",
          "Path to safetensor weights file or directory.");
ABSL_FLAG(std::string, tokenizer, "",
          "Path to SentencePiece tokenizer model file.");
ABSL_FLAG(std::string, prompt, "Write a short poem about coding.",
          "Prompt to run.");
ABSL_FLAG(int, max_tokens, 100, "Maximum number of tokens to generate.");
ABSL_FLAG(int, num_threads, 4, "Number of threads for XNNPack.");
ABSL_FLAG(bool, verbose, false, "Verbose logging.");
ABSL_FLAG(litert::tensor::examples::TokenPrinter::Kind, print,
          litert::tensor::examples::TokenPrinter::Kind::kTokens,
          "Output mode (tokens or progress).");
ABSL_FLAG(std::string, weight_cache, std::string(kAutoWeightCacheFlag),
          "Path to XNNPack weight cache file.");
ABSL_FLAG(std::string, perfetto_output, "",
          "Path to output Perfetto trace file.");
ABSL_FLAG(bool, instruction_tuned, true,
          "Wraps the prompt with turn instruction markers. This is only useful "
          "for instruction tuned models.");

namespace litert::tensor::examples::gemma4 {
namespace {

// Weight mapping hooks that register the weights with the XNNPack weight cache,
// if any.
class XnnpackWeightHooks final : public TensorMappingHooks {
 public:
  // Constructor.
  //
  // - `weight_cache`: if not null, the weights are registered with it. Must
  //   outlive the hooks.
  explicit XnnpackWeightHooks(
      tflite::xnnpack::MMapWeightCacheProvider* weight_cache)
      : weight_cache_(weight_cache) {}

  absl::Status OnLoaded(absl::string_view model_name,
                        TensorHandle& weight) override {
    if (weight_cache_ == nullptr) {
      return absl::OkStatus();
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(Buffer & buffer, weight.GetBuffer());
    auto locked = buffer.Lock();
    const uint64_t identifier =
        static_cast<uint64_t>(std::hash<absl::string_view>{}(model_name));
    if (!weight_cache_->MapBufferIdentifier(locked.data(), locked.size(),
                                            identifier)) {
      return absl::InternalError(
          absl::StrCat("Failed to map weight identifier for ", model_name));
    }
    return absl::OkStatus();
  }

 private:
  tflite::xnnpack::MMapWeightCacheProvider* weight_cache_;
};

// Creates the prefill and decode runners.
//
// - `weight_cache`: if not null, the runners use it and it is built if needed.
absl::StatusOr<CompiledRunners<XnnpackRunner>> CompileRunners(
    BuiltGraphs& graphs, int num_threads,
    tflite::xnnpack::MMapWeightCacheProvider* weight_cache) {
  TRACE_EVENT(kTensorApiCategory, "CompileRunners");
  LRT_TENSOR_ASSIGN_OR_RETURN(
      XnnpackRunner runner,
      XnnpackRunner::Create(graphs.prefill_outputs.GetAllHandles()));
  runner.SetNumThreads(num_threads);

  LRT_TENSOR_ASSIGN_OR_RETURN(
      XnnpackRunner decode_runner,
      XnnpackRunner::Create(graphs.decode_outputs.GetAllHandles()));
  decode_runner.SetNumThreads(num_threads);

  if (weight_cache != nullptr) {
    runner.SetWeightsCache(&weight_cache->GetCacheProvider());
    decode_runner.SetWeightsCache(&weight_cache->GetCacheProvider());

    if (weight_cache->CanStartBuildStep()) {
      ABSL_LOG(INFO) << "Building cache.";
      if (!weight_cache->StartBuildStep()) {
        return absl::InternalError(
            "Failed to start build step for XNNPack weight cache.");
      }
      const absl::Time start = absl::Now();
      LRT_TENSOR_RETURN_IF_ERROR(runner.PrepareRuntime());
      const absl::Time prefill_done = absl::Now();
      LRT_TENSOR_RETURN_IF_ERROR(decode_runner.PrepareRuntime());
      const absl::Time decode_done = absl::Now();
      ABSL_LOG(INFO) << "Prepared XNNPACK runtimes: prefill="
                     << (prefill_done - start)
                     << " decode=" << (decode_done - prefill_done);
      if (!weight_cache->StopBuildStep()) {
        ABSL_LOG(ERROR)
            << "Failed to stop build step for XNNPack weight cache.";
      }
      weight_cache->StopBuild();
    }
  }

  return CompiledRunners<XnnpackRunner>{std::move(runner),
                                        std::move(decode_runner)};
}

absl::Status Run() {
  const std::string& perfetto_out = absl::GetFlag(FLAGS_perfetto_output);
  std::unique_ptr<PerfettoSession> perfetto_session;
  if (!perfetto_out.empty()) {
    LRT_TENSOR_ASSIGN_OR_RETURN(perfetto_session,
                                PerfettoSession::Create(perfetto_out));
  }

  if (xnn_initialize(/*allocator=*/nullptr) != xnn_status_success) {
    return absl::InternalError("Failed to initialize XNNPACK");
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

  std::string weight_cache_path = absl::GetFlag(FLAGS_weight_cache);
  if (weight_cache_path == kAutoWeightCacheFlag) {
    weight_cache_path = absl::StrCat(options.weights_path, ".cache");
  }
  tflite::xnnpack::MMapWeightCacheProvider weight_cache_provider;
  tflite::xnnpack::MMapWeightCacheProvider* weight_cache = nullptr;
  if (!weight_cache_path.empty()) {
    TRACE_EVENT(kTensorApiCategory, "MapWeightCache");
    if (!weight_cache_provider.LoadOrStartBuild(weight_cache_path.c_str())) {
      return absl::InternalError(absl::StrCat(
          "Failed to load or start build for XNNPack weight cache file: ",
          weight_cache_path));
    }
    weight_cache = &weight_cache_provider;
  }

  const int num_threads = absl::GetFlag(FLAGS_num_threads);
  return Generate<XnnpackMixinTag, XnnpackRunner>(
      options,
      [&](LazyTensorMapping& weights) {
        weights.Register<XnnpackWeightHooks>(weight_cache);
      },
      [&](BuiltGraphs& graphs) {
        return CompileRunners(graphs, num_threads, weight_cache);
      });
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

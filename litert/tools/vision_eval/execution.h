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

#ifndef LITERT_TOOLS_VISION_EVAL_EXECUTION_H_
#define LITERT_TOOLS_VISION_EVAL_EXECUTION_H_

#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/container/btree_set.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_options.h"
#include "litert/tools/vision_eval/data.h"

namespace litert::vision {

// Validates accelerator selection, thread count, GPU backend/precision/IR
// settings, and Arm Vulkan ML JIT prerequisites in `backend`.
Expected<void> ValidateBackendOptions(const BackendOptions& backend);

// Derives `BackendOptions` for the reference baseline execution from `config`
// (resetting vendor/GPU/NPU overrides when `baseline_accelerator` is `"cpu"`).
BackendOptions BaselineOptions(const EvaluationConfig& config);

// Translates non-empty runtime, dispatch, and compiler plugin library
// directories in `backend` into LiteRT `EnvironmentOptions`.
EnvironmentOptions BuildEnvironmentOptions(const BackendOptions& backend);

// Selects whether CPU execution uses default/delegate kernels or unaccelerated
// TFLite reference kernels.
enum class CpuKernelPreference {
  kOptimized,
  kReference,
};

// Validates `backend` and builds LiteRT compilation `Options` for the requested
// hardware accelerators and `cpu_kernels` mode.
Expected<Options> CreateOptions(
    const BackendOptions& backend,
    CpuKernelPreference cpu_kernels = CpuKernelPreference::kOptimized);

namespace internal {

// Extracts the CPU hardware description (`model name` or `Hardware`) from
// `/proc/cpuinfo`-formatted `content`.
DeviceIdentity ParseCpuInfoHardware(absl::string_view content);

// Collects UTF-8 paths of existing regular `.so` files from
// `/proc/self/maps`-formatted `maps` text in deterministic sorted order.
absl::btree_set<std::string> ParseProcMapsLibraries(absl::string_view maps);

}  // namespace internal

// Queries host operating system, architecture, and hardware/device metadata for
// run provenance reporting.
DeviceIdentity DeviceInfo();

// Collects `FileIdentity` records for loaded `.so` libraries and shared
// libraries in the runtime, dispatch, and compiler plugin directories of
// `config`.
Expected<std::vector<FileIdentity>> RuntimeLibraries(
    const EvaluationConfig& config);

// Packs finite preprocessed `input` floats into raw tensor bytes matching
// `spec` (`FLOAT32`, or per-tensor quantized `INT8`/`UINT8`).
Expected<std::vector<uint8_t>> EncodeImage(absl::Span<const float> input,
                                           const TensorSpec& spec);

// Dequantized outputs and measured execution latency from a single timed model
// invocation.
struct InvocationResult {
  std::vector<std::vector<float>> outputs;
  double latency_ms = 0.0;
};

// Owns one compiled model and reuses its input and output tensor buffers across
// samples. The enclosing `Environment` must outlive this object.
class Execution {
 public:
  // Compiles the `.tflite` model at `path` for `signature_index` using
  // `environment`, `backend`, and `cpu_kernels`, allocates reusable signature
  // I/O buffers, and records execution provenance accessible via `identity()`.
  static Expected<std::unique_ptr<Execution>> Create(
      Environment& environment, const BackendOptions& backend,
      const std::filesystem::path& path, int signature_index,
      CpuKernelPreference cpu_kernels);

  // Writes raw `inputs` into the bound input buffers, executes `warmups`
  // untimed runs followed by one timed run, and returns finite dequantized
  // `float` outputs paired with execution latency.
  Expected<InvocationResult> Invoke(
      absl::Span<const std::vector<uint8_t>> inputs, int warmups);

  // Returns tensor specifications for the active signature's inputs.
  absl::Span<const TensorSpec> inputs() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return input_specs_;
  }

  // Returns tensor specifications for the active signature's outputs.
  absl::Span<const TensorSpec> outputs() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return output_specs_;
  }

  // Returns the model, backend, delegation, and I/O provenance for this run.
  const ExecutionIdentity& identity() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return identity_;
  }

 private:
  Execution(std::vector<uint8_t> model_bytes, CompiledModel model,
            int signature)
      : model_bytes_(std::move(model_bytes)),
        model_(std::move(model)),
        signature_(signature) {}

  // `model_bytes_` outlives `model_`, and I/O buffers are destroyed before the
  // model that owns their requirements.
  std::vector<uint8_t> model_bytes_;
  CompiledModel model_;
  std::vector<TensorBuffer> inputs_;
  std::vector<TensorBuffer> outputs_;
  std::vector<TensorSpec> input_specs_;
  std::vector<TensorSpec> output_specs_;
  ExecutionIdentity identity_;
  int signature_;
};
}  // namespace litert::vision

#endif  // LITERT_TOOLS_VISION_EVAL_EXECUTION_H_

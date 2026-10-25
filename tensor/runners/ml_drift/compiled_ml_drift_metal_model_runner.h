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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_COMPILED_ML_DRIFT_METAL_MODEL_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_COMPILED_ML_DRIFT_METAL_MODEL_RUNNER_H_

// Plain C++ facade over `CompiledMlDriftModelRunner<MlDriftMetalBackend>`.
//
// The Metal backend types need Objective-C++, so C++ translation units (e.g.
// tests shared with the OpenCL runner) use this class instead. It mirrors the
// `CompiledMlDriftModelRunner` methods used to compile a Tensor API graph,
// feed host inputs, run and read host outputs.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "tensor/tensor.h"

namespace litert::tensor {

class CompiledMlDriftMetalModelRunner {
 public:
  // Like `CompiledMlDriftModelRunner`, aborts if no Metal device is available.
  CompiledMlDriftMetalModelRunner(std::vector<TensorHandle> flat_inputs,
                                  std::vector<TensorHandle> flat_outputs,
                                  ml_drift::CalculationsPrecision precision =
                                      ml_drift::CalculationsPrecision::kF32);
  ~CompiledMlDriftMetalModelRunner();

  CompiledMlDriftMetalModelRunner(const CompiledMlDriftMetalModelRunner&) =
      delete;
  CompiledMlDriftMetalModelRunner& operator=(
      const CompiledMlDriftMetalModelRunner&) = delete;

  absl::Status BuildModel();
  absl::Status SetInput(const std::string& name, TensorHandle& arg);
  absl::Status Enqueue();
  absl::Status Wait();
  absl::Status Run();
  // Outputs in logical BHWC order.
  absl::StatusOr<std::vector<float>> GetFloatOutput(const std::string& name);
  absl::StatusOr<std::vector<int32_t>> GetInt32Output(const std::string& name);
  absl::Duration compilation_time() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_COMPILED_ML_DRIFT_METAL_MODEL_RUNNER_H_

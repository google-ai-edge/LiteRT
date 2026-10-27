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

#include "third_party/odml/litert/tensor/runners/ml_drift/compiled_ml_drift_metal_model_runner.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "third_party/odml/litert/tensor/runners/ml_drift/compiled_ml_drift_model_runner.h"
#include "third_party/odml/litert/tensor/runners/ml_drift/ml_drift_metal_model_runner.h"
#include "third_party/odml/litert/tensor/tensor.h"

namespace litert::tensor {

class CompiledMlDriftMetalModelRunner::Impl
    : public CompiledMlDriftModelRunner<MlDriftMetalBackend> {
 public:
  using CompiledMlDriftModelRunner<MlDriftMetalBackend>::
      CompiledMlDriftModelRunner;
};

CompiledMlDriftMetalModelRunner::CompiledMlDriftMetalModelRunner(
    std::vector<TensorHandle> flat_inputs,
    std::vector<TensorHandle> flat_outputs,
    ml_drift::CalculationsPrecision precision) {
  @autoreleasepool {
    impl_ = std::make_unique<Impl>(std::move(flat_inputs),
                                   std::move(flat_outputs), precision);
  }
}

CompiledMlDriftMetalModelRunner::~CompiledMlDriftMetalModelRunner() {
  @autoreleasepool {
    impl_.reset();
  }
}

absl::Status CompiledMlDriftMetalModelRunner::BuildModel() {
  @autoreleasepool {
    return impl_->BuildModel();
  }
}

absl::Status CompiledMlDriftMetalModelRunner::SetInput(const std::string& name,
                                                       TensorHandle& arg) {
  @autoreleasepool {
    return impl_->SetInput(name, arg);
  }
}

absl::Status CompiledMlDriftMetalModelRunner::Enqueue() {
  @autoreleasepool {
    return impl_->Enqueue();
  }
}

absl::Status CompiledMlDriftMetalModelRunner::Wait() {
  @autoreleasepool {
    return impl_->Wait();
  }
}

absl::Status CompiledMlDriftMetalModelRunner::Run() {
  @autoreleasepool {
    return impl_->Run();
  }
}

absl::StatusOr<std::vector<float>>
CompiledMlDriftMetalModelRunner::GetFloatOutput(const std::string& name) {
  @autoreleasepool {
    return impl_->GetFloatOutput(name);
  }
}

absl::StatusOr<std::vector<int32_t>>
CompiledMlDriftMetalModelRunner::GetInt32Output(const std::string& name) {
  @autoreleasepool {
    return impl_->GetInt32Output(name);
  }
}

absl::Duration CompiledMlDriftMetalModelRunner::compilation_time() const {
  return impl_->compilation_time();
}

}  // namespace litert::tensor

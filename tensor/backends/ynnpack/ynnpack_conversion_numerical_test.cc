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

#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/testing/numerical_test_bridge.h"
#include "tensor/backends/testing/numerical_test_suite.h"
#include "tensor/backends/ynnpack/arithmetic.h"
#include "tensor/runners/ynnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {
namespace {

class YnnpackTestBackendBridge : public TestBackendBridge {
 public:
  ~YnnpackTestBackendBridge() override = default;

  absl::Status Initialize() override {
    // YNNPACK has no global initialization entry point.
    return absl::OkStatus();
  }

  absl::Status BuildGraph(absl::Span<const TensorHandle> inputs,
                          absl::Span<const TensorHandle> outputs) override {
    std::vector<TensorHandle> output_handles(outputs.begin(), outputs.end());
    auto runner_or = YnnpackRunner::Create(output_handles);
    if (!runner_or.ok()) {
      absl::Status s = runner_or.status();
      // Operations without a YNNPACK lowering are reported as invalid
      // arguments by the conversion.
      if (s.code() == absl::StatusCode::kInvalidArgument &&
          absl::StrContains(s.message(), "does not implement")) {
        return absl::UnimplementedError(s.message());
      }
      return s;
    }
    runner_ = std::make_unique<YnnpackRunner>(std::move(*runner_or));
    return absl::OkStatus();
  }

  absl::Status SetInput(const TensorHandle& tensor,
                        absl::Span<const std::byte> data) override {
    if (runner_ == nullptr) {
      return absl::FailedPreconditionError("Runner is not initialized");
    }
    return runner_->SetInput(tensor, data);
  }

  absl::Status Execute() override {
    if (runner_ == nullptr) {
      return absl::FailedPreconditionError("Runner is not initialized");
    }
    return runner_->Run();
  }

  absl::Status GetOutput(const TensorHandle& tensor,
                         absl::Span<std::byte> data) override {
    if (runner_ == nullptr) {
      return absl::FailedPreconditionError("Runner is not initialized");
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(auto output_bytes, runner_->ReadOutput(tensor));
    if (output_bytes.size() != data.size()) {
      return absl::InvalidArgumentError("Output buffer size mismatch");
    }
    std::memcpy(data.data(), output_bytes.data(), data.size());
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<YnnpackRunner> runner_;
};

struct YnnpackTraits {
  using Tag = YnnpackMixinTag;
  static std::unique_ptr<TestBackendBridge> CreateBridge() {
    return std::make_unique<YnnpackTestBackendBridge>();
  }
};

INSTANTIATE_TYPED_TEST_SUITE_P(Ynnpack, NumericalTestSuite, YnnpackTraits);

}  // namespace
}  // namespace litert::tensor

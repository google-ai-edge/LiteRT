// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ODML_LITERT_LITERT_RUNTIME_COMPILED_MODEL_DISPATCH_H_
#define ODML_LITERT_LITERT_RUNTIME_COMPILED_MODEL_DISPATCH_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_scheduling_info.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/cc/litert_expected.h"
#include "litert/runtime/metrics.h"

// Selected only by
// --//third_party/odml/litert/litert/build_common:build_include=qualcomm_aot.
// The model and environment must outlive this object, just as for the normal
// compiled model. Each signature is a single, fixed-shape dispatch operation.
class LiteRtCompiledModelT {
 public:
  using Ptr = std::unique_ptr<LiteRtCompiledModelT>;
  ~LiteRtCompiledModelT();

  static litert::Expected<Ptr> Create(LiteRtEnvironment env, LiteRtModel model,
                                      LiteRtOptions options);
  litert::Expected<const LiteRtTensorBufferRequirementsT*>
  GetInputBufferRequirements(size_t signature_index, size_t input_index);
  litert::Expected<LiteRtTensorBufferRequirements>
  GetOutputBufferRequirementsCApi(size_t signature_index, size_t output_index);
  litert::Expected<LiteRtLayout> GetInputTensorLayout(size_t signature_index,
                                                      size_t input_index);
  litert::Expected<void> GetOutputTensorShapes(
      size_t signature_index, absl::Span<LiteRtLayout>& layouts,
      bool update_allocation = false);
  litert::Expected<void> RunCApi(
      size_t signature_index, size_t num_inputs,
      const LiteRtTensorBuffer* inputs, size_t num_outputs,
      const LiteRtTensorBuffer* outputs, bool* async,
      LiteRtOptions options = nullptr,
      const LiteRtSchedulingInfo* scheduling_info = nullptr);
  litert::Expected<void> RunCApi(size_t signature_index, size_t num_inputs,
                                 const LiteRtTensorBuffer* inputs,
                                 size_t num_outputs,
                                 const LiteRtTensorBuffer* outputs, bool* async,
                                 const LiteRtSchedulingInfo* scheduling_info) {
    return RunCApi(signature_index, num_inputs, inputs, num_outputs, outputs,
                   async, nullptr, scheduling_info);
  }

  litert::Expected<LiteRtEnvironment> GetEnvironment() { return env_; }
  size_t GetNumSignatures() const { return signatures_.size(); }
  bool HasNonDelegatedOps() const { return false; }
  bool IsNonCpuFullyDelegated() const { return true; }

  litert::Expected<void> SetSchedulingInfo(const LiteRtSchedulingInfo*) {
    return Unsupported();
  }
  litert::Expected<void> StartMetricsCollection(int) const {
    return Unsupported();
  }
  litert::Expected<LiteRtMetricsT> StopMetricsCollection() const {
    return Unsupported();
  }
  litert::Expected<LiteRtProfiler> GetProfiler() { return Unsupported(); }
  litert::Expected<void> ResizeInputTensor(size_t, size_t,
                                           absl::Span<const int>) {
    return Unsupported();
  }
  litert::Expected<void> ResizeInputTensorNonStrict(size_t, size_t,
                                                    absl::Span<const int>) {
    return Unsupported();
  }
  void ReportError(const char* format, ...) const;
  litert::Expected<void> ClearErrors() const { return Unsupported(); }
  litert::Expected<std::string> GetErrorMessages() const {
    return Unsupported();
  }

 private:
  struct Signature;
  explicit LiteRtCompiledModelT(LiteRtEnvironment env);
  static litert::Unexpected Unsupported() {
    return litert::Unexpected(kLiteRtStatusErrorUnsupported,
                              "Unavailable in the Qualcomm AOT runtime");
  }
  litert::Expected<Signature*> GetSignature(size_t index);

  LiteRtEnvironment env_;
  std::vector<std::unique_ptr<Signature>> signatures_;
};

#endif  // ODML_LITERT_LITERT_RUNTIME_COMPILED_MODEL_DISPATCH_H_

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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_VERIFIER_H_
#define ODML_LITERT_LITERT_CORE_MODEL_VERIFIER_H_

#include <cstddef>

#include "litert/c/litert_layout.h"
#include "tflite/core/api/error_reporter.h"
#include "tflite/core/api/verifier.h"
#include "tflite/stderr_reporter.h"

namespace litert {

// Options for LiteRT model verification.
struct VerifyOptions {
  // If true, requires all tensors in the model to have a known rank
  // (rejects unranked tensors).
  bool require_all_tensor_shape_ranks_known = true;

  // Runtime-configurable maximum allowed tensor rank for any tensor in the
  // model (both in the serialized FlatBuffer and after shape inference).
  // Applications that do not need rank-8 tensors can set this lower (e.g., 4 or
  // 5) to reject higher-rank tensors. Must not exceed LITERT_TENSOR_MAX_RANK.
  size_t max_rank = LITERT_TENSOR_MAX_RANK;

  // If true, runs LiteRT ShapeInferenceEngine per-op shape and type validation
  // after FlatBuffer verification.
  bool run_shape_inference = true;

  // If true, requires all ops with outputs in the model to have a registered
  // shape inferrer in ShapeInferenceEngine.
  bool require_supported_op_shape_inferrers = false;
};

// Verifies that the serialized model buffer is a valid TFLite FlatBuffer,
// satisfies tensor rank/shape constraints in `options`, can be unpacked into a
// LiteRT model, and passes LiteRT's per-op shape/type inference checks.
bool Verify(const void* buf, size_t len,
            tflite::ErrorReporter* reporter = tflite::DefaultErrorReporter(),
            const VerifyOptions& options = {});

// Convenience overload accepting `options` before `reporter`.
inline bool Verify(
    const void* buf, size_t len, const VerifyOptions& options,
    tflite::ErrorReporter* reporter = tflite::DefaultErrorReporter()) {
  return Verify(buf, len, reporter, options);
}

// TfLiteVerifier adapter for use with
// tflite::FlatBufferModel::VerifyAndBuildFromBuffer.
class LiteRtVerifier : public tflite::TfLiteVerifier {
 public:
  LiteRtVerifier() = default;
  explicit LiteRtVerifier(VerifyOptions options) : options_(options) {}

  bool Verify(const char* data, int length,
              tflite::ErrorReporter* reporter) override;

  const VerifyOptions& options() const { return options_; }
  void set_options(VerifyOptions options) { options_ = options; }

 private:
  VerifyOptions options_;
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CORE_MODEL_VERIFIER_H_

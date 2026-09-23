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

#include "litert/core/model/verifier.h"

#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/core/model/shape_inference.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/core/api/error_reporter.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/stderr_reporter.h"
#include "tflite/tools/verifier.h"

namespace litert {
namespace {

bool ValidateFlatBufferTensors(const tflite::Model* tfl_model,
                               const VerifyOptions& options,
                               tflite::ErrorReporter* reporter) {
  if (tfl_model == nullptr) {
    if (reporter) {
      reporter->Report("Model pointer is null.");
    }
    return false;
  }
  if (tfl_model->subgraphs() == nullptr) {
    return true;
  }

  for (const tflite::SubGraph* subgraph : *tfl_model->subgraphs()) {
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
      continue;
    }
    for (const tflite::Tensor* tensor : *subgraph->tensors()) {
      if (tensor == nullptr) {
        if (reporter) {
          reporter->Report("Subgraph contains null tensor entry.");
        }
        return false;
      }

      const bool has_known_rank =
          (tensor->shape() != nullptr) || tensor->has_rank();
      if (options.require_all_tensor_shape_ranks_known && !has_known_rank) {
        if (reporter) {
          reporter->Report(
              "Tensor has unknown rank and "
              "require_all_tensor_shape_ranks_known is true.");
        }
        return false;
      }

      if (tensor->shape() != nullptr) {
        const size_t rank = tensor->shape()->size();
        if (rank > options.max_rank || rank > LITERT_TENSOR_MAX_RANK) {
          if (reporter) {
            reporter->Report("Tensor rank (%zu) exceeds max_rank (%zu).", rank,
                             options.max_rank);
          }
          return false;
        }
        for (int32_t dim : *tensor->shape()) {
          if (dim < -1) {
            if (reporter) {
              reporter->Report("Tensor has invalid negative dimension (%d).",
                               dim);
            }
            return false;
          }
        }
      }

      if (tensor->shape_signature() != nullptr) {
        const size_t sig_rank = tensor->shape_signature()->size();
        if (sig_rank > options.max_rank || sig_rank > LITERT_TENSOR_MAX_RANK) {
          if (reporter) {
            reporter->Report(
                "Tensor shape_signature rank (%zu) exceeds max_rank (%zu).",
                sig_rank, options.max_rank);
          }
          return false;
        }
        if (tensor->shape() != nullptr && sig_rank != tensor->shape()->size()) {
          if (reporter) {
            reporter->Report(
                "Tensor shape_signature rank (%zu) does not match shape rank "
                "(%zu).",
                sig_rank, tensor->shape()->size());
          }
          return false;
        }
        for (int32_t dim : *tensor->shape_signature()) {
          if (dim < -1) {
            if (reporter) {
              reporter->Report(
                  "Tensor shape_signature has invalid negative dimension (%d).",
                  dim);
            }
            return false;
          }
        }
      }
    }
  }
  return true;
}

bool ValidateLiteRtTensor(const LiteRtTensorT& tensor,
                          const VerifyOptions& options,
                          tflite::ErrorReporter* reporter) {
  if (tensor.Type().first != kLiteRtRankedTensorType) {
    if (options.require_all_tensor_shape_ranks_known) {
      if (reporter) {
        reporter->Report("LiteRT tensor rank is not known.");
      }
      return false;
    }
    return true;
  }

  const auto& layout = tensor.Type().second.ranked_tensor_type.layout;
  if (layout.rank > options.max_rank || layout.rank > LITERT_TENSOR_MAX_RANK) {
    if (reporter) {
      reporter->Report("LiteRT tensor rank (%u) exceeds max_rank (%zu).",
                       layout.rank, options.max_rank);
    }
    return false;
  }

  for (unsigned int d = 0; d < layout.rank; ++d) {
    if (layout.dimensions[d] < -1) {
      if (reporter) {
        reporter->Report("LiteRT tensor has invalid dimension (%d).",
                         layout.dimensions[d]);
      }
      return false;
    }
  }
  return true;
}

}  // namespace

bool Verify(const void* buf, size_t len, tflite::ErrorReporter* reporter,
            const VerifyOptions& options) {
  tflite::ErrorReporter* effective_reporter =
      reporter ? reporter : tflite::DefaultErrorReporter();

  if (buf == nullptr || len == 0) {
    effective_reporter->Report("Model buffer is null or empty.");
    return false;
  }

  if (options.max_rank > LITERT_TENSOR_MAX_RANK) {
    effective_reporter->Report(
        "Configured max_rank (%zu) exceeds LITERT_TENSOR_MAX_RANK (%d).",
        options.max_rank, LITERT_TENSOR_MAX_RANK);
    return false;
  }

  // 1. Verify the TFLite FlatBuffer integrity and buffer bounds.
  if (!::tflite::Verify(buf, len, effective_reporter)) {
    return false;
  }

  // 2. Validate tensor ranks and dimensions in the FlatBuffer.
  const tflite::Model* tfl_model = tflite::GetModel(buf);
  if (!ValidateFlatBufferTensors(tfl_model, options, effective_reporter)) {
    return false;
  }

  // 3. Load into LiteRT's internal model representation.
  BufferRef<uint8_t> model_buf(reinterpret_cast<const uint8_t*>(buf), len);
  auto model = internal::LoadModelFromBuffer(model_buf);
  if (!model) {
    effective_reporter->Report("Failed to unpack LiteRT model (status %d).",
                               static_cast<int>(model.Error().Status()));
    return false;
  }

  // 4. Validate initial unpacked LiteRT tensors and run per-op shape/type
  // inference across all subgraphs.
  for (auto* subgraph : (*model)->Subgraphs()) {
    if (subgraph == nullptr) {
      continue;
    }
    for (const auto* tensor : subgraph->Tensors()) {
      if (tensor == nullptr ||
          !ValidateLiteRtTensor(*tensor, options, effective_reporter)) {
        return false;
      }
    }

    if (!options.run_shape_inference) {
      continue;
    }

    internal::ShapeInferenceEngine engine((*model).get());
    for (auto* op : subgraph->Ops()) {
      if (op == nullptr) {
        continue;
      }
      const LiteRtStatus status =
          engine.InferOpShapes(op, /*validation_only=*/false);
      if (status == kLiteRtStatusErrorUnsupportedOpShapeInferer) {
        if (options.require_supported_op_shape_inferrers &&
            op->NumOutputs() > 0) {
          effective_reporter->Report(
              "Unsupported shape inferrer for op code %d.",
              static_cast<int>(op->OpCode()));
          return false;
        }
        // Mark non-constant output dimensions as dynamic (-1) so downstream ops
        // do not fail on placeholder shapes.
        for (auto* out : op->Outputs()) {
          if (out != nullptr && out->Weights().Buffer().Size() == 0 &&
              out->Type().first == kLiteRtRankedTensorType) {
            const auto& existing = out->Type().second.ranked_tensor_type;
            internal::Dims dynamic_dims(existing.layout.rank, -1);
            out->SetType(MakeRankedTensorType(existing.element_type,
                                              absl::MakeSpan(dynamic_dims)));
          }
        }
        continue;
      }
      if (status == kLiteRtStatusErrorUnsupported) {
        // Runtime-dynamic shape/padding/axis tensor; mark non-constant output
        // dimensions as dynamic (-1) and continue.
        for (auto* out : op->Outputs()) {
          if (out != nullptr && out->Weights().Buffer().Size() == 0 &&
              out->Type().first == kLiteRtRankedTensorType) {
            const auto& existing = out->Type().second.ranked_tensor_type;
            internal::Dims dynamic_dims(existing.layout.rank, -1);
            out->SetType(MakeRankedTensorType(existing.element_type,
                                              absl::MakeSpan(dynamic_dims)));
          }
        }
        continue;
      }
      if (status != kLiteRtStatusOk) {
        // Check if this is a RESHAPE op with a runtime-dynamic 1D shape tensor
        // (no constant buffer and no ReshapeOptions new_shape).
        const auto* reshape_opts =
            internal::GetTflOptions(*op).AsReshapeOptions();
        if (op->OpCode() == kLiteRtOpCodeTflReshape &&
            (!reshape_opts || reshape_opts->new_shape.empty()) &&
            op->NumInputs() >= 2 && op->Inputs()[1] != nullptr &&
            op->Input(1).Weights().Buffer().Size() == 0 &&
            op->Input(1).Type().first == kLiteRtRankedTensorType) {
          const auto& shape_layout =
              op->Input(1).Type().second.ranked_tensor_type.layout;
          if (shape_layout.rank == 1 && shape_layout.dimensions[0] >= 0 &&
              static_cast<size_t>(shape_layout.dimensions[0]) <=
                  options.max_rank) {
            for (auto* out : op->Outputs()) {
              if (out != nullptr && out->Weights().Buffer().Size() == 0 &&
                  out->Type().first == kLiteRtRankedTensorType) {
                const auto& existing = out->Type().second.ranked_tensor_type;
                internal::Dims dynamic_dims(shape_layout.dimensions[0], -1);
                out->SetType(MakeRankedTensorType(
                    existing.element_type, absl::MakeSpan(dynamic_dims)));
              }
            }
            continue;
          }
        }

        effective_reporter->Report(
            "LiteRT per-op shape/type validation failed for op %d (status %d).",
            static_cast<int>(op->OpCode()), static_cast<int>(status));
        return false;
      }

      for (const auto* out : op->Outputs()) {
        if (out == nullptr ||
            !ValidateLiteRtTensor(*out, options, effective_reporter)) {
          return false;
        }
      }
    }
  }

  return true;
}

bool LiteRtVerifier::Verify(const char* data, int length,
                            tflite::ErrorReporter* reporter) {
  if (length < 0) {
    return false;
  }
  return ::litert::Verify(data, static_cast<size_t>(length), reporter,
                          options_);
}

}  // namespace litert

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

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
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

namespace internal {

absl::Status ValidateFlatBufferTensorQuantization(
    const tflite::SubGraph& subgraph, const tflite::Tensor& tensor) {
  const tflite::QuantizationParameters* quant = tensor.quantization();
  if (quant == nullptr) {
    return absl::OkStatus();
  }

  switch (quant->details_type()) {
    case tflite::QuantizationDetails_BlockwiseQuantization: {
      const auto* bw = quant->details_as_BlockwiseQuantization();
      if (bw == nullptr) {
        return absl::InvalidArgumentError(
            "Blockwise quantization details are missing.");
      }
      const size_t num_tensors =
          subgraph.tensors() ? subgraph.tensors()->size() : 0;
      if (bw->scales() < 0 ||
          static_cast<size_t>(bw->scales()) >= num_tensors ||
          subgraph.tensors()->Get(bw->scales()) == nullptr) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Blockwise quantization has invalid scales tensor index (%d).",
            bw->scales()));
      }
      if (bw->zero_points() < -1 ||
          (bw->zero_points() >= 0 &&
           (static_cast<size_t>(bw->zero_points()) >= num_tensors ||
            subgraph.tensors()->Get(bw->zero_points()) == nullptr))) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Blockwise quantization has invalid zero_points tensor index "
            "(%d).",
            bw->zero_points()));
      }
      if (bw->block_shape() != nullptr && !bw->block_shape()->empty()) {
        if (tensor.shape() != nullptr && !tensor.shape()->empty() &&
            bw->block_shape()->size() != tensor.shape()->size()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Blockwise quantization block_shape rank (%u) does not match "
              "tensor rank (%u).",
              bw->block_shape()->size(), tensor.shape()->size()));
        }
        for (int32_t dim : *bw->block_shape()) {
          if (dim <= 0) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Blockwise quantization block_shape entry (%d) must be "
                "positive.",
                dim));
          }
        }
      } else {
        const int32_t block_size = bw->block_size();
        if (block_size <= 0) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Blockwise quantization block_size (%d) must be positive.",
              block_size));
        }
        if ((tensor.type() == tflite::TensorType_INT4 ||
             tensor.type() == tflite::TensorType_UINT4) &&
            block_size % 2 != 0) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Blockwise quantization block_size (%d) must be divisible by 2 "
              "for 4-bit tensors.",
              block_size));
        }
        if (tensor.type() == tflite::TensorType_INT2 && block_size % 4 != 0) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Blockwise quantization block_size (%d) must be divisible by 4 "
              "for 2-bit tensors.",
              block_size));
        }
        if (tensor.shape() != nullptr && !tensor.shape()->empty()) {
          const int32_t last_dim =
              tensor.shape()->Get(tensor.shape()->size() - 1);
          if (last_dim > 0 && last_dim % block_size != 0) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Blockwise quantized tensor last dimension (%d) is not "
                "divisible by block_size (%d).",
                last_dim, block_size));
          }
        }
      }
      return absl::OkStatus();
    }
    case tflite::QuantizationDetails_NONE: {
      const bool scale_empty =
          (quant->scale() == nullptr || quant->scale()->empty());
      const bool zp_empty =
          (quant->zero_point() == nullptr || quant->zero_point()->empty());
      if (!scale_empty || !zp_empty) {
        if (scale_empty != zp_empty ||
            quant->scale()->size() != quant->zero_point()->size()) {
          return absl::InvalidArgumentError(
              "Affine quantization scale and zero_point must both be empty "
              "or have matching non-zero sizes.");
        }
        if (quant->quantized_dimension() < 0) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Affine quantization quantized_dimension (%d) must be "
              "non-negative.",
              quant->quantized_dimension()));
        }
        if (tensor.shape() != nullptr && !tensor.shape()->empty()) {
          if (static_cast<size_t>(quant->quantized_dimension()) >=
              tensor.shape()->size()) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Affine quantization quantized_dimension (%d) out of range "
                "for rank (%u).",
                quant->quantized_dimension(), tensor.shape()->size()));
          }
          if (quant->scale()->size() > 1) {
            const int32_t qdim_size =
                tensor.shape()->Get(quant->quantized_dimension());
            if (qdim_size >= 0 &&
                static_cast<size_t>(qdim_size) != quant->scale()->size()) {
              return absl::InvalidArgumentError(absl::StrFormat(
                  "Per-axis quantization scale count (%u) does not match "
                  "quantized dimension size (%d).",
                  quant->scale()->size(), qdim_size));
            }
          }
        }
      }
      return absl::OkStatus();
    }
    case tflite::QuantizationDetails_MultiAxisQuantization: {
      const auto* ma = quant->details_as_MultiAxisQuantization();
      if (ma == nullptr || ma->quantized_dimensions() == nullptr ||
          ma->quantized_dimensions()->empty()) {
        return absl::InvalidArgumentError(
            "MultiAxisQuantization details or quantized_dimensions are "
            "missing.");
      }
      const size_t num_tensors =
          subgraph.tensors() ? subgraph.tensors()->size() : 0;
      if (ma->scales() < 0 ||
          static_cast<size_t>(ma->scales()) >= num_tensors ||
          subgraph.tensors()->Get(ma->scales()) == nullptr) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "MultiAxisQuantization has invalid scales tensor index (%d).",
            ma->scales()));
      }
      if (ma->zero_points() < -1 ||
          (ma->zero_points() >= 0 &&
           (static_cast<size_t>(ma->zero_points()) >= num_tensors ||
            subgraph.tensors()->Get(ma->zero_points()) == nullptr))) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "MultiAxisQuantization has invalid zero_points tensor index (%d).",
            ma->zero_points()));
      }
      const int32_t block_size = ma->block_size();
      if (block_size < 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "MultiAxisQuantization block_size (%d) must be non-negative.",
            block_size));
      }
      if (block_size > 0) {
        if ((tensor.type() == tflite::TensorType_INT4 ||
             tensor.type() == tflite::TensorType_UINT4) &&
            block_size % 2 != 0) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "MultiAxisQuantization block_size (%d) must be divisible by 2 "
              "for 4-bit tensors.",
              block_size));
        }
        if (tensor.type() == tflite::TensorType_INT2 && block_size % 4 != 0) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "MultiAxisQuantization block_size (%d) must be divisible by 4 "
              "for 2-bit tensors.",
              block_size));
        }
      }
      for (size_t i = 0; i < ma->quantized_dimensions()->size(); ++i) {
        const int32_t qdim = ma->quantized_dimensions()->Get(i);
        if (qdim < 0) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "MultiAxisQuantization quantized_dimension (%d) must be "
              "non-negative.",
              qdim));
        }
        if (tensor.shape() != nullptr && !tensor.shape()->empty() &&
            static_cast<size_t>(qdim) >= tensor.shape()->size()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "MultiAxisQuantization quantized_dimension (%d) out of range "
              "for rank (%u).",
              qdim, tensor.shape()->size()));
        }
        for (size_t j = 0; j < i; ++j) {
          if (ma->quantized_dimensions()->Get(j) == qdim) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "MultiAxisQuantization contains duplicate quantized_dimension "
                "(%d).",
                qdim));
          }
        }
      }
      return absl::OkStatus();
    }
    case tflite::QuantizationDetails_CustomQuantization:
      return absl::UnimplementedError(absl::StrFormat(
          "Unsupported quantization details type (%s).",
          tflite::EnumNameQuantizationDetails(quant->details_type())));
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Unknown quantization details type (%u).",
                      static_cast<unsigned>(quant->details_type())));
}

absl::Status ValidateFlatBufferTensors(const tflite::Model* tfl_model,
                                       const VerifyOptions& options) {
  if (tfl_model == nullptr) {
    return absl::InvalidArgumentError("Model pointer is null.");
  }
  if (tfl_model->subgraphs() == nullptr) {
    return absl::OkStatus();
  }

  for (const tflite::SubGraph* subgraph : *tfl_model->subgraphs()) {
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
      continue;
    }
    for (const tflite::Tensor* tensor : *subgraph->tensors()) {
      if (tensor == nullptr) {
        return absl::InvalidArgumentError(
            "Subgraph contains null tensor entry.");
      }

      const bool has_known_rank =
          (tensor->shape() != nullptr) || tensor->has_rank();
      if (options.require_all_tensor_shape_ranks_known && !has_known_rank) {
        return absl::InvalidArgumentError(
            "Tensor has unknown rank and "
            "require_all_tensor_shape_ranks_known is true.");
      }

      if (tensor->shape() != nullptr) {
        const size_t rank = tensor->shape()->size();
        if (rank > options.max_rank || rank > LITERT_TENSOR_MAX_RANK) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Tensor rank (%zu) exceeds max_rank (%zu).", rank,
                              options.max_rank));
        }
        for (int32_t dim : *tensor->shape()) {
          if (dim < -1) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Tensor has invalid negative dimension (%d).", dim));
          }
        }
      }

      if (tensor->shape_signature() != nullptr) {
        const size_t sig_rank = tensor->shape_signature()->size();
        if (sig_rank > options.max_rank || sig_rank > LITERT_TENSOR_MAX_RANK) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Tensor shape_signature rank (%zu) exceeds max_rank (%zu).",
              sig_rank, options.max_rank));
        }
        if (tensor->shape() != nullptr && sig_rank != tensor->shape()->size()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Tensor shape_signature rank (%zu) does not match shape rank "
              "(%zu).",
              sig_rank, tensor->shape()->size()));
        }
        for (int32_t dim : *tensor->shape_signature()) {
          if (dim < -1) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Tensor shape_signature has invalid negative dimension (%d).",
                dim));
          }
        }
      }

      if (absl::Status quant_status =
              ValidateFlatBufferTensorQuantization(*subgraph, *tensor);
          !quant_status.ok()) {
        return quant_status;
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace internal

namespace {

class StatusErrorReporter : public tflite::ErrorReporter {
 public:
  int Report(const char* format, va_list args) override {
    char buf[1024];
    const int n = std::vsnprintf(buf, sizeof(buf), format, args);
    if (n > 0 && message_.empty()) {
      message_.assign(buf);
    }
    return n;
  }

  const std::string& message() const { return message_; }

 private:
  std::string message_;
};

absl::Status ValidateLiteRtTensor(const LiteRtTensorT& tensor,
                                  const VerifyOptions& options) {
  if (tensor.Type().first != kLiteRtRankedTensorType) {
    if (options.require_all_tensor_shape_ranks_known) {
      return absl::InvalidArgumentError("LiteRT tensor rank is not known.");
    }
    return absl::OkStatus();
  }

  const auto& layout = tensor.Type().second.ranked_tensor_type.layout;
  if (layout.rank > options.max_rank || layout.rank > LITERT_TENSOR_MAX_RANK) {
    return absl::InvalidArgumentError(
        absl::StrFormat("LiteRT tensor rank (%u) exceeds max_rank (%zu).",
                        layout.rank, options.max_rank));
  }

  for (unsigned int d = 0; d < layout.rank; ++d) {
    if (layout.dimensions[d] < -1) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "LiteRT tensor has invalid dimension (%d).", layout.dimensions[d]));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status VerifyModel(const void* buf, size_t len,
                         const VerifyOptions& options) {
  if (buf == nullptr || len == 0) {
    return absl::InvalidArgumentError("Model buffer is null or empty.");
  }

  if (options.max_rank > LITERT_TENSOR_MAX_RANK) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Configured max_rank (%zu) exceeds LITERT_TENSOR_MAX_RANK (%d).",
        options.max_rank, LITERT_TENSOR_MAX_RANK));
  }

  // 1. Verify the TFLite FlatBuffer integrity and buffer bounds.
  StatusErrorReporter tflite_reporter;
  if (!::tflite::Verify(buf, len, &tflite_reporter)) {
    return absl::InvalidArgumentError(
        tflite_reporter.message().empty()
            ? "TFLite FlatBuffer verification failed."
            : tflite_reporter.message());
  }

  // 2. Validate tensor ranks and dimensions in the FlatBuffer.
  const tflite::Model* tfl_model = tflite::GetModel(buf);
  if (absl::Status status =
          internal::ValidateFlatBufferTensors(tfl_model, options);
      !status.ok()) {
    return status;
  }

  // 3. Load into LiteRT's internal model representation.
  BufferRef<uint8_t> model_buf(reinterpret_cast<const uint8_t*>(buf), len);
  auto model = internal::LoadModelFromBuffer(model_buf);
  if (!model) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Failed to unpack LiteRT model (status %d).",
                        static_cast<int>(model.Error().Status())));
  }

  // 4. Validate initial unpacked LiteRT tensors and run per-op shape/type
  // inference across all subgraphs.
  for (auto* subgraph : (*model)->Subgraphs()) {
    if (subgraph == nullptr) {
      continue;
    }
    for (const auto* tensor : subgraph->Tensors()) {
      if (tensor == nullptr) {
        return absl::InvalidArgumentError(
            "LiteRT subgraph contains null tensor.");
      }
      if (absl::Status status = ValidateLiteRtTensor(*tensor, options);
          !status.ok()) {
        return status;
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
          return absl::UnimplementedError(
              absl::StrFormat("Unsupported shape inferrer for op code %d.",
                              static_cast<int>(op->OpCode())));
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

        return absl::InvalidArgumentError(absl::StrFormat(
            "LiteRT per-op shape/type validation failed for op %d (status %d).",
            static_cast<int>(op->OpCode()), static_cast<int>(status)));
      }

      for (const auto* out : op->Outputs()) {
        if (out == nullptr) {
          return absl::InvalidArgumentError(
              "LiteRT op has null output tensor.");
        }
        if (absl::Status out_status = ValidateLiteRtTensor(*out, options);
            !out_status.ok()) {
          return out_status;
        }
      }
    }
  }

  return absl::OkStatus();
}

bool Verify(const void* buf, size_t len, tflite::ErrorReporter* reporter,
            const VerifyOptions& options) {
  const absl::Status status = VerifyModel(buf, len, options);
  if (!status.ok()) {
    tflite::ErrorReporter* effective_reporter =
        reporter ? reporter : tflite::DefaultErrorReporter();
    effective_reporter->Report("%.*s",
                               static_cast<int>(status.message().size()),
                               status.message().data());
    return false;
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

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

#include <algorithm>
#include <climits>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "SafeInt.hpp"  // from @SafeInt
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/numeric/bits.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/string.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "flatbuffers/verifier.h"  // from @flatbuffers
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
#include "tflite/version.h"

namespace litert {

namespace internal {

namespace {

absl::Status ValidateStaticQuantParamTensorShape(
    const tflite::Tensor& param, const char* absl_nonnull label) {
  if (param.shape() == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s tensor must have a known shape.", label));
  }
  for (int32_t dim : *param.shape()) {
    if (dim < 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s tensor must have static dimensions (got %d).", label, dim));
    }
  }
  if (param.shape_signature() != nullptr) {
    for (int32_t dim : *param.shape_signature()) {
      if (dim < 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "%s tensor must have static shape_signature dimensions (got %d).",
            label, dim));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateFlatBufferOperatorCodes(const tflite::Model* tfl_model) {
  if (tfl_model == nullptr) {
    return absl::InvalidArgumentError("Model pointer is null.");
  }
  if (tfl_model->operator_codes() == nullptr) {
    return absl::OkStatus();
  }
  for (size_t i = 0; i < tfl_model->operator_codes()->size(); ++i) {
    const tflite::OperatorCode* code = tfl_model->operator_codes()->Get(i);
    if (code == nullptr) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Model contains null operator_code at index %zu.", i));
    }
    const int32_t builtin_code = code->builtin_code();
    const int32_t dep_code = code->deprecated_builtin_code();
    const int32_t resolved_code =
        dep_code > builtin_code ? dep_code : builtin_code;
    if (builtin_code < 0 || resolved_code < 0 ||
        resolved_code > static_cast<int32_t>(kLiteRtOpCodeShloComposite) ||
        (dep_code < 0 && static_cast<int8_t>(builtin_code) !=
                             code->deprecated_builtin_code())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Operator code at index %zu has invalid builtin_code (%d) / "
          "deprecated_builtin_code (%d).",
          i, builtin_code, dep_code));
    }
  }
  return absl::OkStatus();
}

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
      if (absl::Status status = ValidateStaticQuantParamTensorShape(
              *subgraph.tensors()->Get(bw->scales()),
              "Blockwise quantization scales");
          !status.ok()) {
        return status;
      }
      // Per schema.fbs, zero_points == -1 indicates an omitted optional
      // zero-point tensor (the zero point is assumed to be 0).
      if (bw->zero_points() < -1 ||
          (bw->zero_points() >= 0 &&
           (static_cast<size_t>(bw->zero_points()) >= num_tensors ||
            subgraph.tensors()->Get(bw->zero_points()) == nullptr))) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Blockwise quantization has invalid zero_points tensor index "
            "(%d).",
            bw->zero_points()));
      }
      if (bw->zero_points() >= 0) {
        if (absl::Status status = ValidateStaticQuantParamTensorShape(
                *subgraph.tensors()->Get(bw->zero_points()),
                "Blockwise quantization zero_points");
            !status.ok()) {
          return status;
        }
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
      if (absl::Status status = ValidateStaticQuantParamTensorShape(
              *subgraph.tensors()->Get(ma->scales()),
              "MultiAxisQuantization scales");
          !status.ok()) {
        return status;
      }
      // Per schema.fbs, zero_points == -1 indicates an omitted optional
      // zero-point tensor (the zero point is assumed to be 0).
      if (ma->zero_points() < -1 ||
          (ma->zero_points() >= 0 &&
           (static_cast<size_t>(ma->zero_points()) >= num_tensors ||
            subgraph.tensors()->Get(ma->zero_points()) == nullptr))) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "MultiAxisQuantization has invalid zero_points tensor index (%d).",
            ma->zero_points()));
      }
      if (ma->zero_points() >= 0) {
        if (absl::Status status = ValidateStaticQuantParamTensorShape(
                *subgraph.tensors()->Get(ma->zero_points()),
                "MultiAxisQuantization zero_points");
            !status.ok()) {
          return status;
        }
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
        int32_t positive_elements = 1;
        for (int32_t dim : *tensor->shape()) {
          if (dim < -1) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Tensor has invalid negative dimension (%d).", dim));
          }
          if (dim > 1 &&
              !SafeMultiply(positive_elements, dim, positive_elements)) {
            return absl::InvalidArgumentError(
                "Tensor element count exceeds the 32-bit signed integer "
                "limit.");
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
        int32_t positive_sig_elements = 1;
        for (int32_t dim : *tensor->shape_signature()) {
          if (dim < -1) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Tensor shape_signature has invalid negative dimension (%d).",
                dim));
          }
          if (dim > 1 && !SafeMultiply(positive_sig_elements, dim,
                                       positive_sig_elements)) {
            return absl::InvalidArgumentError(
                "Tensor shape_signature element count exceeds the 32-bit "
                "signed integer limit.");
          }
        }
      }

      if (absl::Status quant_status =
              ValidateFlatBufferTensorQuantization(*subgraph, *tensor);
          !quant_status.ok()) {
        return quant_status;
      }

      if (tensor->sparsity() != nullptr) {
        if (absl::Status sparsity_status =
                VerifyAndCountSparseElements(*tensor).status();
            !sparsity_status.ok()) {
          return sparsity_status;
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace internal

namespace {

constexpr int kOptionalTensorIndex = -1;
constexpr uint32_t kMaxNumString = UINT_MAX / sizeof(int32_t) - 2;

absl::string_view NameOrEmptyString(const flatbuffers::String* str) {
  if (str == nullptr || str->c_str() == nullptr) {
    return "";
  }
  return absl::string_view(str->c_str(), str->size());
}

uint32_t GetIntPtr(const char* absl_nonnull ptr) {
  uint32_t value;
  std::memcpy(&value, ptr, sizeof(value));
  if constexpr (absl::endian::native == absl::endian::big) {
    return absl::byteswap(value);
  }
  return value;
}

tflite::BuiltinOperator ResolveBuiltinCode(
    const tflite::OperatorCode* absl_nonnull opcode) {
  return static_cast<tflite::BuiltinOperator>(
      std::max(opcode->builtin_code(),
               static_cast<tflite::BuiltinOperator>(
                   opcode->deprecated_builtin_code())));
}

// Verifies that the buffer is a valid TFLite Model FlatBuffer and returns the
// in-memory Model pointer, or nullptr on failure.
const tflite::Model* VerifyFlatBufferAndGetModel(const void* buf, size_t len) {
  flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buf), len);
  if (tflite::VerifyModelBuffer(verifier)) {
    return tflite::GetModel(buf);
  }
  return nullptr;
}

// Verifies string tensor has legit buffer contents that follow the schema
// defined in lite/string_util.h.
absl::Status VerifyStringTensorBuffer(const tflite::Tensor& tensor,
                                      const tflite::Buffer& buffer) {
  uint32_t buffer_size = buffer.data()->size();
  if (buffer_size < sizeof(uint32_t)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("String tensor %s is invalid (empty)",
                        NameOrEmptyString(tensor.name())));
  }
  const char* buffer_ptr = reinterpret_cast<const char*>(buffer.data()->data());

  uint32_t num_strings = GetIntPtr(buffer_ptr);
  if (num_strings > kMaxNumString) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "String tensor %s has invalid num of string set: %u",
        NameOrEmptyString(tensor.name()), num_strings));
  }
  uint32_t header_offsets =
      static_cast<uint32_t>(num_strings + 2) * sizeof(int32_t);

  if (buffer_size < header_offsets) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "String tensor %s buffer requires at least %u bytes, but is allocated "
        "with %u bytes",
        NameOrEmptyString(tensor.name()), header_offsets, buffer_size));
  }

  uint32_t prev_ptr = header_offsets;
  uint32_t offset = sizeof(int32_t);

  if (GetIntPtr(buffer_ptr + offset) != header_offsets) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "String tensor %s buffer initial offset must be: %u",
        NameOrEmptyString(tensor.name()), header_offsets));
  }
  offset += sizeof(int32_t);
  for (uint32_t i = 1, end = num_strings; i <= end;
       i++, offset += sizeof(int32_t)) {
    int string_offset = GetIntPtr(buffer_ptr + offset);
    if (string_offset < static_cast<int>(prev_ptr) ||
        string_offset > static_cast<int>(buffer_size)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("String tensor %s buffer is invalid: index %u",
                          NameOrEmptyString(tensor.name()), i));
    }
  }
  if (GetIntPtr(buffer_ptr + offset - sizeof(int32_t)) != buffer_size) {
    return absl::InvalidArgumentError(
        absl::StrFormat("String tensor %s buffer last offset must be %u",
                        NameOrEmptyString(tensor.name()), buffer_size));
  }
  return absl::OkStatus();
}

bool CheckArraySegments(
    const tflite::DimensionMetadata* absl_nonnull dim_metadata) {
  if (dim_metadata->array_segments() == nullptr) {
    return false;
  }
  switch (dim_metadata->array_segments_type()) {
    case tflite::SparseIndexVector_Int32Vector:
      return (dim_metadata->array_segments_as_Int32Vector()->values() !=
              nullptr);
    case tflite::SparseIndexVector_Uint16Vector:
      return (dim_metadata->array_segments_as_Uint16Vector()->values() !=
              nullptr);
    case tflite::SparseIndexVector_Uint8Vector:
      return (dim_metadata->array_segments_as_Uint8Vector()->values() !=
              nullptr);
    default:
      return false;
  }
}

int GetSizeOfSegments(
    const tflite::DimensionMetadata* absl_nonnull dim_metadata) {
  switch (dim_metadata->array_segments_type()) {
    case tflite::SparseIndexVector_Int32Vector:
      return dim_metadata->array_segments_as_Int32Vector()->values()->size();
    case tflite::SparseIndexVector_Uint16Vector:
      return dim_metadata->array_segments_as_Uint16Vector()->values()->size();
    case tflite::SparseIndexVector_Uint8Vector:
      return dim_metadata->array_segments_as_Uint8Vector()->values()->size();
    default:
      return -1;
  }
}

int GetValueOfSegmentsAt(
    const tflite::DimensionMetadata* absl_nonnull dim_metadata, const int i) {
  switch (dim_metadata->array_segments_type()) {
    case tflite::SparseIndexVector_Int32Vector:
      return static_cast<int>(
          dim_metadata->array_segments_as_Int32Vector()->values()->Get(i));
    case tflite::SparseIndexVector_Uint16Vector:
      return static_cast<int>(
          dim_metadata->array_segments_as_Uint16Vector()->values()->Get(i));
    case tflite::SparseIndexVector_Uint8Vector:
      return static_cast<int>(
          dim_metadata->array_segments_as_Uint8Vector()->values()->Get(i));
    default:
      return -1;
  }
}

bool CheckArrayIndices(
    const tflite::DimensionMetadata* absl_nonnull dim_metadata) {
  if (dim_metadata->array_indices() == nullptr) {
    return false;
  }
  switch (dim_metadata->array_indices_type()) {
    case tflite::SparseIndexVector_Int32Vector:
      return (dim_metadata->array_indices_as_Int32Vector()->values() !=
              nullptr);
    case tflite::SparseIndexVector_Uint16Vector:
      return (dim_metadata->array_indices_as_Uint16Vector()->values() !=
              nullptr);
    case tflite::SparseIndexVector_Uint8Vector:
      return (dim_metadata->array_indices_as_Uint8Vector()->values() !=
              nullptr);
    default:
      return false;
  }
}

int GetSizeOfIndices(
    const tflite::DimensionMetadata* absl_nonnull dim_metadata) {
  switch (dim_metadata->array_indices_type()) {
    case tflite::SparseIndexVector_Int32Vector:
      return dim_metadata->array_indices_as_Int32Vector()->values()->size();
    case tflite::SparseIndexVector_Uint16Vector:
      return dim_metadata->array_indices_as_Uint16Vector()->values()->size();
    case tflite::SparseIndexVector_Uint8Vector:
      return dim_metadata->array_indices_as_Uint8Vector()->values()->size();
    default:
      return -1;
  }
}

int GetValueOfIndicesAt(
    const tflite::DimensionMetadata* absl_nonnull dim_metadata, const int i) {
  switch (dim_metadata->array_indices_type()) {
    case tflite::SparseIndexVector_Int32Vector:
      return static_cast<int>(
          dim_metadata->array_indices_as_Int32Vector()->values()->Get(i));
    case tflite::SparseIndexVector_Uint16Vector:
      return static_cast<int>(
          dim_metadata->array_indices_as_Uint16Vector()->values()->Get(i));
    case tflite::SparseIndexVector_Uint8Vector:
      return static_cast<int>(
          dim_metadata->array_indices_as_Uint8Vector()->values()->Get(i));
    default:
      return -1;
  }
}

}  // namespace

namespace internal {

// The sparsity parameter defines a tree structure to map each non-zero element
// stored in the flattened buffer back to its index in the conceptual dense
// tensor.
// Traverse the tree level by level, count total number of elements, and
// validate the sparsity parameters along the way.
absl::StatusOr<uint32_t> VerifyAndCountElements(
    const tflite::SparsityParameters& sparsity, absl::Span<const int> dim_sizes,
    absl::string_view tensor_name) {
  if (sparsity.traversal_order() == nullptr ||
      sparsity.dim_metadata() == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tensor %s has invalid sparsity parameters: missing traversal_order or "
        "dim_metadata",
        tensor_name));
  }
  const int total_level = sparsity.traversal_order()->size();
  if (sparsity.dim_metadata()->size() != static_cast<size_t>(total_level)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tensor %s has invalid sparsity parameters: traversal_order size (%d) "
        "does not match dim_metadata size (%u)",
        tensor_name, total_level, sparsity.dim_metadata()->size()));
  }
  const size_t block_rank =
      sparsity.block_map() != nullptr ? sparsity.block_map()->size() : 0;
  if (block_rank > static_cast<size_t>(total_level)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tensor %s has invalid sparsity parameters: block_map size (%zu) "
        "exceeds traversal_order size (%d)",
        tensor_name, block_rank, total_level));
  }
  const size_t original_rank = static_cast<size_t>(total_level) - block_rank;
  uint32_t num_elements = 1;
  for (int i = 0; i < total_level; i++) {
    const int original_dim = sparsity.traversal_order()->Get(i);
    if (original_dim < 0 ||
        static_cast<size_t>(original_dim) >= dim_sizes.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Tensor %s has invalid sparsity parameters: traversal_order[%d] "
          "(%d) out of bounds [0, %zu)",
          tensor_name, i, original_dim, dim_sizes.size()));
    }
    const auto* dim_metadata = sparsity.dim_metadata()->Get(i);
    if (dim_metadata == nullptr) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Tensor %s has invalid sparsity parameters: dim_metadata[%d] is null",
          tensor_name, i));
    }
    if (dim_metadata->format() == tflite::DimensionType_DENSE) {
      const bool is_block_dim =
          static_cast<size_t>(i) >= original_rank ||
          static_cast<size_t>(original_dim) >= original_rank;
      if (is_block_dim && dim_metadata->dense_size() <= 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Tensor %s has invalid sparsity parameters: dense_size (%d) for "
            "block dimension at level %d must be positive",
            tensor_name, dim_metadata->dense_size(), i));
      }
      if (dim_metadata->dense_size() < 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Tensor %s has invalid sparsity parameters: dense_size (%d) at "
            "level %d must be non-negative",
            tensor_name, dim_metadata->dense_size(), i));
      }
      if (dim_metadata->dense_size() != dim_sizes[original_dim]) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Tensor %s has invalid sparsity parameters: dense_size (%d) at "
            "level %d does not match dimension size (%d)",
            tensor_name, dim_metadata->dense_size(), i,
            dim_sizes[original_dim]));
      }

      // Each index in a dense dimension is stored implicitly.
      if (!SafeMultiply(num_elements, dim_metadata->dense_size(),
                        num_elements)) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Tensor %s dimension overflow", tensor_name));
      }
    } else if (dim_metadata->format() == tflite::DimensionType_SPARSE_CSR) {
      if (!CheckArraySegments(dim_metadata) ||
          !CheckArrayIndices(dim_metadata)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Tensor %s has invalid sparsity parameters: missing array_segments "
            "or array_indices at level %d",
            tensor_name, i));
      }

      int array_segments_size = GetSizeOfSegments(dim_metadata);
      int array_indices_size = GetSizeOfIndices(dim_metadata);

      for (int j = 0; j < array_segments_size - 1; j++) {
        if (GetValueOfSegmentsAt(dim_metadata, j) < 0 ||
            GetValueOfSegmentsAt(dim_metadata, j + 1) < 0 ||
            GetValueOfSegmentsAt(dim_metadata, j) >
                GetValueOfSegmentsAt(dim_metadata, j + 1)) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Tensor %s has invalid sparsity parameters: invalid "
              "array_segments at level %d, index %d",
              tensor_name, i, j));
        }
      }

      if (array_segments_size < 1 ||
          num_elements != static_cast<uint32_t>(array_segments_size - 1)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Tensor %s has invalid sparsity parameters: num_elements (%u) does "
            "not match array_segments_size - 1 (%d) at level %d",
            tensor_name, num_elements, array_segments_size - 1, i));
      }

      if (array_indices_size !=
          GetValueOfSegmentsAt(dim_metadata, array_segments_size - 1)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Tensor %s has invalid sparsity parameters: array_indices_size "
            "(%d) does not match last segment value (%d) at level %d",
            tensor_name, array_indices_size,
            GetValueOfSegmentsAt(dim_metadata, array_segments_size - 1), i));
      }

      for (int j = 0; j < array_indices_size; j++) {
        if (GetValueOfIndicesAt(dim_metadata, j) < 0 ||
            GetValueOfIndicesAt(dim_metadata, j) >= dim_sizes[original_dim]) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Tensor %s has invalid sparsity parameters: array_indices[%d] "
              "(%d) out of bounds [0, %d) at level %d",
              tensor_name, j, GetValueOfIndicesAt(dim_metadata, j),
              dim_sizes[original_dim], i));
        }
      }

      // Need to reset num_elements when seeing a sparse dimension.
      num_elements = static_cast<uint32_t>(array_indices_size);
    } else {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Tensor %s has invalid sparsity parameters: unknown format (%d) at "
          "level %d",
          tensor_name, static_cast<int>(dim_metadata->format()), i));
    }
  }

  return num_elements;
}

absl::StatusOr<uint32_t> VerifyAndCountSparseElements(
    const tflite::Tensor& tensor) {
  const absl::string_view tensor_name = NameOrEmptyString(tensor.name());
  const auto* sparsity = tensor.sparsity();
  if (sparsity == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tensor %s has invalid sparsity parameters: sparsity is null",
        tensor_name));
  }
  if (tensor.shape() == nullptr || tensor.shape()->empty()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tensor %s has invalid sparsity parameters: shape is null or empty",
        tensor_name));
  }
  if (sparsity->traversal_order() == nullptr ||
      sparsity->dim_metadata() == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tensor %s has invalid sparsity parameters: missing traversal_order or "
        "dim_metadata",
        tensor_name));
  }

  const int total_dims = sparsity->traversal_order()->size();
  const int original_rank = tensor.shape()->size();
  const int sparsity_dim_metadata_size = sparsity->dim_metadata()->size();
  if (total_dims < original_rank || sparsity_dim_metadata_size != total_dims) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tensor %s has invalid sparsity parameters: total_dims (%d), "
        "original_rank (%d), and dim_metadata size (%d) are inconsistent",
        tensor_name, total_dims, original_rank, sparsity_dim_metadata_size));
  }

  const int block_rank = total_dims - original_rank;
  if (block_rank > 0) {
    if (sparsity->block_map() == nullptr) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Tensor %s has invalid sparsity parameters: missing block_map for "
          "block_rank %d",
          tensor_name, block_rank));
    }
    const int sparse_rank = sparsity->block_map()->size();
    if (sparse_rank != block_rank) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Tensor %s has invalid sparsity parameters: block_map size (%d) does "
          "not match block_rank (%d)",
          tensor_name, sparse_rank, block_rank));
    }
  }

  // For a n-dimensional tensor (d0, ..., dn-1) with k-dimensional block (dn,
  // ..., dn+k-1), the first n elements in the traversal order should be a
  // permutation of (d0, ..., dn-1), and the last k elements should be a
  // permutation of (dn, ..., dn+k-1).
  std::vector<int> traversal_order(total_dims);
  for (int i = 0; i < total_dims; i++) {
    traversal_order[i] = sparsity->traversal_order()->Get(i);
  }

  std::sort(traversal_order.begin(), traversal_order.begin() + original_rank);
  for (int i = 0; i < original_rank; i++) {
    if (traversal_order[i] != i) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Tensor %s has invalid sparsity parameters: traversal_order is not a "
          "valid permutation of original dimensions",
          tensor_name));
    }
  }

  std::sort(traversal_order.begin() + original_rank, traversal_order.end());
  for (int i = original_rank; i < total_dims; i++) {
    if (traversal_order[i] != i) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Tensor %s has invalid sparsity parameters: traversal_order is not a "
          "valid permutation of block dimensions",
          tensor_name));
    }
  }

  // For a n-dimensional tensor (d0, ..., dn-1) with k-dimensional block (dn,
  // ..., dn+k-1), the expanded_dim_sizes holds the size of each dimension in
  // the order of (d0, ..., dn-1, dn, ..., dn+k-1), not the traversal order.
  // For example, a 4x4 tensor with 2x2 block has expanded_dim_sizes = {2, 2, 2,
  // 2}.
  std::vector<int> expanded_dim_sizes;
  expanded_dim_sizes.resize(total_dims);
  // First go through the original tensor dimensions, populate their sizes.
  for (int i = 0; i < original_rank; i++) {
    expanded_dim_sizes[i] = tensor.shape()->Get(i);
  }
  // Then go through the block dimensions, and
  //   1. populate block dimension size.
  //   2. block_map[i] has the original dimension that block dimension i maps
  //   to. Divide the size of the original dimension by the size of the ith
  //   block dimension.
  for (int i = 0; i < block_rank; i++) {
    int original_block_dim =
        sparsity->traversal_order()->Get(i + original_rank);
    if (original_block_dim < 0 || original_block_dim >= total_dims) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Tensor %s has invalid sparsity parameters: original_block_dim (%d) "
          "out of bounds [0, %d)",
          tensor_name, original_block_dim, total_dims));
    }
    const auto* block_dim_metadata =
        sparsity->dim_metadata()->Get(i + original_rank);
    if (block_dim_metadata == nullptr) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Tensor %s has invalid sparsity parameters: block dim_metadata[%d] "
          "is null",
          tensor_name, i + original_rank));
    }
    int block_dim_size = block_dim_metadata->dense_size();
    // If size is <= 0 we just return as it is invalid.
    if (block_dim_size <= 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Tensor %s has invalid sparsity parameters: block_dim_size (%d) must "
          "be positive",
          tensor_name, block_dim_size));
    }

    expanded_dim_sizes[original_block_dim] = block_dim_size;

    int mapped_block_dim = sparsity->block_map()->Get(i);
    if (mapped_block_dim < 0 || mapped_block_dim >= original_rank) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Tensor %s has invalid sparsity parameters: mapped_block_dim (%d) "
          "out of bounds [0, %d)",
          tensor_name, mapped_block_dim, original_rank));
    }
    expanded_dim_sizes[mapped_block_dim] /= block_dim_size;
  }

  return VerifyAndCountElements(*sparsity, expanded_dim_sizes, tensor_name);
}

}  // namespace internal

namespace {

// Verifies numeric tensor has legit buffer.
absl::Status VerifyNumericTensorBuffer(const tflite::Tensor& tensor,
                                       const tflite::Buffer& buffer) {
  uint32_t bytes_required = 1;
  if (!tensor.shape()) {
    // Empty tensor. Avoid further checks.
    return absl::OkStatus();
  }
  if (tensor.sparsity() != nullptr) {
    absl::StatusOr<uint32_t> num_elements =
        internal::VerifyAndCountSparseElements(tensor);
    if (!num_elements.ok()) {
      return num_elements.status();
    }
    bytes_required = *num_elements;
  } else {
    for (int dim : *tensor.shape()) {
      if (!SafeMultiply(bytes_required, dim, bytes_required)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Tensor %s dimension overflow", NameOrEmptyString(tensor.name())));
      }
    }
  }

  bool overflow = false;
  switch (tensor.type()) {
    case tflite::TensorType_FLOAT32:
      overflow = !SafeMultiply(bytes_required, sizeof(float), bytes_required);
      break;
    case tflite::TensorType_FLOAT16:
      overflow =
          !SafeMultiply(bytes_required, sizeof(uint16_t), bytes_required);
      break;
    case tflite::TensorType_BFLOAT16:
      overflow =
          !SafeMultiply(bytes_required, sizeof(uint16_t), bytes_required);
      break;
    case tflite::TensorType_FLOAT64:
      overflow = !SafeMultiply(bytes_required, sizeof(double), bytes_required);
      break;
    case tflite::TensorType_INT32:
      overflow = !SafeMultiply(bytes_required, sizeof(int32_t), bytes_required);
      break;
    case tflite::TensorType_UINT32:
      overflow =
          !SafeMultiply(bytes_required, sizeof(uint32_t), bytes_required);
      break;
    case tflite::TensorType_INT2:
      overflow = !SafeAdd(bytes_required, 3, bytes_required);
      bytes_required /= 4;
      break;
    case tflite::TensorType_UINT4:
      overflow = !SafeAdd(bytes_required, 1, bytes_required);
      bytes_required /= 2;
      break;
    case tflite::TensorType_INT4:
      overflow = !SafeAdd(bytes_required, 1, bytes_required);
      bytes_required /= 2;
      break;
    case tflite::TensorType_UINT8:
      overflow = !SafeMultiply(bytes_required, sizeof(uint8_t), bytes_required);
      break;
    case tflite::TensorType_FLOAT8_E4M3FN:
    case tflite::TensorType_FLOAT8_E5M2:
      overflow = !SafeMultiply(bytes_required, sizeof(uint8_t), bytes_required);
      break;
    case tflite::TensorType_INT8:
      overflow = !SafeMultiply(bytes_required, sizeof(int8_t), bytes_required);
      break;
    case tflite::TensorType_INT64:
      overflow = !SafeMultiply(bytes_required, sizeof(int64_t), bytes_required);
      break;
    case tflite::TensorType_UINT64:
      overflow =
          !SafeMultiply(bytes_required, sizeof(uint64_t), bytes_required);
      break;
    case tflite::TensorType_BOOL:
      overflow = !SafeMultiply(bytes_required, sizeof(bool), bytes_required);
      break;
    case tflite::TensorType_INT16:
      overflow =
          !SafeMultiply(bytes_required, sizeof(uint16_t), bytes_required);
      break;
    case tflite::TensorType_UINT16:
      overflow =
          !SafeMultiply(bytes_required, sizeof(uint16_t), bytes_required);
      break;
    case tflite::TensorType_COMPLEX64:
      overflow = !SafeMultiply(bytes_required, sizeof(std::complex<float>),
                               bytes_required);
      break;
    case tflite::TensorType_COMPLEX128:
      overflow = !SafeMultiply(bytes_required, sizeof(std::complex<double>),
                               bytes_required);
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Tensor %s invalid type: %d",
                          NameOrEmptyString(tensor.name()), tensor.type()));
  }
  if (overflow) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tensor %s dimension overflow", NameOrEmptyString(tensor.name())));
  }

  if (bytes_required != buffer.data()->size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tensor %s requires %u bytes, but is allocated with %u bytes buffer",
        NameOrEmptyString(tensor.name()), bytes_required,
        buffer.data()->size()));
  }
  return absl::OkStatus();
}

absl::Status VerifyOperators(
    const flatbuffers::Vector<flatbuffers::Offset<tflite::Operator>>& operators,
    size_t len) {
  for (const auto* op : operators) {
    if (op == nullptr) {
      return absl::InvalidArgumentError("Subgraph contains null operator.");
    }
    if (!op->inputs()) {
      return absl::InvalidArgumentError("Missing 'inputs' for operator.");
    }
    if (!op->outputs()) {
      return absl::InvalidArgumentError("Missing 'outputs' for operator.");
    }
    // When custom_options lives outside the flatbuffer (models > 2GB), the
    // offset/size pair points into the raw file allocation. Validate against
    // the total allocation length here, at ingress, using the subtraction
    // form so the check itself cannot overflow: `size > len` first rules out
    // a size larger than the whole allocation, so `len - size` below cannot
    // underflow.
    if (op->large_custom_options_offset() > 1) {
      if (op->large_custom_options_size() > len ||
          op->large_custom_options_offset() >
              len - op->large_custom_options_size()) {
        return absl::InvalidArgumentError(
            "Operator's large_custom_options is out of bound.");
      }
    }
  }
  return absl::OkStatus();
}

bool IsConstantTensor(const tflite::Tensor& tensor,
                      const tflite::Model& model) {
  if (tensor.external_buffer() > 0) return true;
  if (!tensor.buffer() || !model.buffers()) return false;
  if (tensor.buffer() > 0 && tensor.buffer() < model.buffers()->size()) {
    const auto* buffer = model.buffers()->Get(tensor.buffer());
    if (buffer && (buffer->data() || buffer->offset() > 1)) {
      return true;
    }
  }
  return false;
}

// Performs basic consistency checks on a sub-graph.
absl::Status VerifySubGraphConsistency(const tflite::Model& model,
                                       const tflite::SubGraph& subgraph) {
  absl::flat_hash_set<int> subgraph_input_tensors, constant_tensors,
      variable_tensors, output_tensors;
  if (subgraph.tensors()) {
    for (int i = 0, end = subgraph.tensors()->size(); i < end; ++i) {
      const auto* tensor = subgraph.tensors()->Get(i);
      if (tensor == nullptr) {
        return absl::InvalidArgumentError(
            "Subgraph contains null tensor entry.");
      }
      if (IsConstantTensor(*tensor, model)) {
        constant_tensors.insert(i);
      } else if (tensor->is_variable()) {
        variable_tensors.insert(i);
      }
    }
  }
  if (subgraph.inputs()) {
    for (const int tensor_idx : *subgraph.inputs()) {
      subgraph_input_tensors.insert(tensor_idx);
    }
  }

  if (subgraph.operators()) {
    for (int op_idx = 0, end = subgraph.operators()->size(); op_idx < end;
         ++op_idx) {
      const auto* op = subgraph.operators()->Get(op_idx);
      if (op == nullptr) {
        return absl::InvalidArgumentError("Subgraph contains null operator.");
      }
      if (!model.operator_codes() ||
          (op->opcode_index() >= model.operator_codes()->size())) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Operator %u does not exist in model op codes",
            op->opcode_index()));
      }
      const auto* opcode = model.operator_codes()->Get(op->opcode_index());
      if (opcode == nullptr) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Model contains null operator_code at index %u.",
            op->opcode_index()));
      }
      const tflite::BuiltinOperator builtin_code = ResolveBuiltinCode(opcode);
      // Check for invalid inputs by ensuring all exist in produced_tensors.
      for (const int input_idx : *op->inputs()) {
        if (input_idx == kOptionalTensorIndex) continue;
        if (!constant_tensors.contains(input_idx) &&
            !variable_tensors.contains(input_idx) &&
            !subgraph_input_tensors.contains(input_idx) &&
            !output_tensors.contains(input_idx)) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Input tensor %d to op %d (%s) is not produced", input_idx,
              op_idx, tflite::EnumNameBuiltinOperator(builtin_code)));
        }
      }
      // Check for cycles/invalid outputs by ensuring that none exist in
      // produced_tensors.
      for (const int output_idx : *op->outputs()) {
        if (constant_tensors.contains(output_idx)) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Output tensor %d to op %d (%s) is a constant", output_idx,
              op_idx, tflite::EnumNameBuiltinOperator(builtin_code)));
        } else if (variable_tensors.contains(output_idx)) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Output tensor %d to op %d (%s) is a variable", output_idx,
              op_idx, tflite::EnumNameBuiltinOperator(builtin_code)));
        } else if (subgraph_input_tensors.contains(output_idx)) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Output tensor %d to op %d (%s) is a subgraph input", output_idx,
              op_idx, tflite::EnumNameBuiltinOperator(builtin_code)));
        } else if (output_tensors.contains(output_idx)) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Output tensor %d to op %d (%s) is an output from another op. "
              "There is a cycle in the graph",
              output_idx, op_idx,
              tflite::EnumNameBuiltinOperator(builtin_code)));
        }
        // This can be an input to a subsequent op.
        output_tensors.insert(output_idx);
      }
    }
  }
  return absl::OkStatus();
}

absl::Status VerifySubGraphs(const tflite::Model& model, size_t len) {
  if (!model.subgraphs()) {
    return absl::InvalidArgumentError("Missing 'subgraphs' section.");
  }
  for (const auto* subgraph : *model.subgraphs()) {
    if (subgraph == nullptr) {
      return absl::InvalidArgumentError("Model contains null subgraph.");
    }
    const absl::string_view subgraph_name =
        NameOrEmptyString(subgraph->name());
    if (!subgraph->operators()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Subgraph %s: Missing 'operators' section in subgraph.",
          subgraph_name));
    }
    if (absl::Status status = VerifyOperators(*subgraph->operators(), len);
        !status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Subgraph %s: %s", subgraph_name, status.message()));
    }
    if (absl::Status status = VerifySubGraphConsistency(model, *subgraph);
        !status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Subgraph %s: %s", subgraph_name, status.message()));
    }
  }
  return absl::OkStatus();
}

// Verifies tensors have valid properties and legit buffer if set.
absl::Status VerifyTensors(const tflite::Model& model, size_t len) {
  if (!model.subgraphs()) {
    return absl::OkStatus();
  }
  if (!model.buffers()) {
    return absl::InvalidArgumentError("Missing 'buffers' section.");
  }

  for (const auto* subgraph : *model.subgraphs()) {
    if (subgraph == nullptr || !subgraph->tensors()) {
      continue;
    }
    for (const auto* tensor : *subgraph->tensors()) {
      if (tensor == nullptr) {
        return absl::InvalidArgumentError(
            "Subgraph contains null tensor entry.");
      }
      if (tensor->shape()) {
        int32_t positive_elements = 1;
        // Validate that fixed dimensions (> 1) product does not overflow
        // 32-bit int32_t, even when batch size is 0 or 1.
        for (int32_t dim : *tensor->shape()) {
          if (dim > 1 &&
              !SafeMultiply(positive_elements, dim, positive_elements)) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Tensor %s dimension overflow",
                NameOrEmptyString(tensor->name())));
          }
        }
      }

      if (!tensor->buffer()) {
        continue;
      }
      if (tensor->buffer() >= model.buffers()->size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Tensor %s invalid buffer index: %u",
            NameOrEmptyString(tensor->name()), tensor->buffer()));
      }
      const auto* buffer = model.buffers()->Get(tensor->buffer());
      if (!buffer) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Tensor %s buffer %u not set", NameOrEmptyString(tensor->name()),
            tensor->buffer()));
      }

      // Many transient tensors don't have data in the flatbuffer. Their
      // buffers will be allocated by the interpreter at run-time.
      if (buffer->data()) {
        if (tensor->type() == tflite::TensorType_STRING) {
          if (absl::Status status = VerifyStringTensorBuffer(*tensor, *buffer);
              !status.ok()) {
            return status;
          }
        } else {
          if (absl::Status status = VerifyNumericTensorBuffer(*tensor, *buffer);
              !status.ok()) {
            return status;
          }
        }
      } else if (buffer->offset() > 1) {
        // Buffer data lives outside the flatbuffer (models > 2GB), addressed
        // by offset/size into the raw file allocation. Validate against the
        // total allocation length here, at ingress, using the subtraction
        // form so the check itself cannot overflow: `size > len` first rules
        // out a size larger than the whole allocation, so `len - size` below
        // cannot underflow.
        if (buffer->size() > len || buffer->offset() > len - buffer->size()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Tensor %s buffer %u specified an out of range offset.",
              NameOrEmptyString(tensor->name()), tensor->buffer()));
        }
      }
    }
  }
  return absl::OkStatus();
}

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

  int32_t positive_elements = 1;
  for (unsigned int d = 0; d < layout.rank; ++d) {
    const int32_t dim = layout.dimensions[d];
    if (dim < -1) {
      return absl::InvalidArgumentError(
          absl::StrFormat("LiteRT tensor has invalid dimension (%d).", dim));
    }
    if (dim > 1 && !SafeMultiply(positive_elements, dim, positive_elements)) {
      return absl::InvalidArgumentError(
          "LiteRT tensor element count exceeds the 32-bit signed integer "
          "limit.");
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

  // 1. Verify the TFLite FlatBuffer integrity, schema version, subgraph
  // consistency, and buffer bounds.
  const tflite::Model* tfl_model = VerifyFlatBufferAndGetModel(buf, len);
  if (tfl_model == nullptr) {
    return absl::InvalidArgumentError("Invalid flatbuffer format");
  }
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid model version %u", tfl_model->version()));
  }
  if (absl::Status status = VerifySubGraphs(*tfl_model, len); !status.ok()) {
    return status;
  }
  if (absl::Status status = VerifyTensors(*tfl_model, len); !status.ok()) {
    return status;
  }

  // 2. Validate operator codes and tensor ranks/dimensions in the FlatBuffer.
  if (absl::Status status =
          internal::ValidateFlatBufferOperatorCodes(tfl_model);
      !status.ok()) {
    return status;
  }
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

bool Verify(const void* buf, size_t len, const VerifyOptions& options) {
  return VerifyModel(buf, len, options).ok();
}

bool LiteRtVerifier::Verify(const char* data, int length,
                            tflite::ErrorReporter* reporter) {
  if (length < 0) {
    return false;
  }
  const absl::Status status =
      VerifyModel(data, static_cast<size_t>(length), options_);
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

}  // namespace litert

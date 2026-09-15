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

#include "tensor/examples/gemma4/helpers/quantized_embedding.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/log/absl_log.h"         // from @com_google_absl
#include "absl/status/status.h"        // from @com_google_absl
#include "absl/status/statusor.h"      // from @com_google_absl
#include "absl/strings/str_cat.h"      // from @com_google_absl
#include "absl/types/span.h"           // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::examples::gemma4 {

GemmaEmbeddingTable::GemmaEmbeddingTable(TensorHandle tensor, int vocab_size,
                                         int emb_dim, Type type)
    : tensor_(std::move(tensor)),
      vocab_size_(vocab_size),
      emb_dim_(emb_dim),
      type_(type) {}

int32_t GemmaEmbeddingTable::ClampTokenId(int32_t token_id) const {
  if (token_id < 0 || token_id >= vocab_size_) {
    ABSL_LOG(WARNING) << "Token ID " << token_id << " out of range [0, "
                      << vocab_size_ << "), using 0";
    return 0;
  }
  return token_id;
}

absl::Status GemmaEmbeddingTable::Lookup(absl::Span<const int32_t> token_ids,
                                         absl::Span<float> output) const {
  if (token_ids.size() > output.size() / static_cast<size_t>(emb_dim_)) {
    return absl::InvalidArgumentError(
        "Output buffer is smaller than requested lookup size");
  }

  for (size_t i = 0; i < token_ids.size(); ++i) {
    const int32_t token_id = ClampTokenId(token_ids[i]);
    DecodeRow(token_id, 0, emb_dim_, output.data() + i * emb_dim_);
  }
  return absl::OkStatus();
}

absl::Status GemmaEmbeddingTable::LookupPerLayer(
    absl::Span<const int32_t> token_ids, int num_layers, int per_layer_dim,
    absl::Span<std::vector<float>> output_per_layer) const {
  if (output_per_layer.size() < static_cast<size_t>(num_layers)) {
    return absl::InvalidArgumentError("output_per_layer size mismatch");
  }
  if (num_layers <= 0 || per_layer_dim <= 0 ||
      num_layers > emb_dim_ / per_layer_dim) {
    return absl::InvalidArgumentError(
        "Embedding dimension smaller than num_layers * per_layer_dim");
  }
  const size_t seq_len = token_ids.size();
  for (size_t l = 0; l < num_layers; ++l) {
    if (seq_len > output_per_layer[l].size() / per_layer_dim) {
      return absl::InvalidArgumentError(
          absl::StrCat("Layer ", l, " output size is too small."));
    }
  }

  for (size_t s = 0; s < seq_len; ++s) {
    const int32_t token_id = ClampTokenId(token_ids[s]);
    for (int l = 0; l < num_layers; ++l) {
      const int col_start = l * per_layer_dim;
      float* dst = output_per_layer[l].data() + s * per_layer_dim;
      DecodeRow(token_id, col_start, per_layer_dim, dst);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<LockedBufferSpan<const float>>>
GemmaEmbeddingTable::LookupPerLayer(int32_t token_id, int num_layers,
                                    int per_layer_dim) const {
  if (num_layers <= 0 || per_layer_dim <= 0 ||
      num_layers > emb_dim_ / per_layer_dim) {
    return absl::InvalidArgumentError(
        "Embedding dimension smaller than num_layers * per_layer_dim");
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(LockedBufferSpan<const float> full_row,
                              Lookup(token_id));

  std::vector<LockedBufferSpan<const float>> result;
  result.reserve(num_layers);
  for (int l = 0; l < num_layers; ++l) {
    result.push_back(full_row.SubSpan(l * per_layer_dim, per_layer_dim));
  }
  return result;
}

namespace {

// Lookup for FP32 stored embeddings.
class Fp32GemmaEmbeddingTable : public GemmaEmbeddingTable {
 public:
  Fp32GemmaEmbeddingTable(TensorHandle tensor, int vocab_size, int emb_dim,
                          LockedBufferSpan<const float> locked_span)
      : GemmaEmbeddingTable(std::move(tensor), vocab_size, emb_dim,
                            Type::kFP32),
        locked_span_(std::move(locked_span)) {}

  absl::StatusOr<LockedBufferSpan<const float>> Lookup(
      int32_t token_id) const override {
    token_id = ClampTokenId(token_id);
    return locked_span_.SubSpan(static_cast<size_t>(token_id) * emb_dim_,
                                emb_dim_);
  }

 protected:
  void DecodeRow(int32_t row, int col_start, int num_cols,
                 float* dst) const override {
    std::copy_n(
        locked_span_.data() + static_cast<size_t>(row) * emb_dim_ + col_start,
        num_cols, dst);
  }

 private:
  LockedBufferSpan<const float> locked_span_;
};

// Keep the original 16-bit table mapped and convert only requested rows.
// A single-token result owns one FP32 row, including when sliced per layer.
template <Type type>
class Float16GemmaEmbeddingTable : public GemmaEmbeddingTable {
 public:
  using Element = typename NativeStorage<type>::type;

  Float16GemmaEmbeddingTable(TensorHandle tensor, int vocab_size, int emb_dim,
                             LockedBufferSpan<const Element> locked_span)
      : GemmaEmbeddingTable(std::move(tensor), vocab_size, emb_dim, type),
        locked_span_(std::move(locked_span)) {}

  absl::StatusOr<LockedBufferSpan<const float>> Lookup(
      int32_t token_id) const override {
    token_id = ClampTokenId(token_id);
    auto row = std::make_unique<float[]>(emb_dim_);
    DecodeRow(token_id, 0, emb_dim_, row.get());
    return LockedBufferSpan<const float>(std::move(row), emb_dim_);
  }

 protected:
  void DecodeRow(int32_t row, int col_start, int num_cols,
                 float* dst) const override {
    const Element* source =
        locked_span_.data() + static_cast<size_t>(row) * emb_dim_ + col_start;
    for (int col = 0; col < num_cols; ++col) {
      dst[col] = ConvertTo<Type::kFP32>(source[col]);
    }
  }

 private:
  LockedBufferSpan<const Element> locked_span_;
};

// On-the-fly dequantizing embedding table implementation for INT4 and INT8.
class QuantizedGemmaEmbeddingTable : public GemmaEmbeddingTable {
 public:
  QuantizedGemmaEmbeddingTable(
      TensorHandle tensor, int vocab_size, int emb_dim, Type type,
      LockedBufferSpan<const uint8_t> locked_span,
      std::shared_ptr<PerChannelAffineQuantization> per_channel_quant,
      std::shared_ptr<BlockwiseQuantization> blockwise_quant)
      : GemmaEmbeddingTable(std::move(tensor), vocab_size, emb_dim, type),
        locked_span_(std::move(locked_span)),
        per_channel_quant_(std::move(per_channel_quant)),
        blockwise_quant_(std::move(blockwise_quant)) {}

  absl::StatusOr<LockedBufferSpan<const float>> Lookup(
      int32_t token_id) const override {
    token_id = ClampTokenId(token_id);
    auto buffer = std::make_unique<float[]>(emb_dim_);
    DecodeRow(token_id, 0, emb_dim_, buffer.get());
    return LockedBufferSpan<const float>(std::move(buffer), emb_dim_);
  }

 protected:
  void DecodeRow(int32_t row, int col_start, int num_cols,
                 float* dst) const override {
    const uint8_t* raw_bytes = locked_span_.data();

    if (type_ == Type::kI4) {
      // 4-bit packed: 2 elements per byte.
      // Even column index -> low nibble (bits 0..3)
      // Odd column index  -> high nibble (bits 4..7)
      const size_t row_byte_offset = static_cast<size_t>(row) * (emb_dim_ / 2);

      if (blockwise_quant_ != nullptr) {
        const int block_size = blockwise_quant_->block_size;
        const int num_blocks_per_row = emb_dim_ / block_size;
        const float* row_scales = blockwise_quant_->scales.data() +
                                  static_cast<size_t>(row) * num_blocks_per_row;
        const size_t num_zp = blockwise_quant_->zero_points.size();
        const int64_t* zp_data = blockwise_quant_->zero_points.data();
        const size_t row_block_offset =
            static_cast<size_t>(row) * num_blocks_per_row;

        for (int c = 0; c < num_cols; ++c) {
          const int col = col_start + c;
          const size_t byte_idx = row_byte_offset + (col / 2);
          const uint8_t byte_val = raw_bytes[byte_idx];
          const uint8_t nibble =
              (col % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
          // Sign extend 4-bit signed [-8, 7]
          const int8_t val = (nibble & 0x08) ? static_cast<int8_t>(nibble - 16)
                                             : static_cast<int8_t>(nibble);
          const size_t block_in_row = col / block_size;
          const size_t block_idx = row_block_offset + block_in_row;
          int64_t zp = 0;
          if (num_zp == 1) {
            zp = zp_data[0];
          } else if (block_idx < num_zp) {
            zp = zp_data[block_idx];
          }
          const float scale = row_scales[block_in_row];
          dst[c] = static_cast<float>(val - zp) * scale;
        }
      } else if (per_channel_quant_ != nullptr) {
        const float scale = per_channel_quant_->scales[row];
        const size_t num_zp = per_channel_quant_->zero_points.size();
        int64_t zp = 0;
        if (num_zp == 1) {
          zp = per_channel_quant_->zero_points[0];
        } else if (static_cast<size_t>(row) < num_zp) {
          zp = per_channel_quant_->zero_points[row];
        }
        for (int c = 0; c < num_cols; ++c) {
          const int col = col_start + c;
          const size_t byte_idx = row_byte_offset + (col / 2);
          const uint8_t byte_val = raw_bytes[byte_idx];
          const uint8_t nibble =
              (col % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
          const int8_t val = (nibble & 0x08) ? static_cast<int8_t>(nibble - 16)
                                             : static_cast<int8_t>(nibble);
          dst[c] = static_cast<float>(val - zp) * scale;
        }
      }
    } else if (type_ == Type::kI8) {
      const int8_t* int8_bytes = reinterpret_cast<const int8_t*>(raw_bytes);
      const size_t row_offset = static_cast<size_t>(row) * emb_dim_ + col_start;

      if (per_channel_quant_ != nullptr) {
        const float scale = per_channel_quant_->scales[row];
        const size_t num_zp = per_channel_quant_->zero_points.size();
        int64_t zp = 0;
        if (num_zp == 1) {
          zp = per_channel_quant_->zero_points[0];
        } else if (static_cast<size_t>(row) < num_zp) {
          zp = per_channel_quant_->zero_points[row];
        }
        for (int c = 0; c < num_cols; ++c) {
          dst[c] = static_cast<float>(int8_bytes[row_offset + c] - zp) * scale;
        }
      } else if (blockwise_quant_ != nullptr) {
        const int block_size = blockwise_quant_->block_size;
        const int num_blocks_per_row = emb_dim_ / block_size;
        const float* row_scales = blockwise_quant_->scales.data() +
                                  static_cast<size_t>(row) * num_blocks_per_row;
        const size_t num_zp = blockwise_quant_->zero_points.size();
        const int64_t* zp_data = blockwise_quant_->zero_points.data();
        const size_t row_block_offset =
            static_cast<size_t>(row) * num_blocks_per_row;

        for (int c = 0; c < num_cols; ++c) {
          const int col = col_start + c;
          const size_t block_in_row = col / block_size;
          const size_t block_idx = row_block_offset + block_in_row;
          int64_t zp = 0;
          if (num_zp == 1) {
            zp = zp_data[0];
          } else if (block_idx < num_zp) {
            zp = zp_data[block_idx];
          }
          const float scale = row_scales[block_in_row];
          dst[c] = static_cast<float>(int8_bytes[row_offset + c] - zp) * scale;
        }
      }
    }
  }

  LockedBufferSpan<const uint8_t> locked_span_;
  std::shared_ptr<PerChannelAffineQuantization> per_channel_quant_;
  std::shared_ptr<BlockwiseQuantization> blockwise_quant_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<GemmaEmbeddingTable>>
GemmaEmbeddingTable::Create(TensorHandle tensor, int expected_emb_dim) {
  LRT_TENSOR_RETURN_IF_ERROR(tensor.GetStatus());
  if (tensor.GetShape().size() != 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Embedding table must be 2D, got rank ", tensor.GetShape().size()));
  }
  const int vocab_size = tensor.GetShape()[0];
  const int stored_dim = tensor.GetShape()[1];
  if (vocab_size <= 0 || stored_dim <= 0 || expected_emb_dim < 0) {
    return absl::InvalidArgumentError(
        "Embedding dimensions must be positive (expected dimension may be 0)");
  }
  const Type type = tensor.GetType();
  const auto quant = tensor.GetQuantization();
  int logical_dim = expected_emb_dim > 0 ? expected_emb_dim : stored_dim;
  std::shared_ptr<PerChannelAffineQuantization> per_channel_quant;
  std::shared_ptr<BlockwiseQuantization> blockwise_quant;

  if (quant != nullptr) {
    if (type != Type::kI4 && type != Type::kI8) {
      return absl::InvalidArgumentError(
          "Quantized embedding tables must use INT4 or INT8 storage");
    }
    if (auto pc = quant->As<const PerChannelAffineQuantization>(); pc.ok()) {
      per_channel_quant = std::make_shared<PerChannelAffineQuantization>(*pc);
    } else if (auto bw = quant->As<const BlockwiseQuantization>(); bw.ok()) {
      blockwise_quant = std::make_shared<BlockwiseQuantization>(*bw);
    } else {
      return absl::InvalidArgumentError(
          "Unsupported quantization format for embedding table");
    }
    if (blockwise_quant != nullptr) {
      const int block_size = blockwise_quant->block_size;
      if (block_size <= 0 || blockwise_quant->scales.empty() ||
          blockwise_quant->scales.size() % vocab_size != 0) {
        return absl::InvalidArgumentError(
            "Embedding block size and per-row scale counts must be positive");
      }
      // Legacy packed INT4 tensors may report the byte width instead of the
      // logical element width. Block counts disambiguate those two layouts.
      if (type == Type::kI4 && expected_emb_dim == 0) {
        const size_t blocks_per_row =
            blockwise_quant->scales.size() / vocab_size;
        if (blocks_per_row >
            static_cast<size_t>(std::numeric_limits<int>::max() / block_size)) {
          return absl::InvalidArgumentError(
              "Embedding dimension overflows int");
        }
        logical_dim = static_cast<int>(blocks_per_row) * block_size;
      }
      if (logical_dim % block_size != 0) {
        return absl::InvalidArgumentError(
            "Embedding dimension must be divisible by quantization block size");
      }
    }
    const auto& scales = per_channel_quant != nullptr
                             ? per_channel_quant->scales
                             : blockwise_quant->scales;
    const auto& zero_points = per_channel_quant != nullptr
                                  ? per_channel_quant->zero_points
                                  : blockwise_quant->zero_points;
    const int axis = per_channel_quant != nullptr
                         ? per_channel_quant->quantized_dimension
                         : blockwise_quant->quantized_dimension;
    const size_t expected_scales =
        static_cast<size_t>(vocab_size) *
        (blockwise_quant != nullptr ? logical_dim / blockwise_quant->block_size
                                    : 1);
    if (axis != 0 || scales.size() != expected_scales ||
        (zero_points.size() != 1 && zero_points.size() != expected_scales)) {
      return absl::InvalidArgumentError(
          "Embedding quantization requires axis 0, one scale per row or block, "
          "and one zero point or a matching zero-point count");
    }
    for (float scale : scales) {
      if (!std::isfinite(scale) || scale <= 0) {
        return absl::InvalidArgumentError(
            "Embedding quantization scales must be finite and positive");
      }
    }
    const int64_t min_zero = type == Type::kI4 ? -8 : -128;
    const int64_t max_zero = type == Type::kI4 ? 7 : 127;
    for (int64_t zero : zero_points) {
      if (zero < min_zero || zero > max_zero) {
        return absl::InvalidArgumentError(
            "Embedding zero point is outside the storage type's range");
      }
    }
  } else if (type != Type::kFP32 && type != Type::kFP16 &&
             type != Type::kBF16) {
    return absl::InvalidArgumentError(
        "Unquantized embedding table must be FP32, FP16, or BF16");
  }

  const bool packed_width =
      type == Type::kI4 && static_cast<int64_t>(stored_dim) * 2 == logical_dim;
  if (logical_dim != stored_dim && !packed_width) {
    return absl::InvalidArgumentError(
        "Expected embedding dimension does not match the tensor shape");
  }
  if (type == Type::kI4 && logical_dim % 2 != 0) {
    return absl::InvalidArgumentError(
        "Packed INT4 embedding rows must contain an even number of elements");
  }
  const size_t row_bytes = BufferSize(type, logical_dim);
  if (static_cast<size_t>(vocab_size) >
      std::numeric_limits<size_t>::max() / row_bytes) {
    return absl::InvalidArgumentError("Embedding byte size overflows size_t");
  }
  const size_t required_bytes = static_cast<size_t>(vocab_size) * row_bytes;
  LRT_TENSOR_ASSIGN_OR_RETURN(Buffer & buffer, tensor.GetBuffer());
  LRT_TENSOR_ASSIGN_OR_RETURN(const size_t bytes, buffer.ByteSize());
  if (bytes < required_bytes) {
    return absl::InvalidArgumentError(
        "Embedding buffer is smaller than the logical table size");
  }
  auto locked = buffer.Lock();
  if (locked.data() == nullptr || locked.size() < required_bytes) {
    return absl::InvalidArgumentError(
        "Embedding buffer could not expose all required bytes");
  }
  const size_t alignment = type == Type::kFP32 ? alignof(float)
                           : type == Type::kFP16 || type == Type::kBF16
                               ? alignof(uint16_t)
                               : 1;
  if (reinterpret_cast<uintptr_t>(locked.data()) % alignment != 0) {
    return absl::InvalidArgumentError("Embedding buffer is not aligned");
  }
  if (quant != nullptr) {
    return std::make_unique<QuantizedGemmaEmbeddingTable>(
        std::move(tensor), vocab_size, logical_dim, type,
        locked.As<const uint8_t>(), std::move(per_channel_quant),
        std::move(blockwise_quant));
  }
  if (type == Type::kBF16) {
    return std::make_unique<Float16GemmaEmbeddingTable<Type::kBF16>>(
        std::move(tensor), vocab_size, logical_dim, locked.As<const bf16_t>());
  }
  if (type == Type::kFP16) {
    return std::make_unique<Float16GemmaEmbeddingTable<Type::kFP16>>(
        std::move(tensor), vocab_size, logical_dim, locked.As<const fp16_t>());
  }
  return std::make_unique<Fp32GemmaEmbeddingTable>(
      std::move(tensor), vocab_size, logical_dim, locked.As<const float>());
}

}  // namespace litert::tensor::examples::gemma4

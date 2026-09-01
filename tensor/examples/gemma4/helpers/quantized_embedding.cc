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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
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
  if (output.size() < token_ids.size() * static_cast<size_t>(emb_dim_)) {
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
  if (emb_dim_ < num_layers * per_layer_dim) {
    return absl::InvalidArgumentError(
        "Embedding dimension smaller than num_layers * per_layer_dim");
  }
  const size_t seq_len = token_ids.size();
  for (size_t l = 0; l < num_layers; ++l) {
    if (output_per_layer[l].size() < seq_len * per_layer_dim) {
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
  if (emb_dim_ < num_layers * per_layer_dim) {
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
    return locked_span_.SubSpan(token_id * emb_dim_, emb_dim_);
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
  if (tensor.GetShape().size() != 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Embedding table must be 2D, got rank ", tensor.GetShape().size()));
  }
  const int vocab_size = tensor.GetShape()[0];
  int emb_dim = tensor.GetShape()[1];
  if (vocab_size <= 0 || emb_dim <= 0) {
    return absl::InvalidArgumentError(
        "Embedding table dimensions must be positive");
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(Buffer & buffer, tensor.GetBuffer());
  const Type type = tensor.GetType();
  auto quant = tensor.GetQuantization();

  if (quant != nullptr) {
    std::shared_ptr<PerChannelAffineQuantization> per_channel_quant;
    std::shared_ptr<BlockwiseQuantization> blockwise_quant;
    if (auto pc = quant->As<const PerChannelAffineQuantization>(); pc.ok()) {
      per_channel_quant = std::make_shared<PerChannelAffineQuantization>(*pc);
    } else if (auto bw = quant->As<const BlockwiseQuantization>(); bw.ok()) {
      blockwise_quant = std::make_shared<BlockwiseQuantization>(*bw);
    } else {
      return absl::InvalidArgumentError(
          "Unsupported quantization format for embedding table");
    }

    int logical_emb_dim = emb_dim;
    if (expected_emb_dim > 0) {
      logical_emb_dim = expected_emb_dim;
    } else if (blockwise_quant != nullptr && !blockwise_quant->scales.empty() &&
               vocab_size > 0) {
      const int num_blocks_per_row =
          static_cast<int>(blockwise_quant->scales.size() / vocab_size);
      const int computed_dim = num_blocks_per_row * blockwise_quant->block_size;
      if (computed_dim > 0) {
        logical_emb_dim = computed_dim;
      }
    } else if (type == Type::kI4) {
      logical_emb_dim = emb_dim * 2;
    }

    return std::make_unique<QuantizedGemmaEmbeddingTable>(
        std::move(tensor), vocab_size, logical_emb_dim, type,
        buffer.Lock().As<const uint8_t>(), std::move(per_channel_quant),
        std::move(blockwise_quant));
  }

  if (type != Type::kFP32) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unquantized embedding table must be FP32, got ", ToString(type)));
  }

  const int logical_emb_dim =
      (expected_emb_dim > 0) ? expected_emb_dim : emb_dim;
  return std::make_unique<Fp32GemmaEmbeddingTable>(
      std::move(tensor), vocab_size, logical_emb_dim,
      buffer.Lock().As<const float>());
}

}  // namespace litert::tensor::examples::gemma4

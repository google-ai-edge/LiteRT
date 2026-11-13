/* Copyright 2025 Google LLC.

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
#include "tensor/examples/gemma3/safetensor_loader.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/utils/safetensor_loader.h"
#include "tensor/examples/utils/safetensors.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::examples {

namespace {

size_t ElementCount(const Shape& shape) {
  size_t count = 1;
  for (int32_t dim : shape) {
    if (dim <= 0) {
      return 0;
    }
    count *= static_cast<size_t>(dim);
  }
  return count;
}

bool IsZeroCenteredNormWeight(absl::string_view name) {
  return absl::StrContains(name, "layernorm") ||
         absl::EndsWith(name, "norm.weight");
}

absl::StatusOr<std::shared_ptr<OwningCpuBuffer>> ConvertToFp32Buffer(
    absl::string_view name, const TensorHandle& tensor, bool add_unit_offset) {
  const size_t num_elements = ElementCount(tensor.GetShape());
  LRT_TENSOR_ASSIGN_OR_RETURN(Buffer & src_buffer, tensor.GetBuffer());
  LockedBufferSpan<const std::byte> locked = src_buffer.Lock();

  std::shared_ptr<OwningCpuBuffer> out =
      OwningCpuBuffer::Allocate<Type::kFP32>(num_elements);
  if (out == nullptr) {
    return absl::ResourceExhaustedError(
        absl::StrCat("Failed to allocate FP32 buffer for ", name));
  }
  float* dst = out->Span<float>().data();
  const std::byte* src = locked.data();

  switch (tensor.GetType()) {
    case Type::kBF16: {
      if (locked.size() != num_elements * sizeof(uint16_t)) {
        return absl::InvalidArgumentError(
            absl::StrCat("BF16 tensor byte size mismatch for: ", name));
      }
      for (size_t i = 0; i < num_elements; ++i) {
        uint16_t raw = 0;
        std::memcpy(&raw, src + i * sizeof(uint16_t), sizeof(raw));
        dst[i] = safetensors::bfloat16_to_float(raw);
      }
      break;
    }
    case Type::kFP16: {
      if (locked.size() != num_elements * sizeof(uint16_t)) {
        return absl::InvalidArgumentError(
            absl::StrCat("FP16 tensor byte size mismatch for: ", name));
      }
      for (size_t i = 0; i < num_elements; ++i) {
        uint16_t raw = 0;
        std::memcpy(&raw, src + i * sizeof(uint16_t), sizeof(raw));
        dst[i] = safetensors::fp16_to_float(raw);
      }
      break;
    }
    case Type::kFP32: {
      if (locked.size() != num_elements * sizeof(float)) {
        return absl::InvalidArgumentError(
            absl::StrCat("FP32 tensor byte size mismatch for: ", name));
      }
      if (num_elements > 0) {
        std::memcpy(dst, src, locked.size());
      }
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported type for FP32 conversion: ",
                       ToString(tensor.GetType())));
  }

  if (add_unit_offset) {
    for (size_t i = 0; i < num_elements; ++i) {
      dst[i] += 1.0f;
    }
  }
  return out;
}

enum class QuantParamMode {
  kScalar,
  kPerChannelDim0,
  kPerElement,
};

absl::StatusOr<QuantParamMode> DetermineQuantParamMode(
    size_t param_size, const Shape& shape, size_t num_elements,
    absl::string_view param_name) {
  if (param_size == 1) {
    return QuantParamMode::kScalar;
  }
  if (param_size == num_elements) {
    return QuantParamMode::kPerElement;
  }
  if (!shape.empty() && shape[0] > 0 &&
      param_size == static_cast<size_t>(shape[0])) {
    if (num_elements % param_size != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid ", param_name, " count for per-channel quantization"));
    }
    return QuantParamMode::kPerChannelDim0;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unsupported ", param_name, " size ", param_size, " for tensor shape"));
}

size_t QuantParamIndex(QuantParamMode mode, size_t element_index,
                       size_t channel_index) {
  switch (mode) {
    case QuantParamMode::kScalar:
      return 0;
    case QuantParamMode::kPerChannelDim0:
      return channel_index;
    case QuantParamMode::kPerElement:
      return element_index;
  }
  return 0;
}

absl::Status AttachLegacySideTensorQuantization(const SafetensorLoader& loader,
                                                absl::string_view name,
                                                TensorHandle& tensor) {
  std::string scale_name;
  for (absl::string_view suffix : {".scale", ".scales", ".weight_scales"}) {
    std::string candidate = absl::StrCat(name, suffix);
    if (loader.GetTensorInfo(candidate).ok()) {
      scale_name = std::move(candidate);
      break;
    }
  }
  if (scale_name.empty()) {
    return absl::OkStatus();
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> scales,
                              loader.ReadTensorAs<float>(scale_name));
  if (scales.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Scale tensor is empty for: ", name));
  }

  std::vector<int64_t> zero_points(1, 0);
  const std::string zp_name = absl::StrCat(name, ".zero_point");
  if (loader.GetTensorInfo(zp_name).ok()) {
    LRT_TENSOR_ASSIGN_OR_RETURN(zero_points,
                                loader.ReadTensorAs<int64_t>(zp_name));
    if (zero_points.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Zero-point tensor is empty for: ", name));
    }
  }

  // Limitation: Tensor API quantization metadata is exported as per-channel
  // along dim0 for this loader.
  tensor.SetQuantization(std::make_shared<PerChannelAffineQuantization>(
      std::move(scales), std::move(zero_points), /*quantized_dimension=*/0));
  return absl::OkStatus();
}

absl::Status DequantizeInt8ToFp32(absl::string_view name,
                                  TensorHandle& tensor) {
  const Shape& shape = tensor.GetShape();
  const size_t num_elements = ElementCount(shape);
  LRT_TENSOR_ASSIGN_OR_RETURN(Buffer & src_buffer, tensor.GetBuffer());
  LockedBufferSpan<const std::byte> locked = src_buffer.Lock();
  if (locked.size() != num_elements * sizeof(int8_t)) {
    return absl::InvalidArgumentError(
        absl::StrCat("I8 tensor byte size mismatch for: ", name));
  }

  std::shared_ptr<OwningCpuBuffer> out =
      OwningCpuBuffer::Allocate<Type::kFP32>(num_elements);
  if (out == nullptr) {
    return absl::ResourceExhaustedError(
        absl::StrCat("Failed to allocate FP32 buffer for ", name));
  }
  float* dst = out->Span<float>().data();
  const int8_t* quantized_data = reinterpret_cast<const int8_t*>(locked.data());

  const std::shared_ptr<const Quantization> quant = tensor.GetQuantization();
  if (auto pcq = quant->As<PerChannelAffineQuantization>(); pcq.ok()) {
    const std::vector<float>& scales = pcq->scales;
    if (scales.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Scale tensor is empty for: ", name));
    }
    if (pcq->HasFloatZeroPoints()) {
      return absl::UnimplementedError(absl::StrCat(
          "Dequantization with float zero points is not supported for: ",
          name));
    }
    const std::vector<int64_t> default_zp(1, 0);
    const std::vector<int64_t>& zero_points = pcq->IntegerZeroPoints()->empty()
                                                  ? default_zp
                                                  : *pcq->IntegerZeroPoints();

    size_t channel_size = 0;
    if (!shape.empty() && shape[0] > 0) {
      channel_size = num_elements / static_cast<size_t>(shape[0]);
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(
        const QuantParamMode scale_mode,
        DetermineQuantParamMode(scales.size(), shape, num_elements, "scale"));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        const QuantParamMode zp_mode,
        DetermineQuantParamMode(zero_points.size(), shape, num_elements,
                                "zero_point"));

    for (size_t i = 0; i < num_elements; ++i) {
      const size_t channel_index = channel_size == 0 ? 0 : i / channel_size;
      const size_t scale_index =
          QuantParamIndex(scale_mode, i, channel_index);
      const size_t zp_index = QuantParamIndex(zp_mode, i, channel_index);
      const int32_t q = static_cast<int32_t>(quantized_data[i]);
      const int32_t zp = static_cast<int32_t>(zero_points[zp_index]);
      dst[i] = static_cast<float>(q - zp) * scales[scale_index];
    }
  } else if (auto bq = quant->As<BlockwiseQuantization>(); bq.ok()) {
    if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0 ||
        bq->block_size <= 0 || shape[1] % bq->block_size != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid shape or block_size for blockwise dequantization of ",
          name));
    }
    const size_t rows = static_cast<size_t>(shape[0]);
    const size_t cols = static_cast<size_t>(shape[1]);
    const size_t block_size = static_cast<size_t>(bq->block_size);
    const size_t blocks_per_row = cols / block_size;
    const size_t total_blocks = rows * blocks_per_row;
    if (bq->scales.size() != total_blocks) {
      return absl::InvalidArgumentError(
          absl::StrCat("Blockwise scale count mismatch for: ", name));
    }
    if (bq->HasFloatZeroPoints()) {
      return absl::UnimplementedError(absl::StrCat(
          "Dequantization with float zero points is not supported for: ",
          name));
    }
    const std::vector<int64_t>& zero_points = *bq->IntegerZeroPoints();
    if (!zero_points.empty() && zero_points.size() != 1 &&
        zero_points.size() != total_blocks) {
      return absl::InvalidArgumentError(
          absl::StrCat("Blockwise zero-point count mismatch for: ", name));
    }
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < cols; ++c) {
        const size_t idx = r * cols + c;
        const size_t block_idx = r * blocks_per_row + (c / block_size);
        const int64_t zp64 =
            zero_points.empty()
                ? 0
                : (zero_points.size() == 1 ? zero_points[0]
                                           : zero_points[block_idx]);
        const int32_t q = static_cast<int32_t>(quantized_data[idx]);
        const int32_t zp = static_cast<int32_t>(zp64);
        dst[idx] = static_cast<float>(q - zp) * bq->scales[block_idx];
      }
    }
  } else {
    return absl::UnimplementedError(absl::StrCat(
        "Unsupported quantization metadata for FP32 dequantization of ", name));
  }

  tensor.SetType(Type::kFP32);
  tensor.SetBuffer(std::move(out));
  tensor.SetQuantization(nullptr);
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<TensorHandle> LoadGemma3Tensor(
    const SafetensorLoader& loader, absl::string_view name,
    Gemma3QuantizedLoadMode quantized_load_mode) {
  LRT_TENSOR_ASSIGN_OR_RETURN(TensorHandle tensor, loader.LoadTensor(name));
  const bool is_layernorm = IsZeroCenteredNormWeight(name);

  switch (tensor.GetType()) {
    case Type::kBF16:
    case Type::kFP16: {
      LRT_TENSOR_ASSIGN_OR_RETURN(
          std::shared_ptr<OwningCpuBuffer> fp32_buffer,
          ConvertToFp32Buffer(name, tensor, is_layernorm));
      tensor.SetType(Type::kFP32);
      tensor.SetBuffer(std::move(fp32_buffer));
      return tensor;
    }
    case Type::kFP32: {
      if (is_layernorm) {
        LRT_TENSOR_ASSIGN_OR_RETURN(
            std::shared_ptr<OwningCpuBuffer> fp32_buffer,
            ConvertToFp32Buffer(name, tensor, /*add_unit_offset=*/true));
        tensor.SetBuffer(std::move(fp32_buffer));
      }
      return tensor;
    }
    case Type::kI8: {
      if (tensor.GetQuantization() == nullptr) {
        LRT_TENSOR_RETURN_IF_ERROR(
            AttachLegacySideTensorQuantization(loader, name, tensor));
      }
      if (tensor.GetQuantization() == nullptr ||
          quantized_load_mode == Gemma3QuantizedLoadMode::kPreserveQuantized) {
        return tensor;
      }
      LRT_TENSOR_RETURN_IF_ERROR(DequantizeInt8ToFp32(name, tensor));
      return tensor;
    }
    case Type::kI4:
    case Type::kI2: {
      if (quantized_load_mode == Gemma3QuantizedLoadMode::kPreserveQuantized) {
        return tensor;
      }
      return absl::UnimplementedError(
          absl::StrCat("Dequantizing ", ToString(tensor.GetType()),
                       " tensor to FP32 is not supported for: ", name));
    }
    default:
      return tensor;
  }
}

absl::StatusOr<absl::flat_hash_map<std::string, TensorHandle>>
LoadAllGemma3Tensors(const SafetensorLoader& loader,
                     Gemma3QuantizedLoadMode quantized_load_mode) {
  absl::flat_hash_map<std::string, TensorHandle> tensors;
  for (const std::string& name : loader.GetTensorNames()) {
    absl::StatusOr<TensorHandle> tensor_or =
        LoadGemma3Tensor(loader, name, quantized_load_mode);
    if (!tensor_or.ok()) {
      ABSL_LOG(WARNING) << "Failed to load tensor " << name << ": "
                        << tensor_or.status();
      continue;
    }
    tensors[name] = std::move(*tensor_or);
  }
  return tensors;
}

absl::StatusOr<absl::flat_hash_map<std::string, TensorHandle>>
LoadGemma3WeightsWithMapping(
    const SafetensorLoader& loader,
    const absl::flat_hash_map<std::string, std::string>& name_mapping,
    Gemma3QuantizedLoadMode quantized_load_mode) {
  absl::flat_hash_map<std::string, TensorHandle> tensors;
  for (const auto& [hf_name, model_name] : name_mapping) {
    absl::StatusOr<TensorHandle> tensor_or =
        LoadGemma3Tensor(loader, hf_name, quantized_load_mode);
    if (!tensor_or.ok()) {
      ABSL_LOG(WARNING) << "Failed to load tensor " << hf_name << ": "
                        << tensor_or.status();
      continue;
    }
    tensor_or->SetName(model_name);
    tensors[model_name] = std::move(*tensor_or);
  }
  return tensors;
}

absl::flat_hash_map<std::string, std::string> GetGemma3WeightMapping(
    int n_layers) {
  absl::flat_hash_map<std::string, std::string> mapping;

  // Embedding
  mapping["model.embed_tokens.weight"] = "model.embed_tokens.weight";

  // Final norm
  mapping["model.norm.weight"] = "model.norm.weight";

  // Per-layer weights
  for (int i = 0; i < n_layers; ++i) {
    std::string hf_prefix = absl::StrCat("model.layers.", i);
    std::string model_prefix = hf_prefix;

    // Attention weights
    mapping[absl::StrCat(hf_prefix, ".self_attn.q_proj.weight")] =
        absl::StrCat(model_prefix, ".self_attn.q_proj.weight");
    mapping[absl::StrCat(hf_prefix, ".self_attn.k_proj.weight")] =
        absl::StrCat(model_prefix, ".self_attn.k_proj.weight");
    mapping[absl::StrCat(hf_prefix, ".self_attn.v_proj.weight")] =
        absl::StrCat(model_prefix, ".self_attn.v_proj.weight");
    mapping[absl::StrCat(hf_prefix, ".self_attn.o_proj.weight")] =
        absl::StrCat(model_prefix, ".self_attn.o_proj.weight");

    // QK normalization (Gemma3 specific)
    mapping[absl::StrCat(hf_prefix, ".self_attn.q_norm.weight")] =
        absl::StrCat(model_prefix, ".self_attn.q_norm.weight");
    mapping[absl::StrCat(hf_prefix, ".self_attn.k_norm.weight")] =
        absl::StrCat(model_prefix, ".self_attn.k_norm.weight");

    // MLP weights
    mapping[absl::StrCat(hf_prefix, ".mlp.gate_proj.weight")] =
        absl::StrCat(model_prefix, ".mlp.gate_proj.weight");
    mapping[absl::StrCat(hf_prefix, ".mlp.up_proj.weight")] =
        absl::StrCat(model_prefix, ".mlp.up_proj.weight");
    mapping[absl::StrCat(hf_prefix, ".mlp.down_proj.weight")] =
        absl::StrCat(model_prefix, ".mlp.down_proj.weight");

    // Layer norms (Gemma3 has 4 per layer)
    mapping[absl::StrCat(hf_prefix, ".input_layernorm.weight")] =
        absl::StrCat(model_prefix, ".input_layernorm.weight");
    mapping[absl::StrCat(hf_prefix, ".post_attention_layernorm.weight")] =
        absl::StrCat(model_prefix, ".post_attention_layernorm.weight");
    mapping[absl::StrCat(hf_prefix, ".pre_feedforward_layernorm.weight")] =
        absl::StrCat(model_prefix, ".pre_feedforward_layernorm.weight");
    mapping[absl::StrCat(hf_prefix, ".post_feedforward_layernorm.weight")] =
        absl::StrCat(model_prefix, ".post_feedforward_layernorm.weight");
  }

  return mapping;
}

}  // namespace litert::tensor::examples

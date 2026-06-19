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

#include "tensor/examples/gemma3/tflite_loader.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tflite/model_builder.h"
#include "tflite/schema/schema_generated.h"

namespace litert::tensor::examples {

namespace {

std::shared_ptr<Buffer> MakeMappedBuffer(const std::shared_ptr<void>& owner,
                                         const std::byte* data, size_t bytes) {
  return std::shared_ptr<Buffer>(new SpanCpuBuffer(data, bytes),
                                 [owner](Buffer* buffer) {
                                   static_cast<void>(owner);
                                   delete buffer;
                                 });
}

absl::StatusOr<Type> TfliteDtypeToType(tflite::TensorType dtype) {
  switch (dtype) {
    case tflite::TensorType_FLOAT32:
      return Type::kFP32;
    case tflite::TensorType_FLOAT16:
      return Type::kFP16;
    case tflite::TensorType_INT32:
      return Type::kI32;
    case tflite::TensorType_INT8:
      return Type::kI8;
    case tflite::TensorType_UINT8:
      return Type::kU8;
    case tflite::TensorType_INT64:
      return Type::kI64;
    case tflite::TensorType_BOOL:
      return Type::kBOOL;
    case tflite::TensorType_INT16:
      return Type::kI16;
    case tflite::TensorType_INT4:
      return Type::kI4;
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported TFLite dtype: ", tflite::EnumNameTensorType(dtype)));
  }
}

absl::StatusOr<size_t> NumElements(const std::vector<int64_t>& shape) {
  size_t num_elements = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Negative shape dimension: ", dim));
    }
    if (dim == 0) {
      return static_cast<size_t>(0);
    }
    if (num_elements >
        std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      return absl::InvalidArgumentError(
          "Tensor shape overflows total element count");
    }
    num_elements *= static_cast<size_t>(dim);
  }
  return num_elements;
}

}  // namespace

absl::StatusOr<TfliteLoader> TfliteLoader::Load(const std::string& path) {
  TfliteLoader loader;
  loader.model_ = tflite::FlatBufferModel::BuildFromFile(path.c_str());
  if (!loader.model_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to load TFLite model from: ", path));
  }

  const tflite::Model* model = loader.model_->GetModel();
  if (!model) {
    return absl::InvalidArgumentError("Model is null");
  }

  const auto* subgraphs = model->subgraphs();
  if (!subgraphs) {
    return absl::InvalidArgumentError("No subgraphs in model");
  }

  const auto* buffers = model->buffers();
  if (!buffers) {
    return absl::InvalidArgumentError("No buffers in model");
  }

  // Iterate over all subgraphs and tensors to collect metadata.
  for (int sg_idx = 0; sg_idx < subgraphs->size(); ++sg_idx) {
    const auto* subgraph = subgraphs->Get(sg_idx);
    if (!subgraph || !subgraph->tensors()) continue;

    for (int t_idx = 0; t_idx < subgraph->tensors()->size(); ++t_idx) {
      const auto* tensor = subgraph->tensors()->Get(t_idx);
      if (!tensor) continue;

      // Check if tensor has a valid buffer.
      uint32_t buffer_idx = tensor->buffer();
      if (buffer_idx >= buffers->size()) continue;

      const auto* buffer = buffers->Get(buffer_idx);
      if (!buffer || !buffer->data() || buffer->data()->empty()) {
        // Skip tensors without data (e.g. activation tensors).
        continue;
      }

      std::string name = tensor->name() ? tensor->name()->str() : "";
      if (name.empty()) {
        name = absl::StrCat("subgraph_", sg_idx, "_tensor_", t_idx);
      }

      if (loader.tensor_infos_.contains(name)) {
        // Handle duplicate names if any, maybe append subgraph index.
        name = absl::StrCat("subgraph_", sg_idx, "_", name);
      }

      TfliteTensorInfo info;
      info.name = name;

      auto type_or = TfliteDtypeToType(tensor->type());
      if (!type_or.ok()) {
        ABSL_LOG(WARNING) << "Skipping tensor " << name << ": "
                          << type_or.status().message();
        continue;
      }
      info.type = *type_or;

      if (tensor->shape()) {
        info.shape.assign(tensor->shape()->begin(), tensor->shape()->end());
      }

      info.data = reinterpret_cast<const std::byte*>(buffer->data()->data());
      info.data_size = buffer->data()->size();
      info.model_keep_alive = loader.model_;

      // Extract quantization parameters
      const auto* quant = tensor->quantization();
      if (quant) {
        std::vector<float> scales;
        if (quant->scale()) {
          scales.assign(quant->scale()->begin(), quant->scale()->end());
        }
        std::vector<int64_t> zero_points;
        if (quant->zero_point()) {
          zero_points.assign(quant->zero_point()->begin(),
                             quant->zero_point()->end());
        }

        if (!scales.empty()) {
          int quantized_dimension = quant->quantized_dimension();
          info.quantization = std::make_shared<PerChannelAffineQuantization>(
              std::move(scales), std::move(zero_points), quantized_dimension);
        }
      }

      loader.tensor_infos_[name] = std::move(info);
    }
  }

  ABSL_LOG(INFO) << "Loaded TFLite model: " << path << " with "
                 << loader.tensor_infos_.size() << " weight tensors";
  return loader;
}

std::vector<std::string> TfliteLoader::GetTensorNames() const {
  std::vector<std::string> names;
  names.reserve(tensor_infos_.size());
  for (const auto& [name, _] : tensor_infos_) {
    names.push_back(name);
  }
  return names;
}

absl::StatusOr<TfliteTensorInfo> TfliteLoader::GetTensorInfo(
    const std::string& name) const {
  auto it = tensor_infos_.find(name);
  if (it == tensor_infos_.end()) {
    return absl::NotFoundError(absl::StrCat("Tensor not found: ", name));
  }
  return it->second;
}

absl::StatusOr<TensorHandle> TfliteLoader::LoadTensor(
    const std::string& name, QuantizedLoadMode quantized_load_mode) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(TfliteTensorInfo info, GetTensorInfo(name));

  std::shared_ptr<Buffer> buffer;
  std::shared_ptr<Quantization> quantization = info.quantization;
  Type type = info.type;

  bool is_quantized = (quantization != nullptr);

  if (is_quantized &&
      quantized_load_mode == QuantizedLoadMode::kDequantizeToFp32) {
    // Perform dequantization to FP32
    LRT_TENSOR_ASSIGN_OR_RETURN(const size_t num_elements,
                                NumElements(info.shape));
    std::vector<float> dequantized_values(num_elements);

    // Get scales and zero points from quantization
    auto pc_quant =
        std::dynamic_pointer_cast<PerChannelAffineQuantization>(quantization);
    if (!pc_quant) {
      return absl::InternalError(
          "Quantization is not PerChannelAffineQuantization");
    }
    const auto& scales = pc_quant->scales;
    const auto& zero_points = pc_quant->zero_points;
    int quantized_dimension = pc_quant->quantized_dimension;

    if (scales.empty()) {
      return absl::InvalidArgumentError("Quantization scales are empty");
    }

    if (type == Type::kI8) {
      const int8_t* quantized_data = reinterpret_cast<const int8_t*>(info.data);

      if (scales.size() == 1) {
        // Per-tensor
        float scale = scales[0];
        int64_t zp = zero_points.empty() ? 0 : zero_points[0];
        for (size_t i = 0; i < num_elements; ++i) {
          dequantized_values[i] =
              static_cast<float>(quantized_data[i] - zp) * scale;
        }
      } else {
        // Per-channel
        if (info.shape.empty()) {
          return absl::InvalidArgumentError(
              "Shape is empty for per-channel quantization");
        }
        int quant_dim = quantized_dimension;
        if (quant_dim < 0 || quant_dim >= info.shape.size()) {
          return absl::InvalidArgumentError("Invalid quantized dimension");
        }

        size_t num_channels = info.shape[quant_dim];
        if (scales.size() != num_channels) {
          return absl::InvalidArgumentError(
              "Number of scales does not match number of channels");
        }

        // Calculate channel size and stride
        size_t channel_stride = 1;
        for (size_t i = quant_dim + 1; i < info.shape.size(); ++i) {
          channel_stride *= info.shape[i];
        }

        for (size_t i = 0; i < num_elements; ++i) {
          size_t channel_idx = (i / channel_stride) % num_channels;
          float scale = scales[channel_idx];
          int64_t zp = zero_points.empty() ? 0 : zero_points[channel_idx];
          dequantized_values[i] =
              static_cast<float>(quantized_data[i] - zp) * scale;
        }
      }
      buffer = OwningCpuBuffer::Copy<Type::kFP32>(dequantized_values);
      type = Type::kFP32;
      quantization = nullptr;  // No longer quantized
    } else if (type == Type::kI4) {
      const uint8_t* quantized_data =
          reinterpret_cast<const uint8_t*>(info.data);
      if (scales.size() == 1) {
        // Per-tensor
        float scale = scales[0];
        int64_t zp = zero_points.empty() ? 0 : zero_points[0];
        for (size_t i = 0; i < num_elements; ++i) {
          size_t byte_idx = i / 2;
          size_t bit_shift = (i % 2) * 4;
          int8_t q = (quantized_data[byte_idx] >> bit_shift) & 0x0F;
          if (q & 0x08) {
            q |= 0xF0;  // Sign extend
          }
          dequantized_values[i] = static_cast<float>(q - zp) * scale;
        }
      } else {
        // Per-channel
        if (info.shape.empty()) {
          return absl::InvalidArgumentError(
              "Shape is empty for per-channel quantization");
        }
        int quant_dim = quantized_dimension;
        if (quant_dim < 0 || quant_dim >= info.shape.size()) {
          return absl::InvalidArgumentError("Invalid quantized dimension");
        }

        size_t num_channels = info.shape[quant_dim];
        if (scales.size() != num_channels) {
          return absl::InvalidArgumentError(
              "Number of scales does not match number of channels");
        }

        size_t channel_stride = 1;
        for (size_t i = quant_dim + 1; i < info.shape.size(); ++i) {
          channel_stride *= info.shape[i];
        }

        for (size_t i = 0; i < num_elements; ++i) {
          size_t byte_idx = i / 2;
          size_t bit_shift = (i % 2) * 4;
          int8_t q = (quantized_data[byte_idx] >> bit_shift) & 0x0F;
          if (q & 0x08) {
            q |= 0xF0;  // Sign extend
          }
          size_t channel_idx = (i / channel_stride) % num_channels;
          float scale = scales[channel_idx];
          int64_t zp = zero_points.empty() ? 0 : zero_points[channel_idx];
          dequantized_values[i] = static_cast<float>(q - zp) * scale;
        }
      }
      buffer = OwningCpuBuffer::Copy<Type::kFP32>(dequantized_values);
      type = Type::kFP32;
      quantization = nullptr;  // No longer quantized
    } else {
      return absl::UnimplementedError(absl::StrCat(
          "Dequantization from ", ToString(type), " not implemented yet"));
    }
  } else {
    // Wrap data in SpanCpuBuffer
    buffer = MakeMappedBuffer(info.model_keep_alive, info.data, info.data_size);
  }

  return TensorHandle(
      {.name = name,
       .type = type,
       .shape = std::vector<int>(info.shape.begin(), info.shape.end()),
       .buffer = buffer,
       .quantization = quantization});
}

absl::StatusOr<absl::flat_hash_map<std::string, TensorHandle>>
TfliteLoader::LoadAllTensors(QuantizedLoadMode quantized_load_mode) const {
  absl::flat_hash_map<std::string, TensorHandle> tensors;
  for (const auto& [name, info] : tensor_infos_) {
    auto tensor_or = LoadTensor(name, quantized_load_mode);
    if (!tensor_or.ok()) {
      ABSL_LOG(WARNING) << "Failed to load tensor " << name << ": "
                        << tensor_or.status();
      continue;
    }
    tensors[name] = std::move(*tensor_or);
  }
  return tensors;
}

absl::StatusOr<TensorHandle> TfliteLoader::LoadTensorWithSlice(
    const std::string& name, int start_row, int end_row,
    QuantizedLoadMode quantized_load_mode) const {
  LRT_TENSOR_ASSIGN_OR_RETURN(TfliteTensorInfo info, GetTensorInfo(name));

  if (info.shape.size() < 2) {
    return absl::InvalidArgumentError(
        "Slicing is only supported for 2D+ tensors");
  }

  int num_rows = info.shape[0];
  if (start_row < 0 || end_row > num_rows || start_row >= end_row) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid slice range [", start_row, ", ", end_row,
                     "] for tensor rows ", num_rows));
  }

  std::shared_ptr<Buffer> buffer;
  std::shared_ptr<Quantization> quantization = info.quantization;
  Type type = info.type;

  bool is_quantized = (quantization != nullptr);

  // Calculate slice shape
  std::vector<int64_t> slice_shape = info.shape;
  slice_shape[0] = end_row - start_row;

  LRT_TENSOR_ASSIGN_OR_RETURN(const size_t total_elements,
                              NumElements(info.shape));
  LRT_TENSOR_ASSIGN_OR_RETURN(const size_t slice_elements,
                              NumElements(slice_shape));

  size_t row_elements = total_elements / num_rows;

  // Slice quantization first if per-channel
  if (quantization) {
    auto pc_quant =
        std::dynamic_pointer_cast<PerChannelAffineQuantization>(quantization);
    if (pc_quant) {
      if (pc_quant->scales.size() > 1) {
        if (pc_quant->quantized_dimension == 0) {
          std::vector<float> sliced_scales(pc_quant->scales.begin() + start_row,
                                           pc_quant->scales.begin() + end_row);
          std::vector<int64_t> sliced_zero_points;
          if (!pc_quant->zero_points.empty()) {
            sliced_zero_points.assign(pc_quant->zero_points.begin() + start_row,
                                      pc_quant->zero_points.begin() + end_row);
          }
          quantization = std::make_shared<PerChannelAffineQuantization>(
              std::move(sliced_scales), std::move(sliced_zero_points), 0);
        } else {
          return absl::UnimplementedError(
              "Slicing quantization along non-zero dimension is not supported");
        }
      }
    }
  }

  size_t start_element = start_row * row_elements;
  size_t num_elements_to_load = (end_row - start_row) * row_elements;
  size_t start_offset_bytes = 0;
  size_t slice_size_bytes = 0;

  if (type == Type::kI4) {
    start_offset_bytes = (start_element * 4) / 8;
    slice_size_bytes = (num_elements_to_load * 4) / 8;
  } else {
    size_t element_size = BufferSize(type, 1);
    start_offset_bytes = start_element * element_size;
    slice_size_bytes = num_elements_to_load * element_size;
  }

  if (is_quantized &&
      quantized_load_mode == QuantizedLoadMode::kDequantizeToFp32) {
    // Perform dequantization to FP32 for the slice
    std::vector<float> dequantized_values(slice_elements);

    auto pc_quant =
        std::dynamic_pointer_cast<PerChannelAffineQuantization>(quantization);
    if (!pc_quant) {
      return absl::InternalError(
          "Quantization is not PerChannelAffineQuantization");
    }
    const auto& scales = pc_quant->scales;
    const auto& zero_points = pc_quant->zero_points;

    if (scales.empty()) {
      return absl::InvalidArgumentError("Quantization scales are empty");
    }

    if (type == Type::kI8) {
      const int8_t* quantized_data =
          reinterpret_cast<const int8_t*>(info.data) + start_offset_bytes;
      for (size_t i = 0; i < slice_elements; ++i) {
        size_t row_idx = i / row_elements;
        float scale = scales[scales.size() == 1 ? 0 : row_idx];
        int64_t zp = zero_points.empty()
                         ? 0
                         : zero_points[zero_points.size() == 1 ? 0 : row_idx];
        dequantized_values[i] =
            static_cast<float>(quantized_data[i] - zp) * scale;
      }
      buffer = OwningCpuBuffer::Copy<Type::kFP32>(dequantized_values);
      type = Type::kFP32;
      quantization = nullptr;  // No longer quantized
    } else if (type == Type::kI4) {
      const uint8_t* quantized_data =
          reinterpret_cast<const uint8_t*>(info.data) + start_offset_bytes;
      for (size_t i = 0; i < slice_elements; ++i) {
        size_t byte_idx = i / 2;
        size_t bit_shift = (i % 2) * 4;
        int8_t q = (quantized_data[byte_idx] >> bit_shift) & 0x0F;
        if (q & 0x08) {
          q |= 0xF0;  // Sign extend
        }
        size_t row_idx = i / row_elements;
        float scale = scales[scales.size() == 1 ? 0 : row_idx];
        int64_t zp = zero_points.empty()
                         ? 0
                         : zero_points[zero_points.size() == 1 ? 0 : row_idx];
        dequantized_values[i] = static_cast<float>(q - zp) * scale;
      }
      buffer = OwningCpuBuffer::Copy<Type::kFP32>(dequantized_values);
      type = Type::kFP32;
      quantization = nullptr;  // No longer quantized
    } else {
      return absl::UnimplementedError(absl::StrCat(
          "Dequantization from ", ToString(type), " not implemented yet"));
    }
  } else {
    // Wrap data in SpanCpuBuffer
    buffer = MakeMappedBuffer(info.model_keep_alive,
                              info.data + start_offset_bytes, slice_size_bytes);
  }

  return TensorHandle(
      {.name = name,
       .type = type,
       .shape = std::vector<int>(slice_shape.begin(), slice_shape.end()),
       .buffer = buffer,
       .quantization = quantization});
}

absl::StatusOr<absl::flat_hash_map<std::string, TensorHandle>>
TfliteLoader::LoadWeightsWithMapping(
    const TfliteWeightMapping& name_mapping,
    QuantizedLoadMode quantized_load_mode) const {
  absl::flat_hash_map<std::string, TensorHandle> tensors;
  for (const auto& [model_name, entry] : name_mapping) {
    absl::StatusOr<TensorHandle> tensor_or;
    if (entry.slice_range.empty()) {
      tensor_or = LoadTensor(entry.tflite_tensor_name, quantized_load_mode);
    } else {
      if (entry.slice_range.size() != 2) {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid slice range size for ", model_name,
                         ". Expected 2, got ", entry.slice_range.size()));
      }
      tensor_or =
          LoadTensorWithSlice(entry.tflite_tensor_name, entry.slice_range[0],
                              entry.slice_range[1], quantized_load_mode);
    }

    if (!tensor_or.ok()) {
      ABSL_LOG(WARNING) << "Failed to load tensor " << entry.tflite_tensor_name
                        << " for " << model_name << ": " << tensor_or.status();
      continue;
    }
    tensor_or->SetName(model_name);
    tensors[model_name] = std::move(*tensor_or);
  }
  return tensors;
}

TfliteWeightMapping GetGemma3TfliteWeightMapping(int n_layers) {
  volatile int dummy_loader_recompile_token = 33333;
  static_cast<void>(dummy_loader_recompile_token);
  TfliteWeightMapping mapping;

  // Embedding (vocab_size=262144, emb_dim=1152)
  mapping["model.embed_tokens.weight"] = {
      "XlaCallModule/ReadVariableOp_287;StatefulPartitionedCall", {}};

  // Final norm is index 140!
  mapping["model.norm.weight"] = {"arith.constant140", {}};

  auto get_name = [](int idx) {
    if (idx == 0) return std::string("arith.constant");
    return absl::StrCat("arith.constant", idx);
  };

  for (int i = 0; i < n_layers; ++i) {
    std::string prefix = absl::StrCat("model.layers.", i);

    // Layernorms (decreasing from 296)
    mapping[absl::StrCat(prefix, ".input_layernorm.weight")] = {
        get_name(296 - 6 * i), {}};
    mapping[absl::StrCat(prefix, ".post_attention_layernorm.weight")] = {
        get_name(293 - 6 * i), {}};
    mapping[absl::StrCat(prefix, ".pre_feedforward_layernorm.weight")] = {
        get_name(292 - 6 * i), {}};
    mapping[absl::StrCat(prefix, ".post_feedforward_layernorm.weight")] = {
        get_name(291 - 6 * i), {}};
    mapping[absl::StrCat(prefix, ".self_attn.q_norm.weight")] = {
        get_name(295 - 6 * i), {}};
    mapping[absl::StrCat(prefix, ".self_attn.k_norm.weight")] = {
        get_name(294 - 6 * i), {}};

    // 2D weights (decreasing offsets, increasing within block)
    std::string down_name = get_name(125 - 5 * i);
    std::string up_name = get_name(126 - 5 * i);
    std::string gate_name = get_name(127 - 5 * i);
    std::string o_name = get_name(128 - 5 * i);
    std::string qkv_name = get_name(129 - 5 * i);

    // Fused QKV splitting (Q: 1024, K: 256, V: 256)
    mapping[absl::StrCat(prefix, ".self_attn.q_proj.weight")] = {qkv_name,
                                                                 {0, 1024}};
    mapping[absl::StrCat(prefix, ".self_attn.k_proj.weight")] = {qkv_name,
                                                                 {1024, 1280}};
    mapping[absl::StrCat(prefix, ".self_attn.v_proj.weight")] = {qkv_name,
                                                                 {1280, 1536}};

    // MLP
    mapping[absl::StrCat(prefix, ".mlp.down_proj.weight")] = {down_name, {}};
    mapping[absl::StrCat(prefix, ".mlp.gate_proj.weight")] = {gate_name, {}};
    mapping[absl::StrCat(prefix, ".mlp.up_proj.weight")] = {up_name, {}};

    // Out proj
    mapping[absl::StrCat(prefix, ".self_attn.o_proj.weight")] = {o_name, {}};
  }

  return mapping;
}

}  // namespace litert::tensor::examples

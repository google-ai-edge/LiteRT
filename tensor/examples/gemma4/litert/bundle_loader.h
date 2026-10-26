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

// Reads validated published Gemma4 E2B/E4B weight bundles.
#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_BUNDLE_LOADER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_BUNDLE_LOADER_H_

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <iosfwd>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/helpers/quantized_embedding.h"
#include "tensor/examples/gemma4/litert/kv_cache.h"
#include "tensor/examples/utils/minijson.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::examples::gemma4::cpu {
struct LoadedTensors {
  absl::flat_hash_map<std::string, TensorHandle> weights_handle;
  std::unique_ptr<GemmaEmbeddingTable> token_embedding;
  std::unique_ptr<GemmaEmbeddingTable> emb_per_layer_table;
};
namespace published_bundle_detail {

using Object = minijson::object;
using Array = minijson::array;
inline absl::Status Invalid(absl::string_view reason) {
  return absl::InvalidArgumentError(
      absl::StrCat("Published bundle: ", reason));
}
template <class T>
absl::StatusOr<T> Field(const Object& object, absl::string_view name) {
  minijson::value value;
  if (!object.at(std::string(name), &value) || value.as<T>() == nullptr) {
    return Invalid(
        absl::StrCat("missing or incorrectly typed field ", name));
  }
  return *value.as<T>();
}
inline absl::StatusOr<int64_t> Integer(const minijson::value& value) {
  const minijson::number* number = value.as<minijson::number>();
  // JSON numbers are doubles. Only exact, bounded integers are accepted.
  if (number == nullptr || !std::isfinite(*number) || *number < 0 ||
      *number > 9007199254740991.0 || std::floor(*number) != *number) {
    return Invalid("expected exact nonnegative integer");
  }
  return static_cast<int64_t>(*number);
}
inline absl::StatusOr<int64_t> Integer(const Object& object,
                                       absl::string_view name) {
  minijson::value value;
  if (!object.at(std::string(name), &value)) {
    return Invalid(absl::StrCat("missing integer ", name));
  }
  return Integer(value);
}
inline absl::StatusOr<Shape> ReadShape(const Object& object,
                                       absl::string_view name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(Array dimensions, Field<Array>(object, name));
  if (dimensions.size() > 8) {
    return Invalid("unsupported tensor rank");
  }
  Shape shape;
  for (const minijson::value& dimension : dimensions) {
    LRT_TENSOR_ASSIGN_OR_RETURN(int64_t n, Integer(dimension));
    if (n == 0 || n > std::numeric_limits<int>::max()) {
      return Invalid("nonpositive or oversized shape dimension");
    }
    shape.push_back(static_cast<int>(n));
  }
  return shape;
}
inline absl::StatusOr<size_t> Elements(const Shape& shape) {
  size_t count = 1;
  for (int dimension : shape) {
    if (dimension <= 0 ||
        count > std::numeric_limits<size_t>::max() / dimension) {
      return Invalid("invalid or overflowing shape");
    }
    count *= dimension;
  }
  return count;
}
inline absl::Status CheckConfig(const Config& c) {
  const Config e = c.num_kv_heads == 2 ? Config::E4B() : Config::E2B();
  if (c.vocab_size != e.vocab_size || c.embed_dim != e.embed_dim ||
      c.hidden_dim != e.hidden_dim || c.head_dim != e.head_dim ||
      c.num_heads != e.num_heads || c.num_kv_heads != e.num_kv_heads ||
      c.num_layers != e.num_layers || c.global_key_size != e.global_key_size ||
      c.per_layer_input_dim != e.per_layer_input_dim ||
      c.attention_pattern_size != e.attention_pattern_size ||
      c.frac_shared_layers != e.frac_shared_layers || !c.share_global ||
      !c.share_local || !c.use_post_attn_norm || !c.use_post_ffw_norm ||
      c.final_logit_softcap != e.final_logit_softcap ||
      c.rms_norm_eps != e.rms_norm_eps || c.attn_logits_soft_cap.has_value() ||
      c.sliding_window_size != e.sliding_window_size ||
      c.local_base_frequency != e.local_base_frequency ||
      c.global_base_frequency != e.global_base_frequency ||
      c.local_rope_proportion != e.local_rope_proportion ||
      c.global_rope_proportion != e.global_rope_proportion) {
    return Invalid("loader only supports the published E2B/E4B configurations");
  }
  const uint16_t endian = 1;
  if (*reinterpret_cast<const uint8_t*>(&endian) != 1 || sizeof(float) != 4 ||
      !std::numeric_limits<float>::is_iec559) {
    return Invalid("export requires little-endian IEEE float32");
  }
  return absl::OkStatus();
}
inline absl::StatusOr<Object> Manifest(absl::string_view directory,
                                       bool compact_int2 = false,
                                       const Config& config = Config::E2B()) {
  std::ifstream file(absl::StrCat(directory, "/manifest.json"),
                     std::ios::binary);
  if (!file) {
    return Invalid("cannot open manifest.json");
  }
  file.seekg(0, std::ios::end);
  const std::streampos length = file.tellg();
  if (length <= 0 || length > 16 * 1024 * 1024) {
    return Invalid("manifest size must be in (0,16MiB]");
  }
  file.seekg(0);
  std::string text(static_cast<size_t>(length), '\0');
  if (!file.read(text.data(), length)) {
    return Invalid("cannot read manifest");
  }
  minijson::value root;
  const char* cursor = text.c_str();
  if (minijson::parse(cursor, root) != minijson::no_error ||
      root.as<Object>() == nullptr) {
    return Invalid("invalid JSON object");
  }
  while (*cursor == ' ' || *cursor == '\r' || *cursor == '\n' ||
         *cursor == '\t') {
    ++cursor;
  }
  if (*cursor != '\0') {
    return Invalid("trailing data after manifest object");
  }
  const Object& object = *root.as<Object>();
  LRT_TENSOR_ASSIGN_OR_RETURN(int64_t version,
                              Integer(object, "schema_version"));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::string status,
                              Field<std::string>(object, "status"));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      Array unmapped, Field<Array>(object, "unmapped_float_coefficients"));
  if ((version != 1 && version != 2) || status != "complete" ||
      !unmapped.empty()) {
    return Invalid("incomplete, unsupported schema or unmapped coefficients");
  }
  if (version == 2) {
    LRT_TENSOR_ASSIGN_OR_RETURN(Object model,
                                Field<Object>(object, "model_config"));
    LRT_TENSOR_ASSIGN_OR_RETURN(std::string variant,
                                Field<std::string>(model, "variant"));
    if (variant != (config.num_kv_heads == 2 ? "e4b" : "e2b")) {
      return Invalid("bundle and runner model variant differ");
    }
    for (const auto& [key, expected] :
         absl::flat_hash_map<absl::string_view, int>{
             {"embed_dim", config.embed_dim},
             {"hidden_dim", config.hidden_dim},
             {"num_layers", config.num_layers},
             {"num_heads", config.num_heads},
             {"num_kv_heads", config.num_kv_heads},
             {"vocab_size", config.vocab_size},
             {"num_kv_owners", config.num_kv_heads == 2 ? 24 : 15},
             {"attention_pattern_size", config.attention_pattern_size},
             {"head_dim", config.head_dim},
             {"global_key_size", config.global_key_size},
             {"per_layer_input_dim", config.per_layer_input_dim},
             {"sliding_window_size", config.sliding_window_size}}) {
      LRT_TENSOR_ASSIGN_OR_RETURN(int64_t actual, Integer(model, key));
      if (actual != expected) {
        return Invalid(absl::StrCat("incorrect model config: ", key));
      }
    }
    if (compact_int2) {
      LRT_TENSOR_ASSIGN_OR_RETURN(
          std::string storage,
          Field<std::string>(object, "experimental_storage"));
      if (storage != "compact_int2_v1") {
        return Invalid("expected compact INT2 storage");
      }
    } else if (config.num_kv_heads != 1) {
      return Invalid("E4B requires the compact published bundle");
    }
    return object;
  }
  if (config.num_kv_heads != 1) {
    return Invalid("legacy bundle is E2B only");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(Object source,
                              Field<Object>(object, "source_bundle"));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::string hash,
                              Field<std::string>(source, "sha256"));
  LRT_TENSOR_ASSIGN_OR_RETURN(int64_t bytes, Integer(source, "bytes"));
  if (hash !=
          "ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce27911b27e42" ||
      bytes != 2583085056LL) {
    return Invalid(
        "unexpected source bundle identity (not a hash verification)");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(Object counts,
                              Field<Object>(object, "tensor_counts"));
  if (compact_int2) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        std::string storage,
        Field<std::string>(object, "experimental_storage"));
    if (storage != "compact_int2_v1") {
      return Invalid("expected an experimental compact INT2 bundle");
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(int64_t count, Integer(counts, "int2"));
    if (count != 62) {
      return Invalid("expected 62 original INT2 tensors");
    }
  }
  for (const auto& [name, expected] :
       absl::flat_hash_map<absl::string_view, int>{
           {"float32", 814},
           {"int4", compact_int2 ? 146 : 208},
           {"int8", 71}}) {
    LRT_TENSOR_ASSIGN_OR_RETURN(int64_t count, Integer(counts, name));
    if (count != expected) {
      return Invalid("unexpected tensor dtype coverage");
    }
  }
  return object;
}
struct Mapping {
  std::shared_ptr<void> owner;
  const std::byte* data;
  size_t bytes;
};
inline absl::StatusOr<Mapping> Map(absl::string_view directory,
                                   absl::string_view relative, size_t bytes) {
  if (relative.empty() || relative.front() == '/' || bytes == 0) {
    return Invalid("invalid relative file path or byte count");
  }
  for (absl::string_view part : absl::StrSplit(relative, '/')) {
    if (part.empty() || part == ".." || part == ".") {
      return Invalid("unsafe relative file path");
    }
  }
  const std::string full = absl::StrCat(directory, "/", relative);
  const int fd = open(full.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return Invalid(absl::StrCat("open ", full, ": ", std::strerror(errno)));
  }
  struct stat info{};
  if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 0 ||
      static_cast<uint64_t>(info.st_size) != bytes) {
    close(fd);
    return Invalid(
        absl::StrCat("file type or exact byte count mismatch: ", relative));
  }
  void* pointer = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  const int saved_errno = errno;
  close(fd);
  if (pointer == MAP_FAILED) {
    return Invalid(
        absl::StrCat("mmap ", relative, ": ", std::strerror(saved_errno)));
  }
  std::shared_ptr<void> owner(pointer, [bytes](void* p) { munmap(p, bytes); });
  return Mapping{std::move(owner), static_cast<const std::byte*>(pointer),
                 bytes};
}
inline std::shared_ptr<Buffer> BufferFor(const Mapping& mapping) {
  return std::shared_ptr<Buffer>(new SpanCpuBuffer(mapping.data, mapping.bytes),
                                 [owner = mapping.owner](Buffer* buffer) {
                                   (void)owner;
                                   delete buffer;
                                 });
}
struct Expected {
  std::string dtype;
  Shape shape;
};
inline absl::flat_hash_map<std::string, Expected> ExpectedTensors(
    const Config& c, bool compact_int2 = false) {
  absl::flat_hash_map<std::string, Expected> result;
  const auto add = [&](absl::string_view name, absl::string_view dtype,
                       Shape shape) {
    result.emplace(name, Expected{std::string(dtype), std::move(shape)});
  };
  const auto fc = [&](absl::string_view module, absl::string_view dtype,
                      int rows, int columns, bool static_qdq = true) {
    add(absl::StrCat(module, ".weight"), dtype, {rows, columns});
    if (static_qdq) {
      add(absl::StrCat(module, ".input_scale"), "float32", {1});
      add(absl::StrCat(module, ".output_scale"), "float32", {1});
    }
  };
  const int ple_width = c.num_layers * c.per_layer_input_dim;
  const int owner_count = c.num_kv_heads == 2 ? 24 : 15;
  fc("model.per_layer_model_projection", "int8", ple_width, c.embed_dim);
  const char* original_int2 = compact_int2 ? "int2" : "int4";
  fc("lm_head", original_int2, c.vocab_size, c.embed_dim, false);
  add("model.embed_tokens.weight", original_int2, {c.vocab_size, c.embed_dim});
  add("model.embed_tokens_per_layer.weight",
      c.num_kv_heads == 2 ? original_int2 : "int4", {c.vocab_size, ple_width});
  add("model.per_layer_projection_norm.weight", "float32", {256});
  add("model.norm.weight", "float32", {c.embed_dim});
  for (int layer = 0; layer < c.num_layers; ++layer) {
    const std::string prefix = absl::StrCat("model.layers.", layer, ".");
    const int dim =
        c.GetLayerType(layer) == Config::LayerType::kGlobal ? 512 : 256;
    const int hidden =
        c.hidden_dim * (c.num_kv_heads == 1 && layer >= owner_count ? 2 : 1);
    fc(absl::StrCat(prefix, "self_attn.q_proj"), "int4", c.num_heads * dim,
       c.embed_dim);
    fc(absl::StrCat(prefix, "self_attn.o_proj"), "int4", c.embed_dim,
       c.num_heads * dim);
    if (layer < owner_count) {
      fc(absl::StrCat(prefix, "self_attn.k_proj"), "int4", c.num_kv_heads * dim,
         c.embed_dim);
      fc(absl::StrCat(prefix, "self_attn.v_proj"), "int4", c.num_kv_heads * dim,
         c.embed_dim);
      add(absl::StrCat(prefix, "self_attn.k_norm.weight"), "float32", {dim});
    }
    const char* mlp_type =
        c.num_kv_heads == 1 && layer >= owner_count ? original_int2 : "int4";
    fc(absl::StrCat(prefix, "mlp.gate_proj"), mlp_type, hidden, c.embed_dim);
    fc(absl::StrCat(prefix, "mlp.up_proj"), mlp_type, hidden, c.embed_dim);
    fc(absl::StrCat(prefix, "mlp.down_proj"), mlp_type, c.embed_dim, hidden);
    fc(absl::StrCat(prefix, "per_layer_input_gate"), "int8", 256, c.embed_dim);
    fc(absl::StrCat(prefix, "per_layer_projection"), "int8", c.embed_dim, 256);
    for (absl::string_view norm :
         {"input_layernorm", "post_attention_layernorm",
          "pre_feedforward_layernorm", "post_feedforward_layernorm",
          "post_per_layer_input_norm"}) {
      add(absl::StrCat(prefix, norm, ".weight"), "float32", {c.embed_dim});
    }
    add(absl::StrCat(prefix, "self_attn.q_norm.weight"), "float32", {dim});
    add(absl::StrCat(prefix, "layer_scalar"), "float32", {1});
  }
  return result;
}

inline absl::StatusOr<TensorHandle> LoadTensor(absl::string_view directory,
                                               const Object& record,
                                               absl::string_view name,
                                               const Expected& expected,
                                               bool constant = false) {
  LRT_TENSOR_ASSIGN_OR_RETURN(std::string dtype,
                              Field<std::string>(record, "dtype"));
  LRT_TENSOR_ASSIGN_OR_RETURN(Shape shape, ReadShape(record, "shape"));
  if (dtype != expected.dtype || shape != expected.shape) {
    return Invalid(absl::StrCat("dtype/shape mismatch: ", name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(size_t count, Elements(shape));
  Type type = dtype == "float32" ? Type::kFP32
              : dtype == "int8"  ? Type::kI8
              : dtype == "int2"  ? Type::kI2
                                 : Type::kI4;
  if (dtype != "float32" && dtype != "int8" && dtype != "int4" &&
      dtype != "int2") {
    return Invalid(absl::StrCat("unsupported dtype: ", name));
  }
  if (count > std::numeric_limits<size_t>::max() / 4) {
    return Invalid(absl::StrCat("overflowing byte count: ", name));
  }
  const size_t bytes = dtype == "float32" ? count * 4
                       : dtype == "int8"  ? count
                       : dtype == "int2"  ? (count + 3) / 4
                                          : (count + 1) / 2;
  LRT_TENSOR_ASSIGN_OR_RETURN(int64_t declared, Integer(record, "bytes"));
  if (static_cast<uint64_t>(declared) != bytes) {
    return Invalid(absl::StrCat("tensor byte count: ", name));
  }
  if (!constant) {
    LRT_TENSOR_ASSIGN_OR_RETURN(std::string encoding,
                                Field<std::string>(record, "encoding"));
    const char* wanted = dtype == "float32" ? "little_endian_float32"
                         : dtype == "int8"  ? "signed_int8"
                         : dtype == "int2"
                             ? "signed_twos_complement_low_2bits_first"
                             : "signed_twos_complement_low_nibble_first";
    if (encoding != wanted) {
      return Invalid(absl::StrCat("unsupported encoding: ", name));
    }
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(std::string file,
                              Field<std::string>(record, "file"));
  LRT_TENSOR_ASSIGN_OR_RETURN(Mapping mapping, Map(directory, file, bytes));
  std::shared_ptr<Quantization> quantization;
  if (type == Type::kFP32) {
    if (record.count("quantization")) {
      return Invalid(absl::StrCat("unexpected FP32 quantization: ", name));
    }
    const float* values = reinterpret_cast<const float*>(mapping.data);
    const bool scale = absl::EndsWith(name, ".input_scale") ||
                       absl::EndsWith(name, ".output_scale");
    for (size_t i = 0; i < count; ++i) {
      if (!std::isfinite(values[i]) || (scale && !(values[i] > 0))) {
        return Invalid(absl::StrCat(
            "nonfinite coefficient or nonpositive activation scale: ", name));
      }
    }
  } else {
    LRT_TENSOR_ASSIGN_OR_RETURN(Object q,
                                Field<Object>(record, "quantization"));
    LRT_TENSOR_ASSIGN_OR_RETURN(std::string kind,
                                Field<std::string>(q, "kind"));
    LRT_TENSOR_ASSIGN_OR_RETURN(int64_t axis,
                                Integer(q, "quantized_dimension"));
    LRT_TENSOR_ASSIGN_OR_RETURN(Array zps, Field<Array>(q, "zero_points"));
    if (axis != 0 || zps.size() != 1) {
      return Invalid(absl::StrCat(
          "unsupported quantization axis/zero point count: ", name));
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(int64_t zero, Integer(zps.front()));
    if (zero != 0) {
      return Invalid(absl::StrCat("nonzero zero point: ", name));
    }
    const bool blockwise = name == "model.embed_tokens_per_layer.weight";
    const Shape scale_shape =
        blockwise ? Shape{shape[0], shape[1] / 256} : Shape{shape[0]};
    LRT_TENSOR_ASSIGN_OR_RETURN(Shape actual_scale_shape,
                                ReadShape(q, "scales_shape"));
    LRT_TENSOR_ASSIGN_OR_RETURN(std::string scale_dtype,
                                Field<std::string>(q, "scales_dtype"));
    if (actual_scale_shape != scale_shape || scale_dtype != "float32" ||
        kind != (blockwise ? "blockwise" : "per_channel")) {
      return Invalid(absl::StrCat("incorrect scale layout/kind: ", name));
    }
    int block_size = 0;
    if (blockwise) {
      LRT_TENSOR_ASSIGN_OR_RETURN(int64_t block, Integer(q, "block_size"));
      if (block != 256 || shape[1] % block != 0) {
        return Invalid(absl::StrCat("invalid block size: ", name));
      }
      block_size = static_cast<int>(block);
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(size_t scale_count, Elements(scale_shape));
    LRT_TENSOR_ASSIGN_OR_RETURN(int64_t scale_bytes,
                                Integer(q, "scales_bytes"));
    if (static_cast<uint64_t>(scale_bytes) != scale_count * sizeof(float)) {
      return Invalid(
          absl::StrCat("quantization scale byte count: ", name));
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(std::string scale_file,
                                Field<std::string>(q, "scales_file"));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        Mapping scale_mapping,
        Map(directory, scale_file, scale_count * sizeof(float)));
    const float* source = reinterpret_cast<const float*>(scale_mapping.data);
    std::vector<float> scales(source, source + scale_count);
    for (float scale : scales) {
      if (!(scale > 0) || !std::isfinite(scale)) {
        return Invalid(absl::StrCat("invalid quantization scale: ", name));
      }
    }
    if (blockwise) {
      quantization = std::make_shared<BlockwiseQuantization>(
          std::move(scales), std::vector<int64_t>{0}, block_size, 0);
    } else {
      // Author TFLite-compatible metadata: one zero point per channel scale.
      quantization = std::make_shared<PerChannelAffineQuantization>(
          std::move(scales), std::vector<int64_t>(scale_count, zero), 0);
    }
  }
  std::shared_ptr<Buffer> buffer;
  if (type == Type::kI2) {
    LRT_TENSOR_ASSIGN_OR_RETURN(Array sources, Field<Array>(record, "sources"));
    const bool ple = name == "model.embed_tokens_per_layer.weight";
    if (ple) {
      if (sources.size() != shape[1] / 256) {
        return Invalid("INT2 PLE partition count");
      }
      for (size_t i = 0; i < sources.size(); ++i) {
        const Object* part = sources[i].as<Object>();
        if (!part) {
          return Invalid("INT2 PLE source record");
        }
        LRT_TENSOR_ASSIGN_OR_RETURN(
            int64_t partition, Integer(*part, "destination_layer_partition"));
        LRT_TENSOR_ASSIGN_OR_RETURN(std::string dtype,
                                    Field<std::string>(*part, "dtype"));
        LRT_TENSOR_ASSIGN_OR_RETURN(Shape dimensions,
                                    ReadShape(*part, "shape"));
        LRT_TENSOR_ASSIGN_OR_RETURN(int64_t source_bytes,
                                    Integer(*part, "source_bytes"));
        if (partition != i || dtype != "INT2" ||
            dimensions != Shape({shape[0], 256}) ||
            source_bytes != size_t(shape[0]) * 64) {
          return Invalid("INT2 PLE provenance");
        }
      }
    } else {
      if (constant || sources.size() != 1 || !sources[0].as<Object>()) {
        return Invalid(
            absl::StrCat("expected one original INT2 tensor: ", name));
      }
      const Object& source = *sources[0].as<Object>();
      LRT_TENSOR_ASSIGN_OR_RETURN(std::string original_type,
                                  Field<std::string>(source, "dtype"));
      LRT_TENSOR_ASSIGN_OR_RETURN(Shape original_shape,
                                  ReadShape(source, "shape"));
      LRT_TENSOR_ASSIGN_OR_RETURN(int64_t source_bytes,
                                  Integer(source, "source_bytes"));
      if (original_type != "INT2" || original_shape != shape ||
          count % 4 != 0 || source_bytes != bytes) {
        return Invalid(
            absl::StrCat("compact INT2 provenance mismatch: ", name));
      }
    }
  }
  if (!constant && type == Type::kI4 && name != "model.embed_tokens.weight" &&
      name != "model.embed_tokens_per_layer.weight") {
    LRT_TENSOR_ASSIGN_OR_RETURN(Array sources, Field<Array>(record, "sources"));
    if (sources.size() != 1 || !sources[0].as<Object>()) {
      return Invalid(
          absl::StrCat("expected one original quantized tensor: ", name));
    }
    const Object& source = *sources[0].as<Object>();
    LRT_TENSOR_ASSIGN_OR_RETURN(std::string original_type,
                                Field<std::string>(source, "dtype"));
    if (original_type == "INT2") {
      LRT_TENSOR_ASSIGN_OR_RETURN(int64_t source_bytes,
                                  Integer(source, "source_bytes"));
      if (count % 4 != 0 || source_bytes != count / 4) {
        return Invalid(absl::StrCat("original INT2 size mismatch: ", name));
      }
      std::shared_ptr<OwningCpuBuffer> compact =
          OwningCpuBuffer::Allocate<Type::kU8>(count / 4);
      absl::Span<uint8_t> packed = compact->Span<uint8_t>();
      std::fill(packed.begin(), packed.end(), uint8_t{0});
      const uint8_t* widened = reinterpret_cast<const uint8_t*>(mapping.data);
      for (size_t i = 0; i < count; ++i) {
        const int nibble = (widened[i / 2] >> (4 * (i % 2))) & 15;
        const int code = nibble >= 8 ? nibble - 16 : nibble;
        if (code < -2 || code > 1) {
          return Invalid(
              absl::StrCat("original INT2 code outside [-2,1]: ", name));
        }
        packed[i / 4] |= (code & 3) << (2 * (i % 4));
      }
      buffer = std::move(compact);
      type = Type::kI2;
    }
  }
  if (!buffer) {
    buffer = BufferFor(mapping);
  }
  TensorHandle tensor(TensorInit{.name = std::string(name),
                                 .type = type,
                                 .shape = shape,
                                 .buffer = std::move(buffer),
                                 .quantization = std::move(quantization)});
  LRT_TENSOR_RETURN_IF_ERROR(tensor.GetStatus());
  return tensor;
}
}  // namespace published_bundle_detail

inline absl::StatusOr<std::vector<KvOwnerSpec>> LoadPublishedKvSpecs(
    absl::string_view directory, const Config& config,
    bool compact_int2 = false) {
  namespace p = published_bundle_detail;
  LRT_TENSOR_RETURN_IF_ERROR(p::CheckConfig(config));
  LRT_TENSOR_ASSIGN_OR_RETURN(p::Object manifest,
                              p::Manifest(directory, compact_int2, config));
  LRT_TENSOR_ASSIGN_OR_RETURN(p::Array records,
                              p::Field<p::Array>(manifest, "kv_cache_specs"));
  const int owner_count = config.num_kv_heads == 2 ? 24 : 15;
  if (records.size() != owner_count) {
    return p::Invalid("incorrect KV owner count");
  }
  std::vector<KvOwnerSpec> specs(owner_count);
  std::vector<bool> seen(owner_count, false);
  for (const minijson::value& value : records) {
    const p::Object* record = value.as<p::Object>();
    if (!record) {
      return p::Invalid("invalid KV record");
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(int64_t owner, p::Integer(*record, "owner"));
    LRT_TENSOR_ASSIGN_OR_RETURN(int64_t dim, p::Integer(*record, "head_dim"));
    LRT_TENSOR_ASSIGN_OR_RETURN(int64_t zero,
                                p::Integer(*record, "zero_point"));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        minijson::number key, p::Field<minijson::number>(*record, "key_scale"));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        minijson::number val,
        p::Field<minijson::number>(*record, "value_scale"));
    if (owner >= owner_count || seen[owner] || zero != 0 ||
        dim != (config.GetLayerType(owner) == Config::LayerType::kGlobal
                    ? config.global_key_size
                    : config.head_dim) ||
        !(key > 0) || !(val > 0) || !std::isfinite(key) ||
        !std::isfinite(val) ||
        static_cast<double>(static_cast<float>(key)) != key ||
        static_cast<double>(static_cast<float>(val)) != val) {
      return p::Invalid("invalid owner, head width, scale or zero point");
    }
    // Mixed string/integer templates are part of the fixed schema.
    for (absl::string_view field :
         {"key_shape_template", "value_shape_template"}) {
      LRT_TENSOR_ASSIGN_OR_RETURN(p::Array shape,
                                  p::Field<p::Array>(*record, field));
      const bool is_key = field == "key_shape_template";
      if (shape.size() != 4) {
        return p::Invalid("KV shape template rank");
      }
      const std::string* capacity = shape[is_key ? 2 : 3].as<std::string>();
      if (!capacity || *capacity != "capacity") {
        return p::Invalid("KV capacity template");
      }
      for (int i : {0, 1, is_key ? 3 : 2}) {
        LRT_TENSOR_ASSIGN_OR_RETURN(int64_t n, p::Integer(shape[i]));
        if (n != (i == 0 ? 1 : i == 1 ? config.num_kv_heads : dim)) {
          return p::Invalid("KV shape template dimensions");
        }
      }
    }
    seen[owner] = true;
    specs[owner] = KvOwnerSpec{static_cast<int>(owner), static_cast<int>(dim),
                               static_cast<float>(key), static_cast<float>(val),
                               config.num_kv_heads};
  }
  return specs;
}

inline absl::StatusOr<LoadedTensors> LoadPublishedBundle(
    absl::string_view directory, const Config& config,
    bool embeddings_only = false, bool compact_int2 = false) {
  namespace p = published_bundle_detail;
  LRT_TENSOR_RETURN_IF_ERROR(p::CheckConfig(config));
  LRT_TENSOR_ASSIGN_OR_RETURN(p::Object manifest,
                              p::Manifest(directory, compact_int2, config));
  LRT_TENSOR_ASSIGN_OR_RETURN(p::Array records,
                              p::Field<p::Array>(manifest, "tensors"));
  absl::flat_hash_map<std::string, p::Expected> expected =
      p::ExpectedTensors(config, compact_int2);
  if (records.size() != expected.size()) {
    return p::Invalid("incorrect active tensor record count");
  }
  absl::flat_hash_map<std::string, TensorHandle> handles;
  handles.reserve(expected.size() + 1);
  for (const minijson::value& value : records) {
    const p::Object* record = value.as<p::Object>();
    if (!record) {
      return p::Invalid("tensor record must be an object");
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(std::string name,
                                p::Field<std::string>(*record, "name"));
    auto it = expected.find(name);
    if (it == expected.end()) {
      return p::Invalid(
          absl::StrCat("unexpected or duplicate tensor: ", name));
    }
    if (embeddings_only && name != "model.embed_tokens.weight" &&
        name != "model.embed_tokens_per_layer.weight") {
      expected.erase(it);
      continue;
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(
        TensorHandle tensor,
        p::LoadTensor(directory, *record, name, it->second));
    handles.emplace(name, std::move(tensor));
    expected.erase(it);
  }
  if (!expected.empty()) {
    return p::Invalid("missing active tensor names");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(p::Array constants,
                              p::Field<p::Array>(manifest, "constants"));
  std::string softcap_name = "decode.tensor_36";
  if (manifest.count("semantic_constants")) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        p::Object semantic,
        p::Field<p::Object>(manifest, "semantic_constants"));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        softcap_name,
        p::Field<std::string>(semantic, "logits_softcap_reciprocal"));
  }
  bool found = false;
  for (const minijson::value& value : constants) {
    const p::Object* record = value.as<p::Object>();
    if (!record) {
      return p::Invalid("constant record must be an object");
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(std::string name,
                                p::Field<std::string>(*record, "name"));
    if (name != softcap_name) {
      continue;
    }
    if (found) {
      return p::Invalid("duplicate logits softcap reciprocal");
    }
    const std::string key = "matched.logits_softcap_reciprocal";
    LRT_TENSOR_ASSIGN_OR_RETURN(
        TensorHandle tensor,
        p::LoadTensor(directory, *record, key, {"float32", {}}, true));
    LRT_TENSOR_ASSIGN_OR_RETURN(Buffer & buffer, tensor.GetBuffer());
    LockedBufferSpan<const float> span = buffer.Lock().As<const float>();
    if (span.size() != 1 || span.data()[0] != 0.03333333507180214f) {
      return p::Invalid("unexpected exact softcap reciprocal value");
    }
    handles.emplace(key, std::move(tensor));
    found = true;
  }
  if (!found) {
    return p::Invalid("missing logits softcap reciprocal");
  }
  // Finalize handles before embedding wrappers retain their tensor references.
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::unique_ptr<GemmaEmbeddingTable> token_embedding,
      GemmaEmbeddingTable::Create(handles.at("model.embed_tokens.weight"),
                                  config.embed_dim));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::unique_ptr<GemmaEmbeddingTable> ple_embedding,
      GemmaEmbeddingTable::Create(
          handles.at("model.embed_tokens_per_layer.weight"),
          config.num_layers * config.per_layer_input_dim));
  return LoadedTensors{std::move(handles), std::move(token_embedding),
                       std::move(ple_embedding)};
}
}  // namespace litert::tensor::examples::gemma4::cpu
#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_BUNDLE_LOADER_H_

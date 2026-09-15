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

// Artifact-only loader for the validated published Gemma4 E2B bundle export.
// Include after xnnpack_main.cc: LoadedTensors lives in its anonymous
// namespace. This loader never opens a CT checkpoint. File hashes are verified
// separately by matched-bundle/validate_export.py; this is a structural/data
// validator.
#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_MATCHED_BUNDLE_LOADER_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_MATCHED_BUNDLE_LOADER_H_

#include "tensor/examples/gemma4/native/driver_support.h"
#include "tensor/examples/gemma4/native/model/helpers/int8_kv_cache.h"
#include "tensor/examples/gemma4/native/model/helpers/static_int2_fully_connected.h"
#include "tensor/examples/utils/minijson.h"
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace litert::tensor::examples::gemma4::native {
namespace {
namespace published_bundle_detail {

using Object = minijson::object;
using Array = minijson::array;
inline absl::Status Invalid(const std::string &reason) {
  return absl::InvalidArgumentError("Published bundle: " + reason);
}
template <class T>
absl::StatusOr<T> Field(const Object &object, const std::string &name) {
  minijson::value value;
  if (!object.at(name, &value) || value.as<T>() == nullptr)
    return Invalid("missing or incorrectly typed field " + name);
  return *value.as<T>();
}
inline absl::StatusOr<int64_t> Integer(const minijson::value &value) {
  const auto *number = value.as<minijson::number>();
  // JSON numbers are doubles. Only exact, bounded integers are accepted.
  if (number == nullptr || !std::isfinite(*number) || *number < 0 ||
      *number > 9007199254740991.0 || std::floor(*number) != *number)
    return Invalid("expected exact nonnegative integer");
  return static_cast<int64_t>(*number);
}
inline absl::StatusOr<int64_t> Integer(const Object &object,
                                       const std::string &name) {
  minijson::value value;
  if (!object.at(name, &value))
    return Invalid("missing integer " + name);
  return Integer(value);
}
inline absl::StatusOr<Shape> ReadShape(const Object &object,
                                       const std::string &name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto dimensions, Field<Array>(object, name));
  if (dimensions.size() > 8)
    return Invalid("unsupported tensor rank");
  Shape shape;
  for (const auto &dimension : dimensions) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto n, Integer(dimension));
    if (n == 0 || n > std::numeric_limits<int>::max())
      return Invalid("nonpositive or oversized shape dimension");
    shape.push_back(static_cast<int>(n));
  }
  return shape;
}
inline absl::StatusOr<size_t> Elements(const Shape &shape) {
  size_t count = 1;
  for (int dimension : shape) {
    if (dimension <= 0 ||
        count > std::numeric_limits<size_t>::max() / dimension)
      return Invalid("invalid or overflowing shape");
    count *= dimension;
  }
  return count;
}
inline absl::Status CheckConfig(const Config &c) {
  const auto e = Config::E2B();
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
      c.global_rope_proportion != e.global_rope_proportion)
    return Invalid("loader only supports the published E2B configuration");
  const uint16_t endian = 1;
  if (*reinterpret_cast<const uint8_t *>(&endian) != 1 || sizeof(float) != 4 ||
      !std::numeric_limits<float>::is_iec559)
    return Invalid("export requires little-endian IEEE float32");
  return absl::OkStatus();
}
inline absl::StatusOr<Object> Manifest(const std::string &directory) {
  std::ifstream file(directory + "/manifest.json", std::ios::binary);
  if (!file)
    return Invalid("cannot open manifest.json");
  file.seekg(0, std::ios::end);
  const auto length = file.tellg();
  if (length <= 0 || length > 16 * 1024 * 1024)
    return Invalid("manifest size must be in (0,16MiB]");
  file.seekg(0);
  std::string text(static_cast<size_t>(length), '\0');
  if (!file.read(text.data(), length))
    return Invalid("cannot read manifest");
  minijson::value root;
  const char* cursor = text.c_str();
  if (minijson::parse(cursor, root) != minijson::no_error ||
      root.as<Object>() == nullptr)
    return Invalid("invalid JSON object");
  while (*cursor == ' ' || *cursor == '\r' || *cursor == '\n' || *cursor == '\t')
    ++cursor;
  if (*cursor != '\0') return Invalid("trailing data after manifest object");
  const auto &object = *root.as<Object>();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto version, Integer(object, "schema_version"));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto status,
                              Field<std::string>(object, "status"));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto unmapped, Field<Array>(object, "unmapped_float_coefficients"));
  if (version != 1 || status != "complete" || !unmapped.empty())
    return Invalid("incomplete, unsupported schema or unmapped coefficients");
  LRT_TENSOR_ASSIGN_OR_RETURN(auto source,
                              Field<Object>(object, "source_bundle"));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto hash, Field<std::string>(source, "sha256"));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto bytes, Integer(source, "bytes"));
  if (hash !=
          "ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce27911b27e42" ||
      bytes != 2583085056LL)
    return Invalid(
        "unexpected source bundle identity (not a hash verification)");
  LRT_TENSOR_ASSIGN_OR_RETURN(auto counts,
                              Field<Object>(object, "tensor_counts"));
  for (const auto &[name, expected] : std::map<std::string, int>{
           {"float32", 814}, {"int4", 208}, {"int8", 71}}) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto count, Integer(counts, name));
    if (count != expected)
      return Invalid("unexpected tensor dtype coverage");
  }
  return object;
}
struct Mapping {
  std::shared_ptr<void> owner;
  const std::byte *data;
  size_t bytes;
};
inline absl::StatusOr<Mapping> Map(const std::string &directory,
                                   const std::string &relative, size_t bytes) {
  const std::filesystem::path path(relative);
  if (relative.empty() || path.is_absolute() || bytes == 0)
    return Invalid("invalid relative file path or byte count");
  for (const auto &part : path)
    if (part == ".." || part == ".")
      return Invalid("unsafe relative file path");
  const std::string full = (std::filesystem::path(directory) / path).string();
  const int fd = open(full.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return Invalid("open " + full + ": " + std::strerror(errno));
  struct stat info{};
  if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 0 ||
      static_cast<uint64_t>(info.st_size) != bytes) {
    close(fd);
    return Invalid("file type or exact byte count mismatch: " + relative);
  }
  void *pointer = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  const int saved_errno = errno;
  close(fd);
  if (pointer == MAP_FAILED)
    return Invalid("mmap " + relative + ": " + std::strerror(saved_errno));
  auto owner =
      std::shared_ptr<void>(pointer, [bytes](void *p) { munmap(p, bytes); });
  return Mapping{std::move(owner), static_cast<const std::byte *>(pointer),
                 bytes};
}
inline std::shared_ptr<Buffer> BufferFor(const Mapping &mapping) {
  return std::shared_ptr<Buffer>(new SpanCpuBuffer(mapping.data, mapping.bytes),
                                 [owner = mapping.owner](Buffer *buffer) {
                                   (void)owner;
                                   delete buffer;
                                 });
}
struct Expected {
  std::string dtype;
  Shape shape;
};
inline std::map<std::string, Expected> ExpectedTensors(const Config &c) {
  std::map<std::string, Expected> result;
  auto add = [&](const std::string &name, const std::string &dtype,
                 Shape shape) {
    result.emplace(name, Expected{dtype, std::move(shape)});
  };
  auto fc = [&](const std::string &module, const std::string &dtype, int rows,
                int columns, bool static_qdq = true) {
    add(module + ".weight", dtype, {rows, columns});
    if (static_qdq) {
      add(module + ".input_scale", "float32", {1});
      add(module + ".output_scale", "float32", {1});
    }
  };
  fc("model.per_layer_model_projection", "int8", 8960, c.embed_dim);
  fc("lm_head", "int4", c.vocab_size, c.embed_dim, false);
  add("model.embed_tokens.weight", "int4", {c.vocab_size, c.embed_dim});
  add("model.embed_tokens_per_layer.weight", "int4", {c.vocab_size, 8960});
  add("model.per_layer_projection_norm.weight", "float32", {256});
  add("model.norm.weight", "float32", {c.embed_dim});
  for (int layer = 0; layer < c.num_layers; ++layer) {
    const auto prefix = absl::StrCat("model.layers.", layer, ".");
    const int dim =
        c.GetLayerType(layer) == Config::LayerType::kGlobal ? 512 : 256;
    const int hidden = c.hidden_dim * (layer >= 15 ? 2 : 1);
    fc(prefix + "self_attn.q_proj", "int4", c.num_heads * dim, c.embed_dim);
    fc(prefix + "self_attn.o_proj", "int4", c.embed_dim, c.num_heads * dim);
    if (layer < 15) {
      fc(prefix + "self_attn.k_proj", "int4", dim, c.embed_dim);
      fc(prefix + "self_attn.v_proj", "int4", dim, c.embed_dim);
      add(prefix + "self_attn.k_norm.weight", "float32", {dim});
    }
    fc(prefix + "mlp.gate_proj", "int4", hidden, c.embed_dim);
    fc(prefix + "mlp.up_proj", "int4", hidden, c.embed_dim);
    fc(prefix + "mlp.down_proj", "int4", c.embed_dim, hidden);
    fc(prefix + "per_layer_input_gate", "int8", 256, c.embed_dim);
    fc(prefix + "per_layer_projection", "int8", c.embed_dim, 256);
    for (const auto *norm :
         {"input_layernorm", "post_attention_layernorm",
          "pre_feedforward_layernorm", "post_feedforward_layernorm",
          "post_per_layer_input_norm"})
      add(prefix + norm + ".weight", "float32", {c.embed_dim});
    add(prefix + "self_attn.q_norm.weight", "float32", {dim});
    add(prefix + "layer_scalar", "float32", {1});
  }
  return result;
}

// Select the fixed bundle's static MLP group, then REQUIRE its original dtype
// and source identity. Merely finding small numeric codes never enables INT2.
inline int StaticInt2SourceTensor(const std::string& name) {
  for (int layer = 15; layer < 35; ++layer) {
    const auto prefix = absl::StrCat("model.layers.", layer, ".mlp.");
    for (const auto& item : std::map<std::string, int>{
             {"gate_proj.weight", 0}, {"up_proj.weight", 3},
             {"down_proj.weight", 9}})
      if (name == prefix + item.first)
        return 1587 + 57 * (layer - 15) + item.second;
  }
  return -1;
}

inline absl::StatusOr<StaticInt2Provenance> OriginalStaticInt2Provenance(
    const Object& record, const std::string& name, const Shape& shape) {
  const int expected_tensor = StaticInt2SourceTensor(name);
  if (expected_tensor < 0) return Invalid("not a published static INT2 MLP: " + name);
  LRT_TENSOR_ASSIGN_OR_RETURN(auto sources, Field<Array>(record, "sources"));
  if (sources.size() != 1 || !sources.front().as<Object>())
    return Invalid("static INT2 requires one original decode source: " + name);
  const auto& source = *sources.front().as<Object>();
  LRT_TENSOR_ASSIGN_OR_RETURN(auto dtype, Field<std::string>(source, "dtype"));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto model_type, Field<std::string>(source, "section_model_type"));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto source_shape, ReadShape(source, "shape"));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto bytes, Integer(source, "source_bytes"));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto section, Integer(source, "section_index"));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto subgraph, Integer(source, "subgraph_index"));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto tensor, Integer(source, "tensor_index"));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto hash, Field<std::string>(source, "source_data_sha256"));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto count, Elements(shape));
  if (dtype != "INT2" || model_type != "tf_lite_prefill_decode" ||
      source_shape != shape || shape.size() != 2 || shape[1] % 4 != 0 ||
      count / 4 != static_cast<size_t>(bytes) || section != 10 ||
      subgraph != 0 || tensor != expected_tensor || hash.size() != 64 ||
      hash.find_first_not_of("0123456789abcdef") != std::string::npos)
    return Invalid("original static INT2 provenance mismatch: " + name);
  return StaticInt2Provenance{name, dtype, hash, static_cast<int>(section),
                              static_cast<int>(subgraph), static_cast<int>(tensor),
                              shape, static_cast<size_t>(bytes)};
}

inline absl::StatusOr<TensorHandle> LoadTensor(const std::string &directory,
                                               const Object &record,
                                               const std::string &name,
                                               const Expected &expected,
                                               bool constant = false) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto dtype, Field<std::string>(record, "dtype"));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto shape, ReadShape(record, "shape"));
  if (dtype != expected.dtype || shape != expected.shape)
    return Invalid("dtype/shape mismatch: " + name);
  LRT_TENSOR_ASSIGN_OR_RETURN(auto count, Elements(shape));
  Type type = dtype == "float32" ? Type::kFP32
                    : dtype == "int8"  ? Type::kI8
                                       : Type::kI4;
  if (dtype != "float32" && dtype != "int8" && dtype != "int4")
    return Invalid("unsupported dtype: " + name);
  if (count > std::numeric_limits<size_t>::max() / 4)
    return Invalid("overflowing byte count: " + name);
  const size_t bytes = dtype == "float32" ? count * 4
                       : dtype == "int8"  ? count
                                          : (count + 1) / 2;
  LRT_TENSOR_ASSIGN_OR_RETURN(auto declared, Integer(record, "bytes"));
  if (static_cast<uint64_t>(declared) != bytes)
    return Invalid("tensor byte count: " + name);
  if (!constant) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto encoding,
                                Field<std::string>(record, "encoding"));
    const char *wanted = dtype == "float32" ? "little_endian_float32"
                         : dtype == "int8"
                             ? "signed_int8"
                             : "signed_twos_complement_low_nibble_first";
    if (encoding != wanted)
      return Invalid("unsupported encoding: " + name);
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(auto file, Field<std::string>(record, "file"));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto mapping, Map(directory, file, bytes));
  std::shared_ptr<Quantization> quantization;
  if (type == Type::kFP32) {
    if (record.count("quantization"))
      return Invalid("unexpected FP32 quantization: " + name);
    const auto *values = reinterpret_cast<const float *>(mapping.data);
    const bool scale = absl::EndsWith(name, ".input_scale") ||
                       absl::EndsWith(name, ".output_scale");
    for (size_t i = 0; i < count; ++i)
      if (!std::isfinite(values[i]) || (scale && !(values[i] > 0)))
        return Invalid(
            "nonfinite coefficient or nonpositive activation scale: " + name);
  } else {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto q, Field<Object>(record, "quantization"));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto kind, Field<std::string>(q, "kind"));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto axis, Integer(q, "quantized_dimension"));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto zps, Field<Array>(q, "zero_points"));
    if (axis != 0 || zps.size() != 1)
      return Invalid("unsupported quantization axis/zero point count: " + name);
    LRT_TENSOR_ASSIGN_OR_RETURN(auto zero, Integer(zps.front()));
    if (zero != 0)
      return Invalid("nonzero zero point: " + name);
    const bool blockwise = name == "model.embed_tokens_per_layer.weight";
    const Shape scale_shape = blockwise ? Shape{262144, 35} : Shape{shape[0]};
    LRT_TENSOR_ASSIGN_OR_RETURN(auto actual_scale_shape,
                                ReadShape(q, "scales_shape"));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto scale_dtype,
                                Field<std::string>(q, "scales_dtype"));
    if (actual_scale_shape != scale_shape || scale_dtype != "float32" ||
        kind != (blockwise ? "blockwise" : "per_channel"))
      return Invalid("incorrect scale layout/kind: " + name);
    int block_size = 0;
    if (blockwise) {
      LRT_TENSOR_ASSIGN_OR_RETURN(auto block, Integer(q, "block_size"));
      if (block != 256 || shape[1] % block != 0)
        return Invalid("invalid block size: " + name);
      block_size = static_cast<int>(block);
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(auto scale_count, Elements(scale_shape));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto scale_bytes, Integer(q, "scales_bytes"));
    if (static_cast<uint64_t>(scale_bytes) != scale_count * sizeof(float))
      return Invalid("quantization scale byte count: " + name);
    LRT_TENSOR_ASSIGN_OR_RETURN(auto scale_file,
                                Field<std::string>(q, "scales_file"));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto scale_mapping,
        Map(directory, scale_file, scale_count * sizeof(float)));
    const auto *source = reinterpret_cast<const float *>(scale_mapping.data);
    std::vector<float> scales(source, source + scale_count);
    for (float scale : scales)
      if (!(scale > 0) || !std::isfinite(scale))
        return Invalid("invalid quantization scale: " + name);
    if (blockwise)
      quantization = std::make_shared<BlockwiseQuantization>(
          std::move(scales), std::vector<int64_t>{0}, block_size, 0);
    else
      quantization = std::make_shared<PerChannelAffineQuantization>(
          std::move(scales), std::vector<int64_t>{0}, 0);
  }
  std::shared_ptr<Buffer> buffer;
  if (!constant && StaticInt2SourceTensor(name) >= 0) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto provenance,
                                OriginalStaticInt2Provenance(record, name, shape));
    if (preserve_static_int2_weights) {
      if (type != Type::kI4)
        return Invalid("static INT2 export must use signed I4: " + name);
      LRT_TENSOR_ASSIGN_OR_RETURN(auto compact,
          StaticInt2WeightBuffer::FromWidenedI4(
              reinterpret_cast<const uint8_t*>(mapping.data), mapping.bytes,
              std::move(provenance)));
      buffer = std::move(compact);
      type = Type::kI2;
      // mapping is intentionally NOT retained: its widened file is unmapped
      // when this function returns. Only the compact representation survives.
    }
  }
  if (!buffer) buffer = BufferFor(mapping);
  TensorHandle tensor(TensorInit{.name = name,
                                 .type = type,
                                 .shape = shape,
                                 .buffer = std::move(buffer),
                                 .quantization = std::move(quantization)});
  LRT_TENSOR_RETURN_IF_ERROR(tensor.GetStatus());
  return tensor;
}
} // namespace published_bundle_detail

inline absl::StatusOr<std::vector<KvOwnerSpec>>
LoadPublishedKvSpecs(std::string directory, const Config &config) {
  namespace p = published_bundle_detail;
  LRT_TENSOR_RETURN_IF_ERROR(p::CheckConfig(config));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto manifest, p::Manifest(directory));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto records,
                              p::Field<p::Array>(manifest, "kv_cache_specs"));
  if (records.size() != 15)
    return p::Invalid("expected exactly 15 KV owners");
  std::vector<KvOwnerSpec> specs(15);
  std::vector<bool> seen(15, false);
  for (const auto &value : records) {
    const auto *record = value.as<p::Object>();
    if (!record)
      return p::Invalid("invalid KV record");
    LRT_TENSOR_ASSIGN_OR_RETURN(auto owner, p::Integer(*record, "owner"));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto dim, p::Integer(*record, "head_dim"));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto zero, p::Integer(*record, "zero_point"));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto key, p::Field<minijson::number>(*record, "key_scale"));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto val, p::Field<minijson::number>(*record, "value_scale"));
    if (owner >= 15 || seen[owner] || zero != 0 ||
        dim != (owner % 5 == 4 ? 512 : 256) || !(key > 0) || !(val > 0) ||
        !std::isfinite(key) || !std::isfinite(val) ||
        static_cast<double>(static_cast<float>(key)) != key ||
        static_cast<double>(static_cast<float>(val)) != val)
      return p::Invalid("invalid owner, head width, scale or zero point");
    // Mixed string/integer templates are part of the fixed schema.
    for (const auto *field : {"key_shape_template", "value_shape_template"}) {
      LRT_TENSOR_ASSIGN_OR_RETURN(auto shape,
                                  p::Field<p::Array>(*record, field));
      const bool is_key = std::string(field) == "key_shape_template";
      if (shape.size() != 4)
        return p::Invalid("KV shape template rank");
      const auto *capacity = shape[is_key ? 2 : 3].as<std::string>();
      if (!capacity || *capacity != "capacity")
        return p::Invalid("KV capacity template");
      for (int i : {0, 1, is_key ? 3 : 2}) {
        LRT_TENSOR_ASSIGN_OR_RETURN(auto n, p::Integer(shape[i]));
        if (n != (i < 2 ? 1 : dim))
          return p::Invalid("KV shape template dimensions");
      }
    }
    seen[owner] = true;
    specs[owner] =
        KvOwnerSpec{static_cast<int>(owner), static_cast<int>(dim),
                    static_cast<float>(key), static_cast<float>(val)};
  }
  return specs;
}

inline absl::StatusOr<LoadedTensors> LoadPublishedBundle(std::string directory,
                                                         const Config &config) {
  namespace p = published_bundle_detail;
  LRT_TENSOR_RETURN_IF_ERROR(p::CheckConfig(config));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto manifest, p::Manifest(directory));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto records,
                              p::Field<p::Array>(manifest, "tensors"));
  auto expected = p::ExpectedTensors(config);
  if (expected.size() != 1093 || records.size() != expected.size())
    return p::Invalid("expected all 1,093 active tensor records");
  absl::flat_hash_map<std::string, TensorHandle> handles;
  handles.reserve(expected.size() + 1);
  for (const auto &value : records) {
    const auto *record = value.as<p::Object>();
    if (!record)
      return p::Invalid("tensor record must be an object");
    LRT_TENSOR_ASSIGN_OR_RETURN(auto name,
                                p::Field<std::string>(*record, "name"));
    auto it = expected.find(name);
    if (it == expected.end())
      return p::Invalid("unexpected or duplicate tensor: " + name);
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto tensor, p::LoadTensor(directory, *record, name, it->second));
    handles.emplace(name, std::move(tensor));
    expected.erase(it);
  }
  if (!expected.empty())
    return p::Invalid("missing active tensor names");
  const auto int2_audit = GetStaticInt2WeightAudit(handles);
  if (preserve_static_int2_weights &&
      (int2_audit.tensor_count != 60 || int2_audit.compact_bytes != 283115520 ||
       int2_audit.widened_bytes != 566231040))
    return p::Invalid("static INT2 preservation coverage/size mismatch");
  LRT_TENSOR_ASSIGN_OR_RETURN(auto constants,
                              p::Field<p::Array>(manifest, "constants"));
  bool found = false;
  for (const auto &value : constants) {
    const auto *record = value.as<p::Object>();
    if (!record)
      return p::Invalid("constant record must be an object");
    LRT_TENSOR_ASSIGN_OR_RETURN(auto name,
                                p::Field<std::string>(*record, "name"));
    if (name != "decode.tensor_36")
      continue;
    if (found)
      return p::Invalid("duplicate logits softcap reciprocal");
    const std::string key = "matched.logits_softcap_reciprocal";
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto tensor,
        p::LoadTensor(directory, *record, key, {"float32", {}}, true));
    LRT_TENSOR_ASSIGN_OR_RETURN(Buffer & buffer, tensor.GetBuffer());
    auto span = buffer.Lock().As<const float>();
    if (span.size() != 1 || span.data()[0] != 0.03333333507180214f)
      return p::Invalid("unexpected exact softcap reciprocal value");
    handles.emplace(key, std::move(tensor));
    found = true;
  }
  if (!found)
    return p::Invalid("missing logits softcap reciprocal");
  // Finalize handles before embedding wrappers retain their tensor references.
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto token_embedding,
      GemmaEmbeddingTable::Create(handles.at("model.embed_tokens.weight"),
                                  config.embed_dim));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto ple_embedding, GemmaEmbeddingTable::Create(
                              handles.at("model.embed_tokens_per_layer.weight"),
                              config.num_layers * config.per_layer_input_dim));
  return LoadedTensors{std::move(handles), std::move(token_embedding),
                       std::move(ple_embedding)};
}
} // namespace
} // namespace litert::tensor::examples::gemma4::native
#endif // LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_MATCHED_BUNDLE_LOADER_H_

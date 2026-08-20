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
#include "tensor/examples/gemma4/safetensor_loader.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <system_error>  // NOLINT
#include <type_traits>
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
#include "tensor/examples/gemma4/minijson.h"
#include "tensor/examples/gemma4/perfetto_session.h"
#include "tensor/examples/gemma4/safetensors.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::examples {

namespace {

// Creates a `SpanCpuBuffer` over the given data and wraps it in a `shared_ptr`.
//
// `owner` is kept around in the `shared_ptr` to keep the data and the span
// lifetimes in sync.
std::shared_ptr<Buffer> MakeMappedBuffer(const std::shared_ptr<void>& owner,
                                         const std::byte* data, size_t bytes) {
  return std::shared_ptr<Buffer>(new SpanCpuBuffer(data, bytes),
                                 [owner](Buffer* buffer) {
                                   static_cast<void>(owner);
                                   delete buffer;
                                 });
}

const char* ToString(safetensors::dtype dtype) {
  switch (dtype) {
    case safetensors::dtype::kBFLOAT16:
      return "BF16";
    case safetensors::dtype::kFLOAT16:
      return "F16";
    case safetensors::dtype::kFLOAT32:
      return "F32";
    case safetensors::dtype::kFLOAT64:
      return "F64";
    case safetensors::dtype::kINT32:
      return "I32";
    case safetensors::dtype::kINT64:
      return "I64";
    case safetensors::dtype::kINT16:
      return "I16";
    case safetensors::dtype::kINT8:
      return "I8";
    case safetensors::dtype::kUINT8:
      return "U8";
    case safetensors::dtype::kUINT16:
      return "U16";
    case safetensors::dtype::kUINT32:
      return "U32";
    case safetensors::dtype::kUINT64:
      return "U64";
    case safetensors::dtype::kBOOL:
      return "BOOL";
    default:
      return "UNKNOWN";
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

absl::Status ValidateTensorRange(const SafetensorTensorInfo& info,
                                 size_t data_size, absl::string_view name) {
  if (info.data_end < info.data_start) {
    return absl::DataLossError(
        absl::StrCat("Invalid tensor data range for: ", name));
  }
  if (info.data_end > data_size) {
    return absl::DataLossError(
        absl::StrCat("Tensor data out of range for: ", name));
  }
  return absl::OkStatus();
}

// Type trait specialized for types that we know how to convert.
template <safetensors::dtype From, class To>
struct ConvertInfo;

// Type trait to find out whether `ConvertInfo` has been specialized or not.
template <safetensors::dtype From, class To, class SFINAE = void>
struct CanConvert : std::false_type {};

template <safetensors::dtype From, class To>
struct CanConvert<From, To, std::void_t<decltype(ConvertInfo<From, To>{})>>
    : std::true_type {};

// Helper specialized to provide bound checking when converting values.
template <safetensors::dtype From, class To>
struct ConvertBoundCheck;

// Type trait to find out whether `ConvertBoundCheck` has been specialized or
// not.
template <safetensors::dtype From, class To, class SFINAE = void>
struct HasBoundCheck : std::false_type {};

template <safetensors::dtype From, class To>
struct HasBoundCheck<From, To,
                     std::void_t<decltype(ConvertBoundCheck<From, To>{})>>
    : std::true_type {};

#define CONVERT_INFO(ST_TYPE, TARGET_TYPE, STORAGE, CONVERT_FUNC)         \
  template <>                                                             \
  struct ConvertInfo<safetensors::dtype::k##ST_TYPE, TARGET_TYPE> {       \
    using Storage = STORAGE;                                              \
    static TARGET_TYPE Convert(STORAGE val) { return CONVERT_FUNC(val); } \
  }

#define CHECK_INFO(ST_TYPE, TARGET_TYPE, CHECK_EXPR)                      \
  template <>                                                             \
  struct ConvertBoundCheck<safetensors::dtype::k##ST_TYPE, TARGET_TYPE> { \
    template <class T>                                                    \
    static absl::Status Check(T val) {                                    \
      if (!(CHECK_EXPR)) {                                                \
        return absl::InvalidArgumentError(#CHECK_EXPR " is false.");      \
      }                                                                   \
      return absl::OkStatus();                                            \
    };                                                                    \
  }

CONVERT_INFO(FLOAT32, float, float, static_cast<float>);
CONVERT_INFO(BFLOAT16, float, uint16_t, safetensors::bfloat16_to_float);
CONVERT_INFO(FLOAT16, float, uint16_t, safetensors::fp16_to_float);
CONVERT_INFO(INT8, float, int8_t, static_cast<float>);
CONVERT_INFO(INT16, float, int16_t, static_cast<float>);
CONVERT_INFO(INT32, float, int32_t, static_cast<float>);
CONVERT_INFO(INT64, float, int64_t, static_cast<float>);
CONVERT_INFO(UINT8, float, uint8_t, static_cast<float>);
CONVERT_INFO(UINT16, float, uint16_t, static_cast<float>);
CONVERT_INFO(UINT32, float, uint32_t, static_cast<float>);
CONVERT_INFO(UINT64, float, uint64_t, static_cast<float>);
CONVERT_INFO(BOOL, float, bool, static_cast<float>);

CONVERT_INFO(INT8, int64_t, int8_t, static_cast<int64_t>);
CONVERT_INFO(INT16, int64_t, int16_t, static_cast<int64_t>);
CONVERT_INFO(INT32, int64_t, int32_t, static_cast<int64_t>);
CONVERT_INFO(INT64, int64_t, int64_t, static_cast<int64_t>);
CONVERT_INFO(UINT8, int64_t, uint8_t, static_cast<int64_t>);
CONVERT_INFO(UINT16, int64_t, uint16_t, static_cast<int64_t>);
CONVERT_INFO(UINT32, int64_t, uint32_t, static_cast<int64_t>);
CONVERT_INFO(UINT64, int64_t, uint64_t, static_cast<int64_t>);
CHECK_INFO(UINT64, int64_t, val <= std::numeric_limits<int64_t>::max());
CONVERT_INFO(BOOL, int64_t, bool, static_cast<int64_t>);

template <Type type>
struct TypedOwningBuffer {
  using value_type = typename NativeStorage<type>::type;
  void resize(size_t count) { buffer = OwningCpuBuffer::Allocate<type>(count); }
  value_type* data() const noexcept {
    return reinterpret_cast<value_type*>(buffer->data());
  }

  std::shared_ptr<OwningCpuBuffer> buffer;
};

template <class Container>
absl::StatusOr<Container> ConvertTensorTo(const SafetensorTensorInfo& info,
                                          const std::byte* data_base) {
  using T = typename Container::value_type;
  LRT_TENSOR_ASSIGN_OR_RETURN(const size_t num_elements,
                              NumElements(info.shape));
  const size_t bytes = info.data_end - info.data_start;
  const std::byte* data_ptr = data_base + info.data_start;

  Container values;
  values.resize(num_elements);
  auto* values_data = values.data();

#define CONVERT_CASE(ST_TYPE)                                                  \
  case safetensors::dtype::k##ST_TYPE: {                                       \
    if constexpr (CanConvert<safetensors::dtype::k##ST_TYPE, T>::value) {      \
      using Info = ConvertInfo<safetensors::dtype::k##ST_TYPE, T>;             \
      if (bytes != num_elements * sizeof(typename Info::Storage)) {            \
        return absl::InvalidArgumentError(#ST_TYPE                             \
                                          " tensor byte size mismatch");       \
      }                                                                        \
      const typename Info::Storage* src =                                      \
          reinterpret_cast<const typename Info::Storage*>(data_ptr);           \
      for (size_t i = 0; i < num_elements; ++i) {                              \
        if constexpr (HasBoundCheck<safetensors::dtype::k##ST_TYPE,            \
                                    T>::value) {                               \
          LRT_TENSOR_RETURN_IF_ERROR(                                          \
              (ConvertBoundCheck<safetensors::dtype::k##ST_TYPE, T>::Check(    \
                  src[i])));                                                   \
        }                                                                      \
        values_data[i] = Info::Convert(src[i]);                                \
      }                                                                        \
    } else {                                                                   \
      return absl::InvalidArgumentError(                                       \
          absl::StrCat("Unsupported conversion from ", ToString(info.dtype))); \
    }                                                                          \
    break;                                                                     \
  }

  switch (info.dtype) {
    CONVERT_CASE(FLOAT32);
    CONVERT_CASE(BFLOAT16);
    CONVERT_CASE(FLOAT16);
    CONVERT_CASE(INT8);
    CONVERT_CASE(INT16);
    CONVERT_CASE(INT32);
    CONVERT_CASE(INT64);
    CONVERT_CASE(UINT8);
    CONVERT_CASE(UINT16);
    CONVERT_CASE(UINT32);
    CONVERT_CASE(UINT64);
    CONVERT_CASE(BOOL);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported conversion dtype: ", ToString(info.dtype)));
  }
#undef CONVERT_CASE

  return values;
}

enum class QuantParamMode {
  kScalar,
  kPerChannelDim0,
  kPerElement,
};

template <typename T>
struct MinijsonTypeTraits;

template <>
struct MinijsonTypeTraits<std::string> {
  using type = minijson::string;
};

template <>
struct MinijsonTypeTraits<int> {
  using type = minijson::number;
};

template <>
struct MinijsonTypeTraits<bool> {
  using type = minijson::boolean;
};

template <>
struct MinijsonTypeTraits<QuantizationConfig::Method> {
  using type = minijson::string;
};

template <>
struct MinijsonTypeTraits<QuantizationConfig::Format> {
  using type = minijson::string;
};

template <typename TargetType>
struct ValueParser {
  static absl::StatusOr<TargetType> Parse(
      const typename MinijsonTypeTraits<TargetType>::type& raw) {
    return static_cast<TargetType>(raw);
  }
};

template <>
struct ValueParser<QuantizationConfig::Method> {
  static absl::StatusOr<QuantizationConfig::Method> Parse(absl::string_view s) {
    if (s == "compressed-tensors") {
      return QuantizationConfig::Method::kCompressedTensors;
    }
    return QuantizationConfig::Method::kUnknown;
  }
};

template <>
struct ValueParser<QuantizationConfig::Format> {
  static absl::StatusOr<QuantizationConfig::Format> Parse(absl::string_view s) {
    if (s == "pack-quantized") {
      return QuantizationConfig::Format::kPackQuantized;
    }
    if (s == "int-quantized") {
      return QuantizationConfig::Format::kIntQuantized;
    }
    return QuantizationConfig::Format::kUnknown;
  }
};

template <typename TargetType,
          typename MinijsonType = typename MinijsonTypeTraits<TargetType>::type>
absl::StatusOr<TargetType> GetJsonField(const minijson::object& obj,
                                        absl::string_view key) {
  minijson::value v;
  if (!obj.at(std::string(key), &v)) {
    return absl::NotFoundError(
        absl::StrCat("Key '", key, "' not found in JSON object"));
  }
  const MinijsonType* val_ptr = v.as<MinijsonType>();
  if (val_ptr == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Field '", key, "' is not of expected type"));
  }
  return ValueParser<TargetType>::Parse(*val_ptr);
}

#define ASSIGN_IF_OK(DECL, ...)                                             \
  if (auto status_or_##__LINE__ = (__VA_ARGS__); status_or_##__LINE__.ok()) \
  DECL = std::move(*status_or_##__LINE__)

absl::StatusOr<QuantizationConfig> ParseQuantizationConfig(
    std::string& quant_cfg_json) {
  minijson::value val;
  const char* json_str = quant_cfg_json.data();
  if (minijson::parse(json_str, val) != minijson::no_error) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to parse quantization_config JSON: ", quant_cfg_json));
  }

  const minijson::object* root_obj = val.as<minijson::object>();
  if (root_obj == nullptr) {
    return absl::InvalidArgumentError(
        "quantization_config in safetensors header is not a JSON object");
  }

  QuantizationConfig cfg;

  ASSIGN_IF_OK(cfg.quant_method, GetJsonField<QuantizationConfig::Method>(
                                     *root_obj, "quant_method"));
  ASSIGN_IF_OK(cfg.format,
               GetJsonField<QuantizationConfig::Format>(*root_obj, "format"));

  // Look inside config_groups for quantization parameters
  minijson::value config_groups_val;
  if (root_obj->at("config_groups", &config_groups_val)) {
    if (const minijson::object* groups_obj =
            config_groups_val.as<minijson::object>();
        groups_obj != nullptr) {
      if (groups_obj->keys().size() > 1) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Multiple config_groups found (", groups_obj->keys().size(),
            "); currently only a single config_group is supported"));
      }

      for (const std::string& group_name : groups_obj->keys()) {
        minijson::value group_val;
        if (groups_obj->at(group_name, &group_val)) {
          if (const minijson::object* group_obj =
                  group_val.as<minijson::object>();
              group_obj != nullptr) {
            ASSIGN_IF_OK(cfg.format, GetJsonField<QuantizationConfig::Format>(
                                         *group_obj, "format"));
            ASSIGN_IF_OK(cfg.num_bits,
                         GetJsonField<int>(*group_obj, "num_bits"));
            ASSIGN_IF_OK(cfg.group_size,
                         GetJsonField<int>(*group_obj, "group_size"));
            ASSIGN_IF_OK(cfg.symmetric,
                         GetJsonField<bool>(*group_obj, "symmetric"));

            minijson::value weights_val;
            if (group_obj->at("weights", &weights_val)) {
              if (const minijson::object* weights_obj =
                      weights_val.as<minijson::object>();
                  weights_obj != nullptr) {
                ASSIGN_IF_OK(cfg.num_bits,
                             GetJsonField<int>(*weights_obj, "num_bits"));
                ASSIGN_IF_OK(cfg.group_size,
                             GetJsonField<int>(*weights_obj, "group_size"));
                ASSIGN_IF_OK(cfg.symmetric,
                             GetJsonField<bool>(*weights_obj, "symmetric"));
              }
            }
          }
        }
      }
    }
  }

  if (cfg.format == QuantizationConfig::Format::kPackQuantized &&
      (cfg.num_bits <= 0 || cfg.group_size <= 0)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid quantization_config in safetensors header: num_bits=",
        cfg.num_bits, " group_size=", cfg.group_size));
  }

  return cfg;
}

#undef ASSIGN_IF_OK

}  // namespace

// static
absl::StatusOr<Type> SafetensorLoader::DtypeToType(safetensors::dtype dtype) {
  switch (dtype) {
    case safetensors::dtype::kBFLOAT16:
      return Type::kBF16;
    case safetensors::dtype::kFLOAT16:
      return Type::kFP16;
    case safetensors::dtype::kFLOAT32:
      return Type::kFP32;
    case safetensors::dtype::kFLOAT64:
      return Type::kFP64;
    case safetensors::dtype::kINT16:
      return Type::kI16;
    case safetensors::dtype::kINT32:
      return Type::kI32;
    case safetensors::dtype::kINT64:
      return Type::kI64;
    case safetensors::dtype::kINT8:
      return Type::kI8;
    case safetensors::dtype::kUINT8:
      return Type::kU8;
    case safetensors::dtype::kUINT16:
      return Type::kU16;
    case safetensors::dtype::kUINT32:
      return Type::kU32;
    case safetensors::dtype::kUINT64:
      return Type::kU64;
    case safetensors::dtype::kBOOL:
      return Type::kBOOL;
    default:
      break;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported dtype: ", ToString(dtype)));
}

absl::Status SafetensorLoader::AddSafetensorFile(const std::string& path) {
  TRACE_EVENT(gemma4::kGemma4Category, "AddSafetensorFile");
  auto st = std::make_shared<safetensors::safetensors_t>();
  std::string warn, err;
  bool ret = safetensors::mmap_from_file(path, st.get(), &warn, &err);
  if (!ret) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse safetensor file: ", err));
  }

  if (!warn.empty()) {
    ABSL_LOG(WARNING) << "Safetensor warning: " << warn;
  }

  // Validate data offsets
  if (!safetensors::validate_data_offsets(*st, err)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid data offsets: ", err));
  }

  const std::byte* data_base =
      reinterpret_cast<const std::byte*>(st->databuffer_addr);
  const size_t data_size = st->databuffer_size;

  auto storage_info = std::make_shared<TensorStorageInfo>(TensorStorageInfo{
      .file_data = st, .data_base = data_base, .data_size = data_size});

  // Convert safetensors-cpp tensor info to our format.
  const std::vector<std::string>& tensor_keys = st->tensors.keys();
  for (const std::string& name : tensor_keys) {
    TRACE_EVENT(gemma4::kGemma4Category, "AddTensor");
    if (tensor_infos_.contains(name)) {
      return absl::AlreadyExistsError(absl::StrCat(
          "Duplicate tensor name across safetensor files: ", name));
    }

    safetensors::tensor_t tensor_info;
    if (!st->tensors.at(name, &tensor_info)) {
      continue;
    }

    SafetensorTensorInfo info;
    info.name = name;
    info.dtype = tensor_info.dtype;

    // Convert shape
    info.shape.assign(tensor_info.shape.begin(), tensor_info.shape.end());

    info.data_start = tensor_info.data_offsets[0];
    info.data_end = tensor_info.data_offsets[1];
    info.storage = storage_info;

    tensor_infos_[name] = std::move(info);
  }

  // Extract quantization_config from header metadata if available
  std::string quant_cfg_json;
  if (st->metadata.at("quantization_config", &quant_cfg_json)) {
    LRT_TENSOR_ASSIGN_OR_RETURN(quant_config_,
                                ParseQuantizationConfig(quant_cfg_json));
    ABSL_LOG(INFO) << "Parsed header quantization_config: format="
                   << quant_config_->format
                   << " num_bits=" << quant_config_->num_bits
                   << " group_size=" << quant_config_->group_size;
  }

  ABSL_LOG(INFO) << "Loaded safetensor file: " << path
                 << " tensors: " << tensor_keys.size();
  return absl::OkStatus();
}

absl::StatusOr<SafetensorLoader> SafetensorLoader::Load(
    const std::string& path) {
  TRACE_EVENT(gemma4::kGemma4Category, "Initialize weight loader");
  namespace fs = std::filesystem;
  SafetensorLoader loader;

  std::error_code ec;
  const fs::path input_path(path);
  const bool is_directory = fs::is_directory(input_path, ec);
  if (ec) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to inspect path ", path, ": ", ec.message()));
  }

  if (is_directory) {
    std::vector<std::string> safetensor_files;
    for (const auto& entry : fs::directory_iterator(input_path, ec)) {
      if (ec) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Failed to read directory ", path, ": ", ec.message()));
      }
      if (!entry.is_regular_file()) {
        continue;
      }
      const std::string filename = entry.path().filename().string();
      if (!absl::EndsWith(filename, ".safetensors")) {
        continue;
      }
      safetensor_files.push_back(entry.path().string());
    }
    if (safetensor_files.empty()) {
      return absl::NotFoundError(
          absl::StrCat("No .safetensors files found in directory: ", path));
    }
    std::sort(safetensor_files.begin(), safetensor_files.end());
    for (const std::string& file : safetensor_files) {
      absl::Status status = loader.AddSafetensorFile(file);
      if (!status.ok()) {
        return status;
      }
    }
    ABSL_LOG(INFO) << "Loaded " << safetensor_files.size()
                   << " safetensor files from directory " << path << " with "
                   << loader.tensor_infos_.size() << " tensors";
    return loader;
  }

  absl::Status status = loader.AddSafetensorFile(path);
  if (!status.ok()) {
    return status;
  }
  ABSL_LOG(INFO) << "Loaded safetensor file with "
                 << loader.tensor_infos_.size()
                 << " tensors using safetensors-cpp";
  return loader;
}

std::vector<std::string> SafetensorLoader::GetTensorNames() const {
  std::vector<std::string> names;
  names.reserve(tensor_infos_.size());
  for (const auto& [name, info] : tensor_infos_) {
    names.push_back(name);
  }
  return names;
}

absl::StatusOr<SafetensorTensorInfo> SafetensorLoader::GetTensorInfo(
    absl::string_view name) const {
  auto it = tensor_infos_.find(name);
  if (it == tensor_infos_.end()) {
    return absl::NotFoundError(absl::StrCat("Tensor not found: ", name));
  }
  return it->second;
}

absl::StatusOr<TensorHandle> SafetensorLoader::LoadTensor(
    absl::string_view name) const {
  TRACE_EVENT(gemma4::kGemma4Category, "LoadTensor");
  ABSL_VLOG(3) << "Loading tensor " << name;
  LRT_TENSOR_ASSIGN_OR_RETURN(SafetensorTensorInfo info, GetTensorInfo(name));
  LRT_TENSOR_ASSIGN_OR_RETURN(Type type, DtypeToType(info.dtype));

  const TensorStorageInfo& storage = *info.storage;
  if (storage.data_base == nullptr || storage.file_data == nullptr) {
    return absl::FailedPreconditionError("Safetensor storage is invalid");
  }

  LRT_TENSOR_RETURN_IF_ERROR(
      ValidateTensorRange(info, storage.data_size, name));

  auto ReadTensor =
      [&](absl::flat_hash_map<std::string, SafetensorTensorInfo>::const_iterator
              tensor_info_it,
          auto as) -> absl::StatusOr<std::vector<decltype(as)>> {
    absl::string_view tensor_name = tensor_info_it->first;
    LRT_TENSOR_RETURN_IF_ERROR(ValidateTensorRange(
        tensor_info_it->second, tensor_info_it->second.storage->data_size,
        tensor_name));
    return ConvertTensorTo<std::vector<decltype(as)>>(
        tensor_info_it->second, tensor_info_it->second.storage->data_base);
  };

  const std::byte* data_ptr = storage.data_base + info.data_start;
  size_t data_size = info.data_end - info.data_start;
  std::shared_ptr<Buffer> buffer;
  std::shared_ptr<Quantization> quantization;
  switch (type) {
    case Type::kU8:
    case Type::kI8:
    case Type::kI32: {
      buffer = MakeMappedBuffer(storage.file_data, data_ptr, data_size);

      if (!quant_config_.has_value()) {
        break;
      }

      auto FindDataFor =
          [&](std::initializer_list<absl::string_view> suffixes) {
            for (absl::string_view suffix : suffixes) {
              if (auto it = tensor_infos_.find(absl::StrCat(name, suffix));
                  it != tensor_infos_.end()) {
                return it;
              }
            }
            return tensor_infos_.end();
          };

      auto tensor_info_it =
          FindDataFor({".weight_scale", ".scale", ".scales", ".weight_scales"});
      if (tensor_info_it == tensor_infos_.end()) {
        break;
      }
      LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> scales,
                                  ReadTensor(tensor_info_it, /*as=*/float{}));
      if (scales.empty()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Scale tensor is empty for: ", name));
      }

      std::vector<int64_t> zero_points(1, 0);
      if (auto zp_it = FindDataFor({".weight_zero_point", ".zero_point"});
          zp_it != tensor_infos_.end()) {
        LRT_TENSOR_ASSIGN_OR_RETURN(zero_points,
                                    ReadTensor(zp_it, /*as=*/int64_t{}));
        if (zero_points.empty()) {
          return absl::InvalidArgumentError(
              absl::StrCat("Zero-point tensor is empty for: ", name));
        }
      }

      if (quant_config_->format == QuantizationConfig::Format::kPackQuantized &&
          quant_config_->num_bits == 4 && info.shape.size() == 2) {
        const size_t d_out = info.shape[0];
        const size_t d_in_packed = info.shape[1];
        const size_t packed_element_count = BufferSize(type, 1) * 2;
        const size_t d_in = d_in_packed * packed_element_count;

        quantization = std::make_shared<BlockwiseQuantization>(
            std::move(scales), std::move(zero_points),
            static_cast<int>(quant_config_->group_size),
            /*quantized_dimension=*/0);

        return TensorHandle(TensorInit{
            .name = std::string(name),
            .type = Type::kI4,
            .shape = {static_cast<int>(d_out), static_cast<int>(d_in)},
            .buffer = buffer,
            .quantization = quantization});
      } else {
        quantization = std::make_shared<PerChannelAffineQuantization>(
            std::move(scales), std::move(zero_points),
            /*quantized_dimension=*/0);
      }
      break;
    }
    case Type::kBF16:
    case Type::kFP16:
    case Type::kFP32:
    case Type::kFP64:
    case Type::kI64:
    case Type::kI16:
    case Type::kU16:
    case Type::kU32:
    case Type::kU64:
    case Type::kBOOL:
      buffer = MakeMappedBuffer(storage.file_data, data_ptr, data_size);
      break;
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unsupported type for loading: ", ToString(type)));
  }

  return TensorHandle(TensorInit{
      .name = std::string(name),
      .type = type,
      .shape = std::vector<int>(info.shape.begin(), info.shape.end()),
      .buffer = buffer,
      .quantization = quantization});
}

absl::StatusOr<absl::flat_hash_map<std::string, TensorHandle>>
SafetensorLoader::LoadAllTensors() const {
  absl::flat_hash_map<std::string, TensorHandle> tensors;
  for (const auto& [name, info] : tensor_infos_) {
    absl::StatusOr<TensorHandle> tensor_or = LoadTensor(name);
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
SafetensorLoader::LoadWeightsWithMapping(
    const absl::flat_hash_map<std::string, std::string>& name_mapping) const {
  TRACE_EVENT(gemma4::kGemma4Category, "LoadWeightsWithMapping");
  absl::flat_hash_map<std::string, TensorHandle> tensors;
  for (const auto& [hf_name, model_name] : name_mapping) {
    absl::StatusOr<TensorHandle> tensor_or = LoadTensor(hf_name);
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

absl::flat_hash_map<std::string, std::string> GetGemma4WeightMapping(
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

    // Layer norms
    mapping[absl::StrCat(hf_prefix, ".input_layernorm.weight")] =
        absl::StrCat(model_prefix, ".input_layernorm.weight");
    mapping[absl::StrCat(hf_prefix, ".post_attention_layernorm.weight")] =
        absl::StrCat(model_prefix, ".post_attention_layernorm.weight");
    mapping[absl::StrCat(hf_prefix, ".pre_feedforward_layernorm.weight")] =
        absl::StrCat(model_prefix, ".pre_feedforward_layernorm.weight");
    mapping[absl::StrCat(hf_prefix, ".post_feedforward_layernorm.weight")] =
        absl::StrCat(model_prefix, ".post_feedforward_layernorm.weight");

    // Gemma 4 per-layer components
    mapping[absl::StrCat(hf_prefix, ".per_layer_input_gate.weight")] =
        absl::StrCat(model_prefix, ".per_layer_input_gate.weight");
    mapping[absl::StrCat(hf_prefix, ".per_layer_projection.weight")] =
        absl::StrCat(model_prefix, ".per_layer_projection.weight");
    mapping[absl::StrCat(hf_prefix, ".post_per_layer_input_norm.weight")] =
        absl::StrCat(model_prefix, ".post_per_layer_input_norm.weight");
    mapping[absl::StrCat(hf_prefix, ".layer_scalar")] =
        absl::StrCat(model_prefix, ".layer_scalar");
  }

  return mapping;
}

}  // namespace litert::tensor::examples

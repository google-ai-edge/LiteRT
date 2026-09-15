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
#include "tensor/examples/utils/safetensor_loader.h"
#ifndef LITERT_TENSOR_STANDALONE
#include "perfetto/tracing/track_event.h"  // from @perfetto
#include "tensor/examples/utils/perfetto_session.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <regex>
#include <string>
#include <system_error>  // NOLINT
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"             // from @com_google_absl
#include "absl/status/status.h"            // from @com_google_absl
#include "absl/status/statusor.h"          // from @com_google_absl
#include "absl/strings/match.h"            // from @com_google_absl
#include "absl/strings/str_cat.h"          // from @com_google_absl
#include "absl/strings/string_view.h"      // from @com_google_absl
#include "absl/strings/strip.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/utils/minijson.h"
#include "tensor/examples/utils/safetensors.h"
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
struct ValueParser<int> {
  static absl::StatusOr<int> Parse(minijson::number value) {
    if (!std::isfinite(value) || std::trunc(value) != value ||
        value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
      return absl::InvalidArgumentError("JSON integer is out of range");
    }
    return static_cast<int>(value);
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

absl::StatusOr<std::vector<std::string>> ReadTargets(
    const minijson::object& object, absl::string_view field) {
  minijson::value value;
  if (!object.at(std::string(field), &value)) {
    return absl::InvalidArgumentError(absl::StrCat("Missing ", field));
  }
  const auto* array = value.as<minijson::array>();
  if (array == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(field, " must be an array"));
  }
  std::vector<std::string> targets;
  for (const auto& item : *array) {
    const auto* target = item.as<minijson::string>();
    if (target == nullptr || target->empty()) {
      return absl::InvalidArgumentError(
          "Quantization targets must be nonempty strings");
    }
    if (absl::StartsWith(*target, "re:")) {
      try {
        std::regex pattern(target->substr(3));
      } catch (const std::regex_error&) {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid target regex: ", *target));
      }
    }
    targets.push_back(*target);
  }
  return targets;
}

bool MatchesTarget(absl::string_view module, const std::string& target) {
  if (absl::StartsWith(target, "re:")) {
    // compressed-tensors uses re.match: the expression starts at the beginning
    // of the module name; callers can use '$' when they require an exact end.
    return std::regex_search(module.begin(), module.end(),
                             std::regex(target.substr(3)),
                             std::regex_constants::match_continuous);
  }
  return module == target;
}

absl::StatusOr<bool> HasStaticActivations(const minijson::object& group,
                                          absl::string_view field) {
  minijson::value value;
  if (!group.at(std::string(field), &value) || value.as<minijson::null_t>()) {
    return false;
  }
  const auto* config = value.as<minijson::object>();
  if (config == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat(field, " must be an object or null"));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(int bits, GetJsonField<int>(*config, "num_bits"));
  LRT_TENSOR_ASSIGN_OR_RETURN(bool dynamic,
                              GetJsonField<bool>(*config, "dynamic"));
  LRT_TENSOR_ASSIGN_OR_RETURN(bool symmetric,
                              GetJsonField<bool>(*config, "symmetric"));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::string type,
                              GetJsonField<std::string>(*config, "type"));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::string strategy,
                              GetJsonField<std::string>(*config, "strategy"));
  if (bits != 8 || dynamic || !symmetric || type != "int" ||
      strategy != "tensor") {
    return absl::UnimplementedError(
        "Only static symmetric int8 tensor activation quantization is "
        "supported");
  }
  return true;
}

}  // namespace

absl::Status SafetensorLoader::LoadCompanionConfig(const std::string& path) {
  std::error_code ec;
  const bool exists = std::filesystem::exists(path, ec);
  if (ec)
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot inspect ", path, ": ", ec.message()));
  if (!exists) return absl::OkStatus();
  std::ifstream input(path);
  if (!input)
    return absl::InvalidArgumentError(absl::StrCat("Cannot read ", path));
  const std::string json((std::istreambuf_iterator<char>(input)), {});
  minijson::value root;
  const char* text = json.c_str();
  if (minijson::parse(text, root) != minijson::no_error ||
      root.as<minijson::object>() == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid model config JSON: ", path));
  }
  minijson::value value;
  if (!root.as<minijson::object>()->at("quantization_config", &value) ||
      value.as<minijson::null_t>())
    return absl::OkStatus();
  const auto* config = value.as<minijson::object>();
  if (config == nullptr)
    return absl::InvalidArgumentError("quantization_config must be an object");
  auto method =
      GetJsonField<QuantizationConfig::Method>(*config, "quant_method");
  if (!method.ok() ||
      *method != QuantizationConfig::Method::kCompressedTensors) {
    return absl::OkStatus();
  }
  minijson::value ignored;
  if (config->at("ignore", &ignored)) {
    LRT_TENSOR_ASSIGN_OR_RETURN(ignored_targets_,
                                ReadTargets(*config, "ignore"));
  }
  minijson::value groups_value;
  if (!config->at("config_groups", &groups_value) ||
      groups_value.as<minijson::object>() == nullptr) {
    return absl::InvalidArgumentError(
        "compressed-tensors config_groups must be an object");
  }
  const auto& groups = *groups_value.as<minijson::object>();
  for (const auto& name : groups.keys()) {
    minijson::value group_value;
    groups.at(name, &group_value);
    const auto* group = group_value.as<minijson::object>();
    if (group == nullptr)
      return absl::InvalidArgumentError("Quantization group must be an object");
    minijson::value weights_value;
    if (!group->at("weights", &weights_value) ||
        weights_value.as<minijson::object>() == nullptr) {
      return absl::InvalidArgumentError(
          "Quantization group must describe weights");
    }
    const auto& weights = *weights_value.as<minijson::object>();
    TargetedQuantizationConfig parsed;
    parsed.weights.quant_method = *method;
    auto format = GetJsonField<QuantizationConfig::Format>(*group, "format");
    if (absl::IsNotFound(format.status())) {
      format = GetJsonField<QuantizationConfig::Format>(*config, "format");
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(parsed.weights.format, std::move(format));
    LRT_TENSOR_ASSIGN_OR_RETURN(parsed.weights.num_bits,
                                GetJsonField<int>(weights, "num_bits"));
    LRT_TENSOR_ASSIGN_OR_RETURN(parsed.weights.symmetric,
                                GetJsonField<bool>(weights, "symmetric"));
    LRT_TENSOR_ASSIGN_OR_RETURN(parsed.strategy,
                                GetJsonField<std::string>(weights, "strategy"));
    LRT_TENSOR_ASSIGN_OR_RETURN(std::string type,
                                GetJsonField<std::string>(weights, "type"));
    LRT_TENSOR_ASSIGN_OR_RETURN(bool dynamic,
                                GetJsonField<bool>(weights, "dynamic"));
    if ((parsed.weights.num_bits != 2 && parsed.weights.num_bits != 4 &&
         parsed.weights.num_bits != 8) ||
        !parsed.weights.symmetric || dynamic || type != "int" ||
        (parsed.strategy != "channel" && parsed.strategy != "group") ||
        (parsed.weights.format != QuantizationConfig::Format::kPackQuantized &&
         parsed.weights.format != QuantizationConfig::Format::kIntQuantized)) {
      return absl::UnimplementedError(
          absl::StrCat("Unsupported compressed-tensors weight config: ", name));
    }
    if (parsed.weights.num_bits != 8 &&
        parsed.weights.format != QuantizationConfig::Format::kPackQuantized) {
      return absl::UnimplementedError(
          "Two- and four-bit weights must use pack-quantized format");
    }
    if (parsed.strategy == "group") {
      LRT_TENSOR_ASSIGN_OR_RETURN(parsed.weights.group_size,
                                  GetJsonField<int>(weights, "group_size"));
      if (parsed.weights.group_size <= 0)
        return absl::InvalidArgumentError("Weight group_size must be positive");
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(parsed.targets, ReadTargets(*group, "targets"));
    if (parsed.targets.empty())
      return absl::InvalidArgumentError("Quantization group has no targets");
    LRT_TENSOR_ASSIGN_OR_RETURN(
        parsed.input_activations,
        HasStaticActivations(*group, "input_activations"));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        parsed.output_activations,
        HasStaticActivations(*group, "output_activations"));
    targeted_configs_.push_back(std::move(parsed));
  }
  ABSL_LOG(INFO) << "Loaded " << targeted_configs_.size()
                 << " compressed-tensors groups from " << path;
  return absl::OkStatus();
}

absl::StatusOr<const SafetensorLoader::TargetedQuantizationConfig*>
SafetensorLoader::FindWeightConfig(absl::string_view module) const {
  for (const auto& target : ignored_targets_) {
    if (MatchesTarget(module, target)) return nullptr;
  }
  const TargetedQuantizationConfig* result = nullptr;
  for (const auto& group : targeted_configs_) {
    const bool matched = std::any_of(
        group.targets.begin(), group.targets.end(),
        [&](const auto& target) { return MatchesTarget(module, target); });
    if (matched) {
      if (result != nullptr)
        return absl::InvalidArgumentError(
            absl::StrCat("Multiple quantization groups match ", module));
      result = &group;
    }
  }
  return result;
}

absl::StatusOr<TensorHandle> SafetensorLoader::LoadCompressedWeight(
    absl::string_view name, absl::string_view module,
    const TargetedQuantizationConfig& config) const {
  const int bits = config.weights.num_bits;
  const std::string physical_name =
      absl::StrCat(module, bits == 8 ? ".weight" : ".weight_packed");
  LRT_TENSOR_ASSIGN_OR_RETURN(SafetensorTensorInfo info,
                              GetTensorInfo(physical_name));
  LRT_TENSOR_RETURN_IF_ERROR(
      ValidateTensorRange(info, info.storage->data_size, physical_name));
  const auto ReadFloat =
      [&](absl::string_view suffix) -> absl::StatusOr<std::vector<float>> {
    const std::string key = absl::StrCat(module, suffix);
    auto metadata = GetTensorInfo(key);
    if (!metadata.ok())
      return absl::InvalidArgumentError(
          absl::StrCat("Missing quantization tensor: ", key));
    if (metadata->dtype != safetensors::dtype::kFLOAT32 &&
        metadata->dtype != safetensors::dtype::kFLOAT16 &&
        metadata->dtype != safetensors::dtype::kBFLOAT16) {
      return absl::InvalidArgumentError(
          absl::StrCat("Quantization scale must be floating point: ", key));
    }
    LRT_TENSOR_RETURN_IF_ERROR(
        ValidateTensorRange(*metadata, metadata->storage->data_size, key));
    return ConvertTensorTo<std::vector<float>>(*metadata,
                                               metadata->storage->data_base);
  };
  const auto ReadInteger =
      [&](absl::string_view suffix) -> absl::StatusOr<std::vector<int64_t>> {
    const std::string key = absl::StrCat(module, suffix);
    auto metadata = GetTensorInfo(key);
    if (!metadata.ok())
      return absl::InvalidArgumentError(
          absl::StrCat("Missing quantization tensor: ", key));
    LRT_TENSOR_RETURN_IF_ERROR(
        ValidateTensorRange(*metadata, metadata->storage->data_size, key));
    return ConvertTensorTo<std::vector<int64_t>>(*metadata,
                                                 metadata->storage->data_base);
  };
  std::vector<int64_t> shape = info.shape;
  if (bits != 8) {
    auto shape_info = GetTensorInfo(absl::StrCat(module, ".weight_shape"));
    if (!shape_info.ok() || shape_info->dtype != safetensors::dtype::kINT64 ||
        shape_info->shape != std::vector<int64_t>{2}) {
      return absl::InvalidArgumentError(
          absl::StrCat("Packed weight requires I64 weight_shape[2]: ", module));
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(shape, ReadInteger(".weight_shape"));
  }
  if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0 ||
      shape[0] > std::numeric_limits<int32_t>::max() ||
      shape[1] > std::numeric_limits<int32_t>::max()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Quantized weight must have positive int32 matrix dimensions: ",
        module));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const size_t elements, NumElements(shape));
  const size_t rows = shape[0];
  const size_t columns = shape[1];
  const size_t packed_columns = (columns + 32 / bits - 1) / (32 / bits);
  const size_t bytes = info.data_end - info.data_start;
  if (bits == 8) {
    if (info.dtype != safetensors::dtype::kINT8 || bytes != elements) {
      return absl::InvalidArgumentError(
          absl::StrCat("Eight-bit weights require an I8 matrix: ", module));
    }
  } else if (info.dtype != safetensors::dtype::kINT32 ||
             info.shape !=
                 std::vector<int64_t>{static_cast<int64_t>(rows),
                                      static_cast<int64_t>(packed_columns)} ||
             bytes != rows * packed_columns * sizeof(uint32_t)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Packed I32 storage disagrees with weight_shape: ", module));
  }
  size_t groups_per_row = 1;
  if (config.strategy == "group") {
    if (columns % config.weights.group_size != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Weight columns must be divisible by group_size: ", module));
    }
    groups_per_row = columns / config.weights.group_size;
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> scales,
                              ReadFloat(".weight_scale"));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto scale_info, GetTensorInfo(absl::StrCat(module, ".weight_scale")));
  const std::vector<int64_t> expected_scale_shape = {
      static_cast<int64_t>(rows), static_cast<int64_t>(groups_per_row)};
  if ((scale_info.shape != expected_scale_shape &&
       !(groups_per_row == 1 &&
         scale_info.shape ==
             std::vector<int64_t>{static_cast<int64_t>(rows)})) ||
      scales.size() != rows * groups_per_row ||
      std::any_of(scales.begin(), scales.end(), [](float scale) {
        return !std::isfinite(scale) || scale <= 0;
      })) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid scale shape or values for ", module));
  }
  if (tensor_infos_.contains(absl::StrCat(module, ".weight_g_idx"))) {
    return absl::UnimplementedError(absl::StrCat(
        "Reordered quantization groups are unsupported: ", module));
  }
  if (tensor_infos_.contains(absl::StrCat(module, ".weight_zero_point"))) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto zeros, ReadInteger(".weight_zero_point"));
    if ((zeros.size() != 1 && zeros.size() != scales.size()) ||
        std::any_of(zeros.begin(), zeros.end(),
                    [](int64_t zero) { return zero != 0; })) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Symmetric weights require zero-valued zero points: ", module));
    }
  }
  for (const auto& [needed, suffix] :
       {std::pair<bool, absl::string_view>{config.input_activations,
                                           ".input_scale"},
        {config.output_activations, ".output_scale"}}) {
    if (!needed) continue;
    LRT_TENSOR_ASSIGN_OR_RETURN(auto activation_scale, ReadFloat(suffix));
    if (activation_scale.size() != 1 || !std::isfinite(activation_scale[0]) ||
        activation_scale[0] <= 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Static activation scale must be one positive finite value: ", module,
          suffix));
    }
    const std::string zero_suffix =
        absl::StrCat(suffix.substr(0, suffix.size() - 6), "_zero_point");
    if (tensor_infos_.contains(absl::StrCat(module, zero_suffix))) {
      LRT_TENSOR_ASSIGN_OR_RETURN(auto zero, ReadInteger(zero_suffix));
      if (zero.size() != 1 || zero[0] != 0) {
        return absl::InvalidArgumentError(
            "Symmetric activation zero point must be zero");
      }
    }
  }

  const auto* source = reinterpret_cast<const uint8_t*>(
      info.storage->data_base + info.data_start);
  std::shared_ptr<Buffer> buffer;
  if (bits == 8) {
    buffer =
        MakeMappedBuffer(info.storage->file_data,
                         reinterpret_cast<const std::byte*>(source), bytes);
  } else {
    auto unpacked = OwningCpuBuffer::Allocate<Type::kI4>(elements);
    auto* destination = reinterpret_cast<uint8_t*>(unpacked->data());
    if (bits == 4 && columns % 8 == 0) {
      // CT stores offset-binary nibbles; tensor I4 uses signed two's
      // complement.
      for (size_t i = 0; i < bytes; ++i) destination[i] = source[i] ^ 0x88;
    } else if (bits == 2 && columns % 16 == 0) {
      // Each source byte contains four 2-bit values, expanded to two I4 bytes.
      std::array<uint16_t, 256> table{};
      for (int value = 0; value < 256; ++value) {
        for (int slot = 0; slot < 4; ++slot) {
          table[value] |= ((((value >> (slot * 2)) & 3) - 2) & 15)
                          << (slot * 4);
        }
      }
      for (size_t i = 0; i < bytes; ++i) {
        const uint16_t expanded = table[source[i]];
        destination[2 * i] = expanded & 255;
        destination[2 * i + 1] = expanded >> 8;
      }
    } else {
      // Trim row padding recorded by weight_shape, including odd-width rows.
      std::memset(destination, 0, unpacked->size());
      for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
          const size_t source_byte =
              row * packed_columns * 4 + column * bits / 8;
          const int code = (source[source_byte] >> ((column * bits) % 8)) &
                           ((1 << bits) - 1);
          const uint8_t nibble = (code - (1 << (bits - 1))) & 15;
          const size_t index = row * columns + column;
          destination[index / 2] |= nibble << ((index % 2) * 4);
        }
      }
    }
    buffer = std::move(unpacked);
  }
  std::shared_ptr<Quantization> quantization;
  if (config.strategy == "group") {
    quantization = std::make_shared<BlockwiseQuantization>(
        std::move(scales), std::vector<int64_t>{0}, config.weights.group_size,
        0);
  } else {
    quantization = std::make_shared<PerChannelAffineQuantization>(
        std::move(scales), std::vector<int64_t>{0}, 0);
  }
  return TensorHandle(TensorInit{
      .name = std::string(name),
      .type = bits == 8 ? Type::kI8 : Type::kI4,
      .shape = {static_cast<int32_t>(rows), static_cast<int32_t>(columns)},
      .buffer = std::move(buffer),
      .quantization = std::move(quantization)});
}

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
#ifndef LITERT_TENSOR_STANDALONE
  TRACE_EVENT(kTensorApiCategory, "AddSafetensorFile");
#endif
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
#ifndef LITERT_TENSOR_STANDALONE
    TRACE_EVENT(kTensorApiCategory, "AddTensor");
#endif
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
#ifndef LITERT_TENSOR_STANDALONE
  TRACE_EVENT(kTensorApiCategory, "Initialize weight loader");
#endif
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
    LRT_TENSOR_RETURN_IF_ERROR(
        loader.LoadCompanionConfig((input_path / "config.json").string()));
    return loader;
  }

  absl::Status status = loader.AddSafetensorFile(path);
  if (!status.ok()) {
    return status;
  }
  ABSL_LOG(INFO) << "Loaded safetensor file with "
                 << loader.tensor_infos_.size()
                 << " tensors using safetensors-cpp";
  LRT_TENSOR_RETURN_IF_ERROR(loader.LoadCompanionConfig(
      (input_path.parent_path() / "config.json").string()));
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
#ifndef LITERT_TENSOR_STANDALONE
  TRACE_EVENT(kTensorApiCategory, "LoadTensor");
#endif
  ABSL_VLOG(3) << "Loading tensor " << name;
  absl::string_view module = name;
  if (absl::ConsumeSuffix(&module, ".weight") ||
      absl::ConsumeSuffix(&module, ".weight_packed")) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto* config, FindWeightConfig(module));
    if (config != nullptr) return LoadCompressedWeight(name, module, *config);
    if (!targeted_configs_.empty() &&
        tensor_infos_.contains(absl::StrCat(module, ".weight_packed"))) {
      return absl::InvalidArgumentError(absl::StrCat(
          "No quantization group matches packed weight: ", module));
    }
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(SafetensorTensorInfo info, GetTensorInfo(name));
  LRT_TENSOR_ASSIGN_OR_RETURN(Type type, DtypeToType(info.dtype));

  const TensorStorageInfo& storage = *info.storage;
  if (storage.data_base == nullptr || storage.file_data == nullptr) {
    return absl::FailedPreconditionError("Safetensor storage is invalid");
  }

  LRT_TENSOR_RETURN_IF_ERROR(
      ValidateTensorRange(info, storage.data_size, name));
  for (int64_t dim : info.shape) {
    if (dim < 0 || dim > std::numeric_limits<int32_t>::max()) {
      return absl::InvalidArgumentError(
          "Tensor dimensions must fit nonnegative int32 values");
    }
  }

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
    case Type::kFP16: {
      if (absl::EndsWith(name, "embed_tokens_per_layer.weight")) {
        buffer = MakeMappedBuffer(storage.file_data, data_ptr, data_size);
      } else {
        LRT_TENSOR_ASSIGN_OR_RETURN(
            auto converted, ConvertTensorTo<TypedOwningBuffer<Type::kFP32>>(
                                info, storage.data_base));
        buffer = std::move(converted.buffer);
        type = Type::kFP32;
      }
      break;
    }
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
#ifndef LITERT_TENSOR_STANDALONE
  TRACE_EVENT(kTensorApiCategory, "LoadWeightsWithMapping");
#endif
  absl::flat_hash_map<std::string, TensorHandle> tensors;
  for (const auto& [hf_name, model_name] : name_mapping) {
    absl::StatusOr<TensorHandle> tensor_or = LoadTensor(hf_name);
    if (!tensor_or.ok()) {
      if (!absl::IsNotFound(tensor_or.status())) return tensor_or.status();
      ABSL_LOG(WARNING) << "Failed to load tensor " << hf_name << ": "
                        << tensor_or.status();
      continue;
    }
    tensor_or->SetName(model_name);
    tensors[model_name] = std::move(*tensor_or);
    absl::string_view source_module = hf_name;
    absl::string_view model_module = model_name;
    if (absl::ConsumeSuffix(&source_module, ".weight") &&
        absl::ConsumeSuffix(&model_module, ".weight")) {
      for (absl::string_view suffix : {".input_scale", ".output_scale"}) {
        const std::string source_name = absl::StrCat(source_module, suffix);
        if (!tensor_infos_.contains(source_name)) continue;
        LRT_TENSOR_ASSIGN_OR_RETURN(auto scale, LoadTensor(source_name));
        const std::string destination_name = absl::StrCat(model_module, suffix);
        scale.SetName(destination_name);
        tensors[destination_name] = std::move(scale);
      }
    }
  }
  return tensors;
}
}  // namespace litert::tensor::examples

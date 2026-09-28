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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <regex>  // NOLINT
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
#include "absl/strings/strip.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/utils/minijson.h"
#include "tensor/examples/utils/perfetto_session.h"
#include "tensor/examples/utils/safetensors.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "perfetto/tracing/track_event.h"  // from @perfetto

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

absl::Status ValidateTensorRange(const SafetensorTensorInfo& info) {
  if (!info.storage) {
    return absl::FailedPreconditionError("No storage is set for tensor.");
  }
  if (info.data_end < info.data_start) {
    return absl::DataLossError(
        absl::StrCat("Invalid tensor data range for: ", info.name));
  }
  if (info.data_end > info.storage->data_size) {
    return absl::DataLossError(
        absl::StrCat("Tensor data out of range for: ", info.name));
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

template <class Container>
absl::StatusOr<Container> ConvertTensorTo(const SafetensorTensorInfo& info) {
  using T = typename Container::value_type;
  LRT_TENSOR_ASSIGN_OR_RETURN(const size_t num_elements,
                              NumElements(info.shape));
  const size_t bytes = info.data_end - info.data_start;
  LRT_TENSOR_RETURN_IF_ERROR(info.storage != nullptr &&
                             info.storage->data_base != nullptr)
      << "Invalid storage for tensor data.";
  const std::byte* data_ptr = info.storage->data_base + info.data_start;

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
      if (reinterpret_cast<uintptr_t>(data_ptr) %                              \
          alignof(typename Info::Storage)) {                                   \
        return absl::InvalidArgumentError(                                     \
            absl::StrCat("Mapped data at offset ", info.data_start,            \
                         " is not correctly aligned for " #ST_TYPE));          \
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

// Suffix given to weight tensors in checkpoints.
constexpr absl::string_view kWeightSuffix = ".weight";
// Suffix givend to quantized weights.
constexpr absl::string_view kPackedSuffix = ".weight_packed";
// Quantized weights neighbour scale tensor suffix.
constexpr absl::string_view kScaleSuffix = ".weight_scale";
// Quantized weights neighbour shape tensor suffix.
constexpr absl::string_view kShapeSuffix = ".weight_shape";
// Quantized weights neighbour zero point tensor suffix.
constexpr absl::string_view kZeroPointSuffix = ".weight_zero_point";

// Prefix marking a `targets` or `ignore` entry as a regular expression.
constexpr absl::string_view kRegexPrefix = "re:";

// Returns whether `target` names a module class, such as "Linear", rather than
// a specific module. Module names are dotted paths, class names are not.
bool IsModuleClassName(absl::string_view target) {
  return !absl::StrContains(target, '.');
}

// Returns the element type holding `num_bits` wide quantized values.
absl::StatusOr<Type> QuantizedElementType(const int num_bits) {
  switch (num_bits) {
    case 2:
      return Type::kI2;
    case 4:
      return Type::kI4;
    case 8:
      return Type::kI8;
    default:
      break;
  }
  return absl::UnimplementedError(
      absl::StrCat("Unsupported quantized weight width: ", num_bits, " bits"));
}

// Safetensor compressed-tensors shift every packed field into unsigned range by
// adding pow(2, num_bits-1) before packing it into a container.
//
// `(v + pow(2, b-1)) % pow(2, b) == v ^ pow(2, b-1)` so we can XOR the mask
// returned to apply the shift.
constexpr uint8_t OffsetMask(Type type) {
  switch (type) {
    case Type::kI2:
      return 0b10101010;
    case Type::kI4:
      return 0b10001000;
    default:
      return 0;
  }
}

// Returns the width, in bits, of the integer `compressed-tensors` packs
// quantized fields into.
absl::StatusOr<size_t> PackedContainerBits(safetensors::dtype dtype) {
  switch (dtype) {
    case safetensors::dtype::kINT32:
    case safetensors::dtype::kUINT32:
      return 32;
    case safetensors::dtype::kINT8:
    case safetensors::dtype::kUINT8:
      return 8;
    default:
      break;
  }
  return absl::UnimplementedError(absl::StrCat(
      "Unsupported packed weight container type: ", ToString(dtype)));
}

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
struct MinijsonTypeTraits<QuantizationConfig::Strategy> {
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
struct ValueParser<QuantizationConfig::Strategy> {
  static absl::StatusOr<QuantizationConfig::Strategy> Parse(
      absl::string_view s) {
    if (s == "tensor") {
      return QuantizationConfig::Strategy::kTensor;
    }
    if (s == "channel") {
      return QuantizationConfig::Strategy::kChannel;
    }
    if (s == "group") {
      return QuantizationConfig::Strategy::kGroup;
    }
    return QuantizationConfig::Strategy::kUnknown;
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

// Returns the array stored at `key`, or nullptr when `key` is absent or does
// not hold an array. The array is owned by `holder`, which the caller must
// keep alive for as long as it uses the result.
const minijson::array* GetJsonArray(const minijson::object& obj,
                                    absl::string_view key,
                                    minijson::value& holder) {
  if (!obj.at(std::string(key), &holder)) {
    return nullptr;
  }
  return holder.as<minijson::array>();
}

template <class F>
absl::Status ParseFromStringArray(const minijson::array& arr, F&& parser) {
  for (const minijson::value& item : arr) {
    const minijson::string* str = item.as<minijson::string>();
    if (!str) {
      return absl::InvalidArgumentError("Expected an array of strings.");
    }
    LRT_TENSOR_RETURN_IF_ERROR(parser(*str));
  }
  return absl::OkStatus();
}

// Parses a single `config_groups` entry.
absl::StatusOr<QuantizationConfig::Scheme> ParseScheme(
    const minijson::object& group_obj) {
  QuantizationConfig::Scheme scheme;

  // Some checkpoints store the weight parameters directly in the group, others
  // nest them under "weights". Read both, letting the nested form win.
  ASSIGN_IF_OK(scheme.num_bits, GetJsonField<int>(group_obj, "num_bits"));
  ASSIGN_IF_OK(scheme.group_size, GetJsonField<int>(group_obj, "group_size"));
  ASSIGN_IF_OK(scheme.symmetric, GetJsonField<bool>(group_obj, "symmetric"));
  ASSIGN_IF_OK(scheme.strategy, GetJsonField<QuantizationConfig::Strategy>(
                                    group_obj, "strategy"));

  minijson::value weights_val;
  if (group_obj.at("weights", &weights_val)) {
    if (const minijson::object* weights_obj =
            weights_val.as<minijson::object>();
        weights_obj != nullptr) {
      ASSIGN_IF_OK(scheme.num_bits,
                   GetJsonField<int>(*weights_obj, "num_bits"));
      ASSIGN_IF_OK(scheme.group_size,
                   GetJsonField<int>(*weights_obj, "group_size"));
      ASSIGN_IF_OK(scheme.symmetric,
                   GetJsonField<bool>(*weights_obj, "symmetric"));
      ASSIGN_IF_OK(scheme.strategy, GetJsonField<QuantizationConfig::Strategy>(
                                        *weights_obj, "strategy"));
    }
  }

  minijson::value targets_val;
  if (const minijson::array* targets =
          GetJsonArray(group_obj, "targets", targets_val);
      targets != nullptr) {
    LRT_TENSOR_RETURN_IF_ERROR(
        ParseFromStringArray(*targets, [&scheme](absl::string_view target) {
          if (absl::ConsumePrefix(&target, kRegexPrefix)) {
            scheme.patterns.emplace_back(std::string(target),
                                         std::regex_constants::ECMAScript);
          } else if (IsModuleClassName(target)) {
            scheme.matches_any_module = true;
          } else {
            scheme.modules.emplace(target);
          }
          return absl::OkStatus();
        }));
  }
  // A group that names no module at all applies to the whole model.
  if (scheme.modules.empty() && scheme.patterns.empty()) {
    scheme.matches_any_module = true;
  }

  // `strategy` is optional in older checkpoints: a group size implies
  // group-wise quantization, and its absence implies per-channel scales.
  if (scheme.strategy == QuantizationConfig::Strategy::kUnknown) {
    scheme.strategy = scheme.group_size > 0
                          ? QuantizationConfig::Strategy::kGroup
                          : QuantizationConfig::Strategy::kChannel;
  }
  if (scheme.strategy == QuantizationConfig::Strategy::kGroup &&
      scheme.group_size <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Group-wise quantization requires a positive group_size, "
                     "got ",
                     scheme.group_size));
  }
  if (scheme.num_bits <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid quantization_config: scheme requires a positive num_bits, "
        "got ",
        scheme.num_bits));
  }
  return scheme;
}

absl::StatusOr<QuantizationConfig> ParseQuantizationConfigObject(
    const minijson::object& root_obj) {
  QuantizationConfig cfg;

  ASSIGN_IF_OK(cfg.quant_method, GetJsonField<QuantizationConfig::Method>(
                                     root_obj, "quant_method"));

  minijson::value ignore_val;
  if (const minijson::array* ignore =
          GetJsonArray(root_obj, "ignore", ignore_val);
      ignore != nullptr) {
    LRT_TENSOR_RETURN_IF_ERROR(
        ParseFromStringArray(*ignore, [&cfg](absl::string_view pattern) {
          if (absl::ConsumePrefix(&pattern, kRegexPrefix)) {
            cfg.ignore_regexes.emplace_back(pattern.data(), pattern.size());
          } else {
            cfg.ignore.emplace_back(pattern);
          }
          return absl::OkStatus();
        }));
  }

  minijson::value config_groups_val;
  if (root_obj.at("config_groups", &config_groups_val)) {
    if (const minijson::object* groups_obj =
            config_groups_val.as<minijson::object>();
        groups_obj != nullptr) {
      for (const std::string& group_name : groups_obj->keys()) {
        minijson::value group_val;
        if (!groups_obj->at(group_name, &group_val)) {
          continue;
        }
        const minijson::object* group_obj = group_val.as<minijson::object>();
        if (group_obj == nullptr) {
          continue;
        }
        LRT_TENSOR_ASSIGN_OR_RETURN(QuantizationConfig::Scheme scheme,
                                    ParseScheme(*group_obj));
        cfg.schemes.push_back(std::move(scheme));
      }
    }
  }

  // Mirror the first group into the flat fields, for callers that assume a
  // single model-wide scheme.
  if (!cfg.schemes.empty()) {
    const QuantizationConfig::Scheme& first = cfg.schemes.front();
    cfg.num_bits = first.num_bits;
    cfg.group_size = first.group_size;
    cfg.symmetric = first.symmetric;
  }

  return cfg;
}

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

  return ParseQuantizationConfigObject(*root_obj);
}

#undef ASSIGN_IF_OK

}  // namespace

bool QuantizationConfig::Scheme::Matches(absl::string_view module) const {
  if (modules.contains(module)) {
    return true;
  }
  for (const std::regex& pattern : patterns) {
    if (std::regex_search(module.data(), module.data() + module.size(),
                          pattern)) {
      return true;
    }
  }
  return false;
}

bool QuantizationConfig::IsIgnored(absl::string_view module) const {
  for (const std::string& pattern : ignore) {
    if (module == pattern ||
        (absl::StartsWith(module, pattern) && module[pattern.size()] == '.') ||
        (absl::EndsWith(module, pattern) &&
         module[module.size() - pattern.size() - 1] == '.')) {
      return true;
    }
  }
  for (const std::regex& pattern : ignore_regexes) {
    if (std::regex_search(module.data(), module.data() + module.size(),
                          pattern)) {
      return true;
    }
  }
  return false;
}

const QuantizationConfig::Scheme* QuantizationConfig::FindScheme(
    absl::string_view module) const {
  if (IsIgnored(module)) {
    return nullptr;
  }
  // A scheme naming the module wins over one that applies to the whole model.
  const Scheme* catch_all = nullptr;
  for (const Scheme& scheme : schemes) {
    if (scheme.Matches(module)) {
      return &scheme;
    }
    if (scheme.matches_any_module && catch_all == nullptr) {
      catch_all = &scheme;
    }
  }
  return catch_all;
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
  TRACE_EVENT(kTensorApiCategory, "AddSafetensorFile");
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
    TRACE_EVENT(kTensorApiCategory, "AddTensor");
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
    ABSL_LOG(INFO) << "Parsed header quantization_config: method="
                   << quant_config_->quant_method
                   << " num_bits=" << quant_config_->num_bits
                   << " group_size=" << quant_config_->group_size;
  }

  ABSL_LOG(INFO) << "Loaded safetensor file: " << path
                 << " tensors: " << tensor_keys.size();
  return absl::OkStatus();
}

absl::Status SafetensorLoader::AddQuantizationConfigFromJsonFile(
    const std::string& path) {
  TRACE_EVENT(kTensorApiCategory, "AddQuantizationConfigFromJsonFile");
  std::ifstream file(path);
  if (!file.is_open()) {
    return absl::NotFoundError(absl::StrCat("File not found: ", path));
  }
  std::string contents((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());

  minijson::value val;
  const char* json_str = contents.data();
  if (minijson::parse(json_str, val) != minijson::no_error) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse ", path, " as JSON"));
  }
  const minijson::object* root_obj = val.as<minijson::object>();
  if (root_obj == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat(path, " does not hold a JSON object"));
  }

  minijson::value quant_cfg_val;
  if (!root_obj->at("quantization_config", &quant_cfg_val)) {
    return absl::OkStatus();
  }
  const minijson::object* quant_cfg_obj = quant_cfg_val.as<minijson::object>();
  if (quant_cfg_obj == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("quantization_config in ", path, " is not a JSON object"));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(quant_config_,
                              ParseQuantizationConfigObject(*quant_cfg_obj));
  ABSL_LOG(INFO) << "Parsed quantization_config from " << path
                 << ": method=" << quant_config_->quant_method
                 << " config groups=" << quant_config_->schemes.size();
  return absl::OkStatus();
}

absl::StatusOr<SafetensorLoader> SafetensorLoader::Load(
    const std::string& path) {
  TRACE_EVENT(kTensorApiCategory, "Initialize weight loader");
  namespace fs = std::filesystem;
  SafetensorLoader loader;

  std::error_code ec;
  const fs::path input_path(path);
  const bool is_directory = fs::is_directory(input_path, ec);
  if (ec) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to inspect path ", path, ": ", ec.message()));
  }

  // A checkpoint that does not declare its quantization config in the
  // safetensors header keeps it in a config.json next to the weights.
  const fs::path config_path =
      (is_directory ? input_path : input_path.parent_path()) / "config.json";

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
    if (!loader.quant_config_.has_value()) {
      absl::Status status =
          loader.AddQuantizationConfigFromJsonFile(config_path.string());
      // Not every checkpoint ships a config.json; those that do not are either
      // unquantized or carry the config in their header instead.
      if (!status.ok() && !absl::IsNotFound(status)) {
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
  if (!loader.quant_config_.has_value()) {
    absl::Status config_status =
        loader.AddQuantizationConfigFromJsonFile(config_path.string());
    if (!config_status.ok() && !absl::IsNotFound(config_status)) {
      return config_status;
    }
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

absl::StatusOr<const SafetensorTensorInfo&> SafetensorLoader::GetTensorInfo(
    absl::string_view name) const {
  auto it = tensor_infos_.find(name);
  if (it == tensor_infos_.end()) {
    return absl::NotFoundError(absl::StrCat("Tensor not found: ", name));
  }
  return it->second;
}

absl::StatusOr<TensorHandle> SafetensorLoader::LoadTensor(
    absl::string_view name) const {
  TRACE_EVENT(kTensorApiCategory, "LoadTensor");
  ABSL_VLOG(3) << "Loading tensor " << name;

  if (!tensor_infos_.contains(name)) {
    absl::string_view module = name;
    if (absl::ConsumeSuffix(&module, kWeightSuffix) &&
        tensor_infos_.contains(absl::StrCat(module, kPackedSuffix))) {
      return LoadPackedTensor(module, name);
    }
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(const SafetensorTensorInfo info,
                              GetTensorInfo(name));
  LRT_TENSOR_ASSIGN_OR_RETURN(Type type, DtypeToType(info.dtype));

  const TensorStorageInfo& storage = *info.storage;
  if (storage.data_base == nullptr || storage.file_data == nullptr) {
    return absl::FailedPreconditionError("Safetensor storage is invalid");
  }

  LRT_TENSOR_RETURN_IF_ERROR(ValidateTensorRange(info));

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

      // `compressed-tensors` names quantization parameters after the module
      // (`<module>.weight_scale`), while checkpoints written by our own
      // converter name them after the weight (`<module>.weight.weight_scale`).
      // Accept both.
      absl::string_view module = name;
      absl::ConsumeSuffix(&module, kWeightSuffix);

      auto FindDataFor =
          [&](std::initializer_list<absl::string_view> suffixes) {
            std::string path;
            for (absl::string_view prefix : {name, module}) {
              path.assign(prefix);
              for (absl::string_view suffix : suffixes) {
                path.resize(prefix.size());
                absl::StrAppend(&path, suffix);
                if (auto it = tensor_infos_.find(path);
                    it != tensor_infos_.end()) {
                  return it;
                }
              }
            }
            return tensor_infos_.end();
          };

      auto ReadData = [&](auto& vec,
                          std::initializer_list<absl::string_view> suffixes)
          -> absl::Status {
        auto it = FindDataFor(suffixes);
        if (it != tensor_infos_.end()) {
          LRT_TENSOR_RETURN_IF_ERROR(ValidateTensorRange(it->second));
          LRT_TENSOR_ASSIGN_OR_RETURN(
              vec, ConvertTensorTo<std::decay_t<decltype(vec)>>(it->second));
        }
        return !vec.empty()
                   ? absl::OkStatus()
                   : absl::InvalidArgumentError(absl::StrCat(
                         "Data is empty for ", module, *suffixes.begin()));
      };

      std::vector<float> scales;
      std::vector<int64_t> zero_points(1, 0);
      LRT_TENSOR_RETURN_IF_ERROR(ReadData(
          scales, {".weight_scale", ".scale", ".scales", ".weight_scales"}));
      LRT_TENSOR_RETURN_IF_ERROR(
          ReadData(zero_points, {".weight_zero_point", ".zero_point"}));

      // Checkpoints declaring config groups describe each module separately;
      // older ones only carry model-wide parameters.
      const QuantizationConfig::Scheme* scheme =
          quant_config_->FindScheme(module);
      const int num_bits =
          scheme != nullptr ? scheme->num_bits : quant_config_->num_bits;
      const int group_size =
          scheme != nullptr ? scheme->group_size : quant_config_->group_size;
      const QuantizationConfig::Strategy strategy =
          (scheme != nullptr &&
           scheme->strategy != QuantizationConfig::Strategy::kUnknown)
              ? scheme->strategy
              : (group_size > 0 ? QuantizationConfig::Strategy::kGroup
                                : QuantizationConfig::Strategy::kChannel);

      if (strategy == QuantizationConfig::Strategy::kGroup) {
        quantization = std::make_shared<BlockwiseQuantization>(
            std::move(scales), std::move(zero_points), group_size,
            /*quantized_dimension=*/0);
      } else {
        quantization = std::make_shared<PerChannelAffineQuantization>(
            std::move(scales), std::move(zero_points),
            /*quantized_dimension=*/0);
      }

      // Nibble-packed weights hold two elements per byte, so the shape on disk
      // is half as wide as the weight it represents.
      if (num_bits == 4 && info.shape.size() == 2) {
        const size_t d_out = info.shape[0];
        const size_t d_in_packed = info.shape[1];
        const size_t packed_element_count = BufferSize(type, 1) * 2;
        const size_t d_in = d_in_packed * packed_element_count;

        return TensorHandle(TensorInit{
            .name = std::string(name),
            .type = Type::kI4,
            .shape = {static_cast<int>(d_out), static_cast<int>(d_in)},
            .buffer = buffer,
            .quantization = quantization});
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

absl::StatusOr<std::shared_ptr<Quantization>>
SafetensorLoader::LoadQuantizationParams(
    const QuantizationConfig::Scheme& scheme, absl::string_view module) const {
  const std::string scale_name = absl::StrCat(module, kScaleSuffix);
  auto scale_it = tensor_infos_.find(scale_name);
  if (scale_it == tensor_infos_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Missing quantization scales: ", scale_name));
  }
  LRT_TENSOR_RETURN_IF_ERROR(ValidateTensorRange(scale_it->second));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::vector<float> scales,
      ConvertTensorTo<std::vector<float>>(scale_it->second));
  if (scales.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Scale tensor is empty for: ", module));
  }

  // Symmetric quantization, which is the common case, stores no zero points.
  std::vector<int64_t> zero_points(1, 0);
  const std::string zero_point_name = absl::StrCat(module, kZeroPointSuffix);
  if (auto zp_it = tensor_infos_.find(zero_point_name);
      zp_it != tensor_infos_.end()) {
    LRT_TENSOR_RETURN_IF_ERROR(ValidateTensorRange(zp_it->second));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        zero_points, ConvertTensorTo<std::vector<int64_t>>(zp_it->second));
    if (zero_points.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Zero-point tensor is empty for: ", module));
    }
  }

  switch (scheme.strategy) {
    case QuantizationConfig::Strategy::kGroup:
      return std::make_shared<BlockwiseQuantization>(
          std::move(scales), std::move(zero_points), scheme.group_size,
          /*quantized_dimension=*/0);
    case QuantizationConfig::Strategy::kChannel:
    case QuantizationConfig::Strategy::kTensor:
      return std::make_shared<PerChannelAffineQuantization>(
          std::move(scales), std::move(zero_points),
          /*quantized_dimension=*/0);
    default:
      break;
  }
  return absl::UnimplementedError(absl::StrCat(
      "Unsupported quantization strategy for ", module, ": ", scheme.strategy));
}

absl::StatusOr<TensorHandle> SafetensorLoader::LoadPackedTensor(
    absl::string_view module, absl::string_view name) const {
  TRACE_EVENT(kTensorApiCategory, "LoadPackedTensor");
  if (!quant_config_.has_value()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Cannot load packed weight ", module,
        ": the checkpoint declares no quantization config, neither in the "
        "safetensors header nor in a neighbouring config.json"));
  }
  const QuantizationConfig::Scheme* scheme = quant_config_->FindScheme(module);
  if (scheme == nullptr) {
    return absl::NotFoundError(absl::StrCat(
        "No quantization config group applies to packed weight ", module));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const Type type,
                              QuantizedElementType(scheme->num_bits));

  const std::string packed_name = absl::StrCat(module, kPackedSuffix);
  LRT_TENSOR_ASSIGN_OR_RETURN(SafetensorTensorInfo info,
                              GetTensorInfo(packed_name));
  const TensorStorageInfo& storage = *info.storage;
  if (storage.data_base == nullptr || storage.file_data == nullptr) {
    return absl::FailedPreconditionError("Safetensor storage is invalid");
  }
  LRT_TENSOR_RETURN_IF_ERROR(ValidateTensorRange(info));
  if (info.shape.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat(packed_name, ": packed weights must be 2D, got rank ",
                     info.shape.size()));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(const size_t container_bits,
                              PackedContainerBits(info.dtype));
  const int64_t num_rows = info.shape[0];
  const int64_t packed_row_bytes = info.shape[1] * container_bits / 8;
  const int64_t unpacked_cols =
      info.shape[1] * static_cast<int64_t>(container_bits) / scheme->num_bits;

  // The packed shape rounds the weight up to a whole number of containers;
  // `weight_shape` records the shape before that padding.
  int64_t num_cols = unpacked_cols;
  const std::string shape_name = absl::StrCat(module, kShapeSuffix);
  if (auto shape_it = tensor_infos_.find(shape_name);
      shape_it != tensor_infos_.end()) {
    LRT_TENSOR_RETURN_IF_ERROR(ValidateTensorRange(shape_it->second));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        std::vector<int64_t> logical_shape,
        ConvertTensorTo<std::vector<int64_t>>(shape_it->second));
    if (logical_shape.size() != 2) {
      return absl::InvalidArgumentError(absl::StrCat(
          shape_name, ": expected 2 dimensions, got ", logical_shape.size()));
    }
    if (logical_shape[0] != num_rows || logical_shape[1] > unpacked_cols) {
      return absl::InvalidArgumentError(
          absl::StrCat(shape_name, ": logical shape [", logical_shape[0], ", ",
                       logical_shape[1], "] does not fit the packed shape [",
                       num_rows, ", ", unpacked_cols, "]"));
    }
    num_cols = logical_shape[1];
  }

  const int64_t row_bits = num_cols * scheme->num_bits;
  if (row_bits % 8 != 0) {
    return absl::UnimplementedError(
        absl::StrCat(packed_name, ": rows of ", num_cols, " ", scheme->num_bits,
                     "-bit values are not byte aligned"));
  }
  const int64_t row_bytes = row_bits / 8;
  if (static_cast<size_t>(num_rows * packed_row_bytes) >
      info.data_end - info.data_start) {
    return absl::DataLossError(
        absl::StrCat(packed_name, ": packed data is shorter than its shape"));
  }

  std::shared_ptr<OwningCpuBuffer> weights = OwningCpuBuffer::AllocateAs(
      type, static_cast<size_t>(num_rows * num_cols));
  if (weights == nullptr) {
    return absl::ResourceExhaustedError(
        absl::StrCat("Failed to allocate ", num_rows * row_bytes,
                     " bytes for weight ", module));
  }

  // Safetensors files store integer containers in little-endian byte order, and
  // `compressed-tensors` packs fields starting from the least significant bits
  // of each container. Thus the raw bytes in the file are already a stream of
  // fields in order regardless of host endianness, and only their offset has to
  // be undone, one XOR per byte. Rows are copied one by one to drop any padding
  // container.
  const std::byte mask{OffsetMask(type)};
  const std::byte* src = storage.data_base + info.data_start;
  std::byte* dst = weights->data();
  for (int64_t row = 0; row < num_rows; ++row) {
    const std::byte* src_row = src + row * packed_row_bytes;
    std::byte* dst_row = dst + row * row_bytes;
    for (int64_t i = 0; i < row_bytes; ++i) {
      dst_row[i] = src_row[i] ^ mask;
    }
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(std::shared_ptr<Quantization> quantization,
                              LoadQuantizationParams(*scheme, module));

  ABSL_VLOG(3) << "Loaded packed weight " << module << " as " << ToString(type)
               << " [" << num_rows << ", " << num_cols << "]";

  return TensorHandle(TensorInit{
      .name = std::string(name),
      .type = type,
      .shape = {static_cast<int>(num_rows), static_cast<int>(num_cols)},
      .buffer = std::move(weights),
      .quantization = std::move(quantization)});
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
  TRACE_EVENT(kTensorApiCategory, "LoadWeightsWithMapping");
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
}  // namespace litert::tensor::examples

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
#define SAFETENSORS_CPP_IMPLEMENTATION
#include "tensor/examples/gemma3/safetensor_loader.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
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
// copybara:uncomment_begin(google_only)
// #include "tensor/backends/xnnpack/arithmetic.h"
// copybara:uncomment_end
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma3/safetensors.h"
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

absl::StatusOr<QuantParamMode> DetermineQuantParamMode(
    size_t param_size, const std::vector<int64_t>& shape, size_t num_elements,
    absl::string_view param_name) {
  if (param_size == 1) {
    return QuantParamMode::kScalar;
  }
  if (param_size == num_elements) {
    return QuantParamMode::kPerElement;
  }
  // Limitation: Tensor API quantizatoin metadata is is exported as
  // per-channel along dim0 for this loader.
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

  const auto* data_base =
      reinterpret_cast<const std::byte*>(st->databuffer_addr);
  const size_t data_size = st->databuffer_size;

  auto storage_info = std::make_shared<TensorStorageInfo>(TensorStorageInfo{
      .file_data = st, .data_base = data_base, .data_size = data_size});

  // Convert safetensors-cpp tensor info to our format.
  const auto& tensor_keys = st->tensors.keys();
  for (const auto& name : tensor_keys) {
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

  ABSL_LOG(INFO) << "Loaded safetensor file: " << path
                 << " tensors: " << tensor_keys.size();
  return absl::OkStatus();
}

absl::StatusOr<SafetensorLoader> SafetensorLoader::Load(
    const std::string& path) {
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
    for (const auto& file : safetensor_files) {
      auto status = loader.AddSafetensorFile(file);
      if (!status.ok()) {
        return status;
      }
    }
    ABSL_LOG(INFO) << "Loaded " << safetensor_files.size()
                   << " safetensor files from directory " << path << " with "
                   << loader.tensor_infos_.size() << " tensors";
    return loader;
  }

  auto status = loader.AddSafetensorFile(path);
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
    const std::string& name) const {
  auto it = tensor_infos_.find(name);
  if (it == tensor_infos_.end()) {
    return absl::NotFoundError(absl::StrCat("Tensor not found: ", name));
  }
  return it->second;
}

absl::StatusOr<TensorHandle> SafetensorLoader::LoadTensor(
    const std::string& name, QuantizedLoadMode quantized_load_mode) const {
  ABSL_LOG(INFO) << "Loading tensor " << name;
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
    case Type::kBF16:
    case Type::kFP16: {
      LRT_TENSOR_ASSIGN_OR_RETURN(
          TypedOwningBuffer<Type::kFP32> values,
          ConvertTensorTo<TypedOwningBuffer<Type::kFP32>>(info,
                                                          storage.data_base));
      buffer = std::move(values.buffer);
      type = Type::kFP32;
      break;
    }
    case Type::kI8: {
      auto tensor_info_it = [&]() {
        for (absl::string_view suffix :
             {".scale", ".scales", ".weight_scales"}) {
          if (auto it = tensor_infos_.find(absl::StrCat(name, suffix));
              it != tensor_infos_.end()) {
            return it;
          }
        }
        return tensor_infos_.end();
      }();

      buffer = MakeMappedBuffer(storage.file_data, data_ptr, data_size);

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
      if (auto zp_it = tensor_infos_.find(absl::StrCat(name, ".zero_point"));
          zp_it != tensor_infos_.end()) {
        LRT_TENSOR_ASSIGN_OR_RETURN(zero_points,
                                    ReadTensor(zp_it, /*as=*/int64_t{}));
        if (zero_points.empty()) {
          return absl::InvalidArgumentError(
              absl::StrCat("Zero-point tensor is empty for: ", name));
        }
      }

      LRT_TENSOR_ASSIGN_OR_RETURN(const size_t num_elements,
                                  NumElements(info.shape));
      if (num_elements * sizeof(int8_t) != data_size) {
        return absl::InvalidArgumentError(
            absl::StrCat("I8 tensor byte size mismatch for: ", name));
      }

      // Limitation: Tensor API quantization metadata is exported as per-channel
      // along dim0 for this loader.
      if (quantized_load_mode == QuantizedLoadMode::kPreserveQuantized) {
        quantization = std::make_shared<PerChannelAffineQuantization>(
            std::move(scales), std::move(zero_points),
            /*quantized_dimension=*/0);
        break;
      }

      size_t channel_size = 0;
      if (!info.shape.empty() && info.shape[0] > 0) {
        channel_size = num_elements / static_cast<size_t>(info.shape[0]);
      }

      LRT_TENSOR_ASSIGN_OR_RETURN(
          const QuantParamMode scale_mode,
          DetermineQuantParamMode(scales.size(), info.shape, num_elements,
                                  "scale"));
      LRT_TENSOR_ASSIGN_OR_RETURN(
          const QuantParamMode zp_mode,
          DetermineQuantParamMode(zero_points.size(), info.shape, num_elements,
                                  "zero_point"));

      std::vector<float> values(num_elements);
      const int8_t* quantized_data = reinterpret_cast<const int8_t*>(data_ptr);
      for (size_t i = 0; i < num_elements; ++i) {
        const size_t channel_index = channel_size == 0 ? 0 : i / channel_size;
        const size_t scale_index =
            QuantParamIndex(scale_mode, i, channel_index);
        const size_t zp_index = QuantParamIndex(zp_mode, i, channel_index);
        const int32_t q = static_cast<int32_t>(quantized_data[i]);
        const int32_t zp = static_cast<int32_t>(zero_points[zp_index]);
        values[i] = static_cast<float>(q - zp) * scales[scale_index];
      }

      buffer = OwningCpuBuffer::Copy<Type::kFP32>(values);
      type = Type::kFP32;
      break;
    }
    case Type::kFP32:
    case Type::kFP64:
    case Type::kI32:
    case Type::kI64:
    case Type::kI16:
    case Type::kU8:
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

  return TensorHandle(
      {.name = name,
       .type = type,
       .shape = std::vector<int>(info.shape.begin(), info.shape.end()),
       .buffer = buffer,
       .quantization = quantization});
}

absl::StatusOr<absl::flat_hash_map<std::string, TensorHandle>>
SafetensorLoader::LoadAllTensors(QuantizedLoadMode quantized_load_mode) const {
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

absl::StatusOr<absl::flat_hash_map<std::string, TensorHandle>>
SafetensorLoader::LoadWeightsWithMapping(
    const absl::flat_hash_map<std::string, std::string>& name_mapping,
    QuantizedLoadMode quantized_load_mode) const {
  absl::flat_hash_map<std::string, TensorHandle> tensors;
  for (const auto& [hf_name, model_name] : name_mapping) {
    auto tensor_or = LoadTensor(hf_name, quantized_load_mode);
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

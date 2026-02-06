// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_builder.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_google_tensor_options.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/cc/options_helper.h"
#include "litert/vendors/google_tensor/adapter.h"
#include "litert/vendors/google_tensor/compiler/google_tensor_options.pb.h"

//
// Configurations
//

using ::third_party::odml::litert::litert::vendors::google_tensor::compiler::
    DeviceType;
using ::third_party::odml::litert::litert::vendors::google_tensor::compiler::
    GoogleTensorCompilerConfig;
using ::third_party::odml::litert::litert::vendors::google_tensor::compiler::
    GoogleTensorOptions;
using ::third_party::odml::litert::litert::vendors::google_tensor::compiler::
    GoogleTensorOptionsShardingIntensity;
using ::third_party::odml::litert::litert::vendors::google_tensor::compiler::
    GoogleTensorOptionsTruncationType;

namespace google_tensor {

constexpr char kPluginManufacturer[] = "Google";

constexpr const char* kPluginSocModels[] = {
    "Tensor_G3",
    "Tensor_G4",
    "Tensor_G5",
    "Tensor_G6",
};  // get the name for plugin soc model

LiteRtStatus GetDeviceType(absl::string_view soc_model,
                           DeviceType* device_type) {
  if (soc_model == "Tensor_G3") {
    *device_type = ::third_party::odml::litert::litert::vendors::google_tensor::
        compiler::DEVICE_TYPE_TENSOR_G3;
  } else if (soc_model == "Tensor_G4") {
    *device_type = ::third_party::odml::litert::litert::vendors::google_tensor::
        compiler::DEVICE_TYPE_TENSOR_G4;
  } else if (soc_model == "Tensor_G5") {
    *device_type = ::third_party::odml::litert::litert::vendors::google_tensor::
        compiler::DEVICE_TYPE_TENSOR_G5;
  } else if (soc_model == "Tensor_G6") {
    *device_type = ::third_party::odml::litert::litert::vendors::google_tensor::
        compiler::DEVICE_TYPE_TENSOR_G6;
  } else {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return kLiteRtStatusOk;
}

constexpr LiteRtOpCode kUnSupportedOps[] = {
    kLiteRtOpCodeTflAssignVariable,
    kLiteRtOpCodeTflBidirectionalSequenceLstm,
    kLiteRtOpCodeTflBroadcastArgs,
    kLiteRtOpCodeTflBucketize,
    kLiteRtOpCodeTflCallOnce,
    kLiteRtOpCodeTflComplexAbs,
    kLiteRtOpCodeTflConv3d,
    kLiteRtOpCodeTflConv3dTranspose,
    kLiteRtOpCodeTflDensify,
    kLiteRtOpCodeTflFakeQuant,
    kLiteRtOpCodeTflHashtable,
    kLiteRtOpCodeTflHashtableFind,
    kLiteRtOpCodeTflHashtableImport,
    kLiteRtOpCodeTflHashtableSize,
    kLiteRtOpCodeTflImag,
    kLiteRtOpCodeTflLocalResponseNormalization,
    kLiteRtOpCodeTflMatrixDiag,
    kLiteRtOpCodeTflMatrixSetDiag,
    kLiteRtOpCodeTflMultinomial,
    kLiteRtOpCodeTflNonMaxSuppressionV4,
    kLiteRtOpCodeTflNonMaxSuppressionV5,
    kLiteRtOpCodeTflRandomStandardNormal,
    kLiteRtOpCodeTflRandomUniform,
    kLiteRtOpCodeTflRank,
    kLiteRtOpCodeTflReadVariable,
    kLiteRtOpCodeTflReal,
    kLiteRtOpCodeTflReduceProd,
    kLiteRtOpCodeTflReverseSequence,
    kLiteRtOpCodeTflRfft2d,
    kLiteRtOpCodeTflSegmentSum,
    kLiteRtOpCodeTflShape,
    kLiteRtOpCodeTflSparseToDense,
    kLiteRtOpCodeTflSvdf,
    kLiteRtOpCodeTflUnidirectionalSequenceRnn,
    kLiteRtOpCodeTflUnique,
    kLiteRtOpCodeTflUnsortedSegmentMax,
    kLiteRtOpCodeTflUnsortedSegmentMin,
    kLiteRtOpCodeTflUnsortedSegmentProd,
    kLiteRtOpCodeTflUnsortedSegmentSum,
    kLiteRtOpCodeTflVarHandle,
    kLiteRtOpCodeTflWhere,
    kLiteRtOpCodeTflCustom,
    kLiteRtOpCodeShloComposite,
    kLiteRtOpCodeShloScatter,
    kLiteRtOpCodeShloWindow,
};
// clang format on

constexpr auto kNumPluginSocModels =
    sizeof(kPluginSocModels) / sizeof(kPluginSocModels[0]);

}  // namespace google_tensor

LiteRtStatus LiteRtOpaqueOptionsToGoogleTensorOptions(
    LiteRtOpaqueOptions options,
    third_party::odml::litert::litert::vendors::google_tensor::compiler::
        GoogleTensorOptions* google_tensor_options) {
  LiteRtGoogleTensorOptions google_tensor_options_data = nullptr;
  if (LiteRtFindOpaqueOptionsData(
          options, "google_tensor",
          reinterpret_cast<void**>(&google_tensor_options_data)) !=
      kLiteRtStatusOk) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  if (google_tensor_options_data == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  // FLOAT TRUNCATION TYPE
  LiteRtGoogleTensorOptionsTruncationType float_truncation_type;
  if (LiteRtGoogleTensorOptionsGetFloatTruncationType(
          google_tensor_options_data, &float_truncation_type) !=
      kLiteRtStatusOk) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  switch (float_truncation_type) {
    case kLiteRtGoogleTensorFloatTruncationTypeAuto:
      // set value in google_tensor_options proto
      google_tensor_options->set_float_truncation_type(
          GoogleTensorOptionsTruncationType::FLOAT_TRUNCATION_TYPE_AUTO);
      break;
    case kLiteRtGoogleTensorFloatTruncationTypeNoTruncation:
      google_tensor_options->set_float_truncation_type(
          GoogleTensorOptionsTruncationType::
              FLOAT_TRUNCATION_TYPE_NO_TRUNCATION);
      break;
    case kLiteRtGoogleTensorFloatTruncationTypeBfloat16:
      google_tensor_options->set_float_truncation_type(
          GoogleTensorOptionsTruncationType::FLOAT_TRUNCATION_TYPE_BFLOAT16);
      break;
    case kLiteRtGoogleTensorFloatTruncationTypeHalf:
      google_tensor_options->set_float_truncation_type(
          GoogleTensorOptionsTruncationType::FLOAT_TRUNCATION_TYPE_HALF);
      break;
  }

  // INT64 TO INT32 TRUNCATION
  bool int64_to_int32_truncation;
  if (LiteRtGoogleTensorOptionsGetInt64ToInt32Truncation(
          google_tensor_options_data, &int64_to_int32_truncation) !=
      kLiteRtStatusOk) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  google_tensor_options->set_int64_to_int32_truncation(
      int64_to_int32_truncation);

  // DUMP OP TIMINGS
  bool dump_op_timings;
  if (LiteRtGoogleTensorOptionsGetDumpOpTimings(
          google_tensor_options_data, &dump_op_timings) != kLiteRtStatusOk) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  google_tensor_options->set_dump_op_timings(dump_op_timings);

  // ENABLE LARGE MODEL SUPPORT
  bool enable_large_model_support;
  if (LiteRtGoogleTensorOptionsGetEnableLargeModelSupport(
          google_tensor_options_data, &enable_large_model_support) !=
      kLiteRtStatusOk) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  google_tensor_options->set_enable_large_model_support(
      enable_large_model_support);

  // ENABLE 4BIT COMPILATION
  bool enable_4bit_compilation;
  if (LiteRtGoogleTensorOptionsGetEnable4BitCompilation(
          google_tensor_options_data, &enable_4bit_compilation) !=
      kLiteRtStatusOk) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  google_tensor_options->set_enable_four_bit_compilation(
      enable_4bit_compilation);

  // SHARDING INTENSITY
  LiteRtGoogleTensorOptionsShardingIntensity sharding_intensity;
  if (LiteRtGoogleTensorOptionsGetShardingIntensity(
          google_tensor_options_data, &sharding_intensity) != kLiteRtStatusOk) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  switch (sharding_intensity) {
    case kLiteRtGoogleTensorShardingIntensityMinimal:
      google_tensor_options->set_sharding_intensity(
          GoogleTensorOptionsShardingIntensity::SHARDING_INTENSITY_MINIMAL);
      break;
    case kLiteRtGoogleTensorShardingIntensityModerate:
      google_tensor_options->set_sharding_intensity(
          GoogleTensorOptionsShardingIntensity::SHARDING_INTENSITY_MODERATE);
      break;
    case kLiteRtGoogleTensorShardingIntensityExtensive:
      google_tensor_options->set_sharding_intensity(
          GoogleTensorOptionsShardingIntensity::SHARDING_INTENSITY_EXTENSIVE);
      break;
    case kLiteRtGoogleTensorShardingIntensityMaximum:
      google_tensor_options->set_sharding_intensity(
          GoogleTensorOptionsShardingIntensity::SHARDING_INTENSITY_MAXIMUM);
      break;
    default:
      google_tensor_options->set_sharding_intensity(
          GoogleTensorOptionsShardingIntensity::SHARDING_INTENSITY_UNSPECIFIED);
      break;
  }

  // TESTING FLAGS
  std::vector<std::vector<std::string>> testing_flags;
  if (LiteRtGoogleTensorOptionsGetTestingFlags(
          google_tensor_options_data, &testing_flags) != kLiteRtStatusOk) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  for (const auto& flag : testing_flags) {
    google_tensor_options->set_testing_flags(flag[0]);
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginVersion(LiteRtApiVersion* api_version) {
  if (api_version == nullptr) {
    LITERT_LOG(LITERT_ERROR, "%s", "api_version is nullptr");
    return kLiteRtStatusErrorInvalidArgument;
  }
  api_version->major = LITERT_API_VERSION_MAJOR;
  api_version->minor = LITERT_API_VERSION_MINOR;
  api_version->patch = LITERT_API_VERSION_PATCH;
  return kLiteRtStatusOk;
}

const char* LiteRtGetCompilerPluginSocManufacturer() {
  return google_tensor::kPluginManufacturer;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators* supported_hardware) {
  if (!compiler_plugin || !supported_hardware) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "compiler_plugin or supported_hardware is nullptr");
    return kLiteRtStatusErrorInvalidArgument;
  }
  *supported_hardware = kLiteRtHwAcceleratorNpu;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex* num_supported_soc_models) {
  if (compiler_plugin == nullptr || num_supported_soc_models == nullptr) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "compiler_plugin or num_supported_soc_models is nullptr");
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_supported_soc_models = google_tensor::kNumPluginSocModels;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  if (compiler_plugin == nullptr ||
      soc_model_idx >= google_tensor::kNumPluginSocModels ||
      soc_model_name == nullptr) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "compiler_plugin or soc_model_idx or soc_model_name is nullptr");
    return kLiteRtStatusErrorInvalidArgument;
  }
  *soc_model_name = google_tensor::kPluginSocModels[soc_model_idx];
  return kLiteRtStatusOk;
}

//
// Compiled Result Definition
//

// TODO (abhirs): Revisit this struct after updating the compiler api wrapper to
// return multiple bytecodes.
struct LiteRtCompiledResultT {
  std::vector<std::string> byte_codes;
  std::vector<std::string> per_op_data;
};

LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size) {
  if (!compiled_result || !byte_code || !byte_code_size) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "compiled_result or byte_code or byte_code_size is nullptr");
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (byte_code_idx >= compiled_result->byte_codes.size()) {
    LITERT_LOG(LITERT_ERROR, "byte_code_idx (%d) is out of bounds (size %d)",
               static_cast<int>(byte_code_idx),
               static_cast<int>(compiled_result->byte_codes.size()));
    return kLiteRtStatusErrorIndexOOB;
  }
  *byte_code = compiled_result->byte_codes[0].data();
  *byte_code_size = compiled_result->byte_codes[0].size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_byte_code) {
  if (!compiled_result || !num_byte_code) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "compiled_result or num_byte_code is nullptr");
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_byte_code = compiled_result->byte_codes.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size,
    LiteRtParamIndex* byte_code_idx) {
  if (!compiled_result || !call_info || !call_info_size) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "compiled_result or call_info or call_info_size is nullptr");
    return kLiteRtStatusErrorInvalidArgument;
  } else if (call_idx >= compiled_result->per_op_data.size()) {
    LITERT_LOG(LITERT_ERROR, "%s", "call_idx is out of bounds");
    return kLiteRtStatusErrorIndexOOB;
  }

  *call_info = compiled_result->per_op_data.at(call_idx).data();
  *call_info_size = compiled_result->per_op_data.at(call_idx).size();
  *byte_code_idx = 0;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  if (!compiled_result || !num_calls) {
    LITERT_LOG(LITERT_ERROR, "%s", "compiled_result or num_calls is nullptr");
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_calls = compiled_result->per_op_data.size();
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompiledResult(LiteRtCompiledResult compiled_result) {
  delete compiled_result;
}

//
// Plugin Definition
//

// Plugins can hold state.
class LiteRtCompilerPluginT {
 public:
  using GoogleTensorOptions = ::litert::google_tensor::GoogleTensorOptions;

  LiteRtCompilerPluginT(LiteRtEnvironmentOptions env, LiteRtOptions options) {
    std::tie(opts_, opq_, google_tensor_opts_) =
        litert::ParseOptions<GoogleTensorOptions>(options);
  }

  ::litert::Expected<GoogleTensorOptions>& GetGoogleTensorOptions() {
    return google_tensor_opts_;
  }

  ::litert::Expected<litert::OpaqueOptions>& GetOpaqueOptions() { return opq_; }
  void SetLiteRtVersion(LiteRtApiVersion v) { litert_version_ = v; }
  LiteRtApiVersion GetLiteRtVersion() const { return litert_version_; }

 private:
  litert::Expected<litert::Options> opts_ =
      litert::Error(kLiteRtStatusErrorInvalidArgument, "Null options");
  litert::Expected<litert::OpaqueOptions> opq_ =
      litert::Error(kLiteRtStatusErrorInvalidArgument, "Null opaque options");
  litert::Expected<litert::google_tensor::GoogleTensorOptions>
      google_tensor_opts_ = litert::Error(kLiteRtStatusErrorInvalidArgument,
                                          "Null google tensor options");
  LiteRtApiVersion litert_version_;
};

LiteRtStatus LiteRtCreateCompilerPlugin(LiteRtCompilerPlugin* compiler_plugin,
                                        LiteRtEnvironmentOptions env,
                                        LiteRtOptions options) {
  *compiler_plugin = new LiteRtCompilerPluginT(env, options);
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  if (compiler_plugin == nullptr) {
    return;
  }
  delete compiler_plugin;
}

namespace google_tensor {
//  TODO(abhirs): update the function to use the darwinn inbuilt way of
//  finding supportedops
bool IsOpSupported(const litert::Op& op) {
  for (auto unsupported_op : kUnSupportedOps) {
    if (unsupported_op == op.Code()) {
      return false;
    }
  }
  return true;
}

}  // namespace google_tensor

LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           const char* soc_model,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  ::litert::Subgraph graph(subgraph);
  for (const auto& op : graph.Ops()) {
    if (!google_tensor::IsOpSupported(op)) {
      continue;
    }

    LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get(), 0));
  }

  return kLiteRtStatusOk;
}

void MakeUniqueSignatureKeysPerSubgraph(LiteRtModelT* model,
                                        size_t num_subgraphs,
                                        char** signature_keys) {
  for (size_t i = 0; i < num_subgraphs; ++i) {
    signature_keys[i] = strdup(absl::StrCat("subgraph_", i, "_fn").c_str());
  }
}

void FreeSignatureKeys(size_t num_subgraphs, char** signature_keys) {
  if (signature_keys) {
    for (size_t i = 0; i < num_subgraphs; ++i) {
      ::free(signature_keys[i]);
    }
  }
  ::free(signature_keys);
}

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  if (compiler_plugin == nullptr || partitions == nullptr ||
      compiled_result == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto model = litert::ExtendedModel::CreateFromNonOwnedHandle(partitions);
  const auto num_partitions = model.NumSubgraphs();

  // Loading Google Tensor Compiler Adapter
  LITERT_LOG(LITERT_INFO, "%s", "Loading Google Tensor Compiler Adapter");
  LITERT_ASSIGN_OR_RETURN(auto adapter,
                          litert::google_tensor::Adapter::Create(
                              /*shared_library_dir=*/std::nullopt));
  if (adapter->IsAot()) {
    // soc_model is required for AOT mode.
    if (soc_model == nullptr) {
      LITERT_LOG(LITERT_ERROR, "%s", "soc_model is nullptr in AOT mode");
      return kLiteRtStatusErrorInvalidArgument;
    }
  } else {
    // Allow unspecified soc model for ODC mode.
    if (soc_model == nullptr) {
      soc_model = "Unspecified";
    }
    // Currently ODC only supports Single subgraph models.
    if (num_partitions > 1) {
      LITERT_LOG(LITERT_ERROR, "%s",
                 "ODC mode does not support multiple subgraphs");
      return kLiteRtStatusErrorInvalidArgument;
    }
  }

  LITERT_LOG(LITERT_INFO,
             "Starting GoogleTensor Compilation for %d subgraphs, soc_model=%s",
             num_partitions, soc_model);

  if (num_partitions == 0) {
    LITERT_LOG(LITERT_ERROR,
               "No subgraphs selected for GoogleTensor compilation.");
    return kLiteRtStatusErrorInvalidArgument;
  }

  // Serialize model.
  LITERT_LOG(LITERT_INFO, "%s", "Serializing model");
  litert::OwningBufferRef buf;
  auto [data, size, offset] = buf.GetWeak();
  const auto opts = litert::SerializationOptions::Defaults();
  char** signatures =
      static_cast<char**>(calloc(num_partitions, sizeof(char*)));
  if (signatures == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Failed to allocate buffers for signatures.");
    return kLiteRtStatusErrorInvalidArgument;
  }
  absl::Cleanup signatures_cleanup = [num_partitions, signatures]() {
    FreeSignatureKeys(num_partitions, signatures);
  };

  MakeUniqueSignatureKeysPerSubgraph(model.Get(), num_partitions, signatures);
  LITERT_RETURN_IF_ERROR(LiteRtSerializeModelWithSignatures(
      partitions, &data, &size, &offset, false, signatures, num_partitions,
      opts));

  absl::string_view buffer_str(reinterpret_cast<const char*>(buf.Data()),
                               buf.Size());

  // Compile model.
  LITERT_LOG(LITERT_INFO, "%s", "Compiling model...");

  // Resolve custom google tensor options.
  LiteRtOpaqueOptions opaque_options = {};
  std::function<void(LiteRtOpaqueOptions)> deleter = nullptr;
  absl::Cleanup opaque_options_cleanup = [&] {
    if (deleter) {
      deleter(opaque_options);
    }
  };
  if (!compiler_plugin->GetGoogleTensorOptions()) {
    LITERT_LOG(
        LITERT_INFO,
        "No custom google tensor options found, creating default options");
    LITERT_ASSIGN_OR_RETURN(
        auto google_tensor_opts,
        ::litert::google_tensor::GoogleTensorOptions::Create());
    deleter = google_tensor_opts.GetDeleter();
    opaque_options = google_tensor_opts.Release();
  } else {
    LITERT_LOG(LITERT_INFO, "Using custom google tensor options");
    opaque_options = compiler_plugin->GetOpaqueOptions()->Get();
  }

  third_party::odml::litert::litert::vendors::google_tensor::compiler::
      GoogleTensorOptions google_tensor_options;

  // map to opaque options
  LITERT_RETURN_IF_ERROR(LiteRtOpaqueOptionsToGoogleTensorOptions(
      opaque_options, &google_tensor_options));

  // Set litert version string (e.g., "0.1.0")
  LiteRtApiVersion litert_version = compiler_plugin->GetLiteRtVersion();
  std::string api_version_str =
      absl::StrFormat("%d.%d.%d", litert_version.major, litert_version.minor,
                      litert_version.patch);

  // Set compilation configuration.
  auto* compiler_config = google_tensor_options.mutable_compiler_config();
  compiler_config->set_compilation_client(
      GoogleTensorCompilerConfig::COMPILATION_CLIENT_LITERT_PLUGIN);
  compiler_config->set_litert_version(api_version_str);

  // In the ODC flow, LiteRT doesn't set a valid value to soc_model, relying on
  // underlying layers to infer it. This allows the device type to be set as
  // unspecified. On the other hand, the AOT flow requires soc_model to
  // determine the device type for ahead-of-time compilation.
  if (adapter->IsAot()) {
    std::string valid_soc_model(soc_model);
    if (valid_soc_model == "g5" || valid_soc_model == "g4" ||
        valid_soc_model == "g3") {
      LITERT_LOG(LITERT_WARNING,
                 "g3/g4/g5 is deprecated. Please use Tensor_G3/G4/G5 instead.");
      valid_soc_model =
          absl::StrCat("Tensor_", absl::AsciiStrToUpper(valid_soc_model));
    }
    // Set device type.
    DeviceType device_type;
    LiteRtStatus status =
        google_tensor::GetDeviceType(valid_soc_model, &device_type);
    if (status != kLiteRtStatusOk) {
      LITERT_LOG(LITERT_ERROR, "Invalid soc model for device type: %s",
                 valid_soc_model.c_str());
      return kLiteRtStatusErrorInvalidArgument;
    }
    compiler_config->set_device(device_type);
  } else {
    compiler_config->set_device(
        ::third_party::odml::litert::litert::vendors::google_tensor::compiler::
            DEVICE_TYPE_UNSPECIFIED);
  }

  // serialize to string
  std::string google_tensor_options_str;
  if (!google_tensor_options.SerializeToString(&google_tensor_options_str)) {
    LITERT_LOG(LITERT_ERROR, "%s", "Failed to serialize opaque options proto.");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  // TODO(b/398984678): add support for multiple bytecodes
  absl::string_view model_buffer_view(buffer_str);

  char** compiled_code_data = nullptr;
  size_t* compiled_code_sizes = nullptr;
  size_t num_bytecodes = 0;

  // Ensure memory allocated by the C API is freed.
  absl::Cleanup code_cleanup = [&] {
    if (compiled_code_data) {
      adapter->FreeCompiledCode(compiled_code_data, compiled_code_sizes,
                                num_bytecodes);
    }
  };
  auto compile_status = adapter->Compile(
      model_buffer_view.data(), model_buffer_view.size(),
      google_tensor_options_str.data(), google_tensor_options_str.size(),
      &compiled_code_data, &compiled_code_sizes, &num_bytecodes);
  if (!compile_status) {
    LITERT_LOG(LITERT_ERROR, "%s", compile_status.Error().Message().c_str());
    return compile_status.Error().Status();
  }

  // Result
  auto result = std::make_unique<LiteRtCompiledResultT>();

  if (num_bytecodes != 1) {
    LITERT_LOG(LITERT_ERROR,
               "Compiler returned unexpected number of bytecodes.Expected: "
               "1, Actual: %d",
               num_bytecodes);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  // Append the CustomOp TFLite file as the bytecode.
  result->byte_codes.resize(num_bytecodes);
  for (auto i = 0; i < num_bytecodes; ++i) {
    result->byte_codes[i].assign(compiled_code_data[i], compiled_code_sizes[i]);
  }

  // Append signature names as per_op_data.
  for (auto i = 0; i < num_partitions; ++i) {
    result->per_op_data.push_back(signatures[i]);
  }

  *compiled_result = result.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginRegisterAllTransformations(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtTransformation** transformations, LiteRtParamIndex* num_patterns) {
  *num_patterns = 0;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCheckCompilerCompatibility(
    LiteRtApiVersion api_version, LiteRtCompilerPlugin compiler_plugin,
    LiteRtEnvironmentOptions env, LiteRtOptions options,
    const char* soc_model_name) {
  compiler_plugin->SetLiteRtVersion(api_version);
  return kLiteRtStatusOk;
}

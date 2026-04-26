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
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_logging_helper.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_op_options.h"
#include "litert/c/options/litert_google_tensor_options.h"
#include "litert/c/options/litert_google_tensor_options_type.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/internal/litert_opaque_options_wrapper.h"
#include "litert/cc/internal/litert_options_wrapper.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/google_tensor/adapter.h"
#include "litert/vendors/google_tensor/compiler/google_tensor_options.pb.h"
#include "google/protobuf/text_format.h"  // from @com_google_protobuf
#include "re2/re2.h"  // from @com_googlesource_code_re2

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
using ::third_party::odml::litert::litert::vendors::google_tensor::compiler::
    OpFilter;
using ::third_party::odml::litert::litert::vendors::google_tensor::compiler::
    OpFilters;

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
    kLiteRtOpCodeShloScatter,
    kLiteRtOpCodeShloWindow,
};

constexpr const char* kSupportedStableHloCompositeOps[] = {
    "odml.rms_norm", "odml.group_norm", "odml.scaled_dot_product_attention"};
// clang format on

constexpr auto kNumPluginSocModels =
    sizeof(kPluginSocModels) / sizeof(kPluginSocModels[0]);

}  // namespace google_tensor

LiteRtStatus LrtOptionsToGoogleTensorOptions(
    LrtGoogleTensorOptions lrt_options,
    third_party::odml::litert::litert::vendors::google_tensor::compiler::
        GoogleTensorOptions& google_tensor_options) {
  // FLOAT TRUNCATION TYPE
  LrtGoogleTensorOptionsTruncationType float_trunc;
  LITERT_RETURN_IF_ERROR(
      LrtGoogleTensorOptionsGetFloatTruncationType(lrt_options, &float_trunc));
  switch (float_trunc) {
    case kLiteRtGoogleTensorFloatTruncationTypeAuto:
      google_tensor_options.set_float_truncation_type(
          GoogleTensorOptionsTruncationType::FLOAT_TRUNCATION_TYPE_AUTO);
      break;
    case kLiteRtGoogleTensorFloatTruncationTypeNoTruncation:
      google_tensor_options.set_float_truncation_type(
          GoogleTensorOptionsTruncationType::
              FLOAT_TRUNCATION_TYPE_NO_TRUNCATION);
      break;
    case kLiteRtGoogleTensorFloatTruncationTypeBfloat16:
      google_tensor_options.set_float_truncation_type(
          GoogleTensorOptionsTruncationType::FLOAT_TRUNCATION_TYPE_BFLOAT16);
      break;
    case kLiteRtGoogleTensorFloatTruncationTypeHalf:
      google_tensor_options.set_float_truncation_type(
          GoogleTensorOptionsTruncationType::FLOAT_TRUNCATION_TYPE_HALF);
      break;
  }

  // INT64 TO INT32 TRUNCATION
  bool int64_to_int32;
  LITERT_RETURN_IF_ERROR(LrtGoogleTensorOptionsGetInt64ToInt32Truncation(
      lrt_options, &int64_to_int32));
  google_tensor_options.set_int64_to_int32_truncation(int64_to_int32);

  // DUMP OP TIMINGS
  bool dump_op_timings;
  LITERT_RETURN_IF_ERROR(
      LrtGoogleTensorOptionsGetDumpOpTimings(lrt_options, &dump_op_timings));
  google_tensor_options.set_dump_op_timings(dump_op_timings);

  // ENABLE LARGE MODEL SUPPORT
  bool enable_large_model_support;
  LITERT_RETURN_IF_ERROR(LrtGoogleTensorOptionsGetEnableLargeModelSupport(
      lrt_options, &enable_large_model_support));
  google_tensor_options.set_enable_large_model_support(
      enable_large_model_support);

  // ENABLE 4BIT COMPILATION
  bool enable_4bit;
  LITERT_RETURN_IF_ERROR(LrtGoogleTensorOptionsGetEnable4BitCompilation(
      lrt_options, &enable_4bit));
  google_tensor_options.set_enable_four_bit_compilation(enable_4bit);

  // SHARDING INTENSITY
  LrtGoogleTensorOptionsShardingIntensity sharding_intensity;
  LITERT_RETURN_IF_ERROR(LrtGoogleTensorOptionsGetShardingIntensity(
      lrt_options, &sharding_intensity));
  switch (sharding_intensity) {
    case kLiteRtGoogleTensorShardingIntensityMinimal:
      google_tensor_options.set_sharding_intensity(
          GoogleTensorOptionsShardingIntensity::SHARDING_INTENSITY_MINIMAL);
      break;
    case kLiteRtGoogleTensorShardingIntensityModerate:
      google_tensor_options.set_sharding_intensity(
          GoogleTensorOptionsShardingIntensity::SHARDING_INTENSITY_MODERATE);
      break;
    case kLiteRtGoogleTensorShardingIntensityExtensive:
      google_tensor_options.set_sharding_intensity(
          GoogleTensorOptionsShardingIntensity::SHARDING_INTENSITY_EXTENSIVE);
      break;
    case kLiteRtGoogleTensorShardingIntensityMaximum:
      google_tensor_options.set_sharding_intensity(
          GoogleTensorOptionsShardingIntensity::SHARDING_INTENSITY_MAXIMUM);
      break;
    default:
      google_tensor_options.set_sharding_intensity(
          GoogleTensorOptionsShardingIntensity::SHARDING_INTENSITY_UNSPECIFIED);
      break;
  }

  // ENABLE DYNAMIC RANGE QUANTIZATION
  bool enable_drq;
  LITERT_RETURN_IF_ERROR(
      LrtGoogleTensorOptionsGetEnableDynamicRangeQuantization(lrt_options,
                                                              &enable_drq));
  google_tensor_options.set_enable_dynamic_range_quantization(enable_drq);

  // TESTING FLAGS
  std::vector<std::vector<std::string>> testing_flags;
  LITERT_RETURN_IF_ERROR(
      LrtGoogleTensorOptionsGetTestingFlags(lrt_options, &testing_flags));

  std::string merged_testing_flags;
  for (const auto& group : testing_flags) {
    if (group.empty()) {
      continue;
    }
    if (!merged_testing_flags.empty()) {
      merged_testing_flags += ',';
    }
    if (group.size() >= 2) {
      absl::StrAppend(&merged_testing_flags, group[0], "=", group[1]);
    } else {
      merged_testing_flags += group[0];
    }
  }

  if (!merged_testing_flags.empty()) {
    google_tensor_options.set_testing_flags(merged_testing_flags);
    LITERT_LOG(LITERT_INFO,
               "GoogleTensor Compiler Plugin using testing_flags: '%s'",
               merged_testing_flags.c_str());
  }

  // OP FILTERS PROTO TEXT FILE
  const char* op_filters_path;
  LITERT_RETURN_IF_ERROR(
      LrtGoogleTensorOptionsGetOpFiltersProto(lrt_options, &op_filters_path));
  google_tensor_options.set_op_filters_proto(op_filters_path);

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
  explicit LiteRtCompilerPluginT(const LiteRtCompilerContext* ctx,
                                 LiteRtEnvironmentOptions env,
                                 LiteRtOptions options)
      : ctx_(ctx) {
    if (options) {
      opts_ = litert::internal::OptionsWrapper(
          litert::internal::ContextWrapper(ctx), options,
          litert::OwnHandle::kNo);
      if (opts_) {
        opq_ = opts_->GetOpaqueOptions();
      }
    }
  }

  litert::Expected<LrtGoogleTensorOptions> CreateGoogleTensorOptions() const {
    litert::Expected<LrtGoogleTensorOptions> google_tensor_opts =
        litert::Error(kLiteRtStatusErrorNotFound, "No options found");

    if (opq_) {
      auto target_opq_status =
          opq_->FindOpaqueOptions(LrtGoogleTensorOptionsGetIdentifier());
      if (target_opq_status) {
        const char* payload =
            static_cast<const char*>(target_opq_status.Value());
        LrtGoogleTensorOptions options;
        auto status = LrtCreateGoogleTensorOptionsFromToml(payload, &options);
        if (status == kLiteRtStatusOk) {
          google_tensor_opts = options;
        } else {
          google_tensor_opts =
              litert::Error(status, "Failed to parse Google Tensor options");
        }
      }
    }

    if (!google_tensor_opts) {
      LITERT_LOG(LITERT_INFO, "%s",
                 "No custom google tensor options found, creating default "
                 "options");
      LrtGoogleTensorOptions options;
      auto status = LrtCreateGoogleTensorOptions(&options);
      if (status == kLiteRtStatusOk) {
        google_tensor_opts = options;
      } else {
        google_tensor_opts = litert::Error(
            status, "Failed to create default Google Tensor options");
      }
    }

    return google_tensor_opts;
  }

  ::litert::Expected<litert::internal::OpaqueOptionsWrapper>&
  GetOpaqueOptions() {
    return opq_;
  }
  void SetLiteRtVersion(LiteRtApiVersion v) { litert_version_ = v; }
  LiteRtApiVersion GetLiteRtVersion() const { return litert_version_; }

  LiteRtStatus ReadOpFilters(const std::string& path,
                             OpFilters& op_filters) const {
    if (path.empty()) {
      return kLiteRtStatusOk;
    }

#if defined(__ANDROID__) || defined(__APPLE__)
    // On Android and iOS, google::protobuf::TextFormat is not supported
    // due to the use of Proto Lite. We will skip loading OpFilters
    // from a textproto file on these platforms.
    LITERT_LOG(LITERT_INFO,
               "OpFilters textproto parsing is disabled on mobile platforms.");
    // Return OK, effectively using default/empty OpFilters.
    return kLiteRtStatusOk;
#else
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
      LITERT_LOG(LITERT_ERROR, "Failed to open OpFilters file: %s",
                 path.c_str());
      return kLiteRtStatusErrorNotFound;
    }

    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string proto_text = ss.str();

    if (!google::protobuf::TextFormat::ParseFromString(proto_text, &op_filters)) {
      LITERT_LOG(LITERT_ERROR, "Failed to parse OpFilters proto text from: %s",
                 path.c_str());
      return kLiteRtStatusErrorInvalidArgument;
    }
    return kLiteRtStatusOk;
#endif
  }

 private:
  const LiteRtCompilerContext* ctx_;
  litert::Expected<litert::internal::OptionsWrapper> opts_ =
      litert::Error(kLiteRtStatusErrorInvalidArgument, "Null options");
  litert::Expected<litert::internal::OpaqueOptionsWrapper> opq_ =
      litert::Error(kLiteRtStatusErrorInvalidArgument, "Null opaque options");
  LiteRtApiVersion litert_version_{LITERT_API_VERSION_MAJOR,
                                   LITERT_API_VERSION_MINOR,
                                   LITERT_API_VERSION_PATCH};
};

LiteRtStatus LiteRtCreateCompilerPlugin(
    const LiteRtCompilerContext* compiler_context,
    LiteRtCompilerPlugin* compiler_plugin, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  LiteRtPropagateMinLoggerSeverity(env);

  *compiler_plugin = new LiteRtCompilerPluginT(compiler_context, env, options);
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  if (compiler_plugin == nullptr) {
    return;
  }
  delete compiler_plugin;
}

namespace google_tensor {

// Enum to indicate the outcome of applying filters to an operation.
enum class FilterOutcome {
  // Filters indicate the operation should run on the TPU.
  kRunOnTpu,
  // Filters indicate the operation should NOT run on the TPU and should
  // fall back to another delegate or CPU.
  kDoNotRunOnTpu,
};

// Applies the OpFilters to the given op and returns whether the filters
// indicate the op should run on TPU or not.
FilterOutcome GetFilterOutcome(const litert::Op& op,
                               const OpFilters& op_filters) {
  const auto& filters = op_filters.filters();
  // If there are no filters or op outputs to match against, run on TPU if
  // filter behavior is `MATCHES_NOT_RUN_ON_TPU`, otherwise do not run on TPU.
  if (filters.empty() || op.Outputs().empty()) {
    return op_filters.filter_behavior() == OpFilters::MATCHES_NOT_RUN_ON_TPU
               ? FilterOutcome::kRunOnTpu
               : FilterOutcome::kDoNotRunOnTpu;
  }
  if (op_filters.filter_behavior() == OpFilters::MATCHES_NOT_RUN_ON_TPU) {
    // Blocklist behavior: If any filter matches, do not run on TPU.
    for (const auto& filter : filters) {
      if (filter.op_name_pattern().empty()) {
        LITERT_LOG(LITERT_WARNING, "Empty op_name_pattern in OpFilter.");
        continue;
      }
      for (const auto& output : op.Outputs()) {
        const auto& tensor_name = output.Name();
        if (RE2::FullMatch(tensor_name, filter.op_name_pattern())) {
          LITERT_LOG(LITERT_INFO,
                     "Op with output tensor '%.*s' will NOT RUN ON TPU "
                     "due to MATCHES_NOT_RUN_ON_TPU filter pattern '%s'",
                     static_cast<int>(tensor_name.length()), tensor_name.data(),
                     filter.op_name_pattern().c_str());
          return FilterOutcome::kDoNotRunOnTpu;
        }
      }
    }
    // No filter matched -> run on TPU.
    return FilterOutcome::kRunOnTpu;
  } else {
    // Allowlist behavior: If any filter matches, run on TPU.
    for (const auto& filter : filters) {
      if (filter.op_name_pattern().empty()) {
        LITERT_LOG(LITERT_WARNING, "Empty op_name_pattern in OpFilter.");
        continue;
      }
      for (const auto& output : op.Outputs()) {
        const auto& tensor_name = output.Name();
        if (RE2::FullMatch(tensor_name, filter.op_name_pattern())) {
          LITERT_LOG(LITERT_INFO,
                     "Op with output tensor '%.*s' will RUN ON TPU "
                     "due to MATCHES_RUN_ON_TPU filter pattern '%s'",
                     static_cast<int>(tensor_name.length()), tensor_name.data(),
                     filter.op_name_pattern().c_str());
          return FilterOutcome::kRunOnTpu;
        }
      }
    }
    // No filter matched -> do not run on TPU.
    return FilterOutcome::kDoNotRunOnTpu;
  }
}

bool IsShloCompositeOpSupported(const litert::Op& op) {
  if (op.Code() == kLiteRtOpCodeShloComposite) {
    const char* custom_op_name = nullptr;
    if (LiteRtGetSHLOCompositeOpName(op.Get(), &custom_op_name) !=
            kLiteRtStatusOk ||
        custom_op_name == nullptr) {
      return false;
    }
    // check if the name of the composite op is in the list of
    // kSupportedStableHloCompositeOps.
    for (auto supported_op : kSupportedStableHloCompositeOps) {
      if (strcmp(supported_op, custom_op_name) == 0) {
        return true;
      }
    }
    LITERT_LOG(LITERT_INFO, "unsupported composite op: %s", custom_op_name);
  }
  return false;
}

bool IsOpSupported(const litert::Op& op, const OpFilters& op_filters) {
  // Check if the composite op is supported.
  if (op.Code() == kLiteRtOpCodeShloComposite) {
    return IsShloCompositeOpSupported(op);
  }
  // Check if the op is in the list of unsupported ops.
  for (auto unsupported_op : kUnSupportedOps) {
    if (unsupported_op == op.Code()) {
      return false;
    }
  }

  // Check against user-defined OpFilters. An op is supported for TPU
  // delegation if the filters outcome is kRunOnTpu.
  return GetFilterOutcome(op, op_filters) == FilterOutcome::kRunOnTpu;
}

}  // namespace google_tensor

LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           const char* soc_model,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  if (compiler_plugin == nullptr) {
    LITERT_LOG(LITERT_ERROR, "%s", "compiler_plugin is nullptr");
    return kLiteRtStatusErrorInvalidArgument;
  }

  third_party::odml::litert::litert::vendors::google_tensor::compiler::
      GoogleTensorOptions google_tensor_options;

  auto lrt_google_tensor_options_expected =
      compiler_plugin->CreateGoogleTensorOptions();
  if (!lrt_google_tensor_options_expected) {
    LITERT_LOG(LITERT_ERROR, "Failed to create LrtGoogleTensorOptions: %s",
               lrt_google_tensor_options_expected.Error().Message().c_str());
    return lrt_google_tensor_options_expected.Error().Status();
  }
  auto lrt_google_tensor_options = *lrt_google_tensor_options_expected;

  LiteRtStatus status = LrtOptionsToGoogleTensorOptions(
      lrt_google_tensor_options, google_tensor_options);
  LrtDestroyGoogleTensorOptions(lrt_google_tensor_options);

  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "Failed to convert LrtOptions to GoogleTensorOptions");
    return status;
  }

  OpFilters op_filters;
  LITERT_RETURN_IF_ERROR(compiler_plugin->ReadOpFilters(
      google_tensor_options.op_filters_proto(), op_filters));

  ::litert::Subgraph graph(subgraph);
  for (const auto& op : graph.Ops()) {
    if (!google_tensor::IsOpSupported(op, op_filters)) {
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
  litert::OwningBufferRef<uint8_t, litert::Mallocator<uint8_t>> buf;
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

  third_party::odml::litert::litert::vendors::google_tensor::compiler::
      GoogleTensorOptions google_tensor_options;

  // map to opaque options
  LITERT_ASSIGN_OR_RETURN(auto lrt_google_tensor_options,
                          compiler_plugin->CreateGoogleTensorOptions());
  LiteRtStatus lrt_status = LrtOptionsToGoogleTensorOptions(
      lrt_google_tensor_options, google_tensor_options);
  LrtDestroyGoogleTensorOptions(lrt_google_tensor_options);
  LITERT_RETURN_IF_ERROR(lrt_status);

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

// Copyright 2025 Google LLC.
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

#include "litert/c/options/litert_qualcomm_options.h"

#include <memory>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"

// Expose these definitions into header file once we support that functionality.
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum LiteRtQualcommOptionsBackendType {
  kLiteRtQualcommBackendTypeUndefine = 0,
  kLiteRtQualcommBackendTypeGpu,
  kLiteRtQualcommBackendTypeHtp,
  kLiteRtQualcommBackendTypeDsp,
  kLiteRtQualcommBackendTypeIr,
} LiteRtQualcommOptionsBackendType;

typedef enum LiteRtQualcommOptionsGraphPriority {
  kLiteRtQualcommGraphPriorityDefault = 0,
  kLiteRtQualcommGraphPriorityLow,
  kLiteRtQualcommGraphPriorityNormal,
  kLiteRtQualcommGraphPriorityNormalHigh,
  kLiteRtQualcommGraphPriorityHigh,
  kLiteRtQualcommGraphPriorityUndefined,
} LiteRtQualcommOptionsGraphPriority;

/* Gpu Backend */
typedef enum LiteRtQualcommOptionsGpuPrecision {
  kLiteRtQualcommGpuPrecisionUserProvided = 0,
  kLiteRtQualcommGpuPrecisionFp32,
  kLiteRtQualcommGpuPrecisionFp16,
  kLiteRtQualcommGpuPrecisionHybrid,
} LiteRtQualcommOptionsGpuPrecision;

typedef enum LiteRtQualcommOptionsGpuPerformanceMode {
  kLiteRtQualcommGpuPerformanceModeDefault = 0,
  kLiteRtQualcommGpuPerformanceModeHigh,
  kLiteRtQualcommGpuPerformanceModeNormal,
  kLiteRtQualcommGpuPerformanceModeLow,
} LiteRtQualcommOptionsGpuPerformanceMode;

typedef struct {
  LiteRtQualcommOptionsGpuPrecision precision;
  LiteRtQualcommOptionsGpuPerformanceMode performance_mode;
  const char* kernel_repo_dir;
} LiteRtQualcommGpuBackendOptions;

#define LITERT_QUALCOMM_GPU_BACKEND_OPTIONS_INIT                     \
  {                                                                  \
      kLiteRtQualcommGpuPrecisionFp16,          /*precision*/        \
      kLiteRtQualcommGpuPerformanceModeDefault, /*performance_mode*/ \
      ""                                        /*kernel_repo_dir*/  \
  }

/* Dsp Backend */
typedef enum LiteRtQualcommOptionsDspPerformanceMode {
  kLiteRtQualcommDspPerformanceModeDefault = 0,
  kLiteRtQualcommDspPerformanceModeSustainedHighPerformance = 1,
  kLiteRtQualcommDspPerformanceModeBurst = 2,
  kLiteRtQualcommDspPerformanceModeHighPerformance = 3,
  kLiteRtQualcommDspPerformanceModePowerSaver = 4,
  kLiteRtQualcommDspPerformanceModeLowPowerSaver = 5,
  kLiteRtQualcommDspPerformanceModeHighPowerSaver = 6,
  kLiteRtQualcommDspPerformanceModeLowBalanced = 7,
  kLiteRtQualcommDspPerformanceModeBalanced = 8,
} LiteRtQualcommOptionsDspPerformanceMode;

typedef enum LiteRtQualcommOptionsDspPerfCtrlStrategy {
  kLiteRtQualcommDspPerfCtrlStrategyManual = 0,
  kLiteRtQualcommDspPerfCtrlStrategyAuto = 1,
} LiteRtQualcommOptionsDspPerfCtrlStrategy;

typedef enum LiteRtQualcommOptionsDspPdSession {
  kLiteRtQualcommDspPdSessionUnsigned = 0,
  kLiteRtQualcommDspPdSessionSigned,
  kLiteRtQualcommDspPdSessionAdaptive,
} LiteRtQualcommOptionsDspPdSession;

typedef enum LiteRtQualcommOptionsDspEncoding {
  kLiteRtQualcommDspEncodingStatic = 0,
  kLiteRtQualcommDspEncodingDynamic = 1,
  kLiteRtQualcommDspEncodingUnknown = 0x7fffffff,
} LiteRtQualcommOptionsDspEncoding;

typedef struct {
  LiteRtQualcommOptionsDspPerformanceMode performance_mode;
  LiteRtQualcommOptionsDspPerfCtrlStrategy perf_ctrl_strategy;
  LiteRtQualcommOptionsDspPdSession pd_session;
  LiteRtQualcommOptionsDspEncoding encoding;
} LiteRtQualcommDspBackendOptions;

#define LITERT_QUALCOMM_DSP_BACKEND_OPTIONS_INIT                       \
  {                                                                    \
      kLiteRtQualcommDspPerformanceModeDefault, /*performance_mode*/   \
      kLiteRtQualcommDspPerfCtrlStrategyManual, /*perf_ctrl_strategy*/ \
      kLiteRtQualcommDspPdSessionUnsigned,      /*pd_session*/         \
      kLiteRtQualcommDspEncodingStatic,         /*encoding*/           \
  }

/* Htp Backend */
typedef enum LiteRtQualcommOptionsHtpPerfCtrlStrategy {
  kLiteRtQualcommHtpPerfCtrlStrategyManual = 0,
  kLiteRtQualcommHtpPerfCtrlStrategyAuto = 1,
} LiteRtQualcommOptionsHtpPerfCtrlStrategy;

typedef enum LiteRtQualcommOptionsHtpPrecision {
  kLiteRtQualcommHtpPrecisionQuantized = 0,
  kLiteRtQualcommHtpPrecisionFp16,
} LiteRtQualcommOptionsHtpPrecision;

typedef enum LiteRtQualcommOptionsHtpPdSession {
  kLiteRtQualcommHtpPdSessionUnsigned = 0,
  kLiteRtQualcommHtpPdSessionSigned,
} LiteRtQualcommOptionsHtpPdSession;

typedef enum LiteRtQualcommOptionsHtpOptimizationStrategy {
  kLiteRtQualcommHtpOptimizationStrategyForInference = 0,
  kLiteRtQualcommHtpOptimizationStrategyForPrepare,
  kLiteRtQualcommHtpOptimizationStrategyForInferenceO3,
} LiteRtQualcommOptionsHtpOptimizationStrategy;

typedef struct {
  LiteRtQualcommOptionsHtpPerformanceMode performance_mode;
  LiteRtQualcommOptionsHtpPerfCtrlStrategy perf_ctrl_strategy;
  LiteRtQualcommOptionsHtpPrecision precision;
  LiteRtQualcommOptionsHtpPdSession pd_session;
  LiteRtQualcommOptionsHtpOptimizationStrategy optimization_strategy;
  bool use_conv_hmx;
  bool use_fold_relu;
  uint32_t vtcm_size;
  uint32_t num_hvx_threads;
  uint32_t device_id;
} LiteRtQualcommHtpBackendOptions;

#define LITERT_QUALCOMM_HTP_BACKEND_OPTIONS_INIT                                    \
  {                                                                                 \
      kLiteRtQualcommHtpPerformanceModeDefault, /*performance_mode*/                \
      kLiteRtQualcommHtpPerfCtrlStrategyManual, /*perf_ctrl_strategy*/              \
      kLiteRtQualcommHtpPrecisionFp16,          /*precision*/                       \
      kLiteRtQualcommHtpPdSessionUnsigned,      /*pd_session*/                      \
      kLiteRtQualcommHtpOptimizationStrategyForInference, /*optimization_strategy*/ \
      true,                                               /*use_conv_hmx*/          \
      false,                                              /*use_fold_relu*/         \
      0,                                                  /*vtcm_size*/             \
      0,                                                  /*num_hvx_threads*/       \
      0,                                                  /*device_id*/             \
  }

/* Ir Backend */
typedef struct {
  const char* output_path;
} LiteRtQualcommIrBackendOptions;

#define LITERT_QUALCOMM_IR_BACKEND_OPTIONS_INIT \
  {                                             \
      nullptr, /*output_path*/                  \
  }

typedef enum LiteRtQualcommOptionsPerformanceAction {
  kLiteRtQualcommPerformanceActionVote = 0,
  kLiteRtQualcommPerformanceActionRelease = 1,
} LiteRtQualcommOptionsPerformanceAction;

/* Op Package */
typedef struct {
  const char* custom_op_name;
  const char* qnn_op_type_name;
} LiteRtQualcommOpPackageOpMap;

typedef struct {
  const char* op_package_name;
  const char* op_package_path;
  const char* interface_provider;
  const char* target;
  int num_ops_map;
  LiteRtQualcommOpPackageOpMap* ops_map;
} LiteRtQualcommOpPackageInfo;

typedef struct {
  int num_op_package_infos;
  LiteRtQualcommOpPackageInfo* op_package_infos;
} LiteRtQualcommOpPackageOptions;

#define LITERT_QUALCOMM_OP_PACKAGE_OPTIONS_INIT \
  {                                             \
      0,       /*num_op_package_infos*/         \
      nullptr, /*op_package_infos*/             \
  }

typedef struct {
  const int* skip_delegate_ops;
  uint32_t skip_delegate_ops_nr;
  const int* skip_delegate_node_ids;
  uint32_t skip_delegate_node_ids_nr;
} LiteRtQualcommSkipOption;

#define LITERT_QUALCOMM_SKIP_OPTION_INIT     \
  {                                          \
      nullptr, /*skip_delegate_ops*/         \
      0,       /*skip_delegate_ops_nr*/      \
      nullptr, /*skip_delegate_node_ids*/    \
      0,       /*skip_delegate_node_ids_nr*/ \
  }

typedef struct {
  LiteRtQualcommOptionsBackendType backend_type;
  const char* library_path;
  const char* skel_library_dir;

  LiteRtQualcommGpuBackendOptions gpu_options;
  LiteRtQualcommHtpBackendOptions htp_options;
  LiteRtQualcommDspBackendOptions dsp_options;
  LiteRtQualcommIrBackendOptions ir_options;
  LiteRtQualcommOptionsLogLevel log_level;
  LiteRtQualcommOptionsProfiling profiling;
  LiteRtQualcommOpPackageOptions op_package_options;
  const char* tensor_dump_output_path;
  const char* cache_dir;
  const char* model_token;
  LiteRtQualcommSkipOption skip_options;
  LiteRtQualcommOptionsGraphPriority graph_priority;
} LiteRtQualcommDelegateOptions;

#define LITERT_QUALCOMM_DELEGATE_OPTIONS__INIT                              \
  {                                                                         \
      kLiteRtQualcommBackendTypeUndefine,       /*backend_type*/            \
      "",                                       /*library_path*/            \
      "",                                       /*skel_library_dir*/        \
      LITERT_QUALCOMM_GPU_BACKEND_OPTIONS_INIT, /*gpu_options*/             \
      LITERT_QUALCOMM_HTP_BACKEND_OPTIONS_INIT, /*htp_options*/             \
      LITERT_QUALCOMM_DSP_BACKEND_OPTIONS_INIT, /*dsp_options*/             \
      LITERT_QUALCOMM_IR_BACKEND_OPTIONS_INIT,  /*ir_options*/              \
      kLiteRtQualcommLogOff,                    /*log_level*/               \
      kLiteRtQualcommProfilingOff,              /*profiling*/               \
      LITERT_QUALCOMM_OP_PACKAGE_OPTIONS_INIT,  /*op_package_options*/      \
      "",                                       /*tensor_dump_output_path*/ \
      "",                                       /*cache_dir*/               \
      "",                                       /*model_token*/             \
      LITERT_QUALCOMM_SKIP_OPTION_INIT,         /*skip_options*/            \
      kLiteRtQualcommGraphPriorityDefault,      /*graph_priority*/          \
  }

#ifdef __cplusplus
}
#endif  // __cplusplus

struct LiteRtQualcommOptionsT {
  // delegate specified options
  LiteRtQualcommDelegateOptions delegate_options =
      LITERT_QUALCOMM_DELEGATE_OPTIONS__INIT;
  // litert specified options
  bool enable_weight_sharing = true;
};

LiteRtStatus LiteRtQualcommOptionsCreate(LiteRtOpaqueOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto options_data = std::make_unique<LiteRtQualcommOptionsT>();

  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtQualcommOptionsGetIdentifier(), options_data.get(),
      [](void* payload) {
        delete reinterpret_cast<LiteRtQualcommOptions>(payload);
      },
      options));

  options_data.release();
  return kLiteRtStatusOk;
}

const char* LiteRtQualcommOptionsGetIdentifier() { return "qualcomm"; }

LiteRtStatus LiteRtQualcommOptionsGet(LiteRtOpaqueOptions options,
                                      LiteRtQualcommOptions* options_data) {
  if (options_data == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const char* identifier;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetOpaqueOptionsIdentifier(options, &identifier));
  if (absl::NullSafeStringView(identifier) !=
      LiteRtQualcommOptionsGetIdentifier()) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  void* payload;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptionsData(options, &payload));
  *options_data = reinterpret_cast<LiteRtQualcommOptions>(payload);

  return kLiteRtStatusOk;
}

// GLOBAL OPTIONS //////////////////////////////////////////////////////////////

// log_level -------------------------------------------------------------------

LiteRtStatus LiteRtQualcommOptionsSetLogLevel(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsLogLevel log_level) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->delegate_options.log_level = log_level;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetLogLevel(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsLogLevel* log_level) {
  if (log_level == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *log_level = options->delegate_options.log_level;

  return kLiteRtStatusOk;
}

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

// enable_weight_sharing -------------------------------------------------------

LiteRtStatus LiteRtQualcommOptionsSetEnableWeightSharing(
    LiteRtQualcommOptions options, bool enable_weight_sharing) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->enable_weight_sharing = enable_weight_sharing;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetEnableWeightSharing(
    LiteRtQualcommOptions options, bool* enable_weight_sharing) {
  if (enable_weight_sharing == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *enable_weight_sharing = options->enable_weight_sharing;

  return kLiteRtStatusOk;
}

// DISPATCH OPTIONS ////////////////////////////////////////////////////////////

// power_mode ------------------------------------------------------------------

LiteRtStatus LiteRtQualcommOptionsSetPowerMode(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsPowerMode power_mode) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  // options->power_mode = power_mode;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetPowerMode(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsPowerMode* power_mode) {
  if (power_mode == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  // *power_mode = options->power_mode;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsSetHtpPerformanceMode(
    LiteRtQualcommOptions options,
    LiteRtQualcommOptionsHtpPerformanceMode htp_performace_mode) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->delegate_options.htp_options.performance_mode = htp_performace_mode;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetHtpPerformanceMode(
    LiteRtQualcommOptions options,
    LiteRtQualcommOptionsHtpPerformanceMode* htp_performace_mode) {
  if (options == nullptr || htp_performace_mode == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *htp_performace_mode = options->delegate_options.htp_options.performance_mode;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsSetProfiling(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsProfiling profiling) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->delegate_options.profiling = profiling;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetProfiling(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsProfiling* profiling) {
  if (options == nullptr || profiling == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *profiling = options->delegate_options.profiling;

  return kLiteRtStatusOk;
}

// C++ WRAPPERS ////////////////////////////////////////////////////////////////

namespace litert::qualcomm {

LiteRtQualcommOptions QualcommOptions::Data() const {
  LiteRtQualcommOptions options;
  internal::AssertOk(LiteRtQualcommOptionsGet, Get(), &options);
  return options;
}

Expected<QualcommOptions> QualcommOptions::Create() {
  LiteRtOpaqueOptions options;
  LITERT_RETURN_IF_ERROR(LiteRtQualcommOptionsCreate(&options));
  return QualcommOptions(options, litert::OwnHandle::kYes);
}

void QualcommOptions::SetLogLevel(QualcommOptions::LogLevel log_level) {
  internal::AssertOk(LiteRtQualcommOptionsSetLogLevel, Data(), log_level);
}

QualcommOptions::LogLevel QualcommOptions::GetLogLevel() {
  QualcommOptions::LogLevel log_level;
  internal::AssertOk(LiteRtQualcommOptionsGetLogLevel, Data(), &log_level);
  return log_level;
}

void QualcommOptions::SetPowerMode(QualcommOptions::PowerMode power_mode) {
  internal::AssertOk(LiteRtQualcommOptionsSetPowerMode, Data(), power_mode);
}

QualcommOptions::PowerMode QualcommOptions::GetPowerMode() {
  QualcommOptions::PowerMode power_mode;
  internal::AssertOk(LiteRtQualcommOptionsGetPowerMode, Data(), &power_mode);
  return power_mode;
}

void QualcommOptions::SetEnableWeightSharing(bool weight_sharing_enabled) {
  internal::AssertOk(LiteRtQualcommOptionsSetEnableWeightSharing, Data(),
                     weight_sharing_enabled);
}

bool QualcommOptions::GetEnableWeightSharing() {
  bool enable_weight_sharing;
  internal::AssertOk(LiteRtQualcommOptionsGetEnableWeightSharing, Data(),
                     &enable_weight_sharing);
  return enable_weight_sharing;
}

Expected<QualcommOptions> QualcommOptions::Create(OpaqueOptions& options) {
  const auto id = options.GetIdentifier();
  if (!id || *id != Discriminator()) {
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return QualcommOptions(options.Get(), OwnHandle::kNo);
}

namespace {}  // namespace

}  // namespace litert::qualcomm

// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/c/vendors/qualcomm/accelerator_options.h"

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum LiteRtQnnBackendType {
  kUndefinedBackend = 0,
  kGpuBackend,
  kHtpBackend,
  kDspBackend,
  kIrBackend,
} LiteRtQnnBackendType;

typedef enum LiteRtQnnGraphPriority {
  kQnnPriorityDefault = 0,
  kQnnPriorityLow,
  kQnnPriorityNormal,
  kQnnPriorityNormalHigh,
  kQnnPriorityHigh,
  kQnnPriorityUndefined,
} LiteRtQnnGraphPriority;

/* Gpu Backend */
typedef enum LiteRtQnnGpuPrecision {
  kGpuUserProvided = 0,
  kGpuFp32,
  kGpuFp16,
  kGpuHybrid,
} LiteRtQnnGpuPrecision;

typedef enum LiteRtQnnGpuPerformanceMode {
  kGpuDefault = 0,
  kGpuHigh,
  kGpuNormal,
  kGpuLow,
} LiteRtQnnGpuPerformanceMode;

typedef struct {
  LiteRtQnnGpuPrecision precision;
  LiteRtQnnGpuPerformanceMode performance_mode;
  const char* kernel_repo_dir;
} LiteRtQnnGpuBackendOptions;

#define LITERT_QNN_GPU_OPTION_INIT      \
  {                                     \
      kGpuFp16,    /*precision*/        \
      kGpuDefault, /*performance_mode*/ \
      ""           /*kernel_repo_dir*/  \
  }

typedef enum LiteRtQnnDspPerformanceMode {
  kDspDefault = 0,
  kDspSustainedHighPerformance = 1,
  kDspBurst = 2,
  kDspHighPerformance = 3,
  kDspPowerSaver = 4,
  kDspLowPowerSaver = 5,
  kDspHighPowerSaver = 6,
  kDspLowBalanced = 7,
  kDspBalanced = 8,
} LiteRtQnnDspPerformanceMode;

/* Dsp Backend */
typedef enum LiteRtQnnDspPerfCtrlStrategy {
  kDspPerfCtrlManual = 0,
  kDspPerfCtrlAuto = 1,
} LiteRtQnnDspPerfCtrlStrategy;

typedef enum LiteRtQnnDspPdSession {
  kDspUnsignedPd = 0,
  kDspSignedPd,
  kDspAdaptivePd,
} LiteRtQnnDspPdSession;

typedef enum LiteRtQnnDspEncoding {
  kDspStatic = 0,
  kDspDynamic = 1,
  kDspUnknown = 0x7fffffff,
} LiteRtQnnDspEncoding;

typedef struct {
  LiteRtQnnDspPerformanceMode performance_mode;
  LiteRtQnnDspPerfCtrlStrategy perf_ctrl_strategy;
  LiteRtQnnDspPdSession pd_session;
  LiteRtQnnDspEncoding encoding;
} LiteRtQnnDspBackendOptions;

#define LITERT_QNN_DSP_OPTION_INIT               \
  {                                              \
      kDspDefault,        /*performance_mode*/   \
      kDspPerfCtrlManual, /*perf_ctrl_strategy*/ \
      kDspUnsignedPd,     /*pd_session*/         \
      kDspStatic,         /*encoding*/           \
  }

/* Htp Backend */
typedef enum LiteRtQnnHtpPerfCtrlStrategy {
  kHtpPerfCtrlManual = 0,
  kHtpPerfCtrlAuto = 1,
} LiteRtQnnHtpPerfCtrlStrategy;

typedef enum LiteRtQnnHtpPdSession {
  kHtpUnsignedPd = 0,
  kHtpSignedPd,
} LiteRtQnnHtpPdSession;

typedef enum LiteRtQnnHtpPrecision {
  kHtpQuantized = 0,
  kHtpFp16,
} LiteRtQnnHtpPrecision;

typedef enum LiteRtQnnHtpOptimizationStrategy {
  kHtpOptimizeForInference = 0,
  kHtpOptimizeForPrepare,
  kHtpOptimizeForInferenceO3,
} LiteRtQnnHtpOptimizationStrategy;

typedef enum LiteRtQnnPerformanceAction {
  kPerformanceVote = 0,
  kPerformanceRelease = 1,
} LiteRtQnnPerformanceAction;

typedef struct {
  LiteRtQnnHtpPerformanceMode performance_mode;
  LiteRtQnnHtpPerfCtrlStrategy perf_ctrl_strategy;
  LiteRtQnnHtpPrecision precision;
  LiteRtQnnHtpPdSession pd_session;
  LiteRtQnnHtpOptimizationStrategy optimization_strategy;
  bool use_conv_hmx;
  bool use_fold_relu;
  uint32_t vtcm_size;
  uint32_t num_hvx_threads;
  uint32_t device_id;
} LiteRtQnnHtpBackendOptions;

#define LITERT_QNN_HTP_OPTION_INIT                        \
  {                                                       \
      kHtpDefault,              /*performance_mode*/      \
      kHtpPerfCtrlManual,       /*perf_ctrl_strategy*/    \
      kHtpFp16,                 /*precision*/             \
      kHtpUnsignedPd,           /*pd_session*/            \
      kHtpOptimizeForInference, /*optimization_strategy*/ \
      true,                     /*use_conv_hmx*/          \
      false,                    /*use_fold_relu*/         \
      0,                        /*vtcm_size*/             \
      0,                        /*num_hvx_threads*/       \
      0,                        /*device_id*/             \
  }

/* Ir Backend */
typedef struct {
  const char* output_path;
} LiteRtQnnIrBackendOptions;

#define LITERT_QNN_IR_OPTION_INIT \
  {                               \
      nullptr, /*output_path*/    \
  }

/* Op Package */
typedef struct {
  const char* custom_op_name;
  const char* qnn_op_type_name;
} LiteRtQnnOpPackageOpMap;

typedef struct {
  const char* op_package_name;
  const char* op_package_path;
  const char* interface_provider;
  const char* target;
  int num_ops_map;
  LiteRtQnnOpPackageOpMap* ops_map;
} LiteRtQnnOpPackageInfo;

typedef struct {
  int num_op_package_infos;
  LiteRtQnnOpPackageInfo* op_package_infos;
} LiteRtQnnOpPackageOptions;

#define LITERT_QNN_OP_PACKAGE_OPTION_INIT \
  {                                       \
      0,       /*num_op_package_infos*/   \
      nullptr, /*op_package_infos*/       \
  }

typedef struct {
  const int* skip_delegate_ops;
  uint32_t skip_delegate_ops_nr;
  const int* skip_delegate_node_ids;
  uint32_t skip_delegate_node_ids_nr;
} LiteRtQnnSkipOption;

#define LITERT_QNN_SKIP_OPTION_INIT          \
  {                                          \
      nullptr, /*skip_delegate_ops*/         \
      0,       /*skip_delegate_ops_nr*/      \
      nullptr, /*skip_delegate_node_ids*/    \
      0,       /*skip_delegate_node_ids_nr*/ \
  }

typedef struct {
  LiteRtQnnBackendType backend_type;
  const char* library_path;
  const char* skel_library_dir;
  LiteRtQnnGpuBackendOptions gpu_options;
  LiteRtQnnHtpBackendOptions htp_options;
  LiteRtQnnDspBackendOptions dsp_options;
  LiteRtQnnIrBackendOptions ir_options;
  LiteRtQnnLogLevel log_level;
  LiteRtQnnProfilingOptions profiling;
  LiteRtQnnOpPackageOptions op_package_options;
  const char* tensor_dump_output_path;
  const char* cache_dir;
  const char* model_token;
  LiteRtQnnSkipOption skip_options;
  LiteRtQnnGraphPriority graph_priority;
} LiteRtQnnOptions;

#define LITERT_QNN_OPTION_INIT                                       \
  {                                                                  \
      kUndefinedBackend,                 /*backend_type*/            \
      "",                                /*library_path*/            \
      "",                                /*skel_library_dir*/        \
      LITERT_QNN_GPU_OPTION_INIT,        /*gpu_options*/             \
      LITERT_QNN_HTP_OPTION_INIT,        /*htp_options*/             \
      LITERT_QNN_DSP_OPTION_INIT,        /*dsp_options*/             \
      LITERT_QNN_IR_OPTION_INIT,         /*ir_options*/              \
      kLogOff,                           /*log_level*/               \
      kProfilingOff,                     /*profiling*/               \
      LITERT_QNN_OP_PACKAGE_OPTION_INIT, /*op_package_options*/      \
      "",                                /*tensor_dump_output_path*/ \
      "",                                /*cache_dir*/               \
      "",                                /*model_token*/             \
      LITERT_QNN_SKIP_OPTION_INIT,       /*skip_options*/            \
      kQnnPriorityDefault,               /*graph_priority*/          \
  }

#ifdef __cplusplus
}
#endif  // __cplusplus

namespace {
constexpr const char* kPayloadIdentifier = "qnn-accelerator";
constexpr const LiteRtApiVersion kPayloadVersion = {0, 1, 0};

class QnnAccleratorCompilationOptions {
 public:
  static void Create(void** ptr) {
    *ptr = static_cast<void*>(new QnnAccleratorCompilationOptions());
  }

  static void Destroy(void* ptr) {
    delete static_cast<QnnAccleratorCompilationOptions*>(ptr);
  }

 public:
  QnnAccleratorCompilationOptions() = default;

 private:
  LiteRtQnnOptions qnn_options_ = LITERT_QNN_OPTION_INIT;
};

}  // namespace

// C API implementation, ABI stable.
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LiteRtStatus LiteRtCreateQnnAcceleratorCompilationOptions(
    LiteRtAcceleratorCompilationOptions* options) {
  void* payload = nullptr;
  QnnAccleratorCompilationOptions::Create(&payload);

  auto status = LiteRtCreateAcceleratorCompilationOptions(
      &kPayloadVersion, kPayloadIdentifier, payload,
      QnnAccleratorCompilationOptions::Destroy, options);
  return status;
}

LiteRtStatus LiteRtDestroyQnnAcceleratorCompilationOptions(
    LiteRtAcceleratorCompilationOptions options) {
  LiteRtApiVersion payload_version;
  void* payload_data = nullptr;
  auto status = LiteRtFindAcceleratorCompilationOptionsData(
      options, kPayloadIdentifier, &payload_version, &payload_data);
  // TODO: delete payload

  return status;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

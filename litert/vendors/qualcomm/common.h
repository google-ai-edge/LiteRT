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

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMMON_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMMON_H_

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/options/litert_qualcomm_options.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "QnnCommon.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include "System/QnnSystemInterface.h"  // from @qairt

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define LITERT_RETURN_STATUS_IF_QNN_NOT_OK(expr) \
  if (QNN_SUCCESS != (expr)) {                   \
    return kLiteRtStatusErrorNotFound;           \
  }

// Pointers to functions of a dynamically loaded QNN library.
typedef QNN_INTERFACE_VER_TYPE QnnApi;

// Pointers to functions of a dynamically loaded QNN system library.
typedef QNN_SYSTEM_INTERFACE_VER_TYPE QnnSystemApi;

// QNN backend library should be on DT_RUNPATH (-rpath) (for linux).
#if LITERT_WINDOWS_OS
static const char kLibQnnSystemSo[] = "QnnSystem.dll";
#else
static const char kLibQnnSystemSo[] = "libQnnSystem.so";
#endif

// Android only library.
#if LITERT_WINDOWS_OS
static const char kLibQnnHtpPrepareSo[] = "QnnHtpPrepare.dll";
#else
static const char kLibQnnHtpPrepareSo[] = "libQnnHtpPrepare.so";
#endif

// Map LiteRT element type to Qnn counterpart.
inline LiteRtStatus LegalizeElementType(litert::ElementType litert_type,
                                        Qnn_DataType_t* qnn_type) {
  switch (litert_type) {
    case litert::ElementType::Bool:
      *qnn_type = QNN_DATATYPE_BOOL_8;
      break;
    case litert::ElementType::Int4:
      *qnn_type = QNN_DATATYPE_SFIXED_POINT_4;
      break;
    case litert::ElementType::Int8:
      *qnn_type = QNN_DATATYPE_INT_8;
      break;
    case litert::ElementType::Int16:
      *qnn_type = QNN_DATATYPE_INT_16;
      break;
    case litert::ElementType::Int32:
      *qnn_type = QNN_DATATYPE_INT_32;
      break;
    case litert::ElementType::Int64:
      *qnn_type = QNN_DATATYPE_INT_64;
      break;
    case litert::ElementType::UInt8:
      *qnn_type = QNN_DATATYPE_UINT_8;
      break;
    case litert::ElementType::UInt16:
      *qnn_type = QNN_DATATYPE_UINT_16;
      break;
    case litert::ElementType::UInt32:
      *qnn_type = QNN_DATATYPE_UINT_32;
      break;
    case litert::ElementType::UInt64:
      *qnn_type = QNN_DATATYPE_UINT_64;
      break;
    case litert::ElementType::Float16:
      *qnn_type = QNN_DATATYPE_FLOAT_16;
      break;
    case litert::ElementType::Float32:
      *qnn_type = QNN_DATATYPE_FLOAT_32;
      break;
    case litert::ElementType::Float64:
      *qnn_type = QNN_DATATYPE_FLOAT_64;
      break;
    default:
      return kLiteRtStatusErrorUnsupported;
  }
  return kLiteRtStatusOk;
}

#ifdef __cplusplus
}
#endif  // __cplusplus

inline LiteRtStatus InitQnnOptions(
    ::qnn::Options& qnn_options,
    litert::qualcomm::QualcommOptions& qualcomm_options) {
  qnn_options.SetLogLevel(
      static_cast<::qnn::LogLevel>(qualcomm_options.GetLogLevel()));
  ::qnn::QNNLogger::SetLogLevel(qnn_options.GetLogLevel());
  qnn_options.SetProfiling(
      static_cast<::qnn::Profiling>(qualcomm_options.GetProfiling()));
  qnn_options.SetUseInt64BiasAsInt32(qualcomm_options.GetUseInt64BiasAsInt32());
  qnn_options.SetBackendType(
      static_cast<::qnn::BackendType>(qualcomm_options.GetBackend()));
  qnn_options.SetEnableWeightSharing(qualcomm_options.GetEnableWeightSharing());
  qnn_options.SetUseConvHMX(qualcomm_options.GetUseConvHMX());
  qnn_options.SetUseFoldReLU(qualcomm_options.GetUseFoldReLU());
  qnn_options.SetHtpPPoint(qualcomm_options.GetHtpPPoint());
  qnn_options.SetDlbc(qualcomm_options.GetDlbc());
  // DLBC weights is mutually exclusive with weight sharing (see QAIRT release
  // notes for 2.36+). Force-off and warn when both are requested.
  bool dlbc_weights = qualcomm_options.GetDlbcWeights();
  if (dlbc_weights && qualcomm_options.GetEnableWeightSharing()) {
    LITERT_LOG(LITERT_WARNING,
               "DLBC weights cannot be combined with weight sharing; "
               "forcing dlbc_weights = false.");
    dlbc_weights = false;
  }
  qnn_options.SetDlbcWeights(dlbc_weights);
  qnn_options.SetHtpPerformanceMode(static_cast<::qnn::HtpPerformanceMode>(
      qualcomm_options.GetHtpPerformanceMode()));
  qnn_options.SetDspPerformanceMode(static_cast<::qnn::DspPerformanceMode>(
      qualcomm_options.GetDspPerformanceMode()));
  qnn_options.SetIrJsonDir(qualcomm_options.GetIrJsonDir());
  qnn_options.SetDlcDir(qualcomm_options.GetDlcDir());
  qnn_options.SetVtcmSize(qualcomm_options.GetVtcmSize());
  qnn_options.SetNumHvxThreads(qualcomm_options.GetNumHvxThreads());
  qnn_options.SetOptimizationLevel(static_cast<::qnn::OptimizationLevel>(
      qualcomm_options.GetOptimizationLevel()));
  qnn_options.SetGraphPriority(
      static_cast<::qnn::GraphPriority>(qualcomm_options.GetGraphPriority()));
  qnn_options.SetDumpTensorIds(qualcomm_options.GetDumpTensorIds());
  qnn_options.SetSaverOutputDir(qualcomm_options.GetSaverOutputDir());
  qnn_options.SetGraphIOTensorMemType(static_cast<::qnn::GraphIOTensorMemType>(
      qualcomm_options.GetGraphIOTensorMemType()));

  LITERT_LOG(LITERT_INFO, "\n%s", qnn_options.Dump().data());
  return kLiteRtStatusOk;
}

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMMON_H_

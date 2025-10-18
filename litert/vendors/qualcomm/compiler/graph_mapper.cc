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
//
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/compiler/graph_mapper.h"

#include <stdio.h>

#include <array>

#include "IR/QnnIrGraph.h"
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/vendors/qualcomm/common.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/qnn_manager.h"
#include "HTP/QnnHtpGraph.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnGraph.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace litert::qnn {

float GetOptimizationValue(::qnn::OptimizationLevel level) {
  // Default optimization level value is 2
  switch (level) {
    case ::qnn::OptimizationLevel::kHtpOptimizeForInference:
      return 2.0f;
    case ::qnn::OptimizationLevel::kHtpOptimizeForPrepare:
      return 1.0f;
    case ::qnn::OptimizationLevel::kHtpOptimizeForInferenceO3:
      return 3.0f;
    default:
      return 2.0f;
  }
}

inline absl::Span<const QnnGraph_Config_t*> GetDefaultGraphConfigs(
    const ::qnn::Options& options) {
  static std::array<QnnHtpGraph_CustomConfig_t, 6> graph_custom_configs;
  static std::array<QnnGraph_Config_t, 6> graph_configs;
  static std::array<const QnnGraph_Config_t*, 7> result;

  // QNN suggest always enable relax precision.
  graph_custom_configs[0] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[0].option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
  graph_custom_configs[0].precision = QNN_PRECISION_FLOAT16;
  // Default use O3 for now.
  graph_custom_configs[1] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[1].option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
  graph_custom_configs[1].optimizationOption.type =
      QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
  // Change to 2 if you want to use O2 (default).
  graph_custom_configs[1].optimizationOption.floatValue =
      GetOptimizationValue(options.GetOptimizationLevel());
  // VTCM
  graph_custom_configs[2] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[2].option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
  // The default value is 0 which means the MAX value
  graph_custom_configs[2].vtcmSizeInMB = options.GetVtcmSize();
  // FoldRelu Off
  graph_custom_configs[3] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[3].option =
      QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF;
  graph_custom_configs[3].foldReluActivationIntoConvOff =
      !options.GetUseFoldReLU();
  // ConvHMX Off
  graph_custom_configs[4] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[4].option =
      QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF;
  graph_custom_configs[4].shortDepthConvOnHmxOff = !options.GetUseConvHMX();

  // Hvx Thread
  bool has_hvx = options.GetNumHvxThreads() != 0;
  if (has_hvx) {
    graph_custom_configs[5] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    graph_custom_configs[5].option =
        QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
    graph_custom_configs[5].numHvxThreads = options.GetNumHvxThreads();
  }

  size_t num_config = has_hvx ? 6 : 5;
  for (size_t i = 0; i < num_config; ++i) {
    graph_configs[i] = QNN_GRAPH_CONFIG_INIT;
    graph_configs[i].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graph_configs[i].customConfig = &graph_custom_configs[i];
    result[i] = &graph_configs[i];
  }

  result[num_config] = nullptr;
  return absl::MakeSpan(result.data(), num_config + 1);
}

inline absl::Span<const QnnGraph_Config_t*> GetLegacyGraphConfigs(
    const ::qnn::Options& options) {
  static std::array<QnnHtpGraph_CustomConfig_t, 5> graph_custom_configs;
  static std::array<QnnGraph_Config_t, 5> graph_configs;
  static std::array<const QnnGraph_Config_t*, 6> result;
  // Default use O3 for now.
  graph_custom_configs[0] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[0].option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
  graph_custom_configs[0].optimizationOption.type =
      QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
  // Change to 2 if you want to use O2 (default).
  graph_custom_configs[0].optimizationOption.floatValue =
      GetOptimizationValue(options.GetOptimizationLevel());

  // VTCM
  graph_custom_configs[1] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[1].option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
  // The default value is 0 which means the MAX value
  graph_custom_configs[1].vtcmSizeInMB = options.GetVtcmSize();
  // FoldRelu Off
  graph_custom_configs[2] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[2].option =
      QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF;
  graph_custom_configs[2].foldReluActivationIntoConvOff =
      !options.GetUseFoldReLU();
  // ConvHMX Off
  graph_custom_configs[3] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[3].option =
      QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF;
  graph_custom_configs[3].shortDepthConvOnHmxOff = !options.GetUseConvHMX();

  // Hvx Thread
  bool has_hvx = options.GetNumHvxThreads() != 0;
  if (has_hvx) {
    graph_custom_configs[4] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    graph_custom_configs[4].option =
        QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
    graph_custom_configs[4].numHvxThreads = options.GetNumHvxThreads();
  }

  size_t num_config = has_hvx ? 5 : 4;
  for (size_t i = 0; i < num_config; ++i) {
    graph_configs[i] = QNN_GRAPH_CONFIG_INIT;
    graph_configs[i].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graph_configs[i].customConfig = &graph_custom_configs[i];
    result[i] = &graph_configs[i];
  }

  result[num_config] = nullptr;
  return absl::MakeSpan(result.data(), num_config + 1);
}

absl::Span<const QnnGraph_Config_t*> GetDefaultIrGraphConfigs() {
  static std::array<QnnIrGraph_CustomConfig_t, 1> graph_custom_configs;
  // TODO(Alen): pass dlc path by options.
  graph_custom_configs[0].option = QNN_IR_GRAPH_CONFIG_OPTION_SERIALIZATION;
  graph_custom_configs[0].serializationOption.serializationType =
      QNN_IR_GRAPH_SERIALIZATION_TYPE_FLAT_BUFFER;
  graph_custom_configs[0].serializationOption.outputPath = "";

  static std::array<QnnGraph_Config_t, 1> graph_configs;
  graph_configs[0] = QNN_GRAPH_CONFIG_INIT;
  graph_configs[0].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graph_configs[0].customConfig = &graph_custom_configs[0];

  static std::array<const QnnGraph_Config_t*, 2> result = {&graph_configs[0],
                                                           nullptr};

  return absl::MakeSpan(result.data(), result.size());
}

absl::Span<const QnnGraph_Config_t*> GraphMapper::PickGraphConfigHeuristic(
    const ::qnn::Options& options) {
  if (qnn_.IsLegacySocModel()) {
    return GetLegacyGraphConfigs(options);
  } else {
    return GetDefaultGraphConfigs(options);
  }
}

QnnManager& GraphMapper::Qnn() { return qnn_; }

Qnn_GraphHandle_t& GraphMapper::QnnGraph() { return qnn_graph_; }

LiteRtStatus GraphMapper::IsLiteRtSubgraphSupported() {
  // For now, we assume all LiteRt subgraphs are supported.
  // TODO: b/381133565: Implement or remove this function.
  return kLiteRtStatusOk;
}

LiteRtStatus GraphMapper::InitQnnGraph(absl::string_view qnn_graph_name,
                                       const ::qnn::Options& options) {
  switch (options.GetBackendType()) {
    case ::qnn::BackendType::kHtpBackend: {
      LITERT_RETURN_STATUS_IF_QNN_NOT_OK(qnn_.Api()->graphCreate(
          context_handle_, qnn_graph_name.data(),
          PickGraphConfigHeuristic(options).data(), &QnnGraph()));
      break;
    }
    case ::qnn::BackendType::kIrBackend: {
      LITERT_RETURN_STATUS_IF_QNN_NOT_OK(qnn_.Api()->graphCreate(
          context_handle_, qnn_graph_name.data(),
          GetDefaultIrGraphConfigs().data(), &QnnGraph()));
      break;
    }
    default: {
      LITERT_LOG(LITERT_ERROR, "Unsupported Backend to create graph");
      return kLiteRtStatusErrorUnsupported;
    }
  }

  return kLiteRtStatusOk;
}

LiteRtStatus GraphMapper::Finalize() {
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      qnn_.Api()->graphFinalize(QnnGraph(), profile_handle_, nullptr));
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn

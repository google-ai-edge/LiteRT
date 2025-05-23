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

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/qualcomm/common.h"
#include "litert/vendors/qualcomm/qnn_manager.h"
#include "HTP/QnnHtpGraph.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnGraph.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace litert::qnn {

inline absl::Span<const QnnGraph_Config_t*> GetDefaultGraphConfigs() {
  static std::array<QnnHtpGraph_CustomConfig_t, 4> graph_custom_configs;
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
  graph_custom_configs[1].optimizationOption.floatValue = 3;
  // VTCM
  graph_custom_configs[2] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[2].option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
  graph_custom_configs[2].vtcmSizeInMB = QNN_HTP_GRAPH_CONFIG_OPTION_MAX;
  // FoldRelu Off
  graph_custom_configs[3] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[3].option =
      QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF;
  graph_custom_configs[3].foldReluActivationIntoConvOff = true;

  static std::array<QnnGraph_Config_t, 4> graph_configs;
  graph_configs[0] = QNN_GRAPH_CONFIG_INIT;
  graph_configs[0].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graph_configs[0].customConfig = &graph_custom_configs[0];

  graph_configs[1] = QNN_GRAPH_CONFIG_INIT;
  graph_configs[1].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graph_configs[1].customConfig = &graph_custom_configs[1];

  graph_configs[2] = QNN_GRAPH_CONFIG_INIT;
  graph_configs[2].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graph_configs[2].customConfig = &graph_custom_configs[2];

  graph_configs[3] = QNN_GRAPH_CONFIG_INIT;
  graph_configs[3].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graph_configs[3].customConfig = &graph_custom_configs[3];

  static std::array<const QnnGraph_Config_t*, 5> result = {
      &graph_configs[0], &graph_configs[1], &graph_configs[2],
      &graph_configs[3], nullptr};

  return absl::MakeSpan(result.data(), result.size());
}

inline absl::Span<const QnnGraph_Config_t*> GetLegacyGraphConfigs() {
  static std::array<QnnHtpGraph_CustomConfig_t, 3> graph_custom_configs;
  // Default use O3 for now.
  graph_custom_configs[0] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[0].option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
  graph_custom_configs[0].optimizationOption.type =
      QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
  // Change to 2 if you want to use O2 (default).
  graph_custom_configs[0].optimizationOption.floatValue = 3;

  // VTCM
  graph_custom_configs[1] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[1].option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
  graph_custom_configs[1].vtcmSizeInMB = QNN_HTP_GRAPH_CONFIG_OPTION_MAX;
  // FoldRelu Off
  graph_custom_configs[2] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[2].option =
      QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF;
  graph_custom_configs[2].foldReluActivationIntoConvOff = true;

  static std::array<QnnGraph_Config_t, 3> graph_configs;
  graph_configs[0] = QNN_GRAPH_CONFIG_INIT;
  graph_configs[0].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graph_configs[0].customConfig = &graph_custom_configs[0];

  graph_configs[1] = QNN_GRAPH_CONFIG_INIT;
  graph_configs[1].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graph_configs[1].customConfig = &graph_custom_configs[1];

  graph_configs[2] = QNN_GRAPH_CONFIG_INIT;
  graph_configs[2].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graph_configs[2].customConfig = &graph_custom_configs[2];

  static std::array<const QnnGraph_Config_t*, 4> result = {
      &graph_configs[0], &graph_configs[1], &graph_configs[2], nullptr};

  return absl::MakeSpan(result.data(), result.size());
}

absl::Span<const QnnGraph_Config_t*> GraphMapper::PickGraphConfigHeuristic() {
  if (qnn_.IsLegacySocModel()) {
    return GetLegacyGraphConfigs();
  } else {
    return GetDefaultGraphConfigs();
  }
}

QnnManager& GraphMapper::Qnn() { return qnn_; }

Qnn_GraphHandle_t& GraphMapper::QnnGraph() { return qnn_graph_; }

LiteRtStatus GraphMapper::IsLiteRtSubgraphSupported() {
  // For now, we assume all LiteRt subgraphs are supported.
  // TODO: b/381133565: Implement or remove this function.
  return kLiteRtStatusOk;
}

LiteRtStatus GraphMapper::InitQnnGraph(absl::string_view qnn_graph_name) {
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      qnn_.Api()->graphCreate(context_handle_, qnn_graph_name.data(),
                              PickGraphConfigHeuristic().data(), &QnnGraph()));
  return kLiteRtStatusOk;
}

LiteRtStatus GraphMapper::Finalize() {
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      qnn_.Api()->graphFinalize(QnnGraph(), nullptr, nullptr));
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn

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

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/vendors/qualcomm/core/backends/graph_config_builder.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/qnn_manager.h"

namespace litert::qnn {

QnnManager& GraphMapper::Qnn() { return qnn_; }

Qnn_GraphHandle_t& GraphMapper::QnnGraph() { return qnn_graph_; }

LiteRtStatus GraphMapper::InitQnnGraph(absl::string_view qnn_graph_name,
                                       const ::qnn::Options& options) {
  ::qnn::GraphConfigBuilder graph_configs =
      qnn_.BuildGraphConfigs(options, qnn_graph_name);

  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      qnn_.Api()->graphCreate(context_handle_, qnn_graph_name.data(),
                              graph_configs.GetNullTerminatedConfigs().data(), &QnnGraph()));

  return kLiteRtStatusOk;
}

LiteRtStatus GraphMapper::Finalize() {
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      qnn_.Api()->graphFinalize(QnnGraph(), profile_handle_, nullptr));
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn

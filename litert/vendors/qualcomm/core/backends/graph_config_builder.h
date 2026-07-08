// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_GRAPH_CONFIG_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_GRAPH_CONFIG_BUILDER_H_

#include <list>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "GPU/QnnGpuGraph.h"  // from @qairt
#include "HTP/QnnHtpGraph.h"  // from @qairt
#include "IR/QnnIrGraph.h"  // from @qairt
#include "QnnGraph.h"  // from @qairt

namespace qnn {

using ConfigPayload = std::variant<std::string, // For output path of IrGraph.
                                    QnnIrGraph_CustomConfig_t,
                                    QnnGpuGraph_CustomConfig_t,
                                    QnnHtpGraph_CustomConfig_t>;

class GraphConfigBuilder {
 public:
  GraphConfigBuilder() = default;

  GraphConfigBuilder(GraphConfigBuilder&&) = default;
  GraphConfigBuilder& operator=(GraphConfigBuilder&&) = default;
  GraphConfigBuilder(const GraphConfigBuilder&) = delete;
  GraphConfigBuilder& operator=(const GraphConfigBuilder&) = delete;

  template <typename T>
  T& Store(const T& init = T{}) {
    return std::get<T>(storage_.emplace_back(std::in_place_type<T>, init));
  }

  template <typename T>
  void AddCustomConfig(const T& custom_config) {
    T& cc = Store<T>(custom_config);
    auto& gc = graph_configs_.emplace_back();
    gc = QNN_GRAPH_CONFIG_INIT;
    gc.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    gc.customConfig = &cc;
  }

  void AddGraphConfig(const QnnGraph_Config_t& config) {
    graph_configs_.emplace_back(config);
  }

  std::vector<const QnnGraph_Config_t*> GetNullTerminatedConfigs() const {
    std::vector<const QnnGraph_Config_t*> ptrs;
    ptrs.reserve(graph_configs_.size() + 1);
    for (const auto& gc : graph_configs_) {
      ptrs.emplace_back(&gc);
    }
    ptrs.emplace_back(nullptr);
    return ptrs;
  }

 private:
  std::list<ConfigPayload> storage_;
  std::vector<QnnGraph_Config_t> graph_configs_;
};

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_GRAPH_CONFIG_BUILDER_H_

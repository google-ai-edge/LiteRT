// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_GRAPH_CONFIG_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_GRAPH_CONFIG_BUILDER_H_

#include <any>
#include <list>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "QnnGraph.h"  // from @qairt

namespace qnn {

// Owns the storage for one graph's worth of QNN configs and returns the
// null-terminated QnnGraph_Config_t* array graphCreate / graphSetConfig expect.
// A backend builds one per graph (BuildGraphConfigs returns it by value); the
// caller keeps it alive for the duration of the API call. It owns:
//   - custom config structs, each wrapped in a QnnGraph_Config_t (AddCustom);
//   - bare QnnGraph_Config_t entries, e.g. graph priority (AddGraphConfig);
//   - side buffers a custom config points into, e.g. a nested sub-config or a
//     path string (Store).
// Everything lives in std::list nodes, whose addresses stay stable across
// appends and moves, so returning the builder by value keeps the pointers in
// the QnnGraph_Config_t entries valid.
class GraphConfigBuilder {
 public:
  GraphConfigBuilder() = default;

  // Move-only: a copy would leave the QnnGraph_Config_t pointers aimed at the
  // source's storage, dangling once the source dies. Moving std::list is safe.
  GraphConfigBuilder(GraphConfigBuilder&&) = default;
  GraphConfigBuilder& operator=(GraphConfigBuilder&&) = default;
  GraphConfigBuilder(const GraphConfigBuilder&) = delete;
  GraphConfigBuilder& operator=(const GraphConfigBuilder&) = delete;

  // Owns a side buffer (initialized to `init`) that a custom config points
  // into, and returns a reference to it. Use for nested sub-configs or strings
  // whose address must outlive the graphCreate / graphSetConfig call.
  template <typename T>
  T& Store(const T& init = T{}) {
    return std::any_cast<T&>(
        storage_.emplace_back(std::in_place_type<T>, init));
  }

  // Owns a new custom config (initialized to `init`, the SDK *_INIT value),
  // wraps it in a QnnGraph_Config_t{option = CUSTOM, customConfig = &cc}, and
  // returns a reference to the custom struct for the caller to fill in.
  template <typename T>
  T& AddCustom(const T& init = T{}) {
    T& cc = Store<T>(init);
    auto& gc = graph_configs_.emplace_back();
    gc = QNN_GRAPH_CONFIG_INIT;
    gc.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    gc.customConfig = &cc;
    return cc;
  }

  // Owns a bare (non-custom) QnnGraph_Config_t (e.g. HTP graph priority),
  // initialized to QNN_GRAPH_CONFIG_INIT. The caller sets .option and the
  // matching union field.
  QnnGraph_Config_t& AddGraphConfig() {
    auto& gc = graph_configs_.emplace_back();
    gc = QNN_GRAPH_CONFIG_INIT;
    return gc;
  }

  // Returns a null-terminated array view to pass to graphCreate /
  // graphSetConfig. Backed by owned storage; valid until the next mutating
  // call (AddCustom / AddGraphConfig).
  absl::Span<const QnnGraph_Config_t*> Configs() {
    ptrs_.clear();
    ptrs_.reserve(graph_configs_.size() + 1);
    for (const auto& gc : graph_configs_) {
      ptrs_.push_back(&gc);
    }
    ptrs_.push_back(nullptr);
    return absl::MakeSpan(ptrs_);
  }

 private:
  // Type-erased side buffers + custom config structs.
  std::list<std::any> storage_;
  std::list<QnnGraph_Config_t> graph_configs_;
  std::vector<const QnnGraph_Config_t*> ptrs_;
};

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_GRAPH_CONFIG_BUILDER_H_

// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CONTEXT_CONFIG_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CONTEXT_CONFIG_BUILDER_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "QnnCommon.h"  // from @qairt
#include "QnnContext.h"  // from @qairt
#include "GPU/QnnGpuContext.h"  // from @qairt
#include "HTP/QnnHtpContext.h"  // from @qairt
#include "litert/vendors/qualcomm/core/common.h"

namespace qnn {

class HtpWeightSharingConfig {
 public:
  explicit HtpWeightSharingConfig(bool enabled)
      : config_(QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT) {
    config_.option = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
    config_.weightSharingEnabled = enabled;
  }

  QnnContext_CustomConfig_t GetQnnConfig() { return &config_; }

 private:
  QnnHtpContext_CustomConfig_t config_;
};

class HtpFileReadMemoryBudgetConfig {
 public:
  explicit HtpFileReadMemoryBudgetConfig(uint32_t budget_mb)
      : config_(QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT) {
    config_.option = QNN_HTP_CONTEXT_CONFIG_OPTION_FILE_READ_MEMORY_BUDGET;
    config_.fileReadMemoryBudgetInMb = budget_mb;
  }

  QnnContext_CustomConfig_t GetQnnConfig() { return &config_; }

 private:
  QnnHtpContext_CustomConfig_t config_;
};

class HtpInitAccelerationConfig {
 public:
  explicit HtpInitAccelerationConfig(bool enabled)
      : config_(QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT) {
    config_.option = QNN_HTP_CONTEXT_CONFIG_OPTION_INIT_ACCELERATION;
    config_.initAcceleration = enabled;
  }

  QnnContext_CustomConfig_t GetQnnConfig() { return &config_; }

 private:
  QnnHtpContext_CustomConfig_t config_;
};

class GpuPerformanceConfig {
 public:
  explicit GpuPerformanceConfig(QnnGpuContext_PerfHint_t performance_hint)
      : config_(QNN_GPU_CONTEXT_CUSTOM_CONFIG_INIT) {
    config_.option = QNN_GPU_CONTEXT_CONFIG_OPTION_PERF_HINT;
    config_.perfHint = performance_hint;
  }

  QnnContext_CustomConfig_t GetQnnConfig() { return &config_; }

 private:
  QnnGpuContext_CustomConfig_t config_;
};

using ContextCustomConfig =
    std::variant<HtpWeightSharingConfig, HtpFileReadMemoryBudgetConfig,
                 HtpInitAccelerationConfig, GpuPerformanceConfig>;

template <typename T, typename Variant>
struct IsVariantAlternative;

template <typename T, typename... Alternatives>
struct IsVariantAlternative<T, std::variant<Alternatives...>>
    : std::disjunction<std::is_same<T, Alternatives>...> {};

template <typename CustomConfigVariant>
using ContextConfigEntry =
    std::variant<QnnContext_Config_t, CustomConfigVariant>;

template <typename CustomConfigVariant>
class ContextConfigsBuilder;

// Immutable, null-terminated QNN context config list. The container owns all
// custom config payloads referenced by Data().
template <typename CustomConfigVariant>
class ContextConfigs {
 public:
  ContextConfigs(const ContextConfigs&) = delete;
  ContextConfigs& operator=(const ContextConfigs&) = delete;
  ContextConfigs(ContextConfigs&&) = delete;
  ContextConfigs& operator=(ContextConfigs&&) = delete;

  const QnnContext_Config_t* const* Data() const { return config_ptrs_.data(); }

  size_t Size() const { return configs_.size(); }

 private:
  friend class ContextConfigsBuilder<CustomConfigVariant>;

  explicit ContextConfigs(
      std::vector<ContextConfigEntry<CustomConfigVariant>> entries)
      : entries_(std::move(entries)) {
    configs_.reserve(entries_.size());
    for (auto& entry : entries_) {
      if (const auto* config = std::get_if<QnnContext_Config_t>(&entry)) {
        configs_.push_back(*config);
        continue;
      }

      QnnContext_Config_t config = QNN_CONTEXT_CONFIG_INIT;
      config.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
      config.customConfig = std::visit(
          [](auto& custom_config) { return custom_config.GetQnnConfig(); },
          std::get<CustomConfigVariant>(entry));
      configs_.push_back(config);
    }

    config_ptrs_.reserve(configs_.size() + 1);
    for (const auto& config : configs_) {
      config_ptrs_.push_back(&config);
    }
    config_ptrs_.push_back(nullptr);
  }

  std::vector<ContextConfigEntry<CustomConfigVariant>> entries_;
  std::vector<QnnContext_Config_t> configs_;
  std::vector<const QnnContext_Config_t*> config_ptrs_;
};

template <typename CustomConfigVariant>
class ContextConfigsBuilder {
 public:
  ContextConfigsBuilder() = default;
  ContextConfigsBuilder(ContextConfigsBuilder&&) = default;
  ContextConfigsBuilder& operator=(ContextConfigsBuilder&&) = default;
  ContextConfigsBuilder(const ContextConfigsBuilder&) = delete;
  ContextConfigsBuilder& operator=(const ContextConfigsBuilder&) = delete;

  ContextConfigsBuilder& AddContextConfig(QnnContext_Config_t config) {
    entries_.emplace_back(std::in_place_type<QnnContext_Config_t>, config);
    return *this;
  }

  template <typename T, typename... Args>
  ContextConfigsBuilder& EmplaceCustomConfig(Args&&... args) {
    static_assert(IsVariantAlternative<T, CustomConfigVariant>::value,
                  "T must be an alternative of CustomConfigVariant");
    entries_.emplace_back(std::in_place_type<CustomConfigVariant>,
                          std::in_place_type<T>, std::forward<Args>(args)...);
    return *this;
  }

  ContextConfigs<CustomConfigVariant> Build() && {
    return ContextConfigs<CustomConfigVariant>(std::move(entries_));
  }

 private:
  std::vector<ContextConfigEntry<CustomConfigVariant>> entries_;
};

using QnnContextConfigs = ContextConfigs<ContextCustomConfig>;
using QnnContextConfigsBuilder = ContextConfigsBuilder<ContextCustomConfig>;

QnnContextConfigs BuildCompileContextConfigs(
    const Options& options, bool enable_effective_weight_sharing);
QnnContextConfigs BuildDispatchContextConfigs(const Options& options);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CONTEXT_CONFIG_BUILDER_H_

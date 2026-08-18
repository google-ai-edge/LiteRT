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
//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
//

#ifndef ODML_LITERT_LITERT_CC_LITERT_COMPILATION_OPTIONS_H_
#define ODML_LITERT_LITERT_CC_LITERT_COMPILATION_OPTIONS_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_op_kernel.h"
#include "litert/cc/internal/litert_runtime_proxy.h"
#include "litert/cc/internal/scoped_file.h"
#include "litert/cc/internal/scoped_weight_source.h"
#include "litert/cc/litert_api_types.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_custom_op_kernel.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/options/litert_arm_options.h"  // IWYU pragma: keep
#include "litert/cc/options/litert_compiler_options.h"  // IWYU pragma: keep
#include "litert/cc/options/litert_concrete_options_base.h"
#include "litert/cc/options/litert_cpu_options.h"  // IWYU pragma: keep
#include "litert/cc/options/litert_google_tensor_options.h"  // IWYU pragma: keep
#include "litert/cc/options/litert_gpu_options.h"  // IWYU pragma: keep
#include "litert/cc/options/litert_intel_openvino_options.h"  // IWYU pragma: keep
#include "litert/cc/options/litert_mediatek_options.h"  // IWYU pragma: keep
#include "litert/cc/options/litert_qualcomm_options.h"  // IWYU pragma: keep
#include "litert/cc/options/litert_runtime_options.h"  // IWYU pragma: keep
#include "litert/cc/options/litert_samsung_options.h"  // IWYU pragma: keep
#include "litert/core/options.h"

namespace litert {
namespace options_impl {

template <typename OptionType>
Expected<OptionType&> EnsureOption(std::optional<OptionType>& slot) {
  if (!slot) {
    LITERT_ASSIGN_OR_RETURN(auto option, OptionType::Create());
    slot.emplace(std::move(option));
  }
  return slot.value();
}

template <typename OptionType>
LiteRtStatus AppendAndReset(internal::RuntimeProxy* runtime,
                            LiteRtOptions options,
                            std::optional<OptionType>& slot) {
  if (!slot) {
    return kLiteRtStatusOk;
  }
  LiteRtOpaqueOptions opaque = slot->Release();
  slot.reset();
  return runtime->AddOpaqueOptions(options, opaque);
}

template <typename OptionType, typename GetDataFunc>
LiteRtStatus AppendAndResetOpaqueData(internal::RuntimeProxy* runtime,
                                      LiteRtOptions options,
                                      const std::optional<OptionType>& slot,
                                      GetDataFunc get_data_func) {
  if (!slot) {
    return kLiteRtStatusOk;
  }
  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_RETURN_IF_ERROR(
      get_data_func(slot->Get(), &identifier, &payload, &payload_deleter));
  LiteRtOpaqueOptions opaque_opts = nullptr;
  LITERT_RETURN_IF_ERROR(runtime->CreateOpaqueOptions(
      identifier, payload, payload_deleter, &opaque_opts));
  LITERT_RETURN_IF_ERROR(runtime->AddOpaqueOptions(options, opaque_opts));
  return kLiteRtStatusOk;
}

}  // namespace options_impl

class CompiledModel;
class CompiledModelNext;

namespace internal {

// Opaque unique identifier for a C++ option type.
using OptionTypeId = const void*;

// Returns a unique pointer address (&type_tag) for each distinct type T
// instantiated in the program. This serves as a fast, zero-overhead runtime
// type key for option lookup without requiring RTTI or string identifiers.
template <typename T>
OptionTypeId GetOptionTypeId() {
  static const char type_tag = 0;
  return &type_tag;
}

struct LiteRtDestroyOptionsDeleter {
  void (*destroy_options)(LiteRtOptionsT*) = nullptr;
  void operator()(LiteRtOptionsT* options) const {
    if (options && destroy_options) {
      destroy_options(options);
    }
  }
};

using LiteRtOptionsPtr =
    std::unique_ptr<LiteRtOptionsT, internal::LiteRtDestroyOptionsDeleter>;

class LiteRtOptionsPtrBuilder;
}  // namespace internal

/// Manages the configuration options for compiling a LiteRT model.
///
/// This class provides methods to set hardware accelerators, add custom
/// operations, bind external tensors, and configure various backend-specific
/// options (e.g., GPU, CPU, Qualcomm, MediaTek, etc.).
class Options {
 public:
  friend class internal::LiteRtOptionsPtrBuilder;

  /// A map from a group name to a weight section.
  ///
  /// A weight section contains the offset and length of a contiguous region
  /// inside a `ScopedFile` that backs a single external buffer group. This map
  /// provides the mapping between the group name and its section.
  using ScopedWeightSectionMap = FlatHashMap<std::string, ScopedWeightSection>;

  Options() = default;

  /// Creates a new `Options` object.
  static Expected<Options> Create() { return Options(); }

  /// Sets the hardware accelerators to be used for the model.
  /// @param accelerators A bitmask of hardware accelerators.
  /// @return An `Expected` object that is empty on success, or contains an
  /// error.
  Expected<void> SetHardwareAccelerators(HwAccelerators accelerators) {
    lite_rt_hw_accelerator_set_ =
        static_cast<LiteRtHwAcceleratorSet>(accelerators);
    return {};
  }

  /// Sets the hardware accelerators to be used for the model.
  /// @param accelerators A set of hardware accelerators.
  /// @return An `Expected` object that is empty on success, or contains an
  /// error.
  Expected<void> SetHardwareAccelerators(HwAcceleratorSet accelerators) {
    lite_rt_hw_accelerator_set_ =
        static_cast<LiteRtHwAcceleratorSet>(accelerators.value);
    return {};
  }

  /// Retrieves the currently set hardware accelerators.
  /// @return An `Expected` object containing the set of hardware accelerators,
  /// or an error.
  Expected<LiteRtHwAcceleratorSet> GetHardwareAccelerators() const {
    if (lite_rt_hw_accelerator_set_.has_value()) {
      return *lite_rt_hw_accelerator_set_;
    }
    return litert::Error(litert::Status::kErrorInvalidArgument,
                         "Hardware accelerators are not set.");
  }

  Expected<void> AddOpaqueOptions(OpaqueOptions&& options) {
    opaque_options_.push_back(options.Release());
    return {};
  }

  /// Adds a custom operator kernel.
  /// @param custom_op_name The name of the custom operator.
  /// @param custom_op_version The version of the custom operator.
  /// @param custom_op_kernel The custom operator kernel implementation.
  /// @param custom_op_kernel_user_data User data to be passed to the kernel.
  /// @return An `Expected` object that is empty on success, or contains an
  /// error.
  Expected<void> AddCustomOpKernel(const std::string& custom_op_name,
                                   int custom_op_version,
                                   const LiteRtCustomOpKernel& custom_op_kernel,
                                   void* custom_op_kernel_user_data = nullptr) {
    build_actions_.push_back(
        [custom_op_name, custom_op_version, custom_op_kernel,
         custom_op_kernel_user_data](internal::RuntimeProxy* runtime,
                                     LiteRtOptions options) {
          return runtime->AddCustomOpKernelOption(
              options, custom_op_name.c_str(), custom_op_version,
              &custom_op_kernel, custom_op_kernel_user_data);
        });
    return {};
  }

  /// Adds a custom operator kernel.
  /// @param custom_op_kernel The custom operator kernel to add.
  /// @return An `Expected` object that is empty on success, or contains an
  /// error.
  Expected<void> AddCustomOpKernel(CustomOpKernel& custom_op_kernel) {
    return AddCustomOpKernel(custom_op_kernel.OpName(),
                             custom_op_kernel.OpVersion(),
                             custom_op_kernel.GetLiteRtCustomOpKernel(),
                             static_cast<void*>(&custom_op_kernel));
  }

  /// Adds a custom build action.
  /// @internal This is experimental and primarily for internal use.
  Expected<void> AddBuildAction(
      std::function<LiteRtStatus(internal::RuntimeProxy*, LiteRtOptions)>
          action) {
    build_actions_.push_back(std::move(action));
    return {};
  }

  /// Binds an external memory buffer to a specific tensor in the model.
  ///
  /// This function sets the tensor's allocation type to `kTfLiteCustom`,
  /// making it appear as a constant tensor with a pre-allocated buffer.
  ///
  /// @note `data` is owned by the caller and must outlive the lifetime of the
  /// `CompiledModel`. `size_bytes` must match the tensor's expected size.
  /// @param signature_name The name of the signature containing the tensor.
  /// @param tensor_name The name of the tensor to bind.
  /// @param data A pointer to the external memory buffer.
  /// @param size_bytes The size of the external memory buffer in bytes.
  /// @return An `Expected` object that is empty on success, or contains an
  /// error.
  Expected<void> AddExternalTensorBinding(const std::string& signature_name,
                                          const std::string& tensor_name,
                                          void* data, size_t size_bytes) {
    build_actions_.push_back([signature_name, tensor_name, data, size_bytes](
                                 internal::RuntimeProxy* runtime,
                                 LiteRtOptions options) {
      return runtime->AddExternalTensorBinding(options, signature_name.c_str(),
                                               tensor_name.c_str(), data,
                                               size_bytes);
    });
    return {};
  }

  /// Sets the weight loader owned by the client and used for the model.
  /// @param weight_loader The weight loader to be used for the model.
  /// @return An `Expected` object that is empty on success, or contains an
  /// error.
  Expected<void> SetWeightLoader(weight_loader::WeightLoader* weight_loader) {
    build_actions_.push_back([weight_loader](internal::RuntimeProxy* runtime,
                                             LiteRtOptions options) {
      auto* options_impl = reinterpret_cast<LiteRtOptionsT*>(options);
      if (!options_impl) {
        return kLiteRtStatusErrorRuntimeFailure;
      }
      options_impl->weight_loader = weight_loader;
      return kLiteRtStatusOk;
    });
    return {};
  }

#ifndef LITERT_NO_ABSL
  /// Sets the in-memory weights map owned by the client and used for the model.
  ///
  /// This extension uses Abseil container and span types and is therefore only
  /// available in the default C++ API mode.
  /// @param map The weight map mapping group names to contiguous buffers.
  /// @return An `Expected` object that is empty on success, or contains an
  /// error.
  Expected<void> SetWeightInMemoryMap(
      const FlatHashMap<std::string, Span<const std::byte>>* map) {
    build_actions_.push_back(
        [map](internal::RuntimeProxy* runtime, LiteRtOptions options) {
          auto* options_impl = reinterpret_cast<LiteRtOptionsT*>(options);
          if (!options_impl) {
            return kLiteRtStatusErrorRuntimeFailure;
          }
          options_impl->weight_in_memory_map = map;
          return kLiteRtStatusOk;
        });
    return {};
  }
#endif  // LITERT_NO_ABSL

  /// Registers a `ScopedFile` that contains all external buffer groups.
  /// @param scoped_file The file containing the external weights.
  /// @param sections A map from group names to their respective sections in the
  /// file.
  /// @return An `Expected` object that is empty on success, or contains an
  /// error.
  Expected<void> SetExternalWeightScopedFile(ScopedFile& scoped_file,
                                             ScopedWeightSectionMap sections) {
    if (!scoped_file.IsValid()) {
      return Unexpected(Status::kErrorInvalidArgument,
                        "Scoped file handle must be valid");
    }
    if (sections.empty()) {
      return Unexpected(Status::kErrorInvalidArgument,
                        "At least one external buffer group must be provided");
    }
    for (const auto& [name, section] : sections) {
      if (section.length == 0) {
        return Unexpected(Status::kErrorInvalidArgument,
                          "Section length must be positive for group " + name);
      }
    }
    ScopedWeightSource::SectionMap runtime_sections;
    runtime_sections.reserve(sections.size());
    runtime_sections.insert(sections.begin(), sections.end());

    auto scoped_weight_source = std::make_unique<ScopedWeightSource>(
        std::move(scoped_file), std::move(runtime_sections));
    build_actions_.push_back(
        [scoped_weight_source_ptr = scoped_weight_source.release()](
            internal::RuntimeProxy* runtime, LiteRtOptions options) {
          auto* options_impl = reinterpret_cast<LiteRtOptionsT*>(options);
          if (!options_impl) {
            return kLiteRtStatusErrorRuntimeFailure;
          }
          options_impl->scoped_weight_source.reset(scoped_weight_source_ptr);
          return kLiteRtStatusOk;
        });
    return {};
  }

  /// Retrieves or lazy-creates the specified option type.
  ///
  /// This template method allows configuring any Option type without
  /// modifying litert_options.h.
  ///
  /// Example usage:
  ///   LITERT_ASSIGN_OR_RETURN(
  ///       auto& qnn_options,
  ///       options.GetOptions<litert::qualcomm::QualcommOptions>());
  template <typename OptionType>
  Expected<OptionType&> GetOptions() {
    // TODO(b/542809108): Consider switching to std::derived_from once LiteRT
    // migrates to C++20.
    static_assert(std::is_base_of_v<ConcreteOptionsBase, OptionType>,
                  "OptionType must inherit from litert::ConcreteOptionsBase");
    auto type_id = internal::GetOptionTypeId<OptionType>();
    auto it = option_slots_.find(type_id);
    if (it == option_slots_.end()) {
      LITERT_ASSIGN_OR_RETURN(auto new_option, OptionType::Create());
      auto ptr = std::make_unique<OptionType>(std::move(new_option));
      it = option_slots_.emplace(type_id, std::move(ptr)).first;
    }
    return *static_cast<OptionType*>(it->second.get());
  }

  /// Returns a reference to the GPU options.
  ///
  /// Use this to configure GPU-specific settings.
  Expected<GpuOptions&> GetGpuOptions() { return GetOptions<GpuOptions>(); }

  /// Returns a reference to the CPU options.
  ///
  /// Use this to configure CPU-specific settings.
  Expected<CpuOptions&> GetCpuOptions() { return GetOptions<CpuOptions>(); }

  /// Returns a reference to the Arm options.
  ///
  /// Use this to configure Arm-specific settings.
  Expected<arm::ArmOptions&> GetArmOptions() {
    return GetOptions<arm::ArmOptions>();
  }

  /// Returns a reference to the runtime options.
  Expected<RuntimeOptions&> GetRuntimeOptions() {
    return GetOptions<RuntimeOptions>();
  }

  /// Returns a reference to the compiler options.
  Expected<CompilerOptions&> GetCompilerOptions() {
    return GetOptions<CompilerOptions>();
  }

  /// @deprecated Use GetOptions<> API instead.
  Expected<qualcomm::QualcommOptions&> GetQualcommOptions() {
    return GetOptions<qualcomm::QualcommOptions>();
  }

  /// @deprecated Use GetOptions<> API instead.
  Expected<mediatek::MediatekOptions&> GetMediatekOptions() {
    return GetOptions<mediatek::MediatekOptions>();
  }

  /// @deprecated Use GetOptions<> API instead.
  Expected<google_tensor::GoogleTensorOptions&> GetGoogleTensorOptions() {
    return GetOptions<google_tensor::GoogleTensorOptions>();
  }

  /// @deprecated Use GetOptions<> API instead.
  Expected<intel_openvino::IntelOpenVinoOptions&> GetIntelOpenVinoOptions() {
    return GetOptions<intel_openvino::IntelOpenVinoOptions>();
  }

  /// @deprecated Use GetOptions<> API instead.
  Expected<samsung::SamsungOptions&> GetSamsungOptions() {
    return GetOptions<samsung::SamsungOptions>();
  }

 private:
  /// Builds the options object and creates a internal::LiteRtOptionsPtr object.
  ///
  /// This should be called after all setters have been invoked.
  static Expected<internal::LiteRtOptionsPtr> Build(
      const Options& options, const internal::EnvironmentHolder& env) {
    auto* runtime = env.runtime;
    LiteRtOptions litert_options;
    LITERT_RETURN_IF_ERROR(runtime->CreateOptions(&litert_options));

    if (options.lite_rt_hw_accelerator_set_.has_value()) {
      LITERT_RETURN_IF_ERROR(runtime->SetOptionsHardwareAccelerators(
          litert_options, *options.lite_rt_hw_accelerator_set_));
    }

    for (auto& litert_opaque_options : options.opaque_options_) {
      LITERT_RETURN_IF_ERROR(
          runtime->AddOpaqueOptions(litert_options, litert_opaque_options));
    }

    for (const auto& action : options.build_actions_) {
      LITERT_RETURN_IF_ERROR(action(runtime, litert_options));
    }

    for (const auto& [_, option_ptr] : options.option_slots_) {
      const char* identifier = nullptr;
      void* payload = nullptr;
      void (*payload_deleter)(void*) = nullptr;
      LITERT_RETURN_IF_ERROR(option_ptr->GetOpaqueOptionsData(
          &identifier, &payload, &payload_deleter));
      LiteRtOpaqueOptions opaque_opts = nullptr;
      LITERT_RETURN_IF_ERROR(runtime->CreateOpaqueOptions(
          identifier, payload, payload_deleter, &opaque_opts));
      LITERT_RETURN_IF_ERROR(
          runtime->AddOpaqueOptions(litert_options, opaque_opts));
    }

    return internal::LiteRtOptionsPtr(
        litert_options, internal::LiteRtDestroyOptionsDeleter{
                            runtime->runtime_c_api_->litert_destroy_options});
  }

  std::optional<LiteRtHwAcceleratorSet> lite_rt_hw_accelerator_set_;
  std::vector<LiteRtOpaqueOptions> opaque_options_;
  std::vector<
      std::function<LiteRtStatus(internal::RuntimeProxy*, LiteRtOptions)>>
      build_actions_;
  FlatHashMap<internal::OptionTypeId, std::unique_ptr<ConcreteOptionsBase>>
      option_slots_;
};

namespace internal {

/// Helper class to build a LiteRtOptionsPtr object from an Options object.
///
/// @internal This class should be only used by LiteRT internal APIs.
class LiteRtOptionsPtrBuilder {
 public:
  static Expected<internal::LiteRtOptionsPtr> Build(
      const Options& options, const internal::EnvironmentHolder& env) {
    return Options::Build(options, env);
  }
};

}  // namespace internal

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_COMPILATION_OPTIONS_H_

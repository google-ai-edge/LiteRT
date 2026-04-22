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

#ifndef ODML_LITERT_LITERT_CC_LITERT_COMPILATION_OPTIONS_H_
#define ODML_LITERT_LITERT_CC_LITERT_COMPILATION_OPTIONS_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_op_kernel.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/internal/litert_runtime_proxy.h"
#include "litert/cc/internal/scoped_file.h"
#include "litert/cc/internal/scoped_weight_source.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_custom_op_kernel.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/options/litert_compiler_options.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/cc/options/litert_google_tensor_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "litert/cc/options/litert_mediatek_options.h"
#include "litert/cc/options/litert_qualcomm_options.h"
#include "litert/cc/options/litert_runtime_options.h"
#include "litert/cc/options/litert_samsung_options.h"

namespace litert {

class CompiledModel;
class CompiledModelNext;

namespace internal {
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
  using ScopedWeightSectionMap =
      absl::flat_hash_map<std::string, ScopedWeightSection>;

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

  /// Registers a `ScopedFile` that contains all external buffer groups.
  /// @param scoped_file The file containing the external weights.
  /// @param sections A map from group names to their respective sections in the
  /// file.
  /// @return An `Expected` object that is empty on success, or contains an
  /// error.
  Expected<void> SetExternalWeightScopedFile(ScopedFile& scoped_file,
                                             ScopedWeightSectionMap sections);

  /// Returns a reference to the GPU options.
  ///
  /// Use this to configure GPU-specific settings.
  Expected<GpuOptions&> GetGpuOptions();

  /// Returns a reference to the CPU options.
  ///
  /// Use this to configure CPU-specific settings.
  Expected<CpuOptions&> GetCpuOptions();

  /// Returns a reference to the Qualcomm options.
  ///
  /// Use this to configure Qualcomm-specific settings.
  Expected<qualcomm::QualcommOptions&> GetQualcommOptions();

  /// Returns a reference to the MediaTek options.
  ///
  /// Use this to configure MediaTek-specific settings.
  Expected<mediatek::MediatekOptions&> GetMediatekOptions();

  /// Returns a reference to the Google Tensor options.
  ///
  /// Use this to configure Google Tensor-specific settings.
  Expected<google_tensor::GoogleTensorOptions&> GetGoogleTensorOptions();

  /// Returns a reference to the Intel OpenVINO options.
  ///
  /// Use this to configure Intel OpenVINO-specific settings.
  Expected<intel_openvino::IntelOpenVinoOptions&> GetIntelOpenVinoOptions();

  /// Returns a reference to the Samsung options.
  ///
  /// Use this to configure Samsung-specific settings.
  Expected<samsung::SamsungOptions&> GetSamsungOptions();

  /// Returns a reference to the runtime options.
  Expected<RuntimeOptions&> GetRuntimeOptions();

  /// Returns a reference to the compiler options.
  Expected<CompilerOptions&> GetCompilerOptions();

 private:
  /// Builds the options object and creates a internal::LiteRtOptionsPtr object.
  ///
  /// This should be called after all setters have been invoked.
  static Expected<internal::LiteRtOptionsPtr> Build(
      const Options& options, const internal::EnvironmentHolder& env);

  std::optional<LiteRtHwAcceleratorSet> lite_rt_hw_accelerator_set_;
  std::vector<LiteRtOpaqueOptions> opaque_options_;
  std::vector<
      std::function<LiteRtStatus(internal::RuntimeProxy*, LiteRtOptions)>>
      build_actions_;
  std::optional<GpuOptions> gpu_options_;
  std::optional<CpuOptions> cpu_options_;
  std::optional<qualcomm::QualcommOptions> qualcomm_options_;
  std::optional<mediatek::MediatekOptions> mediatek_options_;
  std::optional<google_tensor::GoogleTensorOptions> google_tensor_options_;
  std::optional<intel_openvino::IntelOpenVinoOptions> intel_openvino_options_;
  std::optional<samsung::SamsungOptions> samsung_options_;
  std::optional<RuntimeOptions> runtime_options_;
  std::optional<CompilerOptions> compiler_options_;
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

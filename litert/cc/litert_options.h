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
#include <memory>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_op_kernel.h"
#include "litert/c/litert_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_custom_op_kernel.h"
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
#include "litert/cc/internal/scoped_file.h"
#include "litert/cc/internal/scoped_weight_source.h"

namespace litert {

class CompiledModel;
class CompiledModelNext;

namespace tools {
struct ApplyPluginRun;
LiteRtStatus ApplyPlugin(std::unique_ptr<ApplyPluginRun> run);
}  // namespace tools

class Options : public internal::Handle<LiteRtOptions, LiteRtDestroyOptions> {
 public:
  friend class CompiledModel;
  friend class CompiledModelNext;
  friend LiteRtStatus tools::ApplyPlugin(
      std::unique_ptr<tools::ApplyPluginRun> run);

  // A weight section contains the offset and length of a contiguous region
  // inside a ScopedFile that backs a single external buffer group.
  // This map then provides the mapping between the group name and its
  // section.
  using ScopedWeightSectionMap =
      absl::flat_hash_map<std::string, ScopedWeightSection>;

  Options() = default;

  // Parameter `owned` indicates if the created CompilationOptions object
  // should take ownership of the provided `compilation_options` handle.
  explicit Options(LiteRtOptions compilation_options, OwnHandle owned)
      : internal::Handle<LiteRtOptions, LiteRtDestroyOptions>(
            compilation_options, owned) {}

  static Expected<Options> Create() {
    LiteRtOptions options;
    LITERT_RETURN_IF_ERROR(LiteRtCreateOptions(&options));
    return Options(options, OwnHandle::kYes);
  }

  Expected<void> SetHardwareAccelerators(HwAccelerators accelerators) {
    LITERT_RETURN_IF_ERROR(LiteRtSetOptionsHardwareAccelerators(
        Get(), static_cast<LiteRtHwAcceleratorSet>(accelerators)));
    return {};
  }

  Expected<void> SetHardwareAccelerators(HwAcceleratorSet accelerators) {
    LITERT_RETURN_IF_ERROR(LiteRtSetOptionsHardwareAccelerators(
        Get(), static_cast<LiteRtHwAcceleratorSet>(accelerators.value)));
    return {};
  }

  Expected<LiteRtHwAcceleratorSet> GetHardwareAccelerators() {
    LiteRtHwAcceleratorSet accelerators;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetOptionsHardwareAccelerators(Get(), &accelerators));
    return accelerators;
  }

  [[deprecated("Use the GetXXXOptions() methods instead.")]]
  Expected<void> AddOpaqueOptions(OpaqueOptions&& options) {
    LITERT_RETURN_IF_ERROR(LiteRtAddOpaqueOptions(Get(), options.Release()));
    return {};
  }

  Expected<OpaqueOptions> GetOpaqueOptions() {
    LiteRtOpaqueOptions options;
    LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptions(Get(), &options));
    return OpaqueOptions::WrapCObject(options, OwnHandle::kNo);
  }

  Expected<void> AddCustomOpKernel(const std::string& custom_op_name,
                                   int custom_op_version,
                                   const LiteRtCustomOpKernel& custom_op_kernel,
                                   void* custom_op_kernel_user_data = nullptr) {
    LITERT_RETURN_IF_ERROR(LiteRtAddCustomOpKernelOption(
        Get(), custom_op_name.c_str(), custom_op_version, &custom_op_kernel,
        custom_op_kernel_user_data));
    return {};
  }

  Expected<void> AddCustomOpKernel(CustomOpKernel& custom_op_kernel) {
    return AddCustomOpKernel(custom_op_kernel.OpName(),
                             custom_op_kernel.OpVersion(),
                             custom_op_kernel.GetLiteRtCustomOpKernel(),
                             static_cast<void*>(&custom_op_kernel));
  }

  // Binds an external memory buffer to a specific tensor in the model.
  // This function sets the tensor's allocation type to kTfLiteCustom, making it
  // appear as a constant tensor with a pre-allocated buffer.
  //
  // Note: `data` is owned by the caller and must outlive the lifetime of the
  // CompiledModel. `size_bytes` must match the tensor's expected size.
  Expected<void> AddExternalTensorBinding(const std::string& signature_name,
                                          const std::string& tensor_name,
                                          void* data, size_t size_bytes) {
    LITERT_RETURN_IF_ERROR(LiteRtAddExternalTensorBinding(
        Get(), signature_name.c_str(), tensor_name.c_str(), data, size_bytes));
    return {};
  }

  // Registers a ScopedFile that contains all external buffer groups.
  Expected<void> SetExternalWeightScopedFile(ScopedFile& scoped_file,
                                             ScopedWeightSectionMap sections);

  // Returns the reference to the GPU options. User will use this function to
  // set the GPU options.
  Expected<GpuOptions&> GetGpuOptions();

  // Returns the reference to the CPU options. User will use this function to
  // set the CPU options.
  Expected<CpuOptions&> GetCpuOptions();
  // Returns the reference to the Qualcomm options. User will use this function
  // to set the Qualcomm options.
  Expected<qualcomm::QualcommOptions&> GetQualcommOptions();

  // Returns the reference to the MediaTek options. User will use this function
  // to set the MediaTek options.
  Expected<mediatek::MediatekOptions&> GetMediatekOptions();

  // Returns the reference to the Google Tensor options. User will use this
  // function to set the Google Tensor options.
  Expected<google_tensor::GoogleTensorOptions&> GetGoogleTensorOptions();

  // Returns the reference to the Intel OpenVINO options. User will use this
  // function to set the Intel OpenVINO options.
  Expected<intel_openvino::IntelOpenVinoOptions&> GetIntelOpenVinoOptions();

  // LiteRtRuntimeOptions.
  Expected<RuntimeOptions&> GetRuntimeOptions();

  // LiteRtCompilerOptions.
  Expected<CompilerOptions&> GetCompilerOptions();

 private:
  // Builds the options object. This should be called after all the setters.
  // It's automatically called in CompiledModel::Create.
  Expected<void> Build();

  std::optional<GpuOptions> gpu_options_;
  std::optional<CpuOptions> cpu_options_;
  std::optional<qualcomm::QualcommOptions> qualcomm_options_;
  std::optional<mediatek::MediatekOptions> mediatek_options_;
  std::optional<google_tensor::GoogleTensorOptions> google_tensor_options_;
  std::optional<intel_openvino::IntelOpenVinoOptions> intel_openvino_options_;
  std::optional<RuntimeOptions> runtime_options_;
  std::optional<CompilerOptions> compiler_options_;
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_COMPILATION_OPTIONS_H_

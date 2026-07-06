// Copyright 2026 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_DISPATCH_DISPATCH_DELEGATE_KERNEL_FACADE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_DISPATCH_DISPATCH_DELEGATE_KERNEL_FACADE_H_

#include <memory>
#include <string>
#include <utility>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/runtime/dispatch/dispatch_delegate_kernel.h"
#include "litert/runtime/dispatch/dispatch_kernel_interface.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "tflite/c/c_api_types.h"

namespace litert::internal {

// A facade that defers selection of the specific dispatch
// delegate kernel implementation until Init(). That is the point in the
// execution flow where the necessary context and parameters (including
// bytecode) become available.
class DispatchKernelFacade : public DispatchKernelInterface {
 public:
  static std::unique_ptr<DispatchKernelFacade> Create(
      std::string&& graph_name, LiteRtEnvironmentOptions environment_options,
      LiteRtOptions options, LiteRtDispatchDeviceContext device_context) {
    return std::unique_ptr<DispatchKernelFacade>(new DispatchKernelFacade(
        std::move(graph_name), environment_options, options, device_context));
  }

  ~DispatchKernelFacade() override = default;

  TfLiteStatus Init(TfLiteOpaqueContext* context,
                    const TfLiteOpaqueDelegateParams* params) override;

  TfLiteStatus Prepare(TfLiteOpaqueContext* context,
                       TfLiteOpaqueNode* node) override {
    if (!kernel_) return kTfLiteError;
    return kernel_->Prepare(context, node);
  }

  TfLiteStatus Eval(TfLiteOpaqueContext* context,
                    TfLiteOpaqueNode* node) override {
    if (!kernel_) return kTfLiteError;
    return kernel_->Eval(context, node);
  }

  Expected<void> StartMetricsCollection(int detail_level) override {
    if (!kernel_) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Kernel not initialized");
    }
    return kernel_->StartMetricsCollection(detail_level);
  }

  Expected<LiteRtMetricsT> StopMetricsCollection() override {
    if (!kernel_) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Kernel not initialized");
    }
    return kernel_->StopMetricsCollection();
  }

 private:
  DispatchKernelFacade(std::string&& graph_name,
                               LiteRtEnvironmentOptions environment_options,
                               LiteRtOptions options,
                               LiteRtDispatchDeviceContext device_context)
      : graph_name_(std::move(graph_name)),
        environment_options_(environment_options),
        options_(options),
        device_context_(device_context) {}
  std::string graph_name_;
  LiteRtEnvironmentOptions environment_options_;
  LiteRtOptions options_;
  LiteRtDispatchDeviceContext device_context_;

  std::unique_ptr<DispatchKernelInterface> kernel_;
};

}  // namespace litert::internal

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_DISPATCH_DISPATCH_DELEGATE_KERNEL_FACADE_H_

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

#include "litert/runtime/dispatch/dispatch_kernel_facade.h"

#include <memory>
#include <string>
#include <utility>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/runtime/dispatch/dispatch_delegate_kernel.h"
#include "litert/runtime/dispatch/dispatch_kernel_interface.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"

namespace litert::internal {

namespace {

std::unique_ptr<DispatchKernelInterface> CreateKernel(
    std::string graph_name, LiteRtEnvironmentOptions environment_options,
    LiteRtOptions options, LiteRtDispatchDeviceContext device_context,
    TfLiteOpaqueContext* context, const TfLiteOpaqueDelegateParams* params) {

  // Fallback to standard dispatch delegate kernel if not instantiated.
  auto kernel_res = DispatchDelegateKernel::Create(
      std::move(graph_name), environment_options, options, device_context);
  if (kernel_res) {
    return std::unique_ptr<DispatchKernelInterface>(kernel_res->release());
  }

  LITERT_LOG(LITERT_ERROR, "%s", kernel_res.Error().Message().c_str());
  return nullptr;
}

}  // namespace

TfLiteStatus DispatchKernelFacade::Init(
    TfLiteOpaqueContext* context, const TfLiteOpaqueDelegateParams* params) {
  kernel_ = CreateKernel(graph_name_, environment_options_, options_,
                         device_context_, context, params);
  if (!kernel_) {
    return kTfLiteError;
  }

  return kernel_->Init(context, params);
}

}  // namespace litert::internal

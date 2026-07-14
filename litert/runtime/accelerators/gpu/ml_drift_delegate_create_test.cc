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

#include "litert/runtime/accelerators/gpu/ml_drift_delegate_create.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "litert/c/internal/litert_accelerator_registration.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_options.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_types.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"

namespace {

void DtorHelper(void*) {}

}  // namespace

extern "C" void LiteRtDeleteMockGpuDelegate(TfLiteDelegate* delegate) {
  if (!delegate) return;
  delete delegate;
}

litert::TfLiteDelegatePtr CreateMockGpuDelegate(
    litert::ml_drift::MlDriftDelegateOptionsPtr options,
    LiteRtEnvironment litert_env) {
  litert::TfLiteDelegatePtr delegate(new TfLiteDelegate(TfLiteDelegateCreate()),
                                     LiteRtDeleteMockGpuDelegate);
  return delegate;
}

TEST(MlDriftDelegateCreateTest,
     CreateDelegateNoDelegateOptionsNoGpuOptionsPayload) {
  LiteRtAccelerator accelerator;

  ASSERT_EQ(LiteRtCreateAccelerator(&accelerator), kLiteRtStatusOk);

  LiteRtOptions compilation_options = nullptr;
  ASSERT_EQ(LiteRtCreateOptions(&compilation_options), kLiteRtStatusOk);
  // Has opaque_options but no gpu options.
  int dummy = 0;
  LiteRtOpaqueOptions opaque_options = nullptr;
  ASSERT_EQ(
      LiteRtCreateOpaqueOptions("my key", &dummy, DtorHelper, &opaque_options),
      kLiteRtStatusOk);
  ASSERT_EQ(LiteRtAddOpaqueOptions(compilation_options, opaque_options),
            kLiteRtStatusOk);
  litert::TfLiteDelegatePtr delegate_ptr{nullptr, nullptr};
  LiteRtRuntimeContext* runtime_context = LrtGetRuntimeContext();
  ASSERT_EQ(litert::ml_drift::CreateDelegate(
                runtime_context, nullptr, accelerator,
                litert::ml_drift::GetGpuOptionsPayload(runtime_context,
                                                       compilation_options),
                nullptr, CreateMockGpuDelegate, delegate_ptr),
            kLiteRtStatusOk);
  LiteRtDestroyOptions(compilation_options);
  LiteRtDestroyAccelerator(accelerator);
}

TEST(MlDriftDelegateCreateTest, CreateDelegateNoGpuOptionsPayload) {
  LiteRtAccelerator accelerator;

  ASSERT_EQ(LiteRtCreateAccelerator(&accelerator), kLiteRtStatusOk);

  LiteRtOptions compilation_options = nullptr;
  ASSERT_EQ(LiteRtCreateOptions(&compilation_options), kLiteRtStatusOk);
  // Has opaque_options but no gpu options.
  int dummy = 0;
  LiteRtOpaqueOptions opaque_options = nullptr;
  ASSERT_EQ(
      LiteRtCreateOpaqueOptions("my key", &dummy, DtorHelper, &opaque_options),
      kLiteRtStatusOk);
  ASSERT_EQ(LiteRtAddOpaqueOptions(compilation_options, opaque_options),
            kLiteRtStatusOk);

  auto gpu_delegate_options = std::make_unique<MlDriftDelegateOptions>();
  litert::TfLiteDelegatePtr delegate_ptr{nullptr, nullptr};

  LiteRtRuntimeContext* runtime_context = LrtGetRuntimeContext();
  ASSERT_EQ(
      litert::ml_drift::CreateDelegate(
          runtime_context, nullptr, accelerator,
          litert::ml_drift::GetGpuOptionsPayload(runtime_context,
                                                 compilation_options),
          std::move(gpu_delegate_options), CreateMockGpuDelegate, delegate_ptr),
      kLiteRtStatusOk);

  LiteRtDestroyOptions(compilation_options);
  LiteRtDestroyAccelerator(accelerator);
}

TEST(MlDriftDelegateCreateTest, CreateDelegateNoDelegateOptionsNoPayload) {
  LiteRtAccelerator accelerator;

  ASSERT_EQ(LiteRtCreateAccelerator(&accelerator), kLiteRtStatusOk);

  litert::TfLiteDelegatePtr delegate_ptr{nullptr, nullptr};
  LiteRtRuntimeContext* runtime_context = LrtGetRuntimeContext();
  ASSERT_EQ(litert::ml_drift::CreateDelegate(
                runtime_context, nullptr, accelerator, nullptr, nullptr,
                CreateMockGpuDelegate, delegate_ptr),
            kLiteRtStatusOk);

  LiteRtDestroyAccelerator(accelerator);
}

TEST(MlDriftDelegateCreateTest, CreateDelegateNoPayload) {
  LiteRtAccelerator accelerator;

  ASSERT_EQ(LiteRtCreateAccelerator(&accelerator), kLiteRtStatusOk);
  auto gpu_delegate_options = std::make_unique<MlDriftDelegateOptions>();
  litert::TfLiteDelegatePtr delegate_ptr{nullptr, nullptr};
  LiteRtRuntimeContext* runtime_context = LrtGetRuntimeContext();
  ASSERT_EQ(
      litert::ml_drift::CreateDelegate(runtime_context, nullptr, accelerator,
                                       nullptr, std::move(gpu_delegate_options),
                                       CreateMockGpuDelegate, delegate_ptr),
      kLiteRtStatusOk);

  LiteRtDestroyAccelerator(accelerator);
}

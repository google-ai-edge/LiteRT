// Copyright (C) 2026 Samsung Electronics Co. LTD.
// SPDX-License-Identifier: Apache-2.0
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

#include "litert/vendors/samsung/dispatch/enn_manager.h"
#include <memory>
#include <utility>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/samsung/dispatch/enn_type.h"

// system library in mobile phone with exynos chip
static const char kEnnApiLibName[] = "libenn_public_api_cpp.so";

#define ENN_LOAD_API(LIB, SYM)                                              \
  if (auto symbol = LIB.LookupSymbol<void*>(#SYM); symbol.HasValue()) {     \
    api_->SYM =                                                             \
        reinterpret_cast<decltype(api_->SYM)>(std::move(symbol.Value())); \
  } else {                                                                  \
    LITERT_LOG(LITERT_WARNING, "Failed to load symbol %s: %s", #SYM,        \
               LIB.DlError());                                              \
    return kLiteRtStatusErrorDynamicLoading;                                \
  }

namespace litert::samsung {
EnnManager::EnnManager() : api_(std::make_unique<EnnManager::PublicApi>()) {}

Expected<EnnManager::UniquePtr> EnnManager::Create() {
#if !defined(__ANDROID__)
  return Error(kLiteRtStatusErrorRuntimeFailure,
               "Only support android platform");
#else
  auto enn_manager = std::unique_ptr<EnnManager>(new EnnManager());
  if (auto status = enn_manager->LoadEnnRuntimeLibrary(kEnnApiLibName);
      status != kLiteRtStatusOk) {
    return Error(status, "Fail to load enn runtime.");
  }

  return enn_manager;
#endif
}

const EnnManager::PublicApi& EnnManager::Api() const { return *api_.get(); }

EnnManager::~EnnManager() { Api().EnnDeinitialize(); }

LiteRtStatus EnnManager::LoadEnnRuntimeLibrary(absl::string_view path) {
  auto loading_lib = SharedLibrary::Load(path, RtldFlags::Default());
  LITERT_LOG(LITERT_INFO, "Loading from: %s", path.data());
  if (!loading_lib.HasValue() || !loading_lib->Loaded()) {
    LITERT_LOG(LITERT_INFO, "Failed to load enn runtime.");
    return kLiteRtStatusErrorDynamicLoading;
  }
  enn_runtime_lib_ = std::move(loading_lib.Value());

  ENN_LOAD_API(enn_runtime_lib_, EnnInitialize);
  ENN_LOAD_API(enn_runtime_lib_, EnnOpenModelFromMemory);
  ENN_LOAD_API(enn_runtime_lib_, EnnCreateBufferFromFdWithOffset);
  ENN_LOAD_API(enn_runtime_lib_, EnnBufferCommit);
  ENN_LOAD_API(enn_runtime_lib_, EnnGetBuffersInfo);
  ENN_LOAD_API(enn_runtime_lib_, EnnSetBufferByIndex);
  ENN_LOAD_API(enn_runtime_lib_, EnnReleaseBuffer);
  ENN_LOAD_API(enn_runtime_lib_, EnnExecuteModel);
  ENN_LOAD_API(enn_runtime_lib_, EnnBufferUncommit);
  ENN_LOAD_API(enn_runtime_lib_, EnnUnsetBuffers);
  ENN_LOAD_API(enn_runtime_lib_, EnnCloseModel);
  ENN_LOAD_API(enn_runtime_lib_, EnnDeinitialize);
  ENN_LOAD_API(enn_runtime_lib_, EnnAllocateAllBuffers);

  if (api_->EnnInitialize() != ENN_RET_SUCCESS) {
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

}  // namespace litert::samsung

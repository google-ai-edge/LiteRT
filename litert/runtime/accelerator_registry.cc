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

#include "litert/runtime/accelerator_registry.h"

#include <cstddef>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/runtime/accelerator.h"

namespace litert::internal {
namespace {

const char* GetAcceleratorName(LiteRtAcceleratorT* accelerator) {
  const char* name = nullptr;
  if (accelerator && accelerator->GetName) {
    accelerator->GetName(accelerator, &name);
  }
  return name;
}

}  // namespace

void AcceleratorRegistry::DestroyAccelerator(LiteRtAcceleratorT* accelerator) {
  ABSL_LOG(INFO) << "DestroyAccelerator: ptr=" << accelerator
                 << ", name=" << GetAcceleratorName(accelerator);
  if (accelerator && accelerator->ReleaseData) {
    accelerator->env = nullptr;
    accelerator->ReleaseData(accelerator->data);
  }
  delete accelerator;
}

Expected<LiteRtAcceleratorT*> AcceleratorRegistry::RegisterAccelerator(
    Ptr accelerator) {
  ABSL_LOG(INFO) << "RegisterAccelerator: ptr=" << accelerator.get()
                 << ", name=" << GetAcceleratorName(accelerator.get());
  if (!accelerator) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Cannot register a null accelerator.");
  }
  accelerators_.push_back(std::move(accelerator));
  return accelerators_.back().get();
}

Expected<LiteRtAcceleratorT*> AcceleratorRegistry::Get(LiteRtParamIndex idx) {
  if (idx >= size()) {
    return Error(kLiteRtStatusErrorNotFound, "Cannot find accelerator.");
  }
  return accelerators_[idx].get();
}

Expected<LiteRtParamIndex> AcceleratorRegistry::FindAcceleratorIndex(
    LiteRtAcceleratorT* accelerator) {
  for (size_t idx = 0; idx < accelerators_.size(); ++idx) {
    if (accelerator == accelerators_[idx].get()) {
      return static_cast<LiteRtParamIndex>(idx);
    }
  }
  return Error(kLiteRtStatusErrorNotFound,
               "The accelerator is not registered in the LiteRT environment.");
}

void AcceleratorRegistry::TakeOwnershipOfSharedLibrary(SharedLibrary lib) {
  accelerator_shared_libraries_.push_back(std::move(lib));
}

}  // namespace litert::internal
